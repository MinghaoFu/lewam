"""
Entry point. Builds and runs experiments, resolving checkpoints from wandb when asked.

Run:  python main.py run=<arm> wandb.entity=<team>
      python main.py run=<arm> wandb.mode=disabled     # no wandb

Post-hoc eval (load= with eval-only tasks) scores a finished training run and attaches the
result to it, restoring the algorithm and dataset sections from the run's stored config:

      python main.py run=<arm> load=<run_id|ckpt.pt> experiment.tasks=[validation]
      python main.py run=b601_irl load=<run_id|ckpt.pt>   # an arm naming no algorithm
"""

import getpass
import os
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

# must be set before anything imports torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_wandb_scratch = Path(tempfile.gettempdir()) / f"lewam-wandb-{getpass.getuser()}"
for _var, _sub in (("WANDB_CACHE_DIR", "cache"), ("WANDB_DATA_DIR", "data"), ("WANDB_DIR", "run")):
    os.environ.setdefault(_var, str(_wandb_scratch / _sub))
    Path(os.environ[_var]).mkdir(parents=True, exist_ok=True)


def _restore_source_sections(cfg, eval_source, task_overrides, checkpoint_path=None):
    """Replace cfg.algorithm and cfg.dataset with the source run's stored versions, keeping
    any explicit `algorithm.*` / `dataset.*` CLI overrides on top (e.g. attn_backend for the
    eval machine). A section whose GROUP was named on the CLI (`dataset=<config>`) is left
    alone: that is an explicit request to score the model against something else.

    The already-downloaded checkpoint is the preferred source: it embeds the config with
    declaration order intact (`config_yaml`), which wandb's alphabetized storage loses.
    """
    from lewam_real_robot.utils.ckpt_utils import config_from_checkpoint, fetch_run_config

    named_groups = {o.split("=")[0] for o in task_overrides}
    sections = [s for s in ("algorithm", "dataset") if s not in named_groups]
    if not sections:
        return

    run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{eval_source}"
    try:
        stored = config_from_checkpoint(checkpoint_path) if checkpoint_path else None
        if stored is None:
            stored = fetch_run_config(run_path)
    except Exception as err:  # noqa: BLE001 - wandb raises many flavors; eval can proceed
        missing = [s for s in sections if s not in cfg]
        if missing:
            raise ValueError(
                f"could not fetch the stored config of {run_path} ({err}), and this config "
                f"composes no {', '.join(missing)} section of its own to fall back on. Name "
                f"one on the CLI ({' '.join(f'{s}=<config>' for s in missing)}), or select a "
                f"`run=` that composes one."
            ) from err
        print(f"WARNING: could not fetch stored config of {run_path} ({err}); "
              f"using the locally composed {', '.join(sections)} config")
        return

    with open_dict(cfg):
        for section in sections:
            if section not in stored:
                continue
            # the dotlist merges into the section itself, so the `<section>.` prefix must
            # come off: left on, it merges as a nested key of that name and changes nothing
            overrides = [
                o.lstrip("+").split(".", 1)[1] for o in task_overrides
                if not o.startswith("~") and o.lstrip("+").startswith(f"{section}.")
            ]
            cfg[section] = OmegaConf.merge(
                stored[section], OmegaConf.from_dotlist(overrides)
            )


def _resumed_eval_run_id(cfg):
    """The wandb run an interrupted real-robot eval left in its rollout dir, so resuming it
    scores one run rather than two. `eval.resume` names that dir explicitly.
    """
    eval_cfg = cfg.get("eval") or {}
    if not eval_cfg.get("resume"):
        return None
    from lewam_real_robot.experiments.eval.lerobot import read_session_log

    return read_session_log(eval_cfg["resume"]).get("wandb_run_id")


def run_local(cfg: DictConfig, checkpoint_path, eval_source=None):
    from accelerate.utils import set_seed

    from lewam_real_robot.experiments import build_experiment
    from lewam_real_robot.utils.ckpt_utils import source_run_name
    from lewam_real_robot.utils.wandb import finish_wandb, init_wandb, update_run_summary

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
    with open_dict(cfg):
        for group in ("experiment", "dataset", "algorithm"):
            if choices.get(group) is not None:
                cfg[group]._name = choices[group]

    if eval_source:
        _restore_source_sections(cfg, eval_source, hydra_cfg.overrides.task,
                                 checkpoint_path=checkpoint_path)

    output_dir = Path(hydra_cfg.runtime.output_dir)
    print(f"Outputs will be saved to: {output_dir}")
    latest = output_dir.parents[1] / "latest-run"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(output_dir, target_is_directory=True)
    except OSError as err:
        # some filesystems (NFS homes) refuse symlinks
        print(f"note: could not update {latest}: {err}")

    seed = cfg.get("seed", None)
    if seed is not None:
        # must precede build_experiment so weight init is covered. Every rank takes the
        # same seed: DDP requires identical init, and Accelerate shards the sampler by rank
        set_seed(int(seed))

    experiment = build_experiment(cfg, output_dir, checkpoint_path)

    is_main = os.environ.get("RANK", "0") == "0"
    if is_main:
        source_name = source_run_id = None
        if eval_source:
            source_name = source_run_name(
                f"{cfg.wandb.entity}/{cfg.wandb.project}/{eval_source}", checkpoint_path
            )
            source_run_id = _resumed_eval_run_id(cfg)
        init_wandb(cfg, eval_source=eval_source, source_name=source_name,
                   resume_id=source_run_id)
    try:
        for task in cfg.experiment.tasks:
            experiment.exec_task(task)
    except BaseException:
        # under a multirun the next arm starts in this same process: a run left open here
        # swallows that arm's wandb.init, sending its metrics and checkpoints into this run
        if is_main:
            finish_wandb(exit_code=1)
        raise

    result = getattr(experiment, "last_eval_result", None)
    if is_main and eval_source and result is not None:
        prefix = cfg.get("posthoc_prefix", "posthoc")
        metrics = {f"{prefix}/{k}": v for k, v in result.metrics.items()}
        step = getattr(experiment, "last_eval_step", None)
        if step is not None:
            metrics[f"{prefix}/checkpoint_step"] = step
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{eval_source}"
        # the write-back goes through the wandb PUBLIC api, which an offline worker cannot
        # reach. Losing it must not lose the eval too: the numbers are already computed and
        # logged, and under a multirun an exception here would also strand every later arm
        try:
            update_run_summary(run_path, metrics)
            print(f"attached {result.metrics} to {run_path} under '{prefix}/*'")
        except Exception as err:  # noqa: BLE001 - wandb raises many flavors
            print(f"POSTHOC_UNATTACHED {eval_source} {prefix} {err}")
            print(f"POSTHOC_METRICS {eval_source} {prefix} {result.metrics}")
    if is_main:
        finish_wandb()


def resolve_run_refs(cfg, arm=None):
    """The run's `resume` and `load` references, falling back from the scalar keys to the
    per-run maps. The maps are for multiruns, where one sweep shares every override:
    `load_map`/`resume_map` are passed once and each job picks its own entry.

    An entry is matched against the wandb run name and against `arm`, the `run=` config the
    job was launched with. Both are needed: a jobspec that also passes `name=<label>` keys
    its map by the arm, since that is the only name its sweep line mentions, and matching
    `name` alone missed silently -- which reads as a deliberate fresh run, not an error.
    """
    keys = [k for k in (cfg.get("name", None), arm) if k]

    def from_map(map_key):
        entries = cfg.get(map_key) or {}
        return next((entries[k] for k in keys if k in entries), None)

    resume = cfg.get("resume", None) or from_map("resume_map")
    load = cfg.get("load", None) or from_map("load_map")
    return resume, load


@hydra.main(version_base=None, config_path="configs", config_name="base")
def run(cfg: DictConfig):
    from lewam_real_robot.utils.ckpt_utils import download_latest_checkpoint, is_run_id, split_alias

    arm = hydra.core.hydra_config.HydraConfig.get().runtime.choices.get("run")
    if arm is None:
        arms = sorted(p.stem for p in (Path(__file__).parent / "configs" / "run").glob("*.yaml"))
        raise ValueError(
            "no run selected. Pass `run=<arm>`, one of:\n  " + "\n  ".join(arms)
        )

    resume, load = resolve_run_refs(cfg, arm)
    # `<run id>:best` asks for that run's best-scoring checkpoint
    load_ref, load_alias = split_alias(load) if load else (None, None)
    resume_ref, resume_alias = split_alias(resume) if resume else (None, None)
    posthoc = bool(load) and "training" not in cfg.experiment.tasks

    if "name" not in cfg and not resume and not posthoc:
        raise ValueError(
            "no run name. Select a run with `run=<arm>`, which names "
            "the wandb run after it, or pass `+name=<name>` explicitly."
        )
    if cfg.wandb.mode == "online" and not cfg.wandb.get("entity", None):
        raise ValueError(
            "wandb.mode=online requires wandb.entity (your wandb team/user name). "
            "Set wandb.entity=[entity], or use wandb.mode=offline / wandb.mode=disabled."
        )
    if cfg.wandb.project is None:
        cfg.wandb.project = Path(__file__).parent.name

    # resume WITH a local load path takes the weights from that file: the offline-job
    # resume, where wandb artifacts are unreachable
    if resume and load and is_run_id(load_ref):
        raise ValueError(
            "resume= with load=<run id> is ambiguous. Pass load=<local .pt> to supply "
            "the weights for the resumed run, or drop load= to fetch its own artifact."
        )

    # the model lives on the inference server, so this box neither downloads nor builds one
    remote_eval = posthoc and bool(cfg.eval.get("server"))
    if remote_eval and not cfg.eval.get("run") and not is_run_id(str(load_ref)):
        raise ValueError(
            f"eval.server is set but load='{load}' is a local checkpoint, which an "
            "inference server cannot fetch. Pass load=<run_id>, or name the run for the "
            "server directly with eval.run=<entity>/<project>/<run_id>."
        )

    checkpoint_path = None
    if load and not is_run_id(load_ref):
        checkpoint_path = load_ref
    load_id = resume_ref or (load_ref if load and is_run_id(load_ref) else None)
    if load_id and not remote_eval and checkpoint_path is None:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        checkpoint_path = download_latest_checkpoint(
            run_path, Path("outputs/downloaded"), alias=resume_alias or load_alias
        )

    cfg.resume = resume_ref

    eval_source = None
    if posthoc:
        eval_source = load_ref if is_run_id(load_ref) else _baked_source_run_id(checkpoint_path)
        if eval_source is None:
            print("WARNING: checkpoint has no source_run_id; eval will not attach to a training run")

    run_local(cfg, checkpoint_path, eval_source=eval_source)


def _baked_source_run_id(checkpoint_path):
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("source_run_id")


if __name__ == "__main__":
    run()
