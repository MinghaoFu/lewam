from pathlib import Path
import os

import torch


def save_checkpoint(model, optimizer, out_dir, step, save_intermittent=False, best=False,
                    config=None, final=False):
    """Save weights + optimizer + step to out_dir/model.pt. If a wandb run is active, also
    log it as a 'model' artifact so `resume=`/`load=<run id>` can fetch it later.

    By default only the newest artifact version plus one backup stay on wandb (older
    versions are pruned after each upload); `save_intermittent=True` keeps every version.

    `best=True` marks this as the best-scoring weights the run has produced: it writes
    best.pt instead of model.pt and tags the artifact version `best`, which pruning never
    deletes. That is what keeps a pre-overfit model reachable after training has run past
    it -- the newest checkpoint alone is whatever the run happened to end on.

    `final=True` marks the run's last checkpoint: the artifact version is tagged `final`,
    which pruning never deletes, so consumers can address the end-of-training weights as
    `model-<run id>:final` without knowing when the run stopped.

    `config` (the run's root OmegaConf) is embedded as `config_yaml`, a YAML string, so
    the checkpoint alone can rebuild the exact model. This exists because KEY ORDER is
    semantic (GenericViTPredictor's token and rope layouts follow declaration order) and
    wandb's config storage alphabetizes keys -- every rebuild through the wandb config ran
    a silently scrambled model until ckpt_utils learned to reorder it. YAML preserves
    order; loaders prefer this over the wandb round trip (ckpt_utils.config_from_checkpoint).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("best.pt" if best else "model.pt")
    run_id = _run_id()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "source_run_id": run_id,
        "norm_stats": getattr(model, "norm_stats", None),
        "norm_method": getattr(model, "norm_method", None),
    }
    if config is not None:
        # never fail a checkpoint over config serialization; resolve what the process
        # can (hydra: refs need the hydra runtime, present during training)
        try:
            from omegaconf import OmegaConf
            try:
                payload["config_yaml"] = OmegaConf.to_yaml(config, resolve=True)
            except Exception:  # noqa: BLE001 - interpolation errors outside hydra
                payload["config_yaml"] = OmegaConf.to_yaml(config, resolve=False)
        except Exception as err:  # noqa: BLE001
            print(f"checkpoint: could not embed the run config ({err}); saving without it")
    torch.save(payload, path)
    print(f"saved {'best ' if best else ''}checkpoint: {path}"
          + (f"  (run {run_id})" if run_id else ""))
    aliases = ["best"] if best else (["final"] if final else None)
    _log_artifact(path, prune=not save_intermittent, aliases=aliases)
    return path


def run_ckpt_dir(fallback):
    """Stable checkpoint dir for the active wandb run: outputs/checkpoints/<run id>,
    CWD-relative like the artifact download dir. Restarts and resumes of a run land in
    the same dir and overwrite one model.pt -- saving into the per-process hydra run dir
    instead leaves a full checkpoint behind for every restart. Falls back to the given
    dir when wandb is off (no stable id to key on).
    """
    run_id = _run_id()
    return Path("outputs/checkpoints") / run_id if run_id else Path(fallback)


def load_checkpoint(model, optimizer, path):
    """Load weights (and optimizer state, if an optimizer is given) from a .pt checkpoint.
    Returns the step it was saved at, so training can resume the counter.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("norm_stats") is not None:
        model.norm_stats = ckpt["norm_stats"]
    if ckpt.get("norm_method") is not None:
        model.norm_method = ckpt["norm_method"]
    return ckpt.get("step", 0)


def _run_id():
    from .wandb import wandb_enabled

    if not wandb_enabled():
        return None
    import wandb

    return wandb.run.id if wandb.run is not None else None


def _log_artifact(path, prune=True, aliases=None):
    # offline runs relay model.pt themselves; recording artifacts into the offline log
    # would make the later `wandb sync` drag every checkpoint version through the relay
    if os.environ.get("LEWAM_SKIP_ARTIFACTS"):
        return
    run_id = _run_id()
    if run_id is None:
        return
    import wandb

    artifact = wandb.Artifact(name=f"model-{run_id}", type="model")
    artifact.add_file(str(path))
    # an alias is unique within a collection: tagging a new version `best` moves the
    # tag off the previous holder, which drops back into the prunable pool
    wandb.log_artifact(artifact, aliases=aliases)
    if prune:
        _prune_old_versions(run_id)


def _prune_old_versions(run_id, keep=1):
    """Delete committed versions of this run's model artifact beyond the `keep` newest,
    never touching a version tagged `best` or `final`.

    The version logged just above uploads asynchronously and is not committed yet, so
    it is never counted here: the newest committed version always survives, and an
    upload killed uncleanly can only ever lose the in-flight version. A run therefore
    holds two checkpoints at most -- the newest plus the best-scoring one, when it is
    no longer the newest. `keep=1` rather than 2 because a pi05 checkpoint is ~20GB and
    the spare backup costs more wandb quota than it is worth.

    The `best` tag moves to a new version only once that version commits, so right after
    a best upload the previous holder can still look protected and survive one extra
    round. It is pruned on the next save.
    """
    import wandb

    try:
        # fresh Api() (it caches listings) + run-scoped lookup, which tolerates the
        # collection not existing yet on the very first save
        run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{run_id}")
        versions = [
            a for a in run.logged_artifacts()
            if a.type == "model" and a.state == "COMMITTED"
            and not {"best", "final"} & set(a.aliases or [])
        ]
        versions.sort(key=lambda a: int(a.version[1:]), reverse=True)
        for stale in versions[keep:]:
            stale.delete(delete_aliases=True)
    except Exception as e:
        print(f"checkpoint artifact pruning failed (non-fatal): {e}")
