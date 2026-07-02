"""GIP step-1 success-rate eval (single entry, config-gated).

Reuses eval.py's env/dataset machinery; the policy is chosen by ONE knob:

    gip_eval.mode = bc | guided | planning      (default: bc)

  bc        action head run directly as a reactive policy (action head only)
  guided    intention-guided planning -- action head warm-starts CEM, the
            world-model state head rolls candidates to the goal (JOINT metric)
  planning  plain world-model CEM planning (state head only; LeWM baseline)

Example:
  python eval_gip.py --config-name pusht policy=gip_pusht +gip_eval.mode=guided
"""

import stable_worldmodel.data.formats.hdf5  # HDF5 self-registers on import
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm

import gip


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    mode = cfg.get("gip_eval", {}).get("mode", "bc")
    assert cfg.policy != "random", "set policy=<gip_run_name>"
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "horizon*action_block must be <= eval_budget"

    # -- env + data context (shared helpers)
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)

    # -- model + policy (mode is the single switch)
    # mode=gcidm loads its own frozen-LeWM + GCIDMHead, not the JEPA-checkpoint loader
    if mode == "gcidm":
        policy = gip.build_policy(cfg, None, None, process, transform)
    else:
        model, adim = gip.load_gip_model(cfg.policy, epoch=cfg.get("ckpt_epoch", None))
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        # multi-task ckpt: select this task's conditioning vector + set the env<->trained action pad boundary
        if getattr(model, "task_proj", None) is not None:
            d_raw = int(dataset.get_dim("action"))
            f = int(cfg.plan_config.action_block)
            # registry name from the eval dataset name (e.g. "pusht" in "pusht_expert_train")
            dn = cfg.eval.dataset_name
            model.eval_task = next((t for t in model.mt_task_names if t in dn), dn)
            model.eval_action_pad = (d_raw, f)
            adim = f * d_raw
            print(f"[GIP] multi-task: eval_task={model.eval_task} of {model.mt_task_names}  "
                  f"action pad {d_raw}x{f} -> trained block")
        policy = gip.build_policy(cfg, model, adim, process, transform)
    print(f"[GIP] eval mode={mode} policy={type(policy).__name__}")

    results_path = Path(swm.data.utils.get_cache_dir(), "gip_eval", mode, cfg.policy)
    results_path.mkdir(parents=True, exist_ok=True)

    world.set_policy(policy)
    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts,
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=episodes,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    dt = time.time() - t0
    print(f"==== GIP {mode} RESULTS ====")
    print(metrics)

    with (results_path / f"{mode}_{cfg.policy}_results.txt").open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\n==== RESULTS ({mode}) ====\nmetrics: {metrics}\ntime: {dt:.1f}s\n")
    print("wrote", results_path)


if __name__ == "__main__":
    run()
