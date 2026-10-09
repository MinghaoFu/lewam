"""
Goal-reaching and task-completion of a LeWAM checkpoint.

Runs are configured under configs/eval/base.yaml and the respective task config, but can be overridden on the command line
  python scripts/eval_lewam.py --config-name pusht policy=<run_name>
  python scripts/eval_lewam.py --config-name pusht policy=<run_name> eval.mode=lewam_plan eval.plan_mode=grad \
      eval.grad_all_k=true eval.grad_tr=0.01

eval.mode=lewam_policy runs LeWAM reactively
eval.mode=lewam_plan plans through its dynamics with eval.plan_mode

Check the README for more details on eval protocols and modes
"""

import importlib
import time
from pathlib import Path

import gymnasium
import hydra
import numpy as np
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registers the HDF5 format
import torch
from omegaconf import DictConfig, OmegaConf

from lewam import views
from lewam.eval.build import build_policy
from lewam.eval.episodes import get_dataset, img_transform, sample_eval_episodes
from lewam.eval.loaders import load_lewam

# register non-standard envs
ENV_MODULES = {
    "swm/RoboMimicGC-v0": "lewam.envs.robomimic_gc_env",
    "swm/OGBCube-v0": "lewam.envs.cube_env",
    "swm/DexMimicGen-v0": "lewam.envs.dexmimicgen_env",
    "swm/ReacherVisibleTargetDMControl-v0": "lewam.envs.reacher_visible_target_env",
    "swm/OGBScene-v0": "lewam.envs.scene_env",
    "swm/OGBPuzzle-v0": "lewam.envs.puzzle_env",
    "swm/PointMaze-v0": "lewam.envs.pointmaze_env",
}
# eval-config callables that set the goal in the env; the others set the episode's start state
GOAL_METHODS = {"_set_goal_state", "_set_goal_proprio", "set_target_pos", "set_goal_effector", "set_target_qpos",
                "set_cube_target_pos", "set_target_button_state", "set_target_drawer_pos", "set_target_window_pos",
                "set_target_button_states"}


class BudgetTruncate(gymnasium.Wrapper):
    """Truncate the env at its own step budget, set per env after the World is built."""

    budget: int | None = None

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1
        if self.budget is not None and self.t >= self.budget:
            truncated = True
        return obs, reward, terminated, truncated, info


def register_env(env_name: str) -> None:
    """Import the module that registers env_name with gym, if it is one of ours."""
    module = ENV_MODULES.get(str(env_name))
    if module:
        importlib.import_module(module)


def set_episode_budgets(world: swm.World, goal_offsets: list[int]) -> None:
    """Give each env twice its own episode's goal offset as its step budget."""
    assert world.num_envs == len(goal_offsets), f"{world.num_envs} envs for {len(goal_offsets)} episodes"
    for env, offset in zip(world.envs.envs, goal_offsets):
        while not isinstance(env, BudgetTruncate):
            env = env.env
        env.budget = 2 * int(offset)


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="pusht")
def main(cfg: DictConfig):
    register_env(cfg.world.env_name)
    eval_cfg = cfg.eval

    dataset = get_dataset(eval_cfg.dataset_name, list(cfg.dataset.keys_to_cache), cfg.cache_dir)
    episodes, starts, goal_offsets = sample_eval_episodes(
        dataset, eval_cfg.num_eval, cfg.seed, eval_cfg.goal_offset_steps, full_traj=eval_cfg.full_traj,
        random_start=eval_cfg.random_start, min_episode_len=eval_cfg.min_episode_len
    )
    if eval_cfg.full_traj:
        eval_cfg.eval_budget = 2 * int(max(goal_offsets))
        print(f"[full-traj] {len(goal_offsets)} episodes, offsets {min(goal_offsets)}..{max(goal_offsets)}, "
              f"per-env budgets {2 * min(goal_offsets)}..{eval_cfg.eval_budget}", flush=True)
    cfg.world.max_episode_steps = 2 * eval_cfg.eval_budget

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cfg = load_lewam(cfg.policy, which=cfg.seq_which)
    model = model.to(device).eval().requires_grad_(False)
    transform = {"pixels": img_transform(eval_cfg.img_size), "goal": img_transform(eval_cfg.img_size)}
    policy = build_policy(eval_cfg, model, model_cfg, cfg.seed, transform, goal_offsets)
    print(f"[eval] eval mode={eval_cfg.mode} policy={type(policy).__name__}")

    extra_views = views.extra_cameras(views.columns(model_cfg.get("views")))

    world = swm.World(
        **cfg.world, image_shape=(224, 224), extra_wrappers=[BudgetTruncate] if eval_cfg.full_traj else None,
        **({"extra_views": extra_views} if extra_views else {})
    )
    if eval_cfg.full_traj:
        set_episode_budgets(world, goal_offsets)

    callables = OmegaConf.to_container(eval_cfg.callables, resolve=True)
    if eval_cfg.task_only:
        assert eval_cfg.full_traj, "task_only is the start-to-finish protocol: needs eval.full_traj=true"
        callables = [c for c in callables if c.get("method") not in GOAL_METHODS]
        print(f"[eval] task_only: no goal set in the env -> success = env.is_success()['task'] only; "
              f"callables {[c.get('method') for c in callables]}", flush=True)

    results_path = Path(swm.data.utils.get_cache_dir(), "eval", eval_cfg.mode, cfg.policy)
    results_path.mkdir(parents=True, exist_ok=True)
    world.set_policy(policy)
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts,
        goal_offset=np.asarray(goal_offsets) if eval_cfg.full_traj else eval_cfg.goal_offset_steps,
        eval_budget=eval_cfg.eval_budget,
        episodes_idx=episodes,
        callables=callables,
        video=results_path,
    )
    wall_s = time.time() - start_time
    print(f"==== {eval_cfg.mode} RESULTS ====")
    print(metrics)

    with (results_path / f"{eval_cfg.mode}_{cfg.policy}_results.txt").open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\n==== RESULTS ({eval_cfg.mode}) ====\nmetrics: {metrics}\ntime: {wall_s:.1f}s\n")
    print("wrote", results_path)


if __name__ == "__main__":
    main()
