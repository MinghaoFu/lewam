import stable_worldmodel.data.formats.hdf5  # HDF5 self-registers on import
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    # use .lance table if present (must match the training format)
    _nm = str(dataset_name)[:-6] if str(dataset_name).endswith(".lance") else str(dataset_name)
    _lance = dataset_path / (_nm + ".lance")
    if _lance.exists():
        return swm.data.LanceDataset(path=str(_lance), keys_to_cache=cfg.dataset.keys_to_cache)
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


# WF8 env modules register their gym ids on import (side effect); nothing on this eval path
# imported them (lewam/envs/__init__.py is deliberately empty). Same lookup as eval_gip.py.
_ENV_MODULES = {
    "swm/RoboMimicGC-v0": "lewam.envs.robomimic_gc_env",
    "swm/DexMimicGen-v0": "lewam.envs.dexmimicgen_env",
    "swm/PointMaze-v0": "lewam.envs.pointmaze_env",
    "swm/RoboMimic-v0": "lewam.envs.robomimic_env",
    "swm/ReacherVisibleTargetDMControl-v0": "lewam.envs.reacher_visible_target_env",
}


def _register_env(env_name):
    mod = _ENV_MODULES.get(str(env_name))
    if mod:
        import importlib

        importlib.import_module(mod)


@hydra.main(version_base=None, config_path="../configs/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    _register_env(cfg.world.env_name)
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # episodes eligible for the draw: long enough for the goal offset and at least min_episode_len
    # frames (the same length bar for every goal offset makes one seed pick the same episodes for all)
    step_idx = dataset.get_col_data("step_idx")
    episode_of_row = dataset.get_col_data(col_name)
    length_of_episode = {ep_id: episode_len[i] for i, ep_id in enumerate(ep_indices)}
    frames_per_row = np.array([length_of_episode[ep_id] for ep_id in episode_of_row])
    eligible_row = (max_start_per_row >= 0) & (frames_per_row >= int(cfg.eval.get("min_episode_len", 0)))
    g = np.random.default_rng(cfg.seed)
    if not bool(cfg.eval.get("random_start", True)):
        first_frames = np.nonzero((step_idx == 0) & eligible_row)[0]
        random_episode_indices = np.sort(first_frames[g.choice(len(first_frames), size=cfg.eval.num_eval, replace=False)])
    else:
        valid_indices = np.nonzero((step_idx <= max_start_per_row) & eligible_row)[0]
        print(len(valid_indices), "valid starting points found for evaluation.")
        random_episode_indices = np.sort(valid_indices[g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    print("eval episodes:", np.asarray(eval_episodes).tolist(), "starts:", np.asarray(eval_start_idx).tolist())

    # pairing guard: the LeWAM harness must draw exactly these (episode, start) tuples for the same flags
    from lewam.models.gip import sample_eval_episodes
    from omegaconf import OmegaConf as _OC
    paired_cfg = _OC.create({"seed": int(cfg.seed),
                             "eval": {"num_eval": int(cfg.eval.num_eval), "goal_offset_steps": int(cfg.eval.goal_offset_steps)},
                             "gip_eval": {"random_start": bool(cfg.eval.get("random_start", True)),
                                          "min_episode_len": int(cfg.eval.get("min_episode_len", 0))}})
    paired_episodes, paired_starts, _ = sample_eval_episodes(paired_cfg, dataset)
    assert list(map(int, paired_episodes)) == list(map(int, eval_episodes)) and \
        list(map(int, paired_starts)) == list(map(int, eval_start_idx)), \
        "LeWM eval episodes differ from the LeWAM harness draw: the comparison would not be paired"
    print("pairing guard: the LeWAM harness draws the same episodes and starts")

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    timer = None
    if cfg.policy != "random":
        from lewam.eval_timing import PlanTimer
        # plan phase = the solver call (CEM sampling, dynamics rollouts, cost, all iterations);
        # the goal and observation encodings happen inside it in the world model's cost
        timer = PlanTimer(policy, world.num_envs, torch.cuda.is_available(), plan_attr="solver",
                          encode_attr="solver.model.encode", horizon_blocks=int(cfg.plan_config.horizon))

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()

    print(metrics)
    if timer is not None:
        import json
        timing = dict(policy=str(cfg.policy), seed=int(cfg.seed), num_envs=int(world.num_envs), planner="cem",
                      horizon_blocks=int(cfg.plan_config.horizon), receding_blocks=int(cfg.plan_config.receding_horizon),
                      action_block=int(cfg.plan_config.action_block), cem_samples=int(cfg.solver.num_samples),
                      cem_iters=int(cfg.solver.n_steps), cem_topk=int(cfg.solver.topk),
                      goal_offset_steps=int(cfg.eval.goal_offset_steps), eval_budget=int(cfg.eval.eval_budget),
                      random_start=bool(cfg.eval.get("random_start", True)), min_episode_len=int(cfg.eval.get("min_episode_len", 0)),
                      gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                      wall_total_s=float(end_time - start_time), success_rate=float(metrics.get("success_rate", float("nan"))),
                      **timer.summary())
        tag = f"{cfg.policy}_H{int(cfg.eval.goal_offset_steps)}_seed{int(cfg.seed)}"
        (results_path / f"timing_{tag}.json").write_text(json.dumps(timing, indent=1))
        timer.write_records(results_path / f"timing_records_{tag}.jsonl")
        timer.write_per_episode(results_path / f"timing_episodes_{tag}.json", success=metrics.get("episode_successes"))
        print("[timing-json] " + json.dumps(timing), flush=True)
        for line in timer.record_lines():
            print(line, flush=True)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
