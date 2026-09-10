"""Task-completion eval of a published baseline checkpoint on toolhang, in the LeWAM env stack.

The checkpoint runs exactly as it was trained: its own cameras rendered from the live env, its own
low-dim inputs, its own controller convention. The episodes and the success test are ours: each
picked demo's initial state, the task predicate of the env (latched), budget = budget_mult times the
demo length, the same draw as the task-only rows (rng(seed).choice over the episodes of the h5).

--kind bc_rnn: a robomimic checkpoint (BC-RNN of the model zoo); robomimic's rollout policy keeps the
    LSTM state, crops and normalizes; delta actions on the env's default OSC controller.
--kind dp: a diffusion_policy checkpoint (raw model weights; --weights ema for the EMA copy its own
    eval script uses when training.use_ema is on); the last n_obs_steps observations are stacked,
    the first n_action_steps of each predicted chunk are executed; absolute-pose chunks (position,
    6-d rotation, gripper) are turned into position, axis-angle, gripper as diffusion_policy's runner
    does, and the OSC controller is switched to absolute mode.

Needs on PYTHONPATH: this repo (+ the diffusion_policy repo and its deps for --kind dp); robomimic
and robosuite for the env (ROBOMIMIC_RAW points at the env-meta dir).
"""
import argparse
import json
import os
import time
from collections import deque

import h5py
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=["bc_rnn", "dp"], required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--h5", required=True, help="flat toolhang h5 (states + episode index)")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--budget_mult", type=float, default=2.0)
parser.add_argument("--image_size", type=int, default=240)
parser.add_argument("--weights", choices=["model", "ema"], default="model",
                    help="dp: the raw model weights or the EMA copy")
parser.add_argument("--out", default="/tmp/baseline_eval")
args = parser.parse_args()

from lewam.envs.robomimic_gc_env import RoboMimicGCEnv  # noqa: E402  (registers the obs modalities)

env = RoboMimicGCEnv(task="ToolHang", resolution=224)

IMAGE_KEYS = ["sideview_image", "robot0_eye_in_hand_image"]
LOW_DIM = {"robot0_eef_pos": slice(0, 3), "robot0_eef_quat": slice(3, 7), "robot0_gripper_qpos": slice(7, 9)}


def observe(proprio):
    """The checkpoint's inputs from the live env: its cameras (upright HWC uint8) and low-dim keys."""
    obs = {key: np.asarray(env.env.render(mode="rgb_array", height=args.image_size, width=args.image_size,
                                          camera_name=key[:-len("_image")]), np.uint8) for key in IMAGE_KEYS}
    for key, part in LOW_DIM.items():
        obs[key] = np.asarray(proprio, np.float32)[part]
    return obs


def set_absolute_control():
    """DP's abs_action checkpoints were trained with control_delta=False: switch every OSC part
    controller of the live robots to absolute goals (robosuite 1.4 and 1.5 layouts)."""
    switched = 0
    for robot in env.env.env.robots:
        controllers = [getattr(robot, "controller", None), getattr(robot, "composite_controller", None)]
        controllers += list((getattr(controllers[1], "part_controllers", {}) or {}).values())
        for controller in controllers:
            if controller is not None and hasattr(controller, "use_delta"):
                controller.use_delta = False
                switched += 1
    assert switched > 0, "no OSC controller exposes use_delta"
    print(f"[baseline-eval] {switched} controller(s) switched to absolute goals", flush=True)


if args.kind == "bc_rnn":
    import robomimic.utils.file_utils as file_utils
    import robomimic.utils.obs_utils as obs_utils
    from robomimic.config import config_factory

    def merge_into(config, saved):
        """Set every leaf of the saved config into `config`, recursing through nested sections."""
        for key, value in saved.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                merge_into(config[key], value)
            else:
                config[key] = value

    # module names of the image encoder in robomimic 0.2 checkpoints -> the installed robomimic
    RENAMED_MODULES = ((".vis_core.", ".backbone."), (".pool_net.", ".pool."))

    def load_rollout_policy(ckpt_path, device):
        """A model-zoo checkpoint predates config keys the installed robomimic reads (algo.transformer,
        ...) and names its image encoder modules differently: merge its config into the current
        default config of its algorithm, rename the weights, then build the policy."""
        ckpt_dict = file_utils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        saved = json.loads(ckpt_dict["config"])
        config = config_factory(saved["algo_name"])
        with config.unlocked():
            merge_into(config, saved)
        ckpt_dict["config"] = config.dump()
        weights = {}
        for key, value in ckpt_dict["model"].items():
            for old, new in RENAMED_MODULES:
                key = key.replace(old, new)
            weights[key] = value
        ckpt_dict["model"] = weights
        return file_utils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=False)

    policy, ckpt_dict = load_rollout_policy(args.ckpt, args.device)
    config, _ = file_utils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    obs_keys = set(key for keys in config.observation.modalities.obs.values() for key in keys)
    assert set(IMAGE_KEYS) | set(LOW_DIM) >= obs_keys, f"checkpoint wants {sorted(obs_keys)}"
    print(f"[baseline-eval] BC-RNN loaded: obs {sorted(obs_keys)}", flush=True)

    def policy_input(proprio):
        """robomimic's env wrapper feeds its rollout policy processed images (CHW float in [0, 1])."""
        return {key: (obs_utils.process_obs(obs=value, obs_modality="rgb") if key in IMAGE_KEYS else value)
                for key, value in observe(proprio).items() if key in obs_keys}

    def run_episode(budget):
        policy.start_episode()
        obs_env, _ = env.reset(options={"state": start_state})
        for t in range(budget):
            action = policy(ob=policy_input(obs_env["proprio"]))
            obs_env, _, _, _, info = env.step(action)
            if bool(info.get("task_success", False)):
                return True, t + 1
        return False, budget

else:
    import dill
    import hydra
    import robomimic.models.base_nets as base_nets
    import robomimic.models.obs_core as obs_core
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    if not hasattr(base_nets, "CropRandomizer"):      # robomimic 0.3 moved it; DP's policy uses the old path
        base_nets.CropRandomizer = obs_core.CropRandomizer
    payload = torch.load(open(args.ckpt, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model" if args.weights == "ema" else "model"])
    policy.to(args.device).eval()
    n_obs, n_act = int(cfg.n_obs_steps), int(cfg.n_action_steps)
    obs_keys = set(cfg.task.shape_meta["obs"].keys())
    absolute = bool(cfg.task.get("abs_action", False))
    assert set(IMAGE_KEYS) | set(LOW_DIM) >= obs_keys, f"checkpoint wants {sorted(obs_keys)}"
    if absolute:
        set_absolute_control()
        to_axis_angle = RotationTransformer("axis_angle", "rotation_6d").inverse
    print(f"[baseline-eval] DP loaded ({args.weights} weights, training.use_ema={cfg.training.get('use_ema')}): "
          f"n_obs {n_obs} n_act {n_act} abs_action {absolute} steps {policy.num_inference_steps} "
          f"obs {sorted(obs_keys)}", flush=True)

    def stacked(history):
        """[1, n_obs, ...] tensors: images CHW float in [0, 1], low-dim as recorded."""
        batch = {}
        for key in obs_keys:
            values = np.stack([frame[key] for frame in history])
            if key in IMAGE_KEYS:
                values = np.moveaxis(values.astype(np.float32) / 255.0, -1, 1)
            batch[key] = torch.from_numpy(np.ascontiguousarray(values[None])).to(args.device)
        return batch

    def env_action(action):
        """DP's undo_transform_action: position | 6-d rotation | gripper -> position | axis-angle | gripper."""
        if not absolute:
            return action
        return np.concatenate([action[:3], to_axis_angle(action[3:9]), action[9:]])

    def run_episode(budget):
        obs_env, _ = env.reset(options={"state": start_state})
        history = deque([observe(obs_env["proprio"])] * n_obs, maxlen=n_obs)
        t = 0
        while t < budget:
            with torch.no_grad():
                chunk = policy.predict_action(stacked(history))["action"][0].cpu().numpy()
            for action in chunk[:n_act]:
                obs_env, _, _, _, info = env.step(env_action(action))
                history.append(observe(obs_env["proprio"]))
                t += 1
                if bool(info.get("task_success", False)):
                    return True, t
                if t >= budget:
                    break
        return False, budget


source = h5py.File(args.h5, "r")
ep_offset = np.asarray(source["ep_offset"][:]).reshape(-1)
ep_len = np.asarray(source["ep_len"][:]).reshape(-1)
rng = np.random.default_rng(args.seed)
picks = np.sort(rng.choice(len(ep_len), size=min(args.episodes, len(ep_len)), replace=False))

results, started = [], time.time()
for n, episode in enumerate(picks):
    start_state = np.asarray(source["state"][int(ep_offset[episode])], np.float64)
    budget = int(args.budget_mult * (int(ep_len[episode]) - 1))
    success, steps = run_episode(budget)
    results.append(dict(episode=int(episode), success=bool(success), steps=int(steps), budget=budget))
    print(f"[baseline-eval] {n + 1}/{len(picks)} ep{episode}: success={success} steps={steps}/{budget} "
          f"({(time.time() - started) / 60:.1f} min)", flush=True)

success_rate = 100.0 * np.mean([r["success"] for r in results])
summary = dict(kind=args.kind, ckpt=args.ckpt, seed=args.seed, n=len(results), success_rate=success_rate,
               image_size=args.image_size, budget_mult=args.budget_mult, weights=args.weights,
               successes=[r["success"] for r in results], results=results)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(summary, open(f"{args.out}_{args.kind}_s{args.seed}.json", "w"), indent=1)
print(f"[baseline-eval] BASELINE_EVAL {args.kind} success_rate: {success_rate:.1f} n={len(results)} seed={args.seed}",
      flush=True)
