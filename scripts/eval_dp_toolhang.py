"""Task-completion eval of a diffusion_policy checkpoint on toolhang, in the LeWAM env stack.

Runs DP inference natively (n_obs_steps consecutive RAW frames, n_action_steps-chunk
receding horizon, EMA weights, stock 100-step DDPM) inside our robosuite-1.5.1 env
(RoboMimicGCEnv, no goal set -> terminated == robomimic is_success, latched). Episodes
start from the demo initial states of the flat h5, budget = budget_mult * (ep_len-1) raw
steps -- the same start/budget semantics as the unified full_traj protocol.

Needs on PYTHONPATH: this repo, the diffusion_policy repo, and the DP dep tree; robomimic
0.3.1 + robosuite 1.5.1 for the env (ROBOMIMIC_RAW must point at the env-meta dir).
"""
import argparse, json, os, sys, time
from collections import deque

import numpy as np
import torch

# robomimic 0.3 moved CropRandomizer out of base_nets; DP's policy references the old path
import robomimic.models.base_nets as _rmbn
import robomimic.models.obs_core as _rmoc
if not hasattr(_rmbn, "CropRandomizer"):
    _rmbn.CropRandomizer = _rmoc.CropRandomizer

import dill
import hydra
import h5py

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--h5", required=True, help="flat worldforge tool_hang.h5 (states + ep index)")
p.add_argument("--episodes", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--device", default="cuda")
p.add_argument("--budget_mult", type=float, default=2.0)
p.add_argument("--out", default="/tmp/dp_eval")
args = p.parse_args()

payload = torch.load(open(args.ckpt, "rb"), pickle_module=dill, map_location="cpu")
cfg = payload["cfg"]
policy = hydra.utils.instantiate(cfg.policy)
policy.load_state_dict(payload["state_dicts"]["ema_model"])   # EMA weights, not the raw model
policy.to(args.device).eval()
n_obs = int(cfg.n_obs_steps)
n_act = int(cfg.n_action_steps)
ckpt_epoch = payload.get("pickles", {})
print(f"[dp-eval] policy loaded: n_obs={n_obs} n_act={n_act} "
      f"steps={policy.num_inference_steps} device={args.device}", flush=True)

src = h5py.File(args.h5, "r")
ep_offset = np.asarray(src["ep_offset"][:]).reshape(-1)
ep_len = np.asarray(src["ep_len"][:]).reshape(-1)
rng = np.random.default_rng(args.seed)
picks = np.sort(rng.choice(len(ep_len), size=min(args.episodes, len(ep_len)), replace=False))

from lewam.envs.robomimic_gc_env import RoboMimicGCEnv
env = RoboMimicGCEnv(task="ToolHang", resolution=224)

OBS_KEYS = set(cfg.task.shape_meta["obs"].keys())   # only what THIS checkpoint consumes

def policy_obs(hist):
    frames = np.stack([f for f, _ in hist]).astype(np.float32) / 255.0     # [To,H,W,3]
    frames = np.moveaxis(frames, -1, 1)[None]                              # [1,To,3,H,W]
    props = np.stack([pr for _, pr in hist]).astype(np.float32)[None]      # [1,To,9]
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(args.device)
    full = {"sideview_image": t(frames),
            "robot0_eef_pos": t(props[..., 0:3]),
            "robot0_eef_quat": t(props[..., 3:7]),
            "robot0_gripper_qpos": t(props[..., 7:9])}
    return {k: v for k, v in full.items() if k in OBS_KEYS}

results = []
t_start = time.time()
for n, ep in enumerate(picks):
    s0 = np.asarray(src["state"][int(ep_offset[ep])], np.float64)
    obs, _ = env.reset(options={"state": s0})
    frame = env.render()
    hist = deque([(frame, np.asarray(obs["proprio"], np.float32))] * n_obs, maxlen=n_obs)
    budget = int(args.budget_mult * (int(ep_len[ep]) - 1))
    success, t = False, 0
    while t < budget and not success:
        with torch.no_grad():
            chunk = policy.predict_action(policy_obs(hist))["action"][0].cpu().numpy()
        for a in chunk[:n_act]:
            obs, _, terminated, _, info = env.step(a)
            hist.append((env.render(), np.asarray(obs["proprio"], np.float32)))
            t += 1
            if bool(info.get("task_success", False)):
                success = True
                break
            if t >= budget:
                break
    results.append(dict(episode=int(ep), success=bool(success), steps=int(t), budget=budget))
    print(f"[dp-eval] {n+1}/{len(picks)} ep{ep}: success={success} steps={t}/{budget} "
          f"({(time.time()-t_start)/60:.1f}m elapsed)", flush=True)

sr = 100.0 * np.mean([r["success"] for r in results])
summary = dict(ckpt=args.ckpt, seed=args.seed, n=len(results), success_rate=sr,
               successes=[r["success"] for r in results], results=results)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(summary, open(f"{args.out}_s{args.seed}.json", "w"), indent=1)
print(f"[dp-eval] DP_EVAL success_rate: {sr:.1f} n={len(results)} seed={args.seed}", flush=True)
