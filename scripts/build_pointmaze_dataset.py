"""Render a pointmaze dataset at any OGBench maze size into the WF8 swm HDF5 schema.

Mirrors the existing pointmaze.h5 (medium) exactly: same keys, dtypes, chunking and
compression, so the new cell drops into the training/eval pipeline with no code change
(PointMazeEnv already takes task='large', and configs/eval/pointmaze.yaml already
documents the task switch).

State replay is exact: OGBench's released `observations` column is a 2-D xy feature, NOT
simulator state, so qpos/qvel from the npz are what get set (see offline-rl-obs-not-state).

Usage:  python3 build_pointmaze_large.py --size large [--episodes 1000] [--out PATH]
"""
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="large", choices=["medium", "large", "giant", "teleport"])
ap.add_argument("--npz", default=None, help="OGBench npz; defaults to /tmp/pm_<size>.npz")
ap.add_argument("--episodes", type=int, default=0, help="0 = all")
ap.add_argument("--resolution", type=int, default=224)
ap.add_argument("--out", default=None)
args = ap.parse_args()

npz = args.npz or f"/tmp/pm_{args.size}.npz"
out = args.out or f"/tmp/pointmaze_{args.size}.h5"

d = np.load(npz)
qpos, qvel, act, term = d["qpos"], d["qvel"], d["actions"], d["terminals"]
ends = np.where(term > 0)[0]
ep_bounds = [(0 if k == 0 else ends[k - 1] + 1, ends[k]) for k in range(len(ends))]
if args.episodes:
    ep_bounds = ep_bounds[: args.episodes]
# drop the terminal transition of each episode, matching the medium build (1001 -> 1000)
ep_bounds = [(a, b) for a, b in ep_bounds]
lens = [b - a for a, b in ep_bounds]
N = int(sum(lens))
print(f"size={args.size} episodes={len(ep_bounds)} frames={N} out={out}", flush=True)

import gymnasium  # noqa: E402
from lewam.envs.pointmaze_env import PointMazeEnv  # noqa: E402

env = PointMazeEnv(task=args.size, resolution=args.resolution)
u = env.u

R = args.resolution
with h5py.File(out, "w") as f:
    px = f.create_dataset("pixels", (N, R, R, 3), dtype="uint8",
                          chunks=(1, R, R, 3), compression="gzip")
    ds = {k: f.create_dataset(k, shape, dtype=dt) for k, shape, dt in [
        ("action", (N, act.shape[1]), "float32"),
        ("proprio", (N, 4), "float32"),
        ("state", (N, 4), "float32"),
        ("ep_idx", (N,), "int32"),
        ("step_idx", (N,), "int32"),
        ("id", (N,), "int64"),
    ]}
    f.create_dataset("ep_len", data=np.array(lens, dtype="int64"))
    f.create_dataset("ep_offset", data=np.concatenate([[0], np.cumsum(lens)[:-1]]).astype("int64"))
    f.attrs["source"] = f"ogbench pointmaze-{args.size}-navigate-v0 (add_info=True)"
    f.attrs["renderer"] = "lewam.envs.pointmaze_env.PointMazeEnv (native, no resample)"
    f.attrs["resolution"] = args.resolution

    w = 0
    t0 = time.time()
    for e, (a, b) in enumerate(ep_bounds):
        n = b - a
        for j in range(n):
            i = a + j
            u.set_state(qpos[i], qvel[i])
            px[w] = env.render()
            ds["proprio"][w] = np.concatenate([qpos[i], qvel[i]]).astype("float32")
            ds["state"][w] = ds["proprio"][w]
            ds["action"][w] = act[i]
            ds["ep_idx"][w] = e
            ds["step_idx"][w] = j
            ds["id"][w] = w
            w += 1
        if (e + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  ep {e+1}/{len(ep_bounds)}  {w} frames  {w/el:.1f} fps  "
                  f"eta {(N-w)/(w/el)/3600:.2f} h", flush=True)
print("done", out)
