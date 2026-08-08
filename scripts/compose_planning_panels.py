"""Compose [planning rollout | ground-truth demo | goal] 3-panel videos.

Inputs are the eval harness's own artifacts: env_<i>.mp4 written by world.evaluate(video=...)
during a warm-CEM run with num_eval=5 -- the EXACT rollouts the printed success_rate scored, not a
re-enactment. The (episode, start) each env used is recovered by calling the same deterministic
sampler the eval used (gip.sample_eval_episodes, seeded), and asserted against episode lengths.

Panels: LEFT rollout (harness video), MIDDLE the recorded demo over the same window, RIGHT the
goal frame the policy was conditioned on (static). Shorter panels pad with their last frame.
"""
import argparse
import json
import os

import h5py
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True)
ap.add_argument("--h5", required=True, help="raw pixels h5 (backing file, not the _ev view)")
ap.add_argument("--video_dir", required=True, help="dir holding env_<i>.mp4 from the eval run")
ap.add_argument("--out", required=True)
ap.add_argument("--dataset_name", required=True, help="stem the eval loaded (for the sampler)")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--goal_offset", type=int, default=50)
ap.add_argument("--successes", default="", help="comma list like 1,0,1,1,0 from the eval log")
a = ap.parse_args()

import lewam.models.gip as gip
import stable_worldmodel as swm  # noqa: F401

cfg = OmegaConf.create({
    "seed": a.seed, "cache_dir": None,
    "eval": {"num_eval": a.n, "goal_offset_steps": a.goal_offset, "dataset_name": a.dataset_name},
    "dataset": {"keys_to_cache": ["action"]},
    "gip_eval": {},
})
dataset = gip.get_dataset(cfg, a.dataset_name)
episodes, starts, _ = gip.sample_eval_episodes(cfg, dataset)
print(f"[compose] {a.cell}: envs -> (episode,start) = {list(zip(episodes, starts))}", flush=True)

f = h5py.File(a.h5, "r")
ep_off, ep_len = f["ep_offset"][:], f["ep_len"][:]
px = f["pixels"]
succ = [s == "1" for s in a.successes.split(",")] if a.successes else [None] * a.n
os.makedirs(a.out, exist_ok=True)

def label_strip(w, text, ok=None):
    import cv2
    strip = np.full((20, w, 3), 32, np.uint8)
    col = (240, 240, 240) if ok is None else ((80, 220, 80) if ok else (60, 60, 230))
    cv2.putText(strip, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return strip

manifest = []
for i in range(a.n):
    vid = os.path.join(a.video_dir, f"env_{i}.mp4")
    roll = [np.asarray(fr) for fr in imageio.mimread(vid, memtest=False)]
    T = len(roll)
    ep, s = int(episodes[i]), int(starts[i])
    lo, L = int(ep_off[ep]), int(ep_len[ep])
    assert s + a.goal_offset < L, (a.cell, i, ep, s, L)
    gt_end = min(s + T, L)
    gt = [np.asarray(px[lo + s + t]) for t in range(gt_end - s)]
    gt += [gt[-1]] * (T - len(gt))
    goal = np.asarray(px[lo + s + a.goal_offset])
    H, W = roll[0].shape[:2]
    def fit(img):
        img = np.asarray(img, np.uint8)
        if img.shape[:2] != (H, W):
            import cv2
            img = cv2.resize(img, (W, H))
        return img
    bar = np.full((H, 4, 3), 255, np.uint8)
    ok = succ[i] if i < len(succ) else None
    head = np.concatenate([label_strip(W, f"PLAN {'OK' if ok else 'FAIL' if ok is not None else ''}", ok),
                           np.full((20, 4, 3), 32, np.uint8), label_strip(W, "GT demo"),
                           np.full((20, 4, 3), 32, np.uint8), label_strip(W, "GOAL")], axis=1)
    frames = [np.concatenate([head, np.concatenate([fit(roll[t]), bar, fit(gt[t]), bar, fit(goal)], 1)], 0)
              for t in range(T)]
    tag = "S" if ok else ("F" if ok is not None else "U")
    p = os.path.join(a.out, f"{a.cell}_ep{ep:04d}_s{s:04d}_{tag}.mp4")
    imageio.mimwrite(p, frames, fps=10, quality=6, macro_block_size=1)
    manifest.append(dict(cell=a.cell, env=i, episode=ep, start=s, frames=T,
                         success=ok, path=os.path.basename(p), bytes=os.path.getsize(p)))
    print(f"[compose] env{i} ep{ep} s{s} T={T} {'OK' if ok else 'FAIL' if ok is not None else '?'} -> {p}", flush=True)

json.dump(manifest, open(os.path.join(a.out, f"{a.cell}_manifest.json"), "w"), indent=2)
print("COMPOSE_DONE", flush=True)
