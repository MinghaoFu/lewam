#!/usr/bin/env python3
"""Trajectory-divergence analysis for the context probe. Given two executed-latent dumps from the
SAME starts (ctx_cap=1 vs ctx_cap=full, dumped by gip.LeWAMUnifiedPolicy.dump_latents), report per
obs-step t: the k1-vs-kfull latent divergence ||z_t^k1 - z_t^kfull|| (does the k=1 rollout leave the
good k=full path -> drift), and each run's latent distance-to-goal ||z_t - z_goal|| (does k=1 wander
from the goal = gross drift, or approach it = endgame miss). Envs matched by index (same seed/starts).
"""
import argparse
import numpy as np


def load(path):
    d = np.load(path)
    if d["calls"].size == 0:
        return {}, {}
    calls, envs, z_cur, z_goal = d["calls"], d["envs"], d["z_cur"], d["z_goal"]
    traj, goal = {}, {}
    for j in np.argsort(calls, kind="stable"):        # order each env's latents by call (=obs-step)
        e = int(envs[j])
        traj.setdefault(e, []).append(z_cur[j])
        goal[e] = z_goal[j]
    return {e: np.stack(v) for e, v in traj.items()}, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", required=True)
    ap.add_argument("--kfull", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t1, g1 = load(args.k1)
    tf, gf = load(args.kfull)
    envs = sorted(set(t1) & set(tf))
    maxT = max((max(t1[e].shape[0], tf[e].shape[0]) for e in envs), default=0)
    print(f"[div] {args.tag} envs={len(envs)} maxT={maxT}")
    print(f"[div] {args.tag}   t | diverge | d2goal_k1 | d2goal_kfull | n")
    for t in range(maxT):
        divs, d1, df, n = [], [], [], 0
        for e in envs:
            a, b = t1[e], tf[e]
            if t < a.shape[0]:
                d1.append(np.linalg.norm(a[t] - g1[e]))
            if t < b.shape[0]:
                df.append(np.linalg.norm(b[t] - gf[e]))
            if t < a.shape[0] and t < b.shape[0]:
                divs.append(np.linalg.norm(a[t] - b[t]))
                n += 1
        f = lambda xs: (np.mean(xs) if len(xs) else float("nan"))
        print(f"[div] {args.tag}  {t:2d} | {f(divs):7.3f} | {f(d1):9.3f} | {f(df):12.3f} | {n}")


if __name__ == "__main__":
    main()
