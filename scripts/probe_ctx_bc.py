#!/usr/bin/env python3
"""One-step BC context probe for LeWAM-Unified (open-loop, no rollout).

At FIXED decision points from HELD-OUT expert demos (fixed horizon H), feed L real frames of
context and measure the predicted action-block vs the DEMONSTRATOR'S action-block, as a function
of L. This isolates whether context changes the PREDICTION -- disentangling it from the closed-loop
divergence that confounds the ctx_cap SR sweep (where a few bad early actions can snowball).

Why this is in-distribution for every L: a decision at obs-step p with L context frames and goal at
p+H equals a training window [p-L+1 .. p+H], query at position L-1, horizon H. The aggregator is
causal, so the query's context latent depends only on the L frames up to it -- feeding just those L
frames reproduces the exact training-time prediction. In-distribution as long as H+L-1 <= H_max.

"Expert" = the dataset's recorded actions (NOT the split, NOT any model). Data pipeline (frames
1-per-obs-step, z-scored action blocks, goal = frame p+H, horizon = H/H_max) mirrors
train_lewam_unified.py exactly. Held-out episodes = beyond the training max_eps (default >=4000).

  python scripts/probe_ctx_bc.py --dataset_name reacher.h5 --run_name reacher_uni_res \
      --caps "1 2 4 8 16" --horizons "3 5 10" --ep_start 4000 --n_eps 200
"""
import argparse
import numpy as np
import torch
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # HDF5 self-registers on import
import lewam.models.gip as gip
from lewam.utils import get_img_preprocessor


def load_range(base, img_t, act_mean, act_std, frameskip, ep_start, ep_end):
    """Per-episode (frames [n_obs+1,3,H,W] preprocessed, z-scored action blocks [n_obs,adim]).
    Mirrors train_lewam_unified.preload_frames but over an episode RANGE (to pick held-out demos)."""
    am = torch.tensor(act_mean, dtype=torch.float32)
    asd = torch.tensor(act_std, dtype=torch.float32)
    frames, acts = [], []
    for ep in range(ep_start, ep_end):
        L = int(base.lengths[ep])
        sl = base._load_slice(ep, 0, L)
        pix = sl["pixels"]
        raw = sl["action"]
        pix = pix if torch.is_tensor(pix) else torch.as_tensor(np.asarray(pix))
        raw = raw if torch.is_tensor(raw) else torch.as_tensor(np.asarray(raw))
        pp = img_t({"pixels": pix})["pixels"].float()
        n_obs = L // frameskip
        a = raw[: n_obs * frameskip].reshape(n_obs, frameskip, raw.shape[1])
        a = ((a - am) / asd).reshape(n_obs, frameskip * raw.shape[1])
        frames.append(pp[: n_obs + 1])
        acts.append(a)
    return frames, acts


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--run_name", required=True)          # policy tag; ckpt at cache/<run_name>
    ap.add_argument("--keys_to_load", default="pixels,action")
    ap.add_argument("--caps", default="1 2 4 8 16")
    ap.add_argument("--horizons", default="3 5 10")
    ap.add_argument("--ep_start", type=int, default=4000)  # held-out: beyond training max_eps
    ap.add_argument("--n_eps", type=int, default=200)
    ap.add_argument("--enc_batch", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = gip.load_lewam_unified_model(args.run_name)
    model = model.to(dev).eval()
    model.requires_grad_(False)
    act_mean, act_std = cfg["action_mean"], cfg["action_std"]
    frameskip, H_max, D = int(cfg["frameskip"]), float(cfg["H_max"]), int(cfg["z_dim"])
    caps = [int(x) for x in args.caps.split()]
    Hs = [int(x) for x in args.horizons.split()]
    max_cap = max(caps)

    _ktl = [k.strip() for k in args.keys_to_load.split(",") if k.strip()]
    dataset_cfg = dict(num_steps=4, frameskip=5, keys_to_load=_ktl,
                       keys_to_cache=[k for k in _ktl if k != "pixels"])
    base = swm.data.load_dataset(args.dataset_name, transform=None, cache_dir=None, **dataset_cfg)
    img_t = get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size)
    n_total = len(base.lengths)
    ep_start = min(args.ep_start, max(0, n_total - args.n_eps))
    ep_end = min(ep_start + args.n_eps, n_total)
    if ep_start < args.ep_start:
        print(f"[probe] WARN only {n_total} eps; held-out window shrunk to [{ep_start}:{ep_end}] "
              f"(may overlap training)", flush=True)
    print(f"[probe] {args.run_name}: eps [{ep_start}:{ep_end}] of {n_total}  caps={caps} Hs={Hs} "
          f"H_max={H_max} frameskip={frameskip}", flush=True)

    frames, acts = load_range(base, img_t, act_mean, act_std, frameskip, ep_start, ep_end)
    Z = []
    for f in frames:
        zs = [model.encode(f[i:i + args.enc_batch].to(dev).float()).cpu()
              for i in range(0, f.shape[0], args.enc_batch)]
        Z.append(torch.cat(zs, 0))                                   # (n_obs+1, D)

    for H in Hs:
        decs = []
        for e, z in enumerate(Z):
            nfr, n_obs = z.shape[0], acts[e].shape[0]
            for p in range(max_cap - 1, min(n_obs, nfr - 1 - H)):    # goal p+H exists; action p exists
                decs.append((e, p))
        if not decs:
            print(f"[probe] {args.run_name} H={H}: no valid decisions", flush=True)
            continue
        expert = torch.stack([acts[e][p] for e, p in decs]).to(dev).float()       # (Nd, adim)
        z_goal = torch.stack([Z[e][p + H] for e, p in decs]).to(dev).float()      # (Nd, D)
        h_norm = torch.full((len(decs),), min(H, int(H_max)) / H_max, device=dev)
        preds = {}
        for L in caps:
            seq = torch.stack([Z[e][p - L + 1:p + 1] for e, p in decs]).to(dev).float()  # (Nd,L,D)
            c = model.aggregate(seq)[:, -1]                                               # (Nd,D)
            preds[L] = model.gc_head(c, z_goal, h_norm)                                   # (Nd,adim)
        full = preds[max_cap]
        print(f"[probe] {args.run_name} H={H} Nd={len(decs)} (bc_mse vs expert; z-scored, "
              f"1.0=mean-predictor)", flush=True)
        for L in caps:
            bc = ((preds[L] - expert) ** 2).mean().item()
            chg = ((preds[L] - full) ** 2).mean().item()
            print(f"[probe]   L={L:2d}  bc_mse={bc:.4f}  vs_full={chg:.4f}", flush=True)


if __name__ == "__main__":
    main()
