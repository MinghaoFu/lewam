"""Sampling-diversity probe (owner question 2026-08-25): do K policy samples at the same
state land in distinct modes or one band?

Per anchor: K chunks from model.sample (different noise), plus the imagined next state of
each via the joint sample. Metrics (all relative so they compare across ckpts):
  act_div      mean pairwise L2 between sampled chunks / mean chunk norm
  out_div      mean pairwise ||z_i - z_j|| between imagined states / ||z(t+fs) - z(t)||
               (same scale as the counterfactual probe's `sensitivity`, so
                out_div vs that number = on-policy spread vs cross-context spread)
  mode_ratio   per-anchor max/median pairwise imagined-state distance (>>1 = distinct
               modes; ~1 = one band)
"""
import argparse
import json

import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py

from lewam.models import gip

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dirs", required=True, help="run NAMES under $STABLEWM_HOME/checkpoints")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--n_anchors", type=int, default=200)
    ap.add_argument("--k_samples", type=int, default=32)
    ap.add_argument("--goal_offset_obs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/jf_div_probe.json")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    f = h5py.File(args.h5, "r")
    ep_off = np.asarray(f["ep_offset"][:]).reshape(-1)
    ep_len = np.asarray(f["ep_len"][:]).reshape(-1)
    results = {}

    for name in args.ckpt_dirs.split(","):
        name = name.strip()
        model, cfg = gip.load_jointflow_model(name, which="best")
        model = model.to(device).eval()
        model.requires_grad_(False)
        fs, hl = int(cfg["fs"]), int(cfg["policy_history_len"])
        h_max = int(cfg.get("H_max", 50))
        goal_cond = bool(cfg.get("goal_conditioning", False))
        need = fs * (hl - 1) + max(int(cfg["num_actions_pred"]), fs) + args.goal_offset_obs * fs

        cand = []
        for e in range(len(ep_len)):
            lo, L = int(ep_off[e]), int(ep_len[e])
            t0, t1 = fs * (hl - 1), L - need
            if t1 > t0:
                cand.append((lo, t0, t1))
        picks = [None] * args.n_anchors
        for gi in range(args.n_anchors):
            lo, t0, t1 = cand[int(rng.integers(len(cand)))]
            picks[gi] = lo + int(rng.integers(t0, t1))

        def frames_at(rows):
            px = torch.stack([torch.from_numpy(np.asarray(f["pixels"][r])) for r in rows])
            if px.shape[-1] == 3:
                px = px.permute(0, 3, 1, 2)
            px = px.float() / 255.0 if px.dtype == torch.uint8 else px.float()
            return ((px - _IMG_MEAN) / _IMG_STD).to(device)

        act_div, out_div, mode_ratio = [], [], []
        K = args.k_samples
        for gi in range(args.n_anchors):
            t = picks[gi]
            hist_rows = [t - fs * (hl - 1 - k) for k in range(hl)]
            with torch.no_grad():
                z_all = model.encode(frames_at(hist_rows + [t + fs] +
                                               ([t + args.goal_offset_obs * fs] if goal_cond else [])))
            z_hist = z_all[:hl][None].expand(K, -1, -1)
            d_real = (z_all[hl] - z_all[hl - 1]).norm().clamp(min=1e-6)
            pad = torch.zeros(K, hl, dtype=torch.bool, device=device)
            kw = {}
            if goal_cond:
                kw = dict(z_goal=z_all[hl + 1:hl + 2].expand(K, -1),
                          h_norm=torch.full((K,), min(args.goal_offset_obs, h_max) / h_max,
                                            device=device))
            gen = torch.Generator(device=device).manual_seed(args.seed * 7919 + gi)
            with torch.no_grad():
                acts, z_imag = model.sample(z_hist, pad, generator=gen, **kw)
            z_imag = z_imag[:, 0]
            a_flat = acts.reshape(K, -1)
            pd_a = torch.cdist(a_flat, a_flat)
            iu = torch.triu_indices(K, K, offset=1)
            act_div.append(float(pd_a[iu[0], iu[1]].mean() / a_flat.norm(dim=1).mean().clamp(min=1e-6)))
            pd_z = torch.cdist(z_imag, z_imag)[iu[0], iu[1]]
            out_div.append(float(pd_z.mean() / d_real))
            mode_ratio.append(float(pd_z.max() / pd_z.median().clamp(min=1e-9)))

        results[name] = dict(n=args.n_anchors, k=K,
                             act_div=float(np.mean(act_div)),
                             out_div=float(np.mean(out_div)),
                             mode_ratio_p50=float(np.median(mode_ratio)),
                             mode_ratio_p90=float(np.percentile(mode_ratio, 90)))
        print(f"[div-probe] {name}: {results[name]}", flush=True)

    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
