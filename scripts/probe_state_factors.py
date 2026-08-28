"""Linear state-factor probes on pusht latents (owner spec 2026-08-26).

Targets from the h5 `state` array [agent_pos(2), block_pos(2), block_angle(1),
agent_vel(2)] at anchor time t. Probe inputs per model:
  z1  = z_t                      -> agent_pos (D,2), block_pos (D,2), block_angle (D,1)
  z2  = [z_{t-fs}, z_t]          -> agent_vel (2D,2)   (owner spec; fs-anchor spacing)
Reference extras (labeled, not the spec): vel from z1, pos from z2, and for unified
ctx = aggregate(z_{t-4fs..t}) probed alone -- the increased-context hypothesis.
Ridge closed-form, z-scored targets, test R^2 per factor (mean over target dims).
Angle also probed as sin/cos (2-dim) for the wraparound reference.
"""
import argparse
from pathlib import Path
import json

import h5py
import numpy as np
import torch
import torch.nn as nn

from lewam.models.gip import load_jointflow_model, load_lewam_unified_model

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def encode_frames(model, h5f, idx, device, bs=256):
    """Encode h5 pixel rows (uint8 HWC) at flat indices idx -> (len, D) fp32 cpu."""
    order = np.argsort(idx)
    zs = [None] * len(idx)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            rows = np.sort(idx[sel])
            # h5 fancy indexing needs sorted unique; handle dups via searchsorted
            uniq, inv = np.unique(idx[sel], return_inverse=True)
            px = h5f["pixels"][uniq]                        # (u, H, W, 3) uint8
            x = torch.from_numpy(px).float().permute(0, 3, 1, 2) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            z = model.encode(x.to(device)).float().cpu()
            z = z[inv]
            for j, gi in enumerate(sel):
                zs[gi] = z[j]
    return torch.stack(zs)


def ridge_r2(X_tr, Y_tr, X_te, Y_te, lam_scale=1e-3):
    """Closed-form ridge with bias; returns test R^2 (mean over target dims)."""
    X_tr = torch.cat([X_tr, torch.ones(len(X_tr), 1)], 1)
    X_te = torch.cat([X_te, torch.ones(len(X_te), 1)], 1)
    mu, sd = Y_tr.mean(0), Y_tr.std(0).clamp_min(1e-8)
    Yn_tr = (Y_tr - mu) / sd
    d = X_tr.shape[1]
    A = X_tr.T @ X_tr
    lam = lam_scale * torch.trace(A) / d
    W = torch.linalg.solve(A + lam * torch.eye(d), X_tr.T @ Yn_tr)
    pred = X_te @ W
    Yn_te = (Y_te - mu) / sd
    ss_res = ((Yn_te - pred) ** 2).mean(0)
    ss_tot = Yn_te.var(0).clamp_min(1e-8)
    return float((1 - ss_res / ss_tot).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--jf_ckpts", default="", help="comma run names under $STABLEWM_HOME/checkpoints")
    ap.add_argument("--uni_ckpts", default="", help="comma run names (lewam_unified layout)")
    ap.add_argument("--lewm_dir", default="", help="official LeWM folder (config.json + weights) -> "
                                                    "its encoder+projector as a probe column")
    ap.add_argument("--fs", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=5, help="unified aggregator context (anchors)")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/state_probe.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f = h5py.File(args.h5, "r")
    ep_idx = np.asarray(f["episode_idx"])
    N = len(ep_idx)
    fs, ctx = args.fs, args.ctx
    need = args.n_train + args.n_test
    # anchors t with the full ctx window (t-(ctx-1)*fs .. t) inside one episode
    span = (ctx - 1) * fs
    cand = rng.permutation(N - span) + span
    keep = cand[ep_idx[cand] == ep_idx[cand - span]][:need]
    assert len(keep) == need, f"only {len(keep)} valid anchors"
    state = np.asarray(f["state"][np.sort(keep)])
    order = np.argsort(np.sort(keep))  # identity; state rows align with sorted keep
    keep = np.sort(keep)
    tr = np.zeros(need, bool)
    tr[rng.permutation(need)[:args.n_train]] = True

    S = torch.from_numpy(state).float()
    targets = {
        "agent_pos": S[:, 0:2], "block_pos": S[:, 2:4],
        "block_angle": S[:, 4:5], "agent_vel": S[:, 5:7],
        "block_angle_sincos": torch.stack([S[:, 4].sin(), S[:, 4].cos()], 1),
    }

    models = []
    for name in [s for s in args.jf_ckpts.split(",") if s]:
        m, _cfg = load_jointflow_model(name)
        models.append((f"jf:{name}", m.to(device).eval(), None, None))
    for name in [s for s in args.uni_ckpts.split(",") if s]:
        m, ucfg = load_lewam_unified_model(name)
        models.append((f"uni:{name}", m.to(device).eval(), "agg", ucfg))
    if args.lewm_dir:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from probe_wm_discrim import LeWMAdapter

        class _LeWMEnc(nn.Module):
            def __init__(self, ad):
                super().__init__()
                self.m = ad.m

            def encode(self, px):
                cls = self.m.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                return self.m.projector(cls)

        ad = LeWMAdapter(args.lewm_dir, fs=fs, amean=None, astd=None)
        models.append(("lewm:official", _LeWMEnc(ad).to(device).eval(), None, None))

    results = {}
    for label, model, extra, mcfg in models:
        z_now = encode_frames(model, f, keep, device)
        z_prev = encode_frames(model, f, keep - fs, device)
        feats = {"z1": z_now, "z2": torch.cat([z_prev, z_now], 1)}
        if extra == "agg":
            zs_ctx = [encode_frames(model, f, keep - k * fs, device) for k in range(ctx - 1, 0, -1)]
            window_all = torch.stack(zs_ctx + [z_now], 1)          # (n, ctx, D) cpu
            action_cond = bool(mcfg.get("agg_action_cond", False))
            block_dim = int(mcfg.get("action_dim", 10))
            with torch.no_grad():
                outs = []
                for s in range(0, len(window_all), 512):
                    w = window_all[s:s + 512].to(device)
                    pa = pm = None
                    if action_cond:
                        pa = torch.zeros(w.shape[0], w.shape[1], block_dim, device=device,
                                         dtype=w.dtype)
                        pm = torch.zeros(w.shape[0], w.shape[1], dtype=torch.bool, device=device)
                    c = model.aggregate(w, pa, pm)                 # (b, ctx, D)
                    outs.append(c[:, -1].float().cpu())
            feats["ctx"] = torch.cat(outs)
        res = {}
        spec = {"agent_pos": "z1", "block_pos": "z1", "block_angle": "z1", "agent_vel": "z2"}
        for tgt, Y in targets.items():
            row = {}
            for fname, X in feats.items():
                row[fname] = round(ridge_r2(X[tr], Y[tr], X[~tr], Y[~tr]), 4)
            main_in = spec.get(tgt, "z1")
            print(f"[state-probe] {label} {tgt}: MAIN({main_in})={row.get(main_in)} all={row}",
                  flush=True)
            res[tgt] = row
        results[label] = res
    with open(args.out, "w") as fh:
        json.dump({"n_train": args.n_train, "n_test": args.n_test, "fs": fs,
                   "spec": "pos/angle from z1 (D,*); vel from z2 (2D,2)",
                   "results": results}, fh, indent=1)
    print(f"[state-probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
