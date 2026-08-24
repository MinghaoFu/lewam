"""Counterfactual dynamics probe for jointflow (owner request 2026-08-25).

For N random anchors: encode the true history, then inpaint the state slot under the
TRUE action chunk and K alternative chunks (drawn from other anchors), with the flow
noise held fixed per anchor so imagined states are comparable. Metrics:
  match_top1   fraction of anchors where the true chunk's imagined state is closest to
               the encoded true next state (chance = 1/(K+1))
  match_rank   mean rank of the true chunk by that cost (1 = best)
  sensitivity  mean ||z_imag(alt) - z_imag(true)|| / ||z(t+fs) - z(t)|| -- action-induced
               imagination change relative to the real one-step transition scale
  cost_cv      std/mean of the candidate costs per anchor (spread CEM would select over)
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
    ap.add_argument("--ckpt_dirs", required=True,
                    help="comma-separated run NAMES under $STABLEWM_HOME/checkpoints "
                         "(each containing jointflow_best.pt + jointflow_config.json)")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--n_anchors", type=int, default=200)
    ap.add_argument("--n_alts", type=int, default=15)
    ap.add_argument("--goal_offset_obs", type=int, default=5,
                    help="goal anchor offset in obs steps for GC ckpts (h_norm = this/H_max)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/jf_dyn_probe.json")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    f = h5py.File(args.h5, "r")
    ep_off = np.asarray(f["ep_offset"][:]).reshape(-1)
    ep_len = np.asarray(f["ep_len"][:]).reshape(-1)
    results = {}

    for ckpt_dir in args.ckpt_dirs.split(","):
        model, cfg = gip.load_jointflow_model(ckpt_dir.strip(), which="best")
        model = model.to(device).eval()
        model.requires_grad_(False)
        fs = int(cfg["fs"])
        hl = int(cfg["policy_history_len"])
        na = int(cfg["num_actions_pred"])
        h_max = int(cfg.get("H_max", 50))
        a_mean = torch.tensor(cfg["action_mean"], dtype=torch.float32)
        a_std = torch.tensor(cfg["action_std"], dtype=torch.float32)
        goal_cond = bool(cfg.get("goal_conditioning", False))
        need = fs * (hl - 1) + max(na, fs) + args.goal_offset_obs * fs

        # anchors with full history behind and targets/goal ahead
        cand = []
        for e in range(len(ep_len)):
            lo, L = int(ep_off[e]), int(ep_len[e])
            t0, t1 = fs * (hl - 1), L - need
            if t1 > t0:
                cand.append((lo, t0, t1))
        picks = []
        while len(picks) < args.n_anchors:
            lo, t0, t1 = cand[int(rng.integers(len(cand)))]
            picks.append(lo + int(rng.integers(t0, t1)))

        def frames_at(rows):
            px = torch.stack([torch.from_numpy(np.asarray(f["pixels"][r])) for r in rows])
            if px.shape[-1] == 3:                      # HWC on disk -> CHW
                px = px.permute(0, 3, 1, 2)
            px = px.float() / 255.0 if px.dtype == torch.uint8 else px.float()
            return ((px - _IMG_MEAN) / _IMG_STD).to(device)

        def chunk_at(row):
            a = torch.from_numpy(np.asarray(f["action"][row:row + na], np.float32))
            return (a - a_mean) / a_std

        m_top1, m_rank, sens, cost_cv = [], [], [], []
        for gi in range(args.n_anchors):
            t = picks[gi]
            hist_rows = [t - fs * (hl - 1 - k) for k in range(hl)]
            with torch.no_grad():
                z_all = model.encode(frames_at(hist_rows + [t + fs] +
                                               ([t + args.goal_offset_obs * fs] if goal_cond else [])))
            z_hist = z_all[:hl][None]
            z_next = z_all[hl]
            z_goal1 = z_all[hl + 1:hl + 2] if goal_cond else None

            alts = [chunk_at(picks[int(rng.integers(args.n_anchors))]) for _ in range(args.n_alts)]
            cands = torch.stack([chunk_at(t)] + alts).to(device)          # (K+1, na, adim)
            B = cands.shape[0]
            pad = torch.zeros(B, hl, dtype=torch.bool, device=device)
            gen = torch.Generator(device="cpu").manual_seed(args.seed * 100003 + gi)
            noise_a = torch.randn(1, na, cands.shape[-1], generator=gen).to(device).expand(B, -1, -1)
            noise_s = torch.randn(1, model.num_states, model.z_dim, generator=gen).to(device).expand(B, -1, -1)
            kw = {}
            if goal_cond:
                kw = dict(z_goal=z_goal1.expand(B, -1),
                          h_norm=torch.full((B,), min(args.goal_offset_obs, h_max) / h_max,
                                            device=device))
            with torch.no_grad():
                z_imag = model.sample_inpaint(z_hist.expand(B, -1, -1), pad, cands,
                                              noise_action=noise_a.contiguous(),
                                              noise_state=noise_s.contiguous(), **kw)[:, 0]
            costs = ((z_imag - z_next[None]) ** 2).mean(-1)
            order = torch.argsort(costs)
            rank = int((order == 0).nonzero()[0]) + 1
            m_top1.append(1.0 if rank == 1 else 0.0)
            m_rank.append(rank)
            d_alt = (z_imag[1:] - z_imag[0][None]).norm(dim=-1).mean()
            d_real = (z_next - z_hist[0, -1]).norm().clamp(min=1e-6)
            sens.append(float(d_alt / d_real))
            cost_cv.append(float(costs.std() / costs.mean().clamp(min=1e-9)))

        results[ckpt_dir.strip().split("/")[-1]] = dict(
            n=args.n_anchors, k_alts=args.n_alts,
            match_top1=float(np.mean(m_top1)),
            chance=1.0 / (args.n_alts + 1),
            match_rank_mean=float(np.mean(m_rank)),
            sensitivity=float(np.mean(sens)),
            cost_cv=float(np.mean(cost_cv)))
        print(f"[dyn-probe] {ckpt_dir}: {results[ckpt_dir.strip().split('/')[-1]]}", flush=True)

    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
