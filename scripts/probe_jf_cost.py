"""Cost-to-goal analysis (owner spec 2026-08-27).

Part A -- distribution over samples: at each dataset anchor, K joint policy samples ->
imagined next latent -> cost_k = ||z_imag_k - z_goal||, plus the planner's inpaint score
(shared noise, actions clamped to each sampled chunk). Saved next to the "stay put"
cost ||z_t - z_goal|| and the REAL next-state cost ||z(t+fs) - z_goal|| from the episode,
so histograms can show imagined progress vs real progress in the same units.

Part B -- cost vs unroll step: M autoregressive imagination rollouts of H steps per
anchor (imagined z fed back as history; GC h_norm counts down), cost at every step,
against the expert's real cost curve ||z(t+h*fs) - z_goal|| from the same episode.
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
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--unroll_H", type=int, default=8)
    ap.add_argument("--goal_offset_obs", type=int, default=10,
                    help="GR ckpts: goal frame at t + this many anchors (ignored for "
                         "goal_terminal ckpts, whose goal is the episode's last frame)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/jf_cost_probe.json")
    ap.add_argument("--dump_raw", default="", help="npz path prefix for the raw arrays")
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
        terminal = bool(cfg.get("goal_terminal", False))
        H, K, M = args.unroll_H, args.k_samples, args.rollouts
        goff = args.goal_offset_obs
        need = fs * H + 1 + (0 if terminal else goff * fs)

        cand = []
        for e in range(len(ep_len)):
            lo, L = int(ep_off[e]), int(ep_len[e])
            t0, t1 = fs * (hl - 1), L - need
            if t1 > t0:
                cand.append((e, lo, t0, t1))
        picks = []
        for _ in range(args.n_anchors):
            e, lo, t0, t1 = cand[int(rng.integers(len(cand)))]
            picks.append((e, lo + int(rng.integers(t0, t1))))

        def frames_at(rows):
            px = torch.stack([torch.from_numpy(np.asarray(f["pixels"][r])) for r in rows])
            if px.shape[-1] == 3:
                px = px.permute(0, 3, 1, 2)
            px = px.float() / 255.0 if px.dtype == torch.uint8 else px.float()
            return ((px - _IMG_MEAN) / _IMG_STD).to(device)

        cost_joint = np.zeros((args.n_anchors, K), np.float32)
        cost_inp = np.zeros((args.n_anchors, K), np.float32)
        cost_prev = np.zeros(args.n_anchors, np.float32)
        cost_real1 = np.zeros(args.n_anchors, np.float32)
        roll_cost = np.zeros((args.n_anchors, M, H), np.float32)
        real_cost = np.zeros((args.n_anchors, H), np.float32)

        for gi, (e, t) in enumerate(picks):
            hist_rows = [t - fs * (hl - 1 - k) for k in range(hl)]
            fut_rows = [t + fs * h for h in range(1, H + 1)]
            goal_row = (int(ep_off[e]) + int(ep_len[e]) - 1) if terminal else t + goff * fs
            with torch.no_grad():
                z_all = model.encode(frames_at(hist_rows + fut_rows + [goal_row]))
            z_hist = z_all[:hl]
            z_fut = z_all[hl:hl + H]
            z_goal = z_all[hl + H]
            cost_prev[gi] = float((z_hist[-1] - z_goal).norm())
            real_cost[gi] = (z_fut - z_goal).norm(dim=1).cpu().numpy()
            cost_real1[gi] = real_cost[gi, 0]

            def cond(n_rows, steps_left_obs):
                if not goal_cond:
                    return {}
                hn = 0.0 if terminal else min(max(steps_left_obs, 1), h_max) / h_max
                return dict(z_goal=z_goal[None].expand(n_rows, -1),
                            h_norm=torch.full((n_rows,), hn, device=device))

            gen = torch.Generator(device=device).manual_seed(args.seed * 7919 + gi)
            # ---- Part A: K samples at the anchor, joint cost + inpaint (planner) cost ----
            hK = z_hist[None].expand(K, -1, -1)
            pK = torch.zeros(K, hl, dtype=torch.bool, device=device)
            with torch.no_grad():
                acts, z_imag = model.sample(hK, pK, generator=gen, **cond(K, goff))
                cost_joint[gi] = (z_imag[:, 0] - z_goal).norm(dim=1).cpu().numpy()
                noise_a = torch.randn(1, model.num_actions, model.action_raw_dim, device=device,
                                      generator=gen).expand(K, -1, -1)
                noise_s = torch.randn(1, model.num_states, model.z_dim, device=device,
                                      generator=gen).expand(K, -1, -1)
                z_inp = model.sample_inpaint(hK, pK, acts, noise_a, noise_s, **cond(K, goff))
                cost_inp[gi] = (z_inp[:, 0] - z_goal).norm(dim=1).cpu().numpy()
            # ---- Part B: M autoregressive rollouts of H steps ----
            hM = z_hist[None].expand(M, -1, -1).clone()
            pM = torch.zeros(M, hl, dtype=torch.bool, device=device)
            with torch.no_grad():
                for h in range(H):
                    _, z_next = model.sample(hM, pM, generator=gen, **cond(M, goff - h))
                    z_next = z_next[:, 0]
                    roll_cost[gi, :, h] = (z_next - z_goal).norm(dim=1).cpu().numpy()
                    hM = torch.cat([hM[:, 1:], z_next[:, None]], dim=1)

        # summaries (raw latent units, no ratios; differences = "progress" in the same units)
        prog_joint = cost_joint - cost_prev[:, None]          # <0 = imagined to move closer
        prog_real = cost_real1 - cost_prev
        summary = dict(
            n=args.n_anchors, k=K, rollouts=M, H=H,
            cost_prev_median=float(np.median(cost_prev)),
            cost_real1_median=float(np.median(cost_real1)),
            cost_joint_median=float(np.median(cost_joint)),
            cost_inp_median=float(np.median(cost_inp)),
            per_anchor_std_joint_median=float(np.median(cost_joint.std(1))),
            per_anchor_std_inp_median=float(np.median(cost_inp.std(1))),
            per_anchor_cv_joint_median=float(np.median(cost_joint.std(1) / cost_joint.mean(1))),
            per_anchor_cv_inp_median=float(np.median(cost_inp.std(1) / cost_inp.mean(1))),
            imagined_progress_median=float(np.median(prog_joint)),
            real_progress_median=float(np.median(prog_real)),
            frac_samples_imagined_closer=float((prog_joint < 0).mean()),
            frac_anchors_real_closer=float((prog_real < 0).mean()),
            roll_cost_mean_by_step=[round(float(x), 4) for x in roll_cost.mean((0, 1))],
            real_cost_mean_by_step=[round(float(x), 4) for x in real_cost.mean(0)],
        )
        results[name] = summary
        print(f"[cost-probe] {name}: {json.dumps(summary)}", flush=True)
        if args.dump_raw:
            np.savez(f"{args.dump_raw}_{name}.npz", cost_joint=cost_joint, cost_inp=cost_inp,
                     cost_prev=cost_prev, cost_real1=cost_real1, roll_cost=roll_cost,
                     real_cost=real_cost)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[cost-probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
