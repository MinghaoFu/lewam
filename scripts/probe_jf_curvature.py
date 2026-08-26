"""Flow-path curvature probe (owner hypothesis 2026-08-26: curved fields force Euler to
keep adjusting; early errors drift into unseen field regions).

Per anchor, integrate the joint Euler ODE at n_steps in {8, 64} from the SAME initial
noise, recording the velocity at every step. Metrics per ckpt, action and state
components separately:
  cos_adj       mean cosine between successive velocities (1.0 = straight path)
  cos_ends      cosine between the first and last velocity (total direction change)
  straightness  ||x_T - x_0|| / sum ||dx_i||  (1.0 = straight line)
  profile       per-step-pair mean cos_adj (where along tau the field bends)
n=64 gives the field's intrinsic curvature; n=8 is the deployed regime.
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
    ap.add_argument("--steps", default="8,64")
    ap.add_argument("--goal_offset_obs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/jf_curv_probe.json")
    ap.add_argument("--dump_raw", default="", help="npz path prefix: save per-anchor "
                    "adjacent-cos matrices (n_pairs, B) per ckpt/steps/component")
    return ap.parse_args()


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


def euler_metrics(model, z_hist, pad, kw, noise_a, noise_s, n_steps):
    """Batched Euler integration recording velocities; returns metric dict."""
    B = z_hist.shape[0]
    memory = model._memory(z_hist)
    action, state = noise_a.clone(), noise_s.clone()
    vs_a, vs_s = [], []
    with torch.no_grad():
        for i in range(n_steps):
            tau = torch.full((B,), i / n_steps, device=z_hist.device)
            v_a, v_s = model.velocity(action, state, memory, pad, tau, tau,
                                      kw.get("z_goal"), kw.get("h_norm"))
            vs_a.append(v_a)
            vs_s.append(v_s)
            action = action + v_a / n_steps
            state = state + v_s / n_steps
    out, raw = {}, {}
    for tag, vs, x0, xT in (("action", vs_a, noise_a, action), ("state", vs_s, noise_s, state)):
        adj = torch.stack([_cos(vs[i], vs[i + 1]) for i in range(len(vs) - 1)])  # (n-1, B)
        chord = (xT - x0).flatten(1).norm(dim=1)
        arc = torch.stack([v.flatten(1).norm(dim=1) / n_steps for v in vs]).sum(0)
        ends = _cos(vs[0], vs[-1])
        out[tag] = dict(
            cos_adj=float(adj.mean()),
            cos_adj_p10=float(adj.mean(0).quantile(0.1)),
            cos_ends=float(ends.mean()),
            straightness=float((chord / arc.clamp(min=1e-8)).mean()),
            profile=[round(float(x), 4) for x in adj.mean(1)],
        )
        raw[tag] = dict(adj=adj.cpu().numpy(), ends=ends.cpu().numpy(),
                        straightness=(chord / arc.clamp(min=1e-8)).cpu().numpy())
    return out, raw


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
        picks = [cand[int(rng.integers(len(cand)))] for _ in range(args.n_anchors)]
        picks = [lo + int(rng.integers(t0, t1)) for lo, t0, t1 in picks]

        def frames_at(rows):
            px = torch.stack([torch.from_numpy(np.asarray(f["pixels"][r])) for r in rows])
            if px.shape[-1] == 3:
                px = px.permute(0, 3, 1, 2)
            px = px.float() / 255.0 if px.dtype == torch.uint8 else px.float()
            return ((px - _IMG_MEAN) / _IMG_STD).to(device)

        z_hists, z_goals = [], []
        with torch.no_grad():
            for t in picks:
                hist_rows = [t - fs * (hl - 1 - k) for k in range(hl)]
                rows = hist_rows + ([t + args.goal_offset_obs * fs] if goal_cond else [])
                z = model.encode(frames_at(rows))
                z_hists.append(z[:hl])
                if goal_cond:
                    z_goals.append(z[hl])
        z_hist = torch.stack(z_hists)                          # (B, hl, D)
        B = z_hist.shape[0]
        pad = torch.zeros(B, hl, dtype=torch.bool, device=device)
        kw = {}
        if goal_cond:
            kw = dict(z_goal=torch.stack(z_goals),
                      h_norm=torch.full((B,), min(args.goal_offset_obs, h_max) / h_max,
                                        device=device))
        gen = torch.Generator(device=device).manual_seed(args.seed)
        noise_a = torch.randn(B, model.num_actions, model.action_raw_dim, device=device,
                              generator=gen)
        noise_s = torch.randn(B, model.num_states, model.z_dim, device=device, generator=gen)

        res = {}
        for n_steps in [int(s) for s in args.steps.split(",")]:
            res[f"n{n_steps}"], raw = euler_metrics(model, z_hist, pad, kw, noise_a, noise_s,
                                                    n_steps)
            if args.dump_raw:
                np.savez(f"{args.dump_raw}_{name}_n{n_steps}.npz",
                         adj_action=raw["action"]["adj"], adj_state=raw["state"]["adj"],
                         ends_action=raw["action"]["ends"], ends_state=raw["state"]["ends"],
                         straight_action=raw["action"]["straightness"],
                         straight_state=raw["state"]["straightness"])
            r = res[f"n{n_steps}"]
            print(f"[curv-probe] {name} n{n_steps}: "
                  f"act cos_adj={r['action']['cos_adj']:.4f} ends={r['action']['cos_ends']:.4f} "
                  f"straight={r['action']['straightness']:.4f} | "
                  f"state cos_adj={r['state']['cos_adj']:.4f} ends={r['state']['cos_ends']:.4f} "
                  f"straight={r['state']['straightness']:.4f}", flush=True)
            print(f"[curv-probe] {name} n{n_steps} act profile: {r['action']['profile']}",
                  flush=True)
        results[name] = res

    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[curv-probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
