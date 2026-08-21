"""Post-hoc GCHead tests on a trained jointflow-GC checkpoint: does mixing velocity
prediction (noisy tokens at random tau) with a clean z_goal produce conflict?

On a fixed val batch from the GR cache, with the flow noise held constant:
  1. tau-resolved goal utility: action flow loss at each tau on a grid, under the true
     z_goal vs rolled vs null. delta(tau) = loss(wrong) - loss(true) locates where in the
     noise schedule the goal helps (delta>0), is ignored (~0), or conflicts (delta<0).
  2. per-step steering: inside the 8-step Euler loop, ||v(true) - v(rolled)|| / ||v|| at
     each step on the same intermediate state.
  3. gradient conflict: cosine between GCHead gradients of the action loss computed at
     tau_low vs tau_high, calibrated by the same-tau two-noise-draws cosine.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lewam.models.jointflow import build_model

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--frames_cache", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--taus", default="0.05,0.1,0.2,0.35,0.5,0.65,0.8,0.9,0.95")
    ap.add_argument("--out", default="/tmp/diag_gchead.json")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.run_dir)
    cfg = json.loads((run / "jointflow_config.json").read_text())
    model = build_model(cfg)
    model.load_state_dict(torch.load(run / "jointflow_best.pt", map_location="cpu"), strict=True)
    model.to(device).eval()

    stem = Path(cfg["dataset_name"]).name.replace(".h5", "")
    tag = f"{stem}_fs{cfg['fs']}_i{cfg['img_size']}"
    frames = np.load(f"{args.frames_cache}/{stem}/{tag}.frames.npy", mmap_mode="r")
    aux = np.load(f"{args.frames_cache}/{stem}/{tag}.aux.npz")
    t_gidx, maxh, ep_base = aux["t_gidx"], aux["maxh"], aux["ep_base"]
    a_block = aux["A_flat"]
    fs, hl, h_max = int(cfg["fs"]), int(cfg["policy_history_len"]), int(cfg["H_max"])
    n_act = int(cfg["num_actions_pred"])
    raw_adim = a_block.shape[1] // fs
    n_blocks = n_act // fs

    # trainer's val split, bit-exact
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    perm = torch.randperm(len(t_gidx), generator=gen)
    perm = perm[torch.from_numpy(maxh).long()[perm] >= 1]
    n_val = int(round((1 - float(cfg["train_split"])) * perm.numel()))
    # full-target rows (maxh >= n_blocks) so every chunk position is supervised
    picked = [int(i) for i in perm[:n_val].numpy() if maxh[i] >= max(n_blocks, 1)][:args.batch]
    B = len(picked)

    def px(x_u8):
        x = torch.from_numpy(np.asarray(x_u8)).float()
        return ((x / 255.0 - _IMG_MEAN) / _IMG_STD).to(device)

    rng = np.random.default_rng(0)
    history = torch.zeros(B, hl, *frames.shape[1:], device=device)
    pad = torch.ones(B, hl, dtype=torch.bool, device=device)
    goal_px = torch.empty(B, *frames.shape[1:], device=device)
    state_px = torch.empty(B, model.num_states, *frames.shape[1:], device=device)
    target = torch.zeros(B, n_act, raw_adim, device=device)
    h_norm = torch.zeros(B, device=device)
    for r, idx in enumerate(picked):
        ti = int(t_gidx[idx]); mh = int(maxh[idx])
        lo = max(int(ep_base[idx]), ti - hl + 1)
        recent = px(frames[lo:ti + 1])
        history[r, hl - recent.shape[0]:] = recent
        if recent.shape[0] < hl:
            history[r, :hl - recent.shape[0]] = recent[0]
        pad[r, hl - recent.shape[0]:] = False
        h = min(int(rng.integers(1, h_max + 1)), mh)
        goal_px[r] = px(frames[ti + h:ti + h + 1])[0]
        h_norm[r] = min(h, h_max) / h_max
        for q in range(1, model.num_states + 1):
            state_px[r, q - 1] = px(frames[ti + min(q, mh):ti + min(q, mh) + 1])[0]
        for b in range(n_blocks):
            target[r, b * fs:(b + 1) * fs] = torch.from_numpy(
                a_block[idx + b].reshape(fs, raw_adim)).to(device)

    with torch.no_grad():
        z_hist = model.encode(history.reshape(B * hl, *history.shape[2:])).reshape(B, hl, -1)
        z_goal = model.encode(goal_px)
        # real state targets, as in training (online encoder latents)
        state_tgt = model.encode(state_px.reshape(B * model.num_states, *frames.shape[1:])) \
            .reshape(B, model.num_states, model.z_dim)
        memory = model._memory(z_hist)
    z_roll = torch.roll(z_goal, 1, 0)

    g = torch.Generator(device="cpu").manual_seed(123)
    noise_a = torch.randn(B, n_act, raw_adim, generator=g).to(device)
    noise_s = torch.randn(B, model.num_states, model.z_dim, generator=g).to(device)
    flow_tgt = target - noise_a

    def act_loss(tau, zg):
        t = torch.full((B,), tau, device=device)
        na = torch.lerp(noise_a, target, t[:, None, None])
        ns = torch.lerp(noise_s, state_tgt, t[:, None, None])
        v_a, _ = model.velocity(na, ns, memory, pad, t, t, zg, h_norm)
        return ((v_a - flow_tgt) ** 2).mean()

    res = {"batch": B}
    taus = [float(x) for x in args.taus.split(",")]
    with torch.no_grad():
        for tau in taus:
            lt = act_loss(tau, z_goal).item()
            lr_ = act_loss(tau, z_roll).item()
            ln = act_loss(tau, None).item()
            res[f"tau_{tau}"] = {"loss_true": round(lt, 4),
                                 "d_rolled": round(lr_ - lt, 4),
                                 "d_null": round(ln - lt, 4)}

    # 2) steering per Euler step (true-goal trajectory, rolled compared on same state)
    with torch.no_grad():
        torch.manual_seed(7)
        a = torch.randn(B, n_act, raw_adim, device=device)
        s = torch.randn(B, model.num_states, model.z_dim, device=device)
        steer = []
        for i in range(model.n_flow_steps):
            t = torch.full((B,), i / model.n_flow_steps, device=device)
            v_a, v_s = model.velocity(a, s, memory, pad, t, t, z_goal, h_norm)
            v_r, _ = model.velocity(a, s, memory, pad, t, t, z_roll, h_norm)
            steer.append(round(((v_a - v_r).flatten(1).norm(dim=1)
                                / v_a.flatten(1).norm(dim=1).clamp(min=1e-8)).mean().item(), 4))
            a = a + v_a / model.n_flow_steps
            s = s + v_s / model.n_flow_steps
    res["steer_by_step"] = steer

    # 3) gradient conflict in the GCHead across tau bands
    head_params = [p for p in model.action_out.parameters() if p.requires_grad]

    def head_grad(tau, noise_seed):
        gg = torch.Generator(device="cpu").manual_seed(noise_seed)
        na0 = torch.randn(B, n_act, raw_adim, generator=gg).to(device)
        ns0 = torch.randn(B, model.num_states, model.z_dim, generator=gg).to(device)
        t = torch.full((B,), tau, device=device)
        na = torch.lerp(na0, target, t[:, None, None])
        ns = torch.lerp(ns0, state_tgt, t[:, None, None])
        model.zero_grad(set_to_none=True)
        v_a, _ = model.velocity(na, ns, memory, pad, t, t, z_goal, h_norm)
        ((v_a - (target - na0)) ** 2).mean().backward()
        return torch.cat([p.grad.flatten() for p in head_params]).detach().clone()

    def cos(a_, b_):
        return round(F.cosine_similarity(a_, b_, dim=0).item(), 4)

    g_lo1, g_lo2 = head_grad(0.1, 1), head_grad(0.1, 2)
    g_hi1, g_hi2 = head_grad(0.9, 1), head_grad(0.9, 2)
    res["grad_cos_same_tau_low"] = cos(g_lo1, g_lo2)
    res["grad_cos_same_tau_high"] = cos(g_hi1, g_hi2)
    res["grad_cos_cross_tau"] = cos(g_lo1, g_hi1)
    res["grad_norm_low_over_high"] = round((g_lo1.norm() / g_hi1.norm().clamp(min=1e-12)).item(), 3)

    print(f"[gchead] {run.name}: {json.dumps(res)}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
