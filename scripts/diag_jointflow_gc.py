"""Offline conditioning diagnostics for a jointflow-GC checkpoint.

On a batch of val decision points from the fs-strided GR cache, measures
  1. goal separability in latent space: pairwise distance, per-dim std, effective rank
     of z_goal over the batch (a goal signal the head could in principle use);
  2. conditioning sensitivity of the sampled action chunk, with the flow noise held
     fixed: replace the goal (rolled within batch / the learned null) or force the
     horizon across its range, and compare each variant against the eval-matched
     reference (goal at +5 anchors, h_norm = 0.1). The same-conditioning different-noise
     delta is the floor any real conditioning effect must clear.

Action deltas are in the z-scored action space, relative to the mean chunk norm.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lewam.models.jointflow import build_model

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="dir with jointflow_best.pt + jointflow_config.json")
    ap.add_argument("--frames_cache", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--h_ref", type=int, default=5,
                    help="reference goal offset in anchors (5 = the pusht eval protocol's "
                         "goal_offset 25 raw / fs 5)")
    ap.add_argument("--pixel_scale", default="raw255", choices=["norm", "raw255"])
    ap.add_argument("--out", default="/tmp/diag_gc.json")
    return ap.parse_args()


def erank(z):
    cov = np.cov(np.asarray(z, dtype=np.float64).T)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    p = eig / (eig.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-20)).sum()))


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
    t_gidx, maxh = aux["t_gidx"], aux["maxh"]

    # the trainer's val split, reconstructed bit-exact
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    perm = torch.randperm(len(t_gidx), generator=gen)
    perm = perm[torch.from_numpy(maxh).long()[perm] >= 1]
    n_val = int(round((1 - float(cfg["train_split"])) * perm.numel()))
    val_idx = perm[:n_val].numpy()

    hl, h_max, B = int(cfg["policy_history_len"]), int(cfg["H_max"]), args.batch
    picked = [int(i) for i in val_idx if maxh[i] >= args.h_ref][:B]
    assert len(picked) == B, f"only {len(picked)} val anchors with maxh >= {args.h_ref}"

    def px(x_u8):
        x = torch.from_numpy(np.asarray(x_u8)).float()
        if args.pixel_scale == "norm":
            x = (x / 255.0 - _IMG_MEAN) / _IMG_STD
        return x.to(device)

    history = torch.zeros(B, hl, *frames.shape[1:], device=device)
    pad = torch.ones(B, hl, dtype=torch.bool, device=device)
    goal_ref = torch.empty(B, *frames.shape[1:], device=device)
    ep_base = aux["ep_base"]
    for r, idx in enumerate(picked):
        ti = int(t_gidx[idx])
        lo = max(int(ep_base[idx]), ti - hl + 1)
        recent = px(frames[lo:ti + 1])
        history[r, hl - recent.shape[0]:] = recent
        pad[r, hl - recent.shape[0]:] = False
        goal_ref[r] = px(frames[ti + args.h_ref:ti + args.h_ref + 1])[0]

    with torch.no_grad():
        z_hist = model.encode(history.reshape(B * hl, *history.shape[2:])).reshape(B, hl, -1)
        z_goal = model.encode(goal_ref)

    zg = z_goal.cpu().numpy()
    pair = zg[None, :, :] - zg[:, None, :]
    pair_d = np.sqrt((pair ** 2).sum(-1))
    goal_stats = {
        "z_goal_pairwise_dist_mean": float(pair_d[np.triu_indices(B, 1)].mean()),
        "z_goal_zstd": float(zg.std(0).mean()),
        "z_goal_erank": erank(zg),
        "z_hist_last_zstd": float(z_hist[:, -1].cpu().numpy().std(0).mean()),
    }

    h_ref_norm = torch.full((B,), min(args.h_ref, h_max) / h_max, device=device)

    def sample_with(noise_seed, z_g, h_n):
        torch.manual_seed(noise_seed)
        with torch.no_grad():
            a, _ = model.sample(z_hist, pad, z_goal=z_g, h_norm=h_n)
        return a

    a_ref = sample_with(0, z_goal, h_ref_norm)
    ref_norm = a_ref.flatten(1).norm(dim=1).mean().item()

    def rel(a):
        return float((a - a_ref).flatten(1).norm(dim=1).mean().item() / (ref_norm + 1e-9))

    res = {
        **goal_stats,
        "chunk_norm_ref": ref_norm,
        "delta_noise_reseed": rel(sample_with(1, z_goal, h_ref_norm)),
        "delta_goal_rolled": rel(sample_with(0, torch.roll(z_goal, 1, 0), h_ref_norm)),
        "delta_goal_null": rel(sample_with(0, None, h_ref_norm)),
    }
    for h in (0.02, 0.1, 0.5, 1.0):
        res[f"delta_hnorm_{h}"] = rel(sample_with(0, z_goal, torch.full((B,), h, device=device)))

    print(f"[diag_gc] {run.name}: {json.dumps({k: round(v, 4) for k, v in res.items()})}",
          flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
