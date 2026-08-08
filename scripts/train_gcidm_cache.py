# External baseline: gcidm (arXiv 2605.08732) frozen-latent reproduction, preload-cache flavor.
"""Hindsight trainer for GC-IDM on FROZEN LeWM latents, fed from the preload cache.

Same trainer as train_gcidm.py with phase 1 replaced: instead of decoding the dataset and
re-applying the image transform, frames come from scripts/make_preload_cache.py's output
(<stem>_fs<FS>_i<IMG>.frames.npy + .aux.npz), which already holds the transformed fp16
frames, the z-scored action blocks, and the hindsight bookkeeping (t_gidx/maxh/ep_base)
in exactly the layout flatten_for_training() builds. The frozen encoder runs once over
the mmapped frames; phase 2 (sampling, head, checkpoints) is imported unchanged, so
eval_gip's gcidm mode consumes the output as before.

Run:
  python scripts/train_gcidm_cache.py --dataset_name pointmaze --cache_dir <preload_cache> \
    --weights <lewm_weights.pt> --run_name pointmaze_gcidm --epochs 200 --H_max 50
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import stable_worldmodel as swm

import lewam.models.gcidm as gcidm
from lewam.models.jepa import build_frozen_lewm
from train_gcidm import sample_batch


def load_cache(cache_dir, stem, frameskip, img_size):
    """Resolve the preload cache pair; same two-location lookup as PreloadGoalDataset."""
    tag = f"{stem}_fs{int(frameskip)}_i{int(img_size)}"
    fp = os.path.join(cache_dir, stem, f"{tag}.frames.npy")
    ap = os.path.join(cache_dir, stem, f"{tag}.aux.npz")
    if not (os.path.isfile(fp) and os.path.isfile(ap)):
        fp = os.path.join(cache_dir, f"{tag}.frames.npy")
        ap = os.path.join(cache_dir, f"{tag}.aux.npz")
    if not (os.path.isfile(fp) and os.path.isfile(ap)):
        raise FileNotFoundError(f"preload cache not found for {tag} under {cache_dir} -- "
                                f"run scripts/make_preload_cache.py first")
    frames = np.load(fp, mmap_mode="r")
    aux = np.load(ap)
    return frames, aux


def encode_frames(lewm, frames, device, enc_bs=256, bf16=True):
    """Frozen encoder over the mmapped, already-transformed frames -> (N, emb) fp16."""
    zs, t0 = [], time.time()
    n = frames.shape[0]
    for i in range(0, n, enc_bs):
        chunk = torch.from_numpy(np.array(frames[i:i + enc_bs])).float().to(device)
        with torch.no_grad():
            if bf16 and device != "cpu":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z = lewm.encode({"pixels": chunk.unsqueeze(1)})["emb"][:, 0].float()
            else:
                z = lewm.encode({"pixels": chunk.unsqueeze(1)})["emb"][:, 0]
        zs.append(z.half().cpu())
        if (i // enc_bs) % 50 == 0:
            print(f"[gcidm-cache] encoded {i + chunk.shape[0]}/{n} frames "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[gcidm-cache] encode DONE {n} frames in {time.time() - t0:.0f}s", flush=True)
    return torch.cat(zs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--H_max", type=int, default=50)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--history_size", type=int, default=3)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--ablate_horizon", action="store_true")
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--enc_bs", type=int, default=256)
    ap.add_argument("--dataset_name", type=str, required=True,
                    help="cache stem, e.g. pointmaze / tool_hang (no dataset access needed)")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="preload cache root; default $STABLEWM_HOME/preload_cache")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_root = args.cache_dir or str(Path(swm.data.utils.get_cache_dir()) / "preload_cache")
    frames, aux = load_cache(cache_root, args.dataset_name, args.frameskip, args.img_size)
    A_flat = torch.from_numpy(aux["A_flat"])
    t_gidx = torch.from_numpy(aux["t_gidx"]).long()
    maxh = torch.from_numpy(aux["maxh"]).long()
    ep_base = torch.from_numpy(aux["ep_base"]).long()
    act_mean = np.asarray(aux["act_mean"]).ravel().tolist()
    act_std = np.asarray(aux["act_std"]).ravel().tolist()
    action_block_dim = int(A_flat.shape[1])
    raw_adim = action_block_dim // args.frameskip
    assert int(maxh.min()) >= 1 and int((t_gidx + maxh).max()) < frames.shape[0]
    print(f"[gcidm-cache] {args.dataset_name}: frames={frames.shape} samples={len(t_gidx)} "
          f"action_block={action_block_dim}", flush=True)

    lewm = build_frozen_lewm(args.weights, embed_dim=192, history_size=args.history_size,
                             img_size=args.img_size, action_block_dim=action_block_dim).to(device)
    Z = encode_frames(lewm, frames, device, args.enc_bs).to(device)
    print(f"[gcidm-cache] latent sanity: per-dim std(mean over dims)="
          f"{Z[:2000].float().std(0).mean().item():.4f} (0 => collapsed)", flush=True)

    A_flat = A_flat.to(device)
    t_gidx = t_gidx.to(device); maxh = maxh.to(device); ep_base = ep_base.to(device)
    n_samples = t_gidx.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_samples, generator=g)
    n_val = int(round((1 - args.train_split) * n_samples))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[gcidm-cache] hindsight samples n={n_samples} train={train_idx.numel()} "
          f"val={val_idx.numel()} H_max={args.H_max}", flush=True)

    head = gcidm.GCIDMHead(emb_dim=192, action_dim=action_block_dim,
                           hidden_dim=args.hidden_dim, n_freqs=64, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[gcidm-cache] GCIDMHead params={n_params:,}  "
          f"ablate_horizon={args.ablate_horizon}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = dict(
        head="gcidm", emb_dim=192, action_dim=action_block_dim, hidden_dim=args.hidden_dim,
        n_freqs=64, dropout=0.1, H_max=args.H_max, frameskip=args.frameskip,
        action_raw_dim=raw_adim, weights=args.weights, action_mean=act_mean,
        action_std=act_std, ablate_horizon=args.ablate_horizon, data="preload_cache",
    )
    (run_dir / "gcidm_config.json").write_text(json.dumps(cfg_out, indent=2))

    def run_epoch(idx_pool, train_mode, shuffle_gen=None):
        head.train(train_mode)
        torch.set_grad_enabled(train_mode)
        order = idx_pool[torch.randperm(idx_pool.numel(), generator=shuffle_gen)] \
            if (train_mode and shuffle_gen is not None) else idx_pool
        bs = args.batch_size
        n = order.numel()
        losses, counts = 0.0, 0
        for i in range(0, n - (bs if train_mode else 0) + (0 if train_mode else 1), bs):
            bidx = order[i:i + bs].to(device)
            if bidx.numel() == 0:
                continue
            z_t, z_g, h_norm, a_t = sample_batch(
                Z, A_flat, t_gidx, maxh, ep_base, bidx, args.H_max, device, args.ablate_horizon)
            pred = head(z_t, z_g, h_norm)
            loss = F.mse_loss(pred, a_t)
            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
            losses += loss.item() * bidx.numel(); counts += bidx.numel()
        torch.set_grad_enabled(True)
        return losses / max(counts, 1)

    sh_gen = torch.Generator().manual_seed(args.seed + 1)
    best_val = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(train_idx, True, sh_gen)
        sched.step()
        va = run_epoch(val_idx, False)
        print(f"[gcidm-cache] epoch {ep+1}/{args.epochs}  train_mse={tr:.5f}  val_mse={va:.5f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s", flush=True)
        torch.save(head.state_dict(), run_dir / "gcidm_head_latest.pt")
        if va < best_val:
            best_val = va
            torch.save(head.state_dict(), run_dir / "gcidm_head_best.pt")
    print(f"[gcidm-cache] DONE  best_val_mse={best_val:.5f}  saved -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
