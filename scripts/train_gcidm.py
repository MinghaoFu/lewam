# External baseline: gcidm (arXiv 2605.08732) frozen-latent reproduction, not part of LeWAM infra.
# Precomputes frozen latents to a cache, trains only a separate GC-IDM head.
"""Hindsight trainer for GC-IDM on FROZEN LeWM latents (faithful `gcidm` repro).

Two phases (the encoder is frozen, so we PRECOMPUTE its latents once -- this is
what makes gcidm's "~20 min/env" feasible; re-encoding 224x224 images from the
100GB lance dataset every epoch is I/O-bound at ~5s/batch):

  PHASE 1 -- precompute (cached to disk):
    For every episode, load its frameskip-subsampled frames + raw actions, apply
    the SAME image preproc the LeWM training used, encode under no_grad to z
    (192-d cls), and z-score the action blocks with the dataset action stats.
    Store per-episode (latents[n_obs+1, 192], action_blocks[n_obs, 25]).
    Cache => $STABLEWM_HOME/checkpoints/<run>/latents_cache.pt (reused across runs).

  PHASE 2 -- hindsight MSE on cached latents (pure MLP, fast):
    Sample (ep, t, h), h~Uniform[1,H_max]; tuple (z_t, z_{t+h}, h, a_t) where
    z_{t+h} clamps to the episode's last frame. loss = ||GCIDMHead(z_t,z_{t+h},h)-a_t||^2.
    Encoder never touched here (latents are cached) => seconds/epoch => true convergence.

Action handling matches eval: the GCIDMHead is trained on z-scored actions; the
eval policy un-normalizes with the SAME StandardScaler (gip.build_process), and
the stats are also saved in gcidm_config.json for provenance.

Saves: $STABLEWM_HOME/checkpoints/<run>/{gcidm_head_best.pt, gcidm_head_latest.pt,
gcidm_config.json}. ADDITIVE -- does not touch train.py / the LeWM training path.

Run (L40S, train+eval same box/GPU):
  STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 \
  MPLCONFIGDIR=/tmp/mpl_x TMPDIR=/tmp CUDA_VISIBLE_DEVICES=<g> \
    python train_gcidm.py --epochs 200 --H_max 50 --run_name cube_gcidm
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

from lewam.utils import get_img_preprocessor, get_column_normalizer
import lewam.models.gcidm as gcidm


# phase 1: precompute frozen latents (cached)
def precompute_latents(lewm, base, img_t, act_mean, act_std, frameskip, device,
                       enc_bs=256, bf16=True, max_eps=None):
    """Encode every episode's subsampled frames -> latents; z-score action blocks.

    Returns lists (per episode): lat[ep] (n_frames, 192) fp16, act[ep] (n_obs, 25) fp16.
    n_obs = number of complete frameskip action blocks; n_frames = n_obs (+1 if a
    trailing frame exists). Hindsight goals index into lat by obs-step (clamped)."""
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)  # (5,)
    act_std_t = torch.tensor(act_std, dtype=torch.float32)
    n_eps = len(base.lengths) if max_eps is None else min(max_eps, len(base.lengths))
    lat_list, act_list = [], []
    t0 = time.time()
    for ep in range(n_eps):
        L = int(base.lengths[ep])
        sl = base._load_slice(ep, 0, L)            # pixels (Fsub,C,H,W) uint8, action (L,5)
        pix = sl["pixels"]                         # (Fsub, C, H, W) uint8 (raw, NOT preproc here)
        raw_act = sl["action"]                     # (L, 5) raw actions
        if torch.is_tensor(pix):
            pix = pix
        else:
            pix = torch.as_tensor(np.asarray(pix))
        raw_act = raw_act if torch.is_tensor(raw_act) else torch.as_tensor(np.asarray(raw_act))
        # preprocess each frame exactly like training (img_t expects {'pixels':..})
        pp = img_t({"pixels": pix})["pixels"].float()   # (Fsub, C, H, W) normalized
        zs = []
        for i in range(0, pp.shape[0], enc_bs):
            chunk = pp[i:i + enc_bs].to(device)
            if bf16 and device != "cpu":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                    z = lewm.encode({"pixels": chunk.unsqueeze(1)})["emb"][:, 0].float()
            else:
                with torch.no_grad():
                    z = lewm.encode({"pixels": chunk.unsqueeze(1)})["emb"][:, 0]
            zs.append(z.cpu())
        z_ep = torch.cat(zs, dim=0)                # (Fsub, 192)
        # action blocks: complete frameskip chunks; n_obs = L // frameskip
        n_obs = L // frameskip
        a = raw_act[: n_obs * frameskip].reshape(n_obs, frameskip * raw_act.shape[1])  # (n_obs, 25)
        a = a.reshape(n_obs, frameskip, raw_act.shape[1])
        a = (a - act_mean_t) / act_std_t           # z-score per raw dim
        a = a.reshape(n_obs, frameskip * raw_act.shape[1])  # (n_obs, 25)
        # align latents to obs-steps: z_ep has Fsub = ceil(L/fs) frames at raw 0,fs,2fs...
        # keep the first n_obs+1 frames (so a goal at n_obs is available); clamp later.
        z_keep = z_ep[: n_obs + 1] if z_ep.shape[0] >= n_obs + 1 else z_ep
        lat_list.append(z_keep.half())
        act_list.append(a.half())
        if (ep + 1) % 1000 == 0:
            dt = time.time() - t0
            print(f"[gcidm-precompute] {ep+1}/{n_eps} eps  ({dt:.0f}s, {(ep+1)/dt:.1f} eps/s)", flush=True)
    print(f"[gcidm-precompute] DONE {n_eps} eps in {time.time()-t0:.0f}s", flush=True)
    return lat_list, act_list


# phase 2: flat GPU tensors + vectorized hindsight sampling
def flatten_for_training(lat_list, act_list, device):
    """Stack per-episode latents/actions into flat GPU tensors + an index of
    valid (global_frame_t, max_h, action) tuples for vectorized hindsight sampling.

    Returns:
      Z       : (n_frames_total, 192) fp16 on device -- all episode latents, concatenated
      A_flat  : (n_samples, 25) fp16 -- action block a_t for each valid sample
      t_gidx  : (n_samples,) long   -- global frame index of z_t
      maxh    : (n_samples,) long   -- max hindsight horizon for that sample (>=1)
      ep_base : (n_samples,) long   -- global frame index of the episode's frame 0
    A valid sample = (ep, t) with at least one future frame (t < n_lat-1) and a
    real action block at t (t < n_obs)."""
    offsets = []  # global start of each episode in Z
    Zs = []
    off = 0
    for z in lat_list:
        offsets.append(off)
        Zs.append(z)
        off += z.shape[0]
    Z = torch.cat(Zs, dim=0).to(device)            # (n_frames_total, 192) fp16
    t_gidx, maxh, ep_base, A = [], [], [], []
    for ep, a in enumerate(act_list):
        n_obs = a.shape[0]
        n_lat = lat_list[ep].shape[0]
        base = offsets[ep]
        last = n_lat - 1
        n_valid = min(n_obs, n_lat - 1)            # t in [0, n_valid)
        for t in range(n_valid):
            t_gidx.append(base + t)
            maxh.append(last - t)                  # >=1 by construction
            ep_base.append(base)
            A.append(a[t])
    t_gidx = torch.tensor(t_gidx, dtype=torch.long)
    maxh = torch.tensor(maxh, dtype=torch.long)
    ep_base = torch.tensor(ep_base, dtype=torch.long)
    A_flat = torch.stack(A, dim=0)                 # (n_samples, 25) fp16
    return Z, A_flat, t_gidx, maxh, ep_base


def sample_batch(Z, A_flat, t_gidx, maxh, ep_base, idx, H_max, device, ablate_horizon):
    """Vectorized hindsight tuple for the sample indices `idx` (on device).
      h ~ Uniform[1, H_max] per sample, clamped to maxh (goal <= last frame).
      z_t = Z[t_gidx], z_goal = Z[t_gidx + h], a_t = A_flat[idx], h_norm = min(h,H_max)/H_max."""
    n = idx.shape[0]
    ti = t_gidx[idx]                               # (n,) global frame of z_t
    mh = maxh[idx]                                 # (n,) max horizon
    base = ep_base[idx]
    # h ~ Uniform[1, H_max], then clamp to mh
    h = torch.randint(1, H_max + 1, (n,), device=device)
    h = torch.minimum(h, mh.to(device))
    h = torch.clamp(h, min=1)
    gi = ti.to(device) + h                         # goal frame (already <= last by clamp)
    z_t = Z[ti.to(device)].float()
    z_g = Z[gi].float()
    a_t = A_flat[idx].float()
    h_norm = torch.clamp(h.float(), max=H_max) / H_max
    if ablate_horizon:
        h_norm = torch.zeros_like(h_norm)
    return z_t, z_g, h_norm, a_t


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
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--run_name", type=str, default="cube_gcidm")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--ablate_horizon", action="store_true")
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--enc_bs", type=int, default=256)
    ap.add_argument("--max_eps", type=int, default=0, help="cap episodes (smoke); 0=all")
    ap.add_argument("--cache_run", type=str, default="cube_gcidm",
                    help="run folder whose latents_cache.pt to share (precompute once)")
    ap.add_argument("--dataset_name", type=str, default="ogbench/ogb_cube_single.lance",
                    help="swm dataset name (lance for cube, *.h5 for pusht/robomimic)")
    ap.add_argument("--keys_to_load", type=str, default="pixels,action,observation",
                    help="comma-separated dataset columns to load (pusht: pixels,action)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = Path(swm.data.utils.get_cache_dir())
    weights = args.weights or str(cache / "decoders" / "cube_ours_lewm_weights.pt")

    # dataset (for latents / action stats / episode structure)
    _ktl = [k.strip() for k in args.keys_to_load.split(",") if k.strip()]
    _ktc = [k for k in _ktl if k != "pixels"]  # action(+observation) cached for stats
    dataset_cfg = dict(
        name=args.dataset_name, num_steps=args.history_size + 1, frameskip=5,
        keys_to_load=_ktl, keys_to_cache=_ktc,
    )
    name = dataset_cfg.pop("name")
    base = swm.data.load_dataset(name, transform=None, cache_dir=None, **dataset_cfg)
    frameskip = int(base.frameskip)
    raw_adim = int(base.get_dim("action"))            # 5
    action_block_dim = raw_adim * frameskip           # 25

    # action z-score stats (same normalizer train.py would build)
    act_norm = get_column_normalizer(base, "action", "action")
    _zn = act_norm.lambd
    act_mean = _zn.mean.squeeze(0).cpu().numpy().tolist()
    act_std = _zn.std.squeeze(0).cpu().numpy().tolist()
    # image preprocessor
    img_t = get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size)

    lewm = build_frozen_lewm(weights, embed_dim=192, history_size=args.history_size,
                             img_size=args.img_size, action_block_dim=action_block_dim).to(device)

    # phase 1: precompute (cached on disk; shared across runs)
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), args.cache_run)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "latents_cache.pt"
    max_eps = args.max_eps or None
    if cache_path.exists() and max_eps is None:
        print(f"[gcidm-train] loading cached latents <- {cache_path}", flush=True)
        blob = torch.load(cache_path, map_location="cpu")
        lat_list, act_list = blob["lat"], blob["act"]
        # sanity: cache action stats must match (else re-precompute)
        if blob.get("act_mean") != act_mean:
            print("[gcidm-train] WARNING cached action stats differ; re-precomputing")
            lat_list, act_list = precompute_latents(lewm, base, img_t, act_mean, act_std,
                                                    frameskip, device, args.enc_bs, max_eps=max_eps)
            torch.save({"lat": lat_list, "act": act_list, "act_mean": act_mean,
                        "act_std": act_std}, cache_path)
    else:
        lat_list, act_list = precompute_latents(lewm, base, img_t, act_mean, act_std,
                                                frameskip, device, args.enc_bs, max_eps=max_eps)
        if max_eps is None:
            torch.save({"lat": lat_list, "act": act_list, "act_mean": act_mean,
                        "act_std": act_std}, cache_path)
            print(f"[gcidm-train] cached latents -> {cache_path}", flush=True)

    # sanity: latent geometry not collapsed
    allz = torch.cat([z[:1].float() for z in lat_list[:2000]], dim=0)
    print(f"[gcidm-train] latent sanity: n_eps={len(lat_list)} "
          f"per-dim std(mean over dims)={allz.std(0).mean().item():.4f} "
          f"(0 => collapsed)", flush=True)

    # phase 2: hindsight training on flat GPU tensors (vectorized, num_workers=0)
    Z, A_flat, t_gidx, maxh, ep_base = flatten_for_training(lat_list, act_list, device)
    A_flat = A_flat.to(device)
    # index tensors live on device so device-side batch indices gather correctly
    t_gidx = t_gidx.to(device); maxh = maxh.to(device); ep_base = ep_base.to(device)
    n_samples = t_gidx.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_samples, generator=g)
    n_val = int(round((1 - args.train_split) * n_samples))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[gcidm-train] hindsight samples n={n_samples} train={train_idx.numel()} "
          f"val={val_idx.numel()} frames_total={Z.shape[0]} "
          f"H_max={args.H_max} action_block={action_block_dim}", flush=True)

    head = gcidm.GCIDMHead(emb_dim=192, action_dim=action_block_dim,
                           hidden_dim=args.hidden_dim, n_freqs=64, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[gcidm-train] GCIDMHead params={n_params:,} ({n_params/1e6:.3f}M)  "
          f"ablate_horizon={args.ablate_horizon}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    cfg_out = dict(
        head="gcidm", emb_dim=192, action_dim=action_block_dim, hidden_dim=args.hidden_dim,
        n_freqs=64, dropout=0.1, H_max=args.H_max, frameskip=frameskip, action_raw_dim=raw_adim,
        weights=weights, action_mean=act_mean, action_std=act_std, ablate_horizon=args.ablate_horizon,
    )
    (run_dir / "gcidm_config.json").write_text(json.dumps(cfg_out, indent=2))

    def run_epoch(idx_pool, train_mode, shuffle_gen=None):
        head.train(train_mode)
        torch.set_grad_enabled(train_mode)
        order = idx_pool[torch.randperm(idx_pool.numel(), generator=shuffle_gen)] if (train_mode and shuffle_gen is not None) else idx_pool
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
        dt = time.time() - t0
        print(f"[gcidm-train] epoch {ep+1}/{args.epochs}  train_mse={tr:.5f}  val_mse={va:.5f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {dt:.1f}s", flush=True)
        torch.save(head.state_dict(), run_dir / "gcidm_head_latest.pt")
        if va < best_val:
            best_val = va
            torch.save(head.state_dict(), run_dir / "gcidm_head_best.pt")
    print(f"[gcidm-train] DONE  best_val_mse={best_val:.5f}  saved -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
