# LeWAM-Unified: the split (goal-conditioned action head + goal-conditioned dynamics)
# given a short TEMPORAL state-window and one SHARED context latent feeding both heads.
#
#   pixels_{t-k..t} -> encoder -> [z_{t-k}..z_t]
#   c_t = z_t + Aggr([z_{t-k..t}])                # shallow causal aggregator, ZERO-INIT residual
#   a_pred = gc_head(c_t, z_goal, h)              # SAME head as the split, off the shared c_t
#   z_pred = dynamics(c_t, a_t, z_goal)           # SAME dynamics, off the shared c_t
#
#   L = w_act*MSE(a_pred, a_t) + w_dyn*MSE(z_pred, z_next) + w_reg*SIGReg(z_t)
#       + w_cyc*MSE(dynamics(c_t, a_pred, z_g), sg(z_next))
#
# Forked from scripts/train_lewam_gc.py (LeWAMSplit trainer): same data/optim/loss, the only
# additions are the STATE WINDOW (each sample carries [z_{t-k..t}, goal, next] instead of
# [t, goal, next]) and the aggregator param group. See docs/LEWAM_UNIFIED_HANDOFF.md.
#
#   python scripts/train_lewam_unified.py --dataset_name reacher.h5 \
#       --run_name reacher_lewam_unified --epochs 50 --H_max 50 --window 8 --agg_depth 2 \
#       --ckpt_sync_dir /mnt/hdfs/.../reacher_lewam_unified
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True  # fixed 224x224 input, let cuDNN pick the conv algo

import stable_worldmodel as swm

from lewam.utils import get_img_preprocessor, get_column_normalizer
from lewam.models.module import SIGReg
from lewam.models.lewam_unified import LeWAMUnified


def durable_sync(files, dst_dir):
    """Mirror files into dst_dir (e.g. an HDFS mount) so a checkpoint survives the loss of
    the worker's ephemeral local disk. Writes to a sibling .tmp then os.replace, so a crash
    mid-copy can't leave a truncated checkpoint. Best-effort: never raises into training."""
    if not dst_dir:
        return
    try:
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            f = Path(f)
            if not f.exists():
                continue
            tmp = dst / (f.name + ".tmp")
            shutil.copyfile(f, tmp)
            os.replace(tmp, dst / f.name)
    except Exception as e:
        print(f"[lewam-uni] WARN durable ckpt sync -> {dst_dir} failed: {e}", flush=True)


# --------------------------------------------------------------------------- #
# Data loading (identical to train_lewam_gc.py)                                #
# --------------------------------------------------------------------------- #
def preload_frames(base, img_t, act_mean, act_std, frameskip, max_eps=None):
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)
    act_std_t = torch.tensor(act_std, dtype=torch.float32)
    n_eps = len(base.lengths) if max_eps is None else min(max_eps, len(base.lengths))
    frame_list, act_list = [], []
    t0 = time.time()
    for ep in range(n_eps):
        L = int(base.lengths[ep])
        sl = base._load_slice(ep, 0, L)
        pix = sl["pixels"]
        raw_act = sl["action"]
        if not torch.is_tensor(pix):
            pix = torch.as_tensor(np.asarray(pix))
        raw_act = raw_act if torch.is_tensor(raw_act) else torch.as_tensor(np.asarray(raw_act))
        pp = img_t({"pixels": pix})["pixels"].float()
        n_obs = L // frameskip
        a = raw_act[:n_obs * frameskip].reshape(n_obs, frameskip * raw_act.shape[1])
        a = a.reshape(n_obs, frameskip, raw_act.shape[1])
        a = (a - act_mean_t) / act_std_t
        a = a.reshape(n_obs, frameskip * raw_act.shape[1])
        n_keep = min(pp.shape[0], n_obs + 1)
        frame_list.append(pp[:n_keep].half())
        act_list.append(a.half())
        if (ep + 1) % 500 == 0:
            print(f"[lewam-uni] preload {ep+1}/{n_eps} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[lewam-uni] preload DONE {n_eps} eps in {time.time()-t0:.0f}s", flush=True)
    return frame_list, act_list


def flatten_for_training(frame_list, act_list, device):
    """Build flat Frames tensor + per-sample index (t, action, max horizon, ep base).
    Preallocate + free-as-you-go so peak RAM stays ~1x the frames (torch.cat would double it)."""
    offsets, off = [], 0
    for f in frame_list:
        offsets.append(off)
        off += f.shape[0]
    total = off
    C, H, W = frame_list[0].shape[1:]
    Frames = torch.empty((total, C, H, W), dtype=frame_list[0].dtype)
    for ep in range(len(frame_list)):
        f = frame_list[ep]
        Frames[offsets[ep]:offsets[ep] + f.shape[0]].copy_(f)
        frame_list[ep] = None  # free the source episode tensor incrementally
    if device != "cpu":
        Frames = Frames.to(device)
    # per-FRAME action block: A_frame[base+t] = a[t] (the action AT obs-step t, i.e. the action
    # that led INTO frame t+1). Used by agg_action_cond to look up each window token's previous
    # action a_{tau-1} = A_frame[frame-1]. Zero elsewhere (never read where the mask is False).
    adim = act_list[0].shape[1]
    A_frame = torch.zeros((total, adim), dtype=act_list[0].dtype)
    t_gidx, maxh_list, ep_base_list, A = [], [], [], []
    for ep, a in enumerate(act_list):
        n_obs = a.shape[0]
        n_fr = (offsets[ep + 1] if ep + 1 < len(offsets) else total) - offsets[ep]
        base = offsets[ep]
        last = n_fr - 1
        A_frame[base:base + min(n_obs, n_fr)] = a[: min(n_obs, n_fr)]
        n_valid = min(n_obs, n_fr - 1)
        for t in range(n_valid):
            t_gidx.append(base + t)
            maxh_list.append(last - t)
            ep_base_list.append(base)
            A.append(a[t])
    return (Frames,
            torch.stack(A, dim=0),
            torch.tensor(t_gidx, dtype=torch.long),
            torch.tensor(maxh_list, dtype=torch.long),
            torch.tensor(ep_base_list, dtype=torch.long),
            A_frame)


# --------------------------------------------------------------------------- #
# Windowed sample Dataset. Each sample carries the last W state frames (clamped  #
# to the episode start, left-padded by repeating the first frame = causal        #
# padding), plus the goal (t+h) and dynamics-target (t+1) frames. Sampling:      #
# h ~ U[1, H_max] clamped to the episode tail. Mirrors the eval buffer, which     #
# left-pads the first W-1 obs-steps of an episode with the episode's first frame. #
# --------------------------------------------------------------------------- #
class WindowPairDataset(Dataset):
    def __init__(self, frames, a_flat, t_gidx, maxh, ep_base, a_frame, indices, h_max,
                 window, ablate_horizon):
        self.frames = frames          # [N,3,H,W] fp16, CPU, shared read-only
        self.a_flat = a_flat          # [M,adim] fp16, CPU  action AT sample step t
        self.t_gidx = t_gidx          # [M] long  global frame index of t
        self.maxh = maxh              # [M] long  max horizon (frames to episode end)
        self.ep_base = ep_base        # [M] long  global index of the episode's first frame
        self.a_frame = a_frame        # [N,adim] fp16  per-frame action block (for agg_action_cond)
        self.indices = indices        # [K] long  train or val subset
        self.h_max = int(h_max)
        self.window = int(window)
        self.ablate_horizon = bool(ablate_horizon)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        ti = int(self.t_gidx[idx])
        mh = int(self.maxh[idx])
        base = int(self.ep_base[idx])
        h = int(torch.randint(1, self.h_max + 1, (1,)).item())
        if h > mh:
            h = mh
        if h < 1:
            h = 1
        gi = ti + h
        ni = ti + 1  # == min(ti+1, ti+mh) since mh >= 1
        W = self.window
        start = ti - (W - 1)
        # window oldest..current; left-pad by clamping to the episode's first frame
        frame_idx = [max(base, start + j) for j in range(W)]
        win = [self.frames[f] for f in frame_idx]
        # previous-action per token: a_{f-1} = A_frame[f-1], valid iff f > base (else null via mask)
        a_prev = torch.stack(
            [self.a_frame[f - 1] if f > base else torch.zeros_like(self.a_frame[0])
             for f in frame_idx], dim=0)                       # [W, adim]
        a_prev_mask = torch.tensor([f > base for f in frame_idx], dtype=torch.bool)  # [W]
        # stack [win_0..win_{W-1}, goal, next] -> collate -> [B, W+2, 3, H, W]; fp16 halves H2D bytes
        trip = torch.stack(win + [self.frames[gi], self.frames[ni]], dim=0)
        a_t = self.a_flat[idx]
        h_norm = 0.0 if self.ablate_horizon else min(h, self.h_max) / self.h_max
        return trip, a_t, h_norm, a_prev, a_prev_mask


# --------------------------------------------------------------------------- #
# Main training                                                                #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--dynamics_lr", type=float, default=3e-4)
    ap.add_argument("--agg_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--H_max", type=int, default=50)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--run_name", type=str, default="reacher_lewam_unified")
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--dataset_name", type=str, default="reacher.h5")
    ap.add_argument("--keys_to_load", type=str, default="pixels,action")
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--ckpt_sync_dir", type=str, default=None,
                    help="durable dir (e.g. an HDFS mount) to mirror config + best ckpt into "
                         "on each improvement, so losing the worker's ephemeral disk never "
                         "costs the run. run_dir stays the fast local write target.")
    # unified-specific (ablation arms; ALL present so every ckpt strict-loads under one adapter)
    ap.add_argument("--window", type=int, default=8,
                    help="state-window length W (# frames incl. current); 1 = no temporal context")
    ap.add_argument("--agg_depth", type=int, default=2,
                    help="causal-transformer depth of the aggregator (keep SHALLOW: 1-2)")
    ap.add_argument("--agg_heads", type=int, default=4)
    ap.add_argument("--agg_max_len", type=int, default=128,
                    help="positional-embedding capacity of the causal aggregator (>= window; "
                         ">= the longest sequence a future seq-parallel trainer will feed)")
    ap.add_argument("--no_agg_residual", action="store_true",
                    help="c_t = Aggr(window) instead of z_t + Aggr(window) (drops the zero-init "
                         "residual that boots the model as the split)")
    ap.add_argument("--agg_gate", action="store_true",
                    help="input-dependent sigmoid gate on the residual correction "
                         "(c_t = z_t + g(z_t)*Aggr; still boots as split). residual only.")
    ap.add_argument("--agg_action_cond", action="store_true",
                    help="condition each window token z_tau (AdaLN) on the embedded previous "
                         "action a_{tau-1} (learnable null-action at episode start); actions "
                         "enter as conditioning, NOT sequence tokens")
    # loss weights
    ap.add_argument("--w_act", type=float, default=1.0)
    ap.add_argument("--w_dyn", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=0.04)
    ap.add_argument("--ablate_horizon", action="store_true")
    ap.add_argument("--ablate_dynamics", action="store_true",
                    help="disable dynamics loss (w_dyn=0), keep gc_head only")
    ap.add_argument("--w_cyc", type=float, default=0.0,
                    help="FDM-IDM consistency loss weight (0=without, 1.0=with)")
    # data-pipeline knobs (no effect on loss/model logic)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--prefetch_factor", type=int, default=3)
    args = ap.parse_args()

    if args.ablate_dynamics:
        args.w_dyn = 0.0

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dataset ----
    _ktl = [k.strip() for k in args.keys_to_load.split(",") if k.strip()]
    _ktc = [k for k in _ktl if k != "pixels"]
    dataset_cfg = dict(name=args.dataset_name, num_steps=4, frameskip=5,
                       keys_to_load=_ktl, keys_to_cache=_ktc)
    name = dataset_cfg.pop("name")
    base = swm.data.load_dataset(name, transform=None, cache_dir=None, **dataset_cfg)
    frameskip = int(base.frameskip)
    raw_adim = int(base.get_dim("action"))
    action_block_dim = raw_adim * frameskip

    act_norm = get_column_normalizer(base, "action", "action")
    _zn = act_norm.lambd
    act_mean = _zn.mean.squeeze(0).cpu().numpy().tolist()
    act_std = _zn.std.squeeze(0).cpu().numpy().tolist()
    img_t = get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    max_eps = args.max_eps or None

    # ---- build model ----
    model = LeWAMUnified(embed_dim=192, action_dim=action_block_dim,
                         hidden_dim=args.hidden_dim, img_size=args.img_size, dropout=0.1,
                         window=args.window, agg_depth=args.agg_depth,
                         agg_heads=args.agg_heads, agg_residual=not args.no_agg_residual,
                         agg_gate=args.agg_gate, agg_action_cond=args.agg_action_cond,
                         agg_max_len=args.agg_max_len).to(device)
    sigreg = SIGReg().to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_agg = sum(p.numel() for p in model.aggregator.parameters())
    n_head = sum(p.numel() for p in model.gc_head.parameters())
    n_dyn = sum(p.numel() for p in model.dynamics.parameters())
    print(f"[lewam-uni] encoder={n_enc/1e6:.2f}M  agg={n_agg/1e6:.2f}M  gc_head={n_head/1e6:.2f}M  "
          f"dynamics={n_dyn/1e6:.2f}M  total={(n_enc+n_agg+n_head+n_dyn)/1e6:.2f}M  "
          f"window={args.window} agg_depth={args.agg_depth}", flush=True)

    # ---- data ----
    frame_list, act_list = preload_frames(base, img_t, act_mean, act_std,
                                          frameskip, max_eps=max_eps)
    Frames, A_flat, t_gidx, maxh, ep_base, A_frame = flatten_for_training(
        frame_list, act_list, "cpu")
    del frame_list
    n_samples = t_gidx.shape[0]
    print(f"[lewam-uni] frames={Frames.shape} samples={n_samples}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_samples, generator=g)
    n_val = int(round((1 - args.train_split) * n_samples))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[lewam-uni] train={train_idx.numel()} val={val_idx.numel()} "
          f"H_max={args.H_max} action_block={action_block_dim}", flush=True)

    # ---- DataLoaders ----
    train_ds = WindowPairDataset(Frames, A_flat, t_gidx, maxh, ep_base, A_frame, train_idx,
                                 args.H_max, args.window, args.ablate_horizon)
    val_ds = WindowPairDataset(Frames, A_flat, t_gidx, maxh, ep_base, A_frame, val_idx,
                               args.H_max, args.window, args.ablate_horizon)
    _loader_common = dict(
        batch_size=args.batch_size, pin_memory=True, drop_last=False,
        num_workers=args.num_workers,
    )
    if args.num_workers > 0:
        _loader_common.update(prefetch_factor=args.prefetch_factor,
                              persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True, **_loader_common)
    val_loader = DataLoader(val_ds, shuffle=False, **_loader_common)
    print(f"[lewam-uni] loaders ready: workers={args.num_workers} "
          f"prefetch={args.prefetch_factor} pin_memory=True", flush=True)

    # ---- optimizer: 4 param groups ----
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": args.encoder_lr},
        {"params": list(model.aggregator.parameters()), "lr": args.agg_lr},
        {"params": model.gc_head.parameters(), "lr": args.head_lr},
        {"params": model.dynamics.parameters(), "lr": args.dynamics_lr},
    ], weight_decay=args.weight_decay)

    total_epochs = args.epochs
    warmup = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup:
            return max(epoch / max(warmup, 1), 1e-2)
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---- save config ----
    cfg_out = dict(
        model="lewam_unified", z_dim=192, action_dim=action_block_dim,
        hidden_dim=args.hidden_dim, n_freqs=64, dropout=0.1,
        window=args.window, agg_depth=args.agg_depth, agg_heads=args.agg_heads,
        agg_max_len=args.agg_max_len, agg_residual=not args.no_agg_residual,
        agg_gate=args.agg_gate, agg_action_cond=args.agg_action_cond,
        H_max=args.H_max, frameskip=frameskip, action_raw_dim=raw_adim,
        action_mean=act_mean, action_std=act_std,
        w_act=args.w_act, w_dyn=args.w_dyn, w_reg=args.w_reg, w_cyc=args.w_cyc,
        ablate_horizon=args.ablate_horizon, ablate_dynamics=args.ablate_dynamics,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr,
        dynamics_lr=args.dynamics_lr, agg_lr=args.agg_lr,
    )
    (run_dir / "lewam_unified_config.json").write_text(json.dumps(cfg_out, indent=2))
    durable_sync([run_dir / "lewam_unified_config.json"], args.ckpt_sync_dir)

    W = args.window

    def encode_split(trip, n):
        """trip: [B, W+2, 3, H, W] -> (window [B,W,D], z_g [B,D], z_n [B,D]).
        One encoder pass over all (W+2)*B frames; ordering matches the stack
        [win_0..win_{W-1}, goal, next]."""
        frames_all = trip.permute(1, 0, 2, 3, 4).reshape((W + 2) * n, *trip.shape[2:]).float()
        z_all = model.encode(frames_all)                       # [(W+2)*n, D]
        window = z_all[: W * n].reshape(W, n, -1).permute(1, 0, 2)  # [n, W, D]
        z_g = z_all[W * n:(W + 1) * n]
        z_n = z_all[(W + 1) * n:(W + 2) * n]
        return window, z_g, z_n

    # ---- training loop ----
    best_val = float("inf")

    for ep in range(args.epochs):
        t0 = time.time()

        # ---- train ----
        model.train()
        tr_act, tr_dyn, tr_reg, tr_cyc, tr_count = 0.0, 0.0, 0.0, 0.0, 0

        for trip, a_t, h_norm, a_prev, a_prev_mask in train_loader:
            n = trip.shape[0]
            trip = trip.to(device, non_blocking=True)
            a_t = a_t.to(device, non_blocking=True).float()
            h_norm = h_norm.to(device, non_blocking=True).float()
            if model.agg_action_cond:
                a_prev = a_prev.to(device, non_blocking=True).float()
                a_prev_mask = a_prev_mask.to(device, non_blocking=True)
            else:
                a_prev = a_prev_mask = None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                window, z_g, z_n = encode_split(trip, n)
                window_f = window.float()
                z_g_f = z_g.float()
                z_n_f = z_n.float()
                z_t_f = window_f[:, -1]  # current-frame latent (for SIGReg)

                # no stop-grad: the encoder also learns from the dynamics target
                a_pred, z_n_pred = model(window_f, z_g_f, h_norm, a_t, a_prev, a_prev_mask)
                loss_act = F.mse_loss(a_pred, a_t)
                loss_dyn = F.mse_loss(z_n_pred, z_n_f)

                loss_reg = sigreg(z_t_f.unsqueeze(0))

                loss = (args.w_act * loss_act
                        + args.w_dyn * loss_dyn
                        + args.w_reg * loss_reg)

                # consistency: the predicted action through dynamics should reach z_n
                loss_cyc = torch.tensor(0.0, device=device)
                if args.w_cyc > 0:
                    c_t = model.aggregate(window_f, a_prev, a_prev_mask)[:, -1]  # last position
                    z_n_cyc = model.dynamics(c_t, a_pred, z_g_f)
                    loss_cyc = F.mse_loss(z_n_cyc, z_n_f.detach())
                    loss = loss + args.w_cyc * loss_cyc

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_act += loss_act.item() * n
            tr_dyn += loss_dyn.item() * n
            tr_reg += loss_reg.item() * n
            tr_cyc += loss_cyc.item() * n
            tr_count += n

        sched.step()

        # ---- val ----
        model.eval()
        va_act, va_dyn, va_count = 0.0, 0.0, 0
        with torch.no_grad():
            for trip, a_t, h_norm, a_prev, a_prev_mask in val_loader:
                n = trip.shape[0]
                trip = trip.to(device, non_blocking=True)
                a_t = a_t.to(device, non_blocking=True).float()
                h_norm = h_norm.to(device, non_blocking=True).float()
                if model.agg_action_cond:
                    a_prev = a_prev.to(device, non_blocking=True).float()
                    a_prev_mask = a_prev_mask.to(device, non_blocking=True)
                else:
                    a_prev = a_prev_mask = None

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    window, z_g, z_n = encode_split(trip, n)
                    window_f, z_g_f, z_n_f = window.float(), z_g.float(), z_n.float()
                    a_pred, z_n_pred = model(window_f, z_g_f, h_norm, a_t, a_prev, a_prev_mask)
                    loss_act = F.mse_loss(a_pred, a_t)
                    loss_dyn = F.mse_loss(z_n_pred, z_n_f)

                va_act += loss_act.item() * n
                va_dyn += loss_dyn.item() * n
                va_count += n

        tr_a = tr_act / max(tr_count, 1)
        tr_d = tr_dyn / max(tr_count, 1)
        tr_r = tr_reg / max(tr_count, 1)
        tr_c = tr_cyc / max(tr_count, 1)
        va_a = va_act / max(va_count, 1)
        va_d = va_dyn / max(va_count, 1)
        lrs = sched.get_last_lr()
        dt = time.time() - t0
        cyc_str = f"  cyc={tr_c:.5f}" if args.w_cyc > 0 else ""
        print(f"[lewam-uni] ep {ep+1}/{args.epochs}  "
              f"act={tr_a:.5f}/{va_a:.5f}  dyn={tr_d:.5f}/{va_d:.5f}  reg={tr_r:.5f}{cyc_str}  "
              f"lr_enc={lrs[0]:.2e}  {dt:.1f}s", flush=True)

        # ---- save checkpoints ----
        # key prefixes (encoder./aggregator./gc_head./dynamics.) are load-bearing: the
        # unified eval loader (gip.load_lewam_unified_model) reconstructs from them, strict.
        full_sd = {}
        for k, v in model.encoder.state_dict().items():
            full_sd[f"encoder.{k}"] = v
        for k, v in model.aggregator.state_dict().items():
            full_sd[f"aggregator.{k}"] = v
        for k, v in model.gc_head.state_dict().items():
            full_sd[f"gc_head.{k}"] = v
        for k, v in model.dynamics.state_dict().items():
            full_sd[f"dynamics.{k}"] = v
        torch.save(full_sd, run_dir / "lewam_unified_latest.pt")

        combined_val = va_a + va_d
        if combined_val < best_val:
            best_val = combined_val
            torch.save(full_sd, run_dir / "lewam_unified_best.pt")
            durable_sync([run_dir / "lewam_unified_config.json",
                          run_dir / "lewam_unified_best.pt"], args.ckpt_sync_dir)

    print(f"[lewam-uni] DONE  best_val={best_val:.5f}  saved -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
