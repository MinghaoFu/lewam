# LeWAM-Unified: the split (goal-conditioned action head + goal-conditioned dynamics) trained
# SEQUENCE-PARALLEL over a start->goal trajectory, with ONE shared context latent per position.
#
#   sample a start t + goal distance h~U[1,H_max] (clamped to the episode tail);
#   trajectory frames [t .. t+h] -> encoder -> [z_0 .. z_h]   (ONE encode)
#   c_tau = z_tau + g*Aggr(z_<=tau)                          # causal aggregator, per position
#   a_pred_tau = gc_head(c_tau, z_goal, h_tau)               # z_goal = z_h (endpoint, shared)
#   z_pred_tau = dynamics(c_tau, a_tau, z_goal)              # target z_{tau+1}
#
#   L = w_act*MSE(a_pred, a) + w_dyn*MSE(z_pred, z[1:]) + w_reg*SIGReg(z) + w_cyc*consistency
#       (all per-position, masked to the valid (non-pad) positions)
#
# Forked from train_lewam_gc.py: preload/flatten/optim/schedule/ckpt-sync are unchanged; the only
# new pieces are (a) the dataset samples a start->goal TRAJECTORY, (b) a padding collate for
# variable-length trajectories, (c) the loss is per-position (masked) instead of one-step. The goal
# is z[-1] of the trajectory (no separate goal encode); dynamics targets are z[1:]. Each start is a
# FRESH start (no prior history), matching the eval episode start -- so no warmup/prefix.
#
#   python scripts/train_lewam_unified.py --dataset_name reacher.h5 \
#       --run_name reacher_lewam_unified --epochs 50 --H_max 50 --agg_depth 2 \
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
    """Build flat Frames tensor + a per-FRAME action tensor + per-start index (t, max horizon).
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
    # A_frame[base+t] = action block AT obs-step t (== the action that led INTO frame t+1). Zero at
    # the episode's last frame. t_gidx = valid START frames; maxh = frames from the start to the
    # episode end (the max goal distance sampleable from that start).
    adim = act_list[0].shape[1]
    A_frame = torch.zeros((total, adim), dtype=act_list[0].dtype)
    t_gidx, maxh_list = [], []
    for ep, a in enumerate(act_list):
        n_obs = a.shape[0]
        n_fr = (offsets[ep + 1] if ep + 1 < len(offsets) else total) - offsets[ep]
        base = offsets[ep]
        last = n_fr - 1
        A_frame[base:base + min(n_obs, n_fr)] = a[: min(n_obs, n_fr)]
        n_valid = min(n_obs, n_fr - 1)
        for t in range(n_valid):
            t_gidx.append(base + t)
            maxh_list.append(last - t)   # frames from t to the episode end
    return (Frames, A_frame,
            torch.tensor(t_gidx, dtype=torch.long),
            torch.tensor(maxh_list, dtype=torch.long))


# --------------------------------------------------------------------------- #
# Trajectory Dataset: each item is a start->goal sub-trajectory. Sample h~U[1,H_max] clamped to    #
# the episode tail; return frames [t .. t+h] (h+1 frames) + the z-scored action blocks a_t..a_{t+h-1}#
# + the length h. The goal is the trajectory endpoint (z[-1]); dynamics targets are z[1:]; horizons #
# per position run h..1. A fresh start (no prior history) matches the eval episode start.           #
# --------------------------------------------------------------------------- #
class SeqTrajDataset(Dataset):
    def __init__(self, frames, a_frame, t_gidx, maxh, indices, h_max):
        self.frames = frames          # [N,3,H,W] fp16, CPU, shared read-only
        self.a_frame = a_frame        # [N,adim] fp16, CPU  per-frame action block
        self.t_gidx = t_gidx          # [M] long  valid start frames
        self.maxh = maxh              # [M] long  frames from start to episode end
        self.indices = indices        # [K] long  train or val subset
        self.h_max = int(h_max)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        t = int(self.t_gidx[idx])
        mh = int(self.maxh[idx])
        h = int(torch.randint(1, self.h_max + 1, (1,)).item())
        h = min(h, mh)
        if h < 1:
            h = 1
        frames = self.frames[t:t + h + 1]     # (h+1, 3, H, W)  z_0..z_h (z_h = goal)
        actions = self.a_frame[t:t + h]       # (h, adim)       a_t..a_{t+h-1}
        return frames, actions, h


def collate_pad(batch):
    """Pad variable-length trajectories to the batch's max length. Returns
    frames (B, Lmax+1, 3, H, W), actions (B, Lmax, adim), lengths (B,). Pad frames/actions are
    zeros and are masked out of the loss by `lengths` in the loop."""
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    Lmax = int(lengths.max().item())
    C, H, W = batch[0][0].shape[1:]
    adim = batch[0][1].shape[1]
    B = len(batch)
    frames = torch.zeros((B, Lmax + 1, C, H, W), dtype=batch[0][0].dtype)
    actions = torch.zeros((B, Lmax, adim), dtype=batch[0][1].dtype)
    for i, (fr, ac, h) in enumerate(batch):
        frames[i, : h + 1] = fr
        actions[i, : h] = ac
    return frames, actions, lengths


# --------------------------------------------------------------------------- #
# Main training                                                                #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128,
                    help="trajectories per batch (each carries up to H_max decision points, so the "
                         "effective decision-point batch is much larger; raise/lower per GPU mem)")
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--dynamics_lr", type=float, default=3e-4)
    ap.add_argument("--agg_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--H_max", type=int, default=50,
                    help="max goal distance in OBS-STEPS (frames are 1-per-obs-step, post-frameskip; "
                         "NOT divided by frameskip); == the max trajectory length sampled")
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
                    help="durable dir (e.g. an HDFS mount) to mirror config + best ckpt into on "
                         "each improvement, so losing the worker's ephemeral disk never costs the run.")
    # unified-specific (ablation arms; ALL present so every ckpt strict-loads under one adapter)
    ap.add_argument("--agg_depth", type=int, default=4,
                    help="causal-transformer depth of the aggregator")
    ap.add_argument("--agg_heads", type=int, default=4)
    ap.add_argument("--agg_residual", action="store_true",
                    help="c = z + Aggr(z) with a ZERO-INIT correction (boots as the split, "
                         "identity at init); off (default) = c = Aggr(z), no residual")
    ap.add_argument("--agg_gate", action="store_true",
                    help="input-dependent sigmoid gate on the residual correction "
                         "(c = z + g(z)*Aggr(z); still boots as split). residual only.")
    ap.add_argument("--agg_action_cond", action="store_true",
                    help="condition each token z_tau (AdaLN) on the embedded previous action "
                         "a_{tau-1} (null-action at the sequence start); conditioning, NOT tokens")
    # loss weights
    ap.add_argument("--w_act", type=float, default=1.0)
    ap.add_argument("--w_dyn", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=0.04)
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
                         agg_depth=args.agg_depth, agg_heads=args.agg_heads,
                         agg_residual=args.agg_residual, agg_gate=args.agg_gate,
                         agg_action_cond=args.agg_action_cond).to(device)
    sigreg = SIGReg().to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_agg = sum(p.numel() for p in model.aggregator.parameters())
    n_head = sum(p.numel() for p in model.gc_head.parameters())
    n_dyn = sum(p.numel() for p in model.dynamics.parameters())
    print(f"[lewam-uni] encoder={n_enc/1e6:.2f}M  agg={n_agg/1e6:.2f}M  gc_head={n_head/1e6:.2f}M  "
          f"dynamics={n_dyn/1e6:.2f}M  total={(n_enc+n_agg+n_head+n_dyn)/1e6:.2f}M  "
          f"agg_depth={args.agg_depth} action_cond={args.agg_action_cond}", flush=True)

    # ---- data ----
    frame_list, act_list = preload_frames(base, img_t, act_mean, act_std,
                                          frameskip, max_eps=max_eps)
    Frames, A_frame, t_gidx, maxh = flatten_for_training(frame_list, act_list, "cpu")
    del frame_list
    n_starts = t_gidx.shape[0]
    print(f"[lewam-uni] frames={Frames.shape} starts={n_starts}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_starts, generator=g)
    n_val = int(round((1 - args.train_split) * n_starts))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[lewam-uni] train={train_idx.numel()} val={val_idx.numel()} "
          f"H_max={args.H_max} action_block={action_block_dim}", flush=True)

    # ---- DataLoaders ----
    train_ds = SeqTrajDataset(Frames, A_frame, t_gidx, maxh, train_idx, args.H_max)
    val_ds = SeqTrajDataset(Frames, A_frame, t_gidx, maxh, val_idx, args.H_max)
    _loader_common = dict(
        batch_size=args.batch_size, pin_memory=True, drop_last=False,
        num_workers=args.num_workers, collate_fn=collate_pad,
    )
    if args.num_workers > 0:
        _loader_common.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
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
        agg_depth=args.agg_depth, agg_heads=args.agg_heads,
        agg_residual=args.agg_residual, agg_gate=args.agg_gate,
        agg_action_cond=args.agg_action_cond,
        H_max=args.H_max, frameskip=frameskip, action_raw_dim=raw_adim,
        action_mean=act_mean, action_std=act_std,
        w_act=args.w_act, w_dyn=args.w_dyn, w_reg=args.w_reg, w_cyc=args.w_cyc,
        ablate_dynamics=args.ablate_dynamics,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr,
        dynamics_lr=args.dynamics_lr, agg_lr=args.agg_lr,
    )
    (run_dir / "lewam_unified_config.json").write_text(json.dumps(cfg_out, indent=2))
    durable_sync([run_dir / "lewam_unified_config.json"], args.ckpt_sync_dir)

    D = 192
    Hmax = float(args.H_max)

    def run_batch(frames, actions, lengths, train):
        """One sequence-parallel forward: encode the trajectory once, aggregate causally, apply the
        heads at every position, and return per-position masked (act, dyn) losses + valid latents."""
        B, Lp1 = frames.shape[0], frames.shape[1]
        Lmax = Lp1 - 1
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True).float()
        lengths = lengths.to(device, non_blocking=True)
        z = model.encode(frames.reshape(B * Lp1, *frames.shape[2:]).float()).reshape(B, Lp1, D)
        states = z[:, :Lmax].float()                    # (B,Lmax,D) decision-point states
        next_tgt = z[:, 1:Lp1].float()                  # (B,Lmax,D) dynamics targets z[k+1]
        goal = z[torch.arange(B, device=device), lengths].float()       # (B,D) endpoint z[h]
        goal_bc = goal.unsqueeze(1).expand(B, Lmax, D)
        pos = torch.arange(Lmax, device=device)
        valid = pos.unsqueeze(0) < lengths.unsqueeze(1)                 # (B,Lmax) bool
        horizon = (lengths.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=1)
        h_norm = horizon.clamp(max=args.H_max).float() / Hmax          # (B,Lmax)
        a_prev = a_prev_mask = None
        if model.agg_action_cond:
            a_prev = torch.zeros_like(actions)
            a_prev[:, 1:] = actions[:, :-1]                            # a_{tau-1}; position 0 -> null
            a_prev_mask = valid & (pos.unsqueeze(0) >= 1)
        a_pred, z_pred = model.forward_seq(states, goal_bc, h_norm, actions, a_prev, a_prev_mask)
        m = valid.unsqueeze(-1).float()
        nval = valid.sum().clamp(min=1)
        loss_act = ((a_pred - actions) ** 2 * m).sum() / (nval * actions.shape[-1])
        loss_dyn = ((z_pred - next_tgt) ** 2 * m).sum() / (nval * D)
        loss_cyc = torch.zeros((), device=device)
        if train and args.w_cyc > 0:
            c = model.aggregate(states, a_prev, a_prev_mask)
            z_cyc = model.dynamics(c.reshape(B * Lmax, D), a_pred.reshape(B * Lmax, -1),
                                   goal_bc.reshape(B * Lmax, D)).reshape(B, Lmax, D)
            loss_cyc = ((z_cyc - next_tgt.detach()) ** 2 * m).sum() / (nval * D)
        loss_reg = sigreg(states[valid].unsqueeze(0)) if train else torch.zeros((), device=device)
        return loss_act, loss_dyn, loss_reg, loss_cyc, int(nval.item())

    # ---- training loop ----
    best_val = float("inf")

    for ep in range(args.epochs):
        t0 = time.time()

        # ---- train ----
        model.train()
        tr_act, tr_dyn, tr_reg, tr_cyc, tr_count = 0.0, 0.0, 0.0, 0.0, 0
        for frames, actions, lengths in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_act, loss_dyn, loss_reg, loss_cyc, nval = run_batch(
                    frames, actions, lengths, train=True)
                loss = (args.w_act * loss_act + args.w_dyn * loss_dyn
                        + args.w_reg * loss_reg + args.w_cyc * loss_cyc)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_act += loss_act.item() * nval
            tr_dyn += loss_dyn.item() * nval
            tr_reg += loss_reg.item() * nval
            tr_cyc += loss_cyc.item() * nval
            tr_count += nval
        sched.step()

        # ---- val ----
        model.eval()
        va_act, va_dyn, va_count = 0.0, 0.0, 0
        with torch.no_grad():
            for frames, actions, lengths in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_act, loss_dyn, _, _, nval = run_batch(
                        frames, actions, lengths, train=False)
                va_act += loss_act.item() * nval
                va_dyn += loss_dyn.item() * nval
                va_count += nval

        tr_a = tr_act / max(tr_count, 1); tr_d = tr_dyn / max(tr_count, 1)
        tr_r = tr_reg / max(tr_count, 1); tr_c = tr_cyc / max(tr_count, 1)
        va_a = va_act / max(va_count, 1); va_d = va_dyn / max(va_count, 1)
        lrs = sched.get_last_lr()
        dt = time.time() - t0
        cyc_str = f"  cyc={tr_c:.5f}" if args.w_cyc > 0 else ""
        print(f"[lewam-uni] ep {ep+1}/{args.epochs}  "
              f"act={tr_a:.5f}/{va_a:.5f}  dyn={tr_d:.5f}/{va_d:.5f}  reg={tr_r:.5f}{cyc_str}  "
              f"lr_enc={lrs[0]:.2e}  {dt:.1f}s", flush=True)

        # ---- save checkpoints ----
        # the full model state_dict (encoder./aggregator./gc_head./dynamics.[/gate_proj.]) loads
        # strict into a LeWAMUnified rebuilt from the config (gip.load_lewam_unified_model).
        full_sd = model.state_dict()
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
