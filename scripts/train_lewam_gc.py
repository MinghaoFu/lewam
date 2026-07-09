# LeWAM-GC: goal-conditioned action head + goal-conditioned dynamics, trained
# end-to-end with a learnable ViT-tiny encoder and a SIGReg anti-collapse term.
#
#   pixels_t -> encoder -> z_t --+-- gc_head(z_t, z_goal, h) -> a_pred
#                                +-- dynamics(z_t, a_t, z_goal) -> z_{t+1}
#
#   L = w_act*MSE(a_pred, a_t) + w_dyn*MSE(z_pred, z_next) + w_reg*SIGReg(z)
#       + w_cyc*MSE(dynamics(z_t, a_pred, z_g), sg(z_next))    # consistency, off by default
#
# This is the default LeWAM-GC trainer (fast DataLoader path). Ablations are flags:
#   --w_cyc W          consistency (cycle) loss weight   [0 = without, e.g. 1.0 = with]
#   --ablate_dynamics  drop the dynamics loss (gc_head only)
#   --ablate_horizon   drop horizon conditioning
#   --w_act --w_dyn --w_reg   per-term loss weights
#
#   python scripts/train_lewam_gc.py --dataset_name ogbench/cube_single_expert.h5 \
#       --run_name cube_lewam_gc --epochs 100 --H_max 50
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True  # fixed 224x224 input, let cuDNN pick the conv algo

import stable_worldmodel as swm
from hydra.utils import instantiate as hydra_instantiate

from lewam.utils import get_img_preprocessor, get_column_normalizer


# --------------------------------------------------------------------------- #
# Sinusoidal embedding (shared by GCHead and HorizonModulator)                #
# --------------------------------------------------------------------------- #
def _sinusoidal_embedding(h_norm, n_freqs=64):
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(n_freqs, device=h_norm.device).float()
        / max(n_freqs - 1, 1)
    )
    ang = h_norm.float()[:, None] * freqs[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


# --------------------------------------------------------------------------- #
# AdaLN-Zero block                                                            #
# --------------------------------------------------------------------------- #
class AdaLNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False, eps=1e-6)
        self.cond_proj = nn.Linear(cond_dim, 2 * in_dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond):
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.drop(self.act(self.fc(x)))


# --------------------------------------------------------------------------- #
# GC action head                                                              #
# --------------------------------------------------------------------------- #
class GCHead(nn.Module):
    """cat[z_t, z_goal] -> 3 AdaLN blocks -> action."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512,
                 n_freqs=64, cond_dim=128, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 2 * z_dim
        sin_dim = 2 * n_freqs
        self.horizon_mlp = nn.Sequential(
            nn.Linear(sin_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.block1 = AdaLNBlock(in_dim, hidden_dim, cond_dim, dropout)
        self.block2 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.block3 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, z_t, z_goal, h_norm):
        x = torch.cat([z_t, z_goal], dim=-1)
        cond = self.horizon_mlp(_sinusoidal_embedding(h_norm, self.n_freqs))
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)
        return self.out(x)


# --------------------------------------------------------------------------- #
# Goal-conditioned dynamics predictor                                          #
# --------------------------------------------------------------------------- #
class GoalCondDynamics(nn.Module):
    """cat[z_t, a_t, z_goal] -> z_{t+1}."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512):
        super().__init__()
        in_dim = 2 * z_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_t, a_t, z_goal):
        return self.net(torch.cat([z_t, a_t, z_goal], dim=-1))


# --------------------------------------------------------------------------- #
# SIGReg (anti-collapse regularizer)                                           #
# --------------------------------------------------------------------------- #
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# --------------------------------------------------------------------------- #
# Encoder builder (random-init ViT-tiny JEPA)                                  #
# --------------------------------------------------------------------------- #
def build_encoder(embed_dim=192, img_size=224, action_block_dim=25):
    cfg = OmegaConf.create({
        "_target_": "lewam.models.jepa.JEPA",
        "use_action_history": True, "use_proprio": False,
        "encoder": {
            "_target_": "stable_pretraining.backbone.utils.vit_hf",
            "size": "tiny", "patch_size": 14, "image_size": img_size,
            "pretrained": False, "use_mask_token": False,
        },
        "predictor": {
            "_target_": "lewam.models.module.ARPredictor", "num_frames": 3,
            "input_dim": embed_dim, "hidden_dim": embed_dim, "output_dim": embed_dim,
            "depth": 6, "heads": 16, "mlp_dim": 2048, "dim_head": 64,
            "dropout": 0.1, "emb_dropout": 0.0,
        },
        "action_encoder": {
            "_target_": "lewam.models.module.Embedder", "input_dim": action_block_dim,
            "emb_dim": embed_dim,
        },
        "projector": {
            "_target_": "lewam.models.module.MLP", "input_dim": embed_dim, "output_dim": embed_dim,
            "hidden_dim": 2048,
            "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
        "pred_proj": {
            "_target_": "lewam.models.module.MLP", "input_dim": embed_dim, "output_dim": embed_dim,
            "hidden_dim": 2048,
            "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
    })
    model = hydra_instantiate(cfg)
    model.interpolate_pos_encoding = True
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"[lewam-gc] encoder (random init): {n_enc:,} params")
    return model


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
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
            print(f"[lewam-gc] preload {ep+1}/{n_eps} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[lewam-gc] preload DONE {n_eps} eps in {time.time()-t0:.0f}s", flush=True)
    return frame_list, act_list


def flatten_for_training(frame_list, act_list):
    """Build the per-sample index (episode-local t, action, max horizon, episode id).

    Frames are kept as the per-episode list (no ``torch.cat`` into one giant
    contiguous tensor): concatenation transiently holds both the sources and the
    result, doubling peak RAM (~2x the dataset) right at the point where several
    big preloads coincide — the cause of repeated cgroup OOM-kills. Indexing into
    the list is numerically identical: frame (ep, t) here == Frames[offset[ep]+t]
    before. g = t+h and n = t+1 always fall inside the same episode (h <= maxh =
    last-t), so a single episode id per sample is enough."""
    t_local, ep_idx, maxh_list, A = [], [], [], []
    for ep, a in enumerate(act_list):
        n_obs = a.shape[0]
        n_fr = frame_list[ep].shape[0]
        last = n_fr - 1
        n_valid = min(n_obs, n_fr - 1)
        for t in range(n_valid):
            t_local.append(t)
            ep_idx.append(ep)
            maxh_list.append(last - t)
            A.append(a[t])
    return (frame_list,
            torch.stack(A, dim=0),
            torch.tensor(t_local, dtype=torch.long),
            torch.tensor(ep_idx, dtype=torch.long),
            torch.tensor(maxh_list, dtype=torch.long))


# --------------------------------------------------------------------------- #
# Sample Dataset. Workers gather frames off the shared RAM-resident fp16       #
# tensor (read-only, copy-on-write across forked workers), so the DataLoader   #
# overlaps the gather + pinning with GPU compute. Sampling: h ~ U[1, H_max]    #
# clamped to the episode tail, goal at t+h, dynamics target at t+1.            #
# --------------------------------------------------------------------------- #
class FramePairDataset(Dataset):
    def __init__(self, frames_list, a_flat, t_local, ep_idx, maxh, indices, h_max, ablate_horizon):
        self.frames_list = frames_list  # list of [n_fr_ep,3,H,W] fp16, CPU, shared read-only
        self.a_flat = a_flat          # [M,adim] fp16, CPU
        self.t_local = t_local        # [M] long — t within its episode
        self.ep_idx = ep_idx          # [M] long — episode id
        self.maxh = maxh              # [M] long
        self.indices = indices        # [K] long — train or val subset
        self.h_max = int(h_max)
        self.ablate_horizon = bool(ablate_horizon)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        ep = int(self.ep_idx[idx])
        t = int(self.t_local[idx])
        mh = int(self.maxh[idx])
        h = int(torch.randint(1, self.h_max + 1, (1,)).item())
        if h > mh:
            h = mh
        if h < 1:
            h = 1
        g = t + h
        n = t + 1  # == min(t+1, t+mh) since mh >= 1
        fe = self.frames_list[ep]  # [n_fr_ep,3,H,W]; g,n stay inside this episode
        # stack [t, g, n] so collate -> [B,3,3,H,W]; kept fp16 to halve H2D bytes
        trip = torch.stack([fe[t], fe[g], fe[n]], dim=0)
        a_t = self.a_flat[idx]
        h_norm = 0.0 if self.ablate_horizon else min(h, self.h_max) / self.h_max
        return trip, a_t, h_norm


# --------------------------------------------------------------------------- #
# Main training                                                                #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--dynamics_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--H_max", type=int, default=50)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--run_name", type=str, default="cube_lewam_gc")
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--dataset_name", type=str, default="ogbench/cube_single_expert.h5")
    ap.add_argument("--keys_to_load", type=str, default="pixels,action,observation")
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--w_act", type=float, default=1.0)
    ap.add_argument("--w_dyn", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=0.04)
    ap.add_argument("--ablate_horizon", action="store_true")
    ap.add_argument("--ablate_dynamics", action="store_true",
                    help="disable dynamics loss (w_dyn=0), keep gc_head only")
    ap.add_argument("--w_cyc", type=float, default=0.0,
                    help="FDM-IDM consistency loss weight (0=without, 1.0=with)")
    # data-pipeline knobs (no effect on loss/model logic)
    ap.add_argument("--num_workers", type=int, default=6,
                    help="DataLoader worker processes")
    ap.add_argument("--prefetch_factor", type=int, default=3,
                    help="batches prefetched per worker")
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

    # ---- build models ----
    lewm = build_encoder(embed_dim=192, img_size=args.img_size,
                         action_block_dim=action_block_dim).to(device)
    gc_head = GCHead(z_dim=192, action_dim=action_block_dim,
                     hidden_dim=args.hidden_dim, dropout=0.1).to(device)
    dynamics = GoalCondDynamics(z_dim=192, action_dim=action_block_dim,
                                hidden_dim=args.hidden_dim).to(device)
    sigreg = SIGReg().to(device)

    n_enc = sum(p.numel() for p in lewm.encoder.parameters())
    n_head = sum(p.numel() for p in gc_head.parameters())
    n_dyn = sum(p.numel() for p in dynamics.parameters())
    print(f"[lewam-gc] encoder={n_enc/1e6:.2f}M  gc_head={n_head/1e6:.2f}M  "
          f"dynamics={n_dyn/1e6:.2f}M  total={(n_enc+n_head+n_dyn)/1e6:.2f}M", flush=True)

    # ---- data ----
    frame_list, act_list = preload_frames(base, img_t, act_mean, act_std,
                                          frameskip, max_eps=max_eps)
    frames_list, A_flat, t_local, ep_idx, maxh = flatten_for_training(
        frame_list, act_list)
    n_frames = sum(f.shape[0] for f in frames_list)
    n_samples = t_local.shape[0]
    print(f"[lewam-gc] frames={n_frames} (over {len(frames_list)} eps) samples={n_samples}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_samples, generator=g)
    n_val = int(round((1 - args.train_split) * n_samples))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[lewam-gc] train={train_idx.numel()} val={val_idx.numel()} "
          f"H_max={args.H_max} action_block={action_block_dim}", flush=True)

    # ---- DataLoaders (overlap CPU gather + pinning with GPU compute) ----
    train_ds = FramePairDataset(frames_list, A_flat, t_local, ep_idx, maxh, train_idx,
                                args.H_max, args.ablate_horizon)
    val_ds = FramePairDataset(frames_list, A_flat, t_local, ep_idx, maxh, val_idx,
                              args.H_max, args.ablate_horizon)
    _loader_common = dict(
        batch_size=args.batch_size, pin_memory=True, drop_last=False,
        num_workers=args.num_workers,
    )
    if args.num_workers > 0:
        _loader_common.update(prefetch_factor=args.prefetch_factor,
                              persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True, **_loader_common)
    val_loader = DataLoader(val_ds, shuffle=False, **_loader_common)
    print(f"[lewam-gc] loaders ready: workers={args.num_workers} "
          f"prefetch={args.prefetch_factor} pin_memory=True", flush=True)

    # ---- optimizer: 3 param groups ----
    enc_params = list(lewm.encoder.parameters()) + list(lewm.projector.parameters())
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": gc_head.parameters(), "lr": args.head_lr},
        {"params": dynamics.parameters(), "lr": args.dynamics_lr},
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
        model="lewam_gc", z_dim=192, action_dim=action_block_dim,
        hidden_dim=args.hidden_dim, n_freqs=64, dropout=0.1,
        H_max=args.H_max, frameskip=frameskip, action_raw_dim=raw_adim,
        action_mean=act_mean, action_std=act_std,
        w_act=args.w_act, w_dyn=args.w_dyn, w_reg=args.w_reg, w_cyc=args.w_cyc,
        ablate_horizon=args.ablate_horizon, ablate_dynamics=args.ablate_dynamics,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr,
        dynamics_lr=args.dynamics_lr,
    )
    (run_dir / "lewam_gc_config.json").write_text(json.dumps(cfg_out, indent=2))

    # ---- training loop ----
    best_val = float("inf")
    all_params = enc_params + list(gc_head.parameters()) + list(dynamics.parameters())

    for ep in range(args.epochs):
        t0 = time.time()

        # ---- train ----
        lewm.train(); gc_head.train(); dynamics.train()
        tr_act, tr_dyn, tr_reg, tr_cyc, tr_count = 0.0, 0.0, 0.0, 0.0, 0

        for trip, a_t, h_norm in train_loader:
            n = trip.shape[0]
            # trip: [B,3,3,H,W] fp16 pinned -> async H2D, cast on GPU (halves PCIe bytes)
            trip = trip.to(device, non_blocking=True)
            a_t = a_t.to(device, non_blocking=True).float()
            h_norm = h_norm.to(device, non_blocking=True).float()
            # [B,3(t/g/n),3,H,W] -> [t...,g...,n...] to match the cat+chunk order
            frames_all = trip.permute(1, 0, 2, 3, 4).reshape(3 * n, *trip.shape[2:]).float()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # encode all 3 frames in one pass
                z_all = lewm.encode({"pixels": frames_all.unsqueeze(1)})["emb"][:, 0]
                z_t, z_g, z_n = z_all.chunk(3, dim=0)
                z_t_f = z_t.float()
                z_g_f = z_g.float()
                z_n_f = z_n.float()

                a_pred = gc_head(z_t_f, z_g_f, h_norm)
                loss_act = F.mse_loss(a_pred, a_t)

                # no stop-grad: the encoder also learns from the dynamics target
                z_n_pred = dynamics(z_t_f, a_t, z_g_f)
                loss_dyn = F.mse_loss(z_n_pred, z_n_f)

                loss_reg = sigreg(z_t_f.unsqueeze(0))

                loss = (args.w_act * loss_act
                        + args.w_dyn * loss_dyn
                        + args.w_reg * loss_reg)

                # consistency: the predicted action through dynamics should reach z_n
                loss_cyc = torch.tensor(0.0, device=device)
                if args.w_cyc > 0:
                    z_n_cyc = dynamics(z_t_f, a_pred, z_g_f)
                    loss_cyc = F.mse_loss(z_n_cyc, z_n_f.detach())
                    loss = loss + args.w_cyc * loss_cyc

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            tr_act += loss_act.item() * n
            tr_dyn += loss_dyn.item() * n
            tr_reg += loss_reg.item() * n
            tr_cyc += loss_cyc.item() * n
            tr_count += n

        sched.step()

        # ---- val ----
        lewm.eval(); gc_head.eval(); dynamics.eval()
        va_act, va_dyn, va_count = 0.0, 0.0, 0
        with torch.no_grad():
            for trip, a_t, h_norm in val_loader:
                n = trip.shape[0]
                trip = trip.to(device, non_blocking=True)
                a_t = a_t.to(device, non_blocking=True).float()
                h_norm = h_norm.to(device, non_blocking=True).float()
                frames_all = trip.permute(1, 0, 2, 3, 4).reshape(3 * n, *trip.shape[2:]).float()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_all = lewm.encode({"pixels": frames_all.unsqueeze(1)})["emb"][:, 0]
                    z_t, z_g, z_n = z_all.chunk(3, dim=0)
                    z_t_f, z_g_f, z_n_f = z_t.float(), z_g.float(), z_n.float()

                    a_pred = gc_head(z_t_f, z_g_f, h_norm)
                    loss_act = F.mse_loss(a_pred, a_t)

                    z_n_pred = dynamics(z_t_f, a_t, z_g_f)
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
        print(f"[lewam-gc] ep {ep+1}/{args.epochs}  "
              f"act={tr_a:.5f}/{va_a:.5f}  dyn={tr_d:.5f}/{va_d:.5f}  reg={tr_r:.5f}{cyc_str}  "
              f"lr_enc={lrs[0]:.2e}  {dt:.1f}s", flush=True)

        # ---- save checkpoints ----
        full_sd = {}
        for k, v in lewm.state_dict().items():
            full_sd[f"encoder.{k}"] = v
        for k, v in gc_head.state_dict().items():
            full_sd[f"gc_head.{k}"] = v
        for k, v in dynamics.state_dict().items():
            full_sd[f"dynamics.{k}"] = v
        torch.save(full_sd, run_dir / "lewam_gc_latest.pt")
        torch.save(gc_head.state_dict(), run_dir / "gc_head_latest.pt")

        combined_val = va_a + va_d
        if combined_val < best_val:
            best_val = combined_val
            torch.save(full_sd, run_dir / "lewam_gc_best.pt")
            torch.save(gc_head.state_dict(), run_dir / "gc_head_best.pt")

    print(f"[lewam-gc] DONE  best_val={best_val:.5f}  saved -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
