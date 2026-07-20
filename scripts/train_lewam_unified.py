# LeWAM-Unified: the split (goal-conditioned action head + goal-conditioned dynamics) trained
# SEQUENCE-PARALLEL over a CONTEXT window, with ONE shared context latent per position.
#
#   sample a start t; context window = frames [t .. t+W-1], W = context_len (episode-clamped);
#   strip frames [t .. t+W-1+H_max] -> encoder -> [z_0 .. z_E]   (ONE encode; the +H_max tail
#   frames exist only as goal/target candidates, they are NOT aggregator inputs)
#   c_tau = z_tau + g*Aggr(z_<=tau)                          # causal aggregator, tau < W
#   a_pred_tau = gc_head(c_tau, z_goal_tau, h_tau)           # goal PER POSITION: z_{tau+h},
#   z_pred_tau = dynamics(c_tau, a_tau, z_goal_tau)          #   h~U[1,H_max] (episode-clamped);
#                                                            # dynamics target z_{tau+1}
#   L = w_act*MSE(a_pred, a) + w_dyn*MSE(z_pred, z[1:]) + w_reg*SIGReg(z) + w_cyc*consistency
#       (all per-position, masked to the valid (non-pad) positions)
#
# The window is NOT a start->goal cut (the original design made the sampled goal the window
# endpoint, so the last position always trained on horizon 1 with goal == its dynamics target and
# the horizon marginal was short-skewed; at eval the goal is a separately-encoded far frame that is
# never in the sequence). Here the window is pure context and every position draws its goal
# h~U[1,H_max] uniformly, exactly like the split's pair sampling -- goals just come from the same
# encoded strip. Each start is a FRESH start (no prior history), matching the eval episode start.
#
#   python scripts/train_lewam_unified.py --dataset_name reacher.h5 \
#       --run_name reacher_lewam_unified --epochs 50 --H_max 5 --context_len 5 --agg_depth 2 \
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
# Context-window Dataset: each item is a context window of W = context_len decision points        #
# (episode-clamped) PLUS an H_max-frame goal tail. Return frames [t .. t+E] where                  #
# E = min(W-1+H_max, maxh) (covers every position's dynamics target z_{p+1} and any goal          #
# z_{p+h}, h<=H_max), the z-scored action blocks a_t..a_{t+W-1}, and W. Goals/horizons are        #
# sampled per position on-GPU in run_batch (uniform h~U[1,H_max], episode-clamped) -- the window  #
# does NOT terminate at a goal. A fresh start (no prior history) matches the eval episode start.  #
# --------------------------------------------------------------------------- #
class SeqTrajDataset(Dataset):
    def __init__(self, frames, a_frame, t_gidx, maxh, indices, h_max, ctx_len):
        self.frames = frames          # [N,3,H,W] fp16, CPU, shared read-only
        self.a_frame = a_frame        # [N,adim] fp16, CPU  per-frame action block
        self.t_gidx = t_gidx          # [M] long  valid start frames
        self.maxh = maxh              # [M] long  frames from start to episode end
        self.indices = indices        # [K] long  train or val subset
        self.h_max = int(h_max)
        self.ctx_len = int(ctx_len)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        t = int(self.t_gidx[idx])
        mh = int(self.maxh[idx])                  # frames from t to episode end (>=1)
        w = max(1, min(self.ctx_len, mh))         # decision points (frame t+w exists: w<=mh)
        e = min(w - 1 + self.h_max, mh)           # last frame offset: targets + goal tail
        frames = self.frames[t:t + e + 1]         # (e+1, 3, H, W)  z_0..z_e
        actions = self.a_frame[t:t + w]           # (w, adim)       a_t..a_{t+w-1}
        return frames, actions, w


def collate_pad(batch):
    """Pad variable-length strips to the batch's max. Returns frames (B, Fmax, 3, H, W),
    actions (B, Pmax, adim), w_len (B,) valid decision points, f_len (B,) valid frames.
    Pad actions are zeros; positions >= w_len are masked out of the loss in run_batch."""
    w_len = torch.tensor([b[2] for b in batch], dtype=torch.long)
    f_len = torch.tensor([b[0].shape[0] for b in batch], dtype=torch.long)
    Pmax = int(w_len.max().item())
    Fmax = int(f_len.max().item())
    C, H, W = batch[0][0].shape[1:]
    adim = batch[0][1].shape[1]
    B = len(batch)
    frames = torch.zeros((B, Fmax, C, H, W), dtype=batch[0][0].dtype)
    actions = torch.zeros((B, Pmax, adim), dtype=batch[0][1].dtype)
    for i, (fr, ac, w) in enumerate(batch):
        f = fr.shape[0]
        frames[i, :f] = fr
        if f < Fmax:
            frames[i, f:] = fr[-1]       # pad with a REAL frame (repeat the endpoint), NOT zeros:
                                         # the ViT projector has a BatchNorm, so zero-padding frames
                                         # would corrupt the batch statistics and the real latents.
        actions[i, :w] = ac
    return frames, actions, w_len, f_len


# --------------------------------------------------------------------------- #
# Streaming path (--stream): read ONLY actions + episode lengths up front (tiny), enumerate valid    #
# starts, and read each trajectory's frame window from the h5 on demand in __getitem__. No preload / #
# flatten -> constant, low RAM (fits a normal worker), at the cost of per-batch h5 reads (epoch 1    #
# slow; page cache warms after). Produces the SAME (frames, actions, h) items as SeqTrajDataset, so  #
# collate_pad + run_batch are unchanged. Mirrors the proven stream path in train_lewam_seq.py.       #
# --------------------------------------------------------------------------- #
def build_traj_index_stream(base, frameskip, act_mean, act_std, raw_adim, max_eps=None):
    """Read actions + lengths only; z-score actions per episode; enumerate (ep, start, maxh) starts.
    n_fr (obs-frames) matches preload's min(ceil(L/fs), n_obs+1); maxh = last frame - start."""
    import h5py
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)
    act_std_t = torch.tensor(act_std, dtype=torch.float32)
    lengths = np.asarray(base.lengths)
    n_eps = len(lengths) if max_eps is None else min(max_eps, len(lengths))
    with h5py.File(base.h5_path, "r", swmr=True) as hf:
        act_all = torch.from_numpy(np.asarray(hf["action"][:]))
        offsets = (np.asarray(hf["ep_offset"][:]).astype(np.int64) if "ep_offset" in hf
                   else np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64))
    acts_by_ep, index = [], []
    t0 = time.time()
    for ep in range(n_eps):
        L = int(lengths[ep]); off = int(offsets[ep]); n_obs = L // frameskip
        a = act_all[off:off + n_obs * frameskip].reshape(n_obs, frameskip, raw_adim)
        a = ((a - act_mean_t) / act_std_t).reshape(n_obs, frameskip * raw_adim).half()
        n_fr = min((L + frameskip - 1) // frameskip, n_obs + 1)
        acts_by_ep.append(a)
        last = n_fr - 1
        for t in range(min(n_obs, n_fr - 1)):     # valid starts: need >=1 frame after the start
            index.append((ep, t, last - t))       # (ep, start, maxh = frames from start to ep end)
    print(f"[lewam-uni] stream index: {len(index)} starts over {n_eps} eps ({time.time()-t0:.0f}s)",
          flush=True)
    return acts_by_ep, index


class StreamTrajDataset(Dataset):
    """Same item as SeqTrajDataset, but frames are read from the h5 on demand (no preloaded tensor)."""

    def __init__(self, acts_by_ep, index, indices, h_max, ctx_len, base, img_t, frameskip):
        self.acts = acts_by_ep
        self.index = index
        self.indices = indices
        self.h_max = int(h_max)
        self.ctx_len = int(ctx_len)
        self.base = base
        self.img_t = img_t
        self.fs = int(frameskip)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        ep, t, mh = self.index[int(self.indices[i])]
        w = max(1, min(self.ctx_len, mh))         # decision points (frame t+w exists: w<=mh)
        e = min(w - 1 + self.h_max, mh)           # last frame offset: targets + goal tail
        # obs-frames t .. t+e (e+1 frames), strided by frameskip inside _load_slice
        pix = self.base._load_slice(ep, t * self.fs, (t + e) * self.fs + 1)["pixels"]
        if not torch.is_tensor(pix):
            pix = torch.as_tensor(np.asarray(pix))
        frames = self.img_t({"pixels": pix})["pixels"].float().half()[: e + 1]   # (e+1, 3, H, W)
        actions = self.acts[ep][t:t + w]                                          # (w, adim)
        return frames, actions, w


# --------------------------------------------------------------------------- #
# Main training                                                                #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128,
                    help="windows per batch (each carries up to context_len decision points, so the "
                         "effective decision-point batch is larger; raise/lower per GPU mem)")
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--dynamics_lr", type=float, default=3e-4)
    ap.add_argument("--agg_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--H_max", type=int, default=50,
                    help="max goal distance in OBS-STEPS (frames are 1-per-obs-step, post-frameskip; "
                         "NOT divided by frameskip); per-position h~U[1,H_max], episode-clamped. "
                         "DECOUPLED from the window length (see --context_len)")
    ap.add_argument("--context_len", type=int, default=5,
                    help="decision points per context window (the aggregator's max sequence length "
                         "at train; eval defaults its ctx_cap to this). Frames per item = "
                         "context_len + H_max, so this x H_max sets GPU memory")
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
    ap.add_argument("--stream", action="store_true",
                    help="read frames from the h5 on demand (no preload/flatten) -> constant low RAM, "
                         "fits a normal worker; epoch 1 slow (h5 reads), page cache warms after")
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
    if args.stream:
        acts_by_ep, stream_index = build_traj_index_stream(base, frameskip, act_mean, act_std,
                                                           raw_adim, max_eps=max_eps)
        n_starts = len(stream_index)
        print(f"[lewam-uni] stream mode: starts={n_starts} (no preload)", flush=True)
    else:
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
          f"H_max={args.H_max} context_len={args.context_len} "
          f"action_block={action_block_dim}", flush=True)

    # ---- DataLoaders ----
    # Each item is a context window (~context_len decision points), so iterating EVERY start
    # supervises each decision point ~context_len times per epoch. Draw ~1x coverage per epoch
    # instead: n_starts / context_len windows. Re-randomized each epoch (RandomSampler), so over
    # many epochs all starts are still seen.
    if args.stream:
        train_ds = StreamTrajDataset(acts_by_ep, stream_index, train_idx, args.H_max,
                                     args.context_len, base, img_t, frameskip)
        val_ds = StreamTrajDataset(acts_by_ep, stream_index, val_idx, args.H_max,
                                   args.context_len, base, img_t, frameskip)
    else:
        train_ds = SeqTrajDataset(Frames, A_frame, t_gidx, maxh, train_idx, args.H_max, args.context_len)
        val_ds = SeqTrajDataset(Frames, A_frame, t_gidx, maxh, val_idx, args.H_max, args.context_len)
    avg_cover = max(1.0, float(args.context_len))
    n_tr_ep = max(args.batch_size, int(train_idx.numel() / avg_cover))
    n_va_ep = max(args.batch_size, int(val_idx.numel() / avg_cover))
    train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=n_tr_ep)
    val_sampler = torch.utils.data.RandomSampler(val_ds, replacement=True, num_samples=n_va_ep)
    _loader_common = dict(
        batch_size=args.batch_size, pin_memory=True, drop_last=False,
        num_workers=args.num_workers, collate_fn=collate_pad,
    )
    if args.num_workers > 0:
        _loader_common.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    train_loader = DataLoader(train_ds, sampler=train_sampler, **_loader_common)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **_loader_common)
    print(f"[lewam-uni] loaders ready: workers={args.num_workers} prefetch={args.prefetch_factor} "
          f"pin_memory=True  trajectories/epoch: train={n_tr_ep} val={n_va_ep} (~1x coverage)", flush=True)

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
        H_max=args.H_max, context_len=args.context_len,
        frameskip=frameskip, action_raw_dim=raw_adim,
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

    def run_batch(frames, actions, w_len, f_len, train):
        """One sequence-parallel forward: encode the strip once, aggregate causally over the first
        Pmax positions, apply the heads at every valid position, and return per-position masked
        (act, dyn) losses. The strip's goal-tail frames beyond Pmax are encoded ONLY as goal/target
        candidates -- they are never aggregator inputs, so the goal is not part of the sequence."""
        B, Fmax = frames.shape[0], frames.shape[1]
        Pmax = actions.shape[1]                          # max decision points in the batch
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True).float()
        w_len = w_len.to(device, non_blocking=True)
        f_len = f_len.to(device, non_blocking=True)
        z = model.encode(frames.reshape(B * Fmax, *frames.shape[2:]).float()).reshape(B, Fmax, D)
        states = z[:, :Pmax].float()                     # (B,Pmax,D) decision-point states
        next_tgt = z[:, 1:Pmax + 1].float()              # (B,Pmax,D) dynamics targets z[p+1]
        pos = torch.arange(Pmax, device=device)
        valid = pos.unsqueeze(0) < w_len.unsqueeze(1)                   # (B,Pmax) bool
        # Per-position goal h~U[1,H_max], clamped to the episode tail (same sampling as the split's
        # pair dataset). goal for pos p = z[p+h]; the strip carries H_max frames beyond the window
        # so unclamped goals always exist. Pad positions get an arbitrary in-strip goal and are
        # masked from the loss.
        pos_e = pos.unsqueeze(0)                                       # (1,Pmax)
        last = (f_len - 1).unsqueeze(1)                                # (B,1) last valid frame idx
        span = (last - pos_e).clamp(min=1, max=args.H_max)             # max h at pos p
        g_idx = (pos_e + 1 + (torch.rand(B, Pmax, device=device) * span).long())
        g_idx = torch.minimum(g_idx, last.expand_as(g_idx)).clamp(max=Fmax - 1)
        goal_bc = torch.gather(z, 1, g_idx.unsqueeze(-1).expand(B, Pmax, D)).float()  # (B,Pmax,D)
        horizon = (g_idx - pos_e).clamp(min=1)                          # == h (uniform in [1,H_max])
        h_norm = horizon.clamp(max=args.H_max).float() / Hmax          # (B,Pmax)
        a_prev = a_prev_mask = None
        if model.agg_action_cond:
            a_prev = torch.zeros_like(actions)
            a_prev[:, 1:] = actions[:, :-1]                            # a_{tau-1}; position 0 -> null
            a_prev_mask = valid & (pos.unsqueeze(0) >= 1)
        a_pred, z_pred = model.forward_seq(states, goal_bc, h_norm, actions, a_prev, a_prev_mask)
        m = valid.unsqueeze(-1).float()
        nval = valid.sum().clamp(min=1)
        loss_act = ((a_pred - actions) ** 2 * m).sum() / (nval * actions.shape[-1])
        loss_dyn = ((z_pred - next_tgt) ** 2 * m).sum() / (nval * D)   # no stop-grad (like seq/split)
        loss_cyc = torch.zeros((), device=device)
        if train and args.w_cyc > 0:
            c = model.aggregate(states, a_prev, a_prev_mask)
            z_cyc = model.dynamics(c.reshape(B * Pmax, D), a_pred.reshape(B * Pmax, -1),
                                   goal_bc.reshape(B * Pmax, D)).reshape(B, Pmax, D)
            loss_cyc = ((z_cyc - next_tgt.detach()) ** 2 * m).sum() / (nval * D)
        # SIGReg anti-collapse over ALL valid state latents (like the seq trainer's
        # sigreg(z.reshape(-1,D))). The earlier reg~200 blow-up was a symptom of the BatchNorm-
        # padding bug corrupting the latents, not a reason to sub-sample.
        loss_reg = sigreg(states[valid].unsqueeze(0)) if train else torch.zeros((), device=device)
        return loss_act, loss_dyn, loss_reg, loss_cyc, int(nval.item())

    # ---- training loop ----
    best_val = float("inf")

    for ep in range(args.epochs):
        t0 = time.time()

        # ---- train ----
        model.train()
        tr_act, tr_dyn, tr_reg, tr_cyc, tr_count = 0.0, 0.0, 0.0, 0.0, 0
        for frames, actions, w_len, f_len in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_act, loss_dyn, loss_reg, loss_cyc, nval = run_batch(
                    frames, actions, w_len, f_len, train=True)
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
            for frames, actions, w_len, f_len in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_act, loss_dyn, _, _, nval = run_batch(
                        frames, actions, w_len, f_len, train=False)
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
