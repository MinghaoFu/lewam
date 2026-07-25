# LeWAM-Unified: the split (goal-conditioned action head + goal-conditioned dynamics) trained
# SEQUENCE-PARALLEL over a CONTEXT window, with ONE shared context latent per position.
#
#   sample a start t; context window = frames [t .. t+W-1], W = context_len (episode-clamped);
#   encode window frames + each position's ONE goal frame in a single pass (2W+1 frames per item,
#   independent of H_max -- goals are looked up by index, never a contiguous tail)
#   c_tau = z_tau + g*Aggr(z_<=tau)                          # causal aggregator over states ONLY
#   a_pred_tau = gc_head(c_tau, z_goal_tau, h_tau)           # per-position goal + horizon
#   z_pred_tau = dynamics(c_tau, a_tau, z_goal_tau)          # dyn target z_{tau+1}, ALWAYS 1 step
#
#   L = w_act*MSE(a_pred, a) + w_dyn*MSE(z_pred, z[1:]) + w_reg*SIGReg(z) + w_cyc*consistency
#       (all per-position, masked to the valid (non-pad) positions)
#
# Goal sampling, per window, two modes mixed by --p_shared:
#   RANDOM (prob 1-p): each position independently draws h~U[1,H_max] and clamps its goal frame to
#     the episode's last frame -- the split FramePairDataset's exact sampling (incl. the mass
#     pile-up on the final frame near episode ends), so per position this IS a split example, just
#     with an aggregated-context state. Maximal (goal,horizon) coverage; horizon uncorrelated with
#     context length.
#   SHARED (prob p): ONE goal for the whole window, drawn h~U[1,H_max] ahead of the LAST position
#     (episode-clamped); horizons count down toward it across positions (h_norm clamps at 1.0 like
#     eval's min(steps,H_max)/H_max). This is the structure eval runs -- one fixed goal, horizon
#     decreasing over consecutive replans -- and what a future rollout loss would need.
#
# The window is NOT a start->goal cut (the original design made the sampled goal the window
# endpoint: the last position always trained on horizon 1 with goal == its dynamics target, the
# horizon marginal was short-skewed, and the goal sat inside the fed sequence, unlike eval).
# Goals are encoder inputs but NEVER aggregator inputs. Each window is a FRESH start (no prior
# history), matching both the eval episode start and the eval adapter's ctx_cap sliding window.
#
#   python scripts/train_lewam_unified.py --dataset_name reacher.h5 \
#       --run_name reacher_lewam_unified --epochs 50 --H_max 50 --context_len 5 --p_shared 0.5 \
#       --agg_depth 2 --ckpt_sync_dir /mnt/hdfs/.../reacher_lewam_unified
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
def preload_flat(base, img_t, act_mean, act_std, frameskip, max_eps=None):
    """Preload episodes DIRECTLY into ONE preallocated flat frame tensor (never a per-episode
    frame_list), so peak host RAM stays ~1x the frames. The old frame_list + separate flat-copy
    design doubled it: the freed per-episode tensors are not returned to the OS before the flat
    tensor is allocated and filled, so peak ~= 2x the frames and a screening pod sized for ~1x
    OOM-kills inside the copy. Here the flat buffer is a lazily-committed np.empty (pages commit as
    written), each episode is decoded into its slice then dropped, and A_frame/t_gidx/maxh are built
    in the same pass. Returns (Frames, A_frame, t_gidx, maxh), identical semantics to the old
    preload_frames + flatten_for_training pair.

      A_frame[base+t] = action block AT obs-step t (the action leading INTO frame t+1); zero at the
      episode's last frame. t_gidx = valid START frames; maxh = frames from that start to the
      episode's last frame (the max goal distance sampleable there)."""
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)
    act_std_t = torch.tensor(act_std, dtype=torch.float32)
    n_eps = len(base.lengths) if max_eps is None else min(max_eps, len(base.lengths))
    # Upper bound on total frames (n_keep <= n_obs+1 always), from lengths alone -- no pixel load.
    upper = int(sum(int(base.lengths[ep]) // frameskip + 1 for ep in range(n_eps)))

    def _load_ep(ep):
        L = int(base.lengths[ep])
        sl = base._load_slice(ep, 0, L)
        pix = sl["pixels"]
        raw_act = sl["action"]
        if not torch.is_tensor(pix):
            pix = torch.as_tensor(np.asarray(pix))
        raw_act = raw_act if torch.is_tensor(raw_act) else torch.as_tensor(np.asarray(raw_act))
        pp = img_t({"pixels": pix})["pixels"].float()
        n_obs = L // frameskip
        a = raw_act[:n_obs * frameskip].reshape(n_obs, frameskip, raw_act.shape[1])
        a = (a - act_mean_t) / act_std_t
        a = a.reshape(n_obs, frameskip * raw_act.shape[1]).half()
        return pp, a, n_obs

    pp0, a0, _ = _load_ep(0)
    C, H, W = pp0.shape[1:]
    adim = a0.shape[1]
    Frames_np = np.empty((upper, C, H, W), dtype=np.float16)   # anonymous mmap -> lazy commit
    A_frame_np = np.zeros((upper, adim), dtype=np.float16)
    t_gidx, maxh_list = [], []
    off = 0
    t0 = time.time()
    for ep in range(n_eps):
        pp, a, n_obs = (pp0, a0, a0.shape[0]) if ep == 0 else _load_ep(ep)
        n_keep = min(pp.shape[0], n_obs + 1)
        Frames_np[off:off + n_keep] = pp[:n_keep].half().numpy()
        na = min(n_obs, n_keep)
        A_frame_np[off:off + na] = a[:na].numpy()
        last = n_keep - 1
        for t in range(min(n_obs, n_keep - 1)):
            t_gidx.append(off + t)
            maxh_list.append(last - t)
        off += n_keep
        if (ep + 1) % 500 == 0:
            print(f"[lewam-uni] preload {ep+1}/{n_eps} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[lewam-uni] preload DONE {n_eps} eps in {time.time()-t0:.0f}s (flat, ~1x)", flush=True)
    Frames = torch.from_numpy(Frames_np[:off])       # zero-copy; keeps Frames_np alive
    A_frame = torch.from_numpy(A_frame_np[:off])
    return (Frames, A_frame,
            torch.tensor(t_gidx, dtype=torch.long),
            torch.tensor(maxh_list, dtype=torch.long))


# --------------------------------------------------------------------------- #
# Context-window Dataset. Each item is one window of up to `context_len` decision points plus     #
# exactly the frames those decision points use:                                                   #
#   window  frames [start .. start+n_pos]   (n_pos+1: each position's state AND its z_{p+1} target)#
#   goals   one frame per position, at the sampled goal offsets (RANDOM or SHARED mode, see the   #
#           file header) -- looked up by index, so memory per item is ~2*context_len+1 frames     #
#           REGARDLESS of H_max                                                                   #
#   actions the action block taken AT each decision point                                         #
#   horizon each position's distance to its goal in prediction steps                              #
# Windows are clamped at the episode end, never cross it. Each window is a fresh start (no prior  #
# history), like an eval episode / the eval adapter's sliding window.                             #
# --------------------------------------------------------------------------- #
def sample_goal_offsets(n_pos, frames_left, h_max, p_shared, close_bias=0.0):
    """Sample each position's goal offset (obs-frames from the window start) + horizon.

    RANDOM mode (prob 1-p_shared): per position p, draw h~U[1,h_max]; goal offset = p+h clamped
    to the episode's last frame -- the split FramePairDataset's sample-then-clamp, per position.
    SHARED mode (prob p_shared): draw ONE h~U[1,h_max] for the LAST position; every position
    points at that same frame, horizons counting down toward it (>= 1 by construction since the
    goal sits at or beyond the window end even after the episode clamp).

    close_bias>0 skews the horizon draw toward SMALL h (more close-to-goal supervision, where the
    precision-limited tasks plateau): h = 1 + floor((h_max-1) * u^(1+close_bias)), u~U[0,1];
    close_bias=0 recovers the uniform U[1,h_max]. Higher bias -> more mass near h=1.

    Returns (goal_offsets (n_pos,) long, horizon (n_pos,) long). horizon can exceed h_max in
    SHARED mode (earlier positions are farther); the trainer clamps h_norm at 1.0 exactly like
    the eval countdown's min(steps, H_max)/H_max."""
    def _draw(shape):
        if close_bias > 0:
            u = torch.rand(shape)
            return 1 + (u.pow(1.0 + close_bias) * (h_max - 1)).long()
        return torch.randint(1, h_max + 1, shape)
    positions = torch.arange(n_pos)
    if float(torch.rand(())) < p_shared:
        h_last = int(_draw(()))
        goal_offsets = torch.full((n_pos,), min(n_pos - 1 + h_last, frames_left))
    else:
        h_draw = _draw((n_pos,))
        goal_offsets = torch.minimum(positions + h_draw,
                                     torch.tensor(frames_left, dtype=torch.long))
    horizon = (goal_offsets - positions).clamp(min=1)
    return goal_offsets, horizon


class SeqTrajDataset(Dataset):
    def __init__(self, frames, a_frame, t_gidx, maxh, indices, h_max, ctx_len, p_shared,
                 close_bias=0.0):
        self.frames = frames          # [N,3,H,W] fp16, CPU, shared read-only; N = all obs-frames
        self.a_frame = a_frame        # [N,adim] fp16, CPU; a_frame[t] = z-scored block taken AT t
        self.t_gidx = t_gidx          # [M] long; episode-global obs-frame index of each valid start
        self.maxh = maxh              # [M] long; frames from that start to the episode's last frame
        self.indices = indices        # [K] long; train or val subset of the M starts
        self.h_max = int(h_max)       # max goal distance (obs-steps)
        self.ctx_len = int(ctx_len)   # decision points per window
        self.p_shared = float(p_shared)
        self.close_bias = float(close_bias)  # >0 -> over-sample close goals (near-goal precision)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        start = int(self.t_gidx[idx])
        frames_left = int(self.maxh[idx])                        # >= 1 for every valid start
        n_pos = min(self.ctx_len, frames_left)                   # frame start+n_pos exists (the
                                                                 # last position's target), since
                                                                 # n_pos <= frames_left
        goal_offsets, horizon = sample_goal_offsets(n_pos, frames_left, self.h_max, self.p_shared,
                                                    self.close_bias)
        window = self.frames[start:start + n_pos + 1]            # (n_pos+1, 3, H, W)
        goals = self.frames[start + goal_offsets]                # (n_pos, 3, H, W)
        actions = self.a_frame[start:start + n_pos]              # (n_pos, adim)
        return window, goals, actions, horizon, n_pos


def collate_pad(batch):
    """Pad the batch's variable-length items to a common shape.

    Returns:
      window   (B, max_pos+1, 3, H, W)  states + next-state targets, tail-padded
      goals    (B, max_pos, 3, H, W)    each position's goal frame, tail-padded
      actions  (B, max_pos, adim)       zero-padded past each item's n_pos
      horizon  (B, max_pos) long        pad value 1 (masked from the loss anyway)
      n_pos    (B,) long                valid decision points per item (the loss mask)

    Frame padding repeats each item's LAST REAL frame instead of zeros: the ViT projector has a
    BatchNorm, and in train mode zero frames would corrupt the batch statistics for the real
    frames (the ep-1 "collapse" bug, see the handoff). Pad positions never enter the loss."""
    n_pos = torch.tensor([item[4] for item in batch], dtype=torch.long)
    max_pos = int(n_pos.max())
    B = len(batch)
    C, H, W = batch[0][0].shape[1:]
    adim = batch[0][2].shape[1]
    window = torch.zeros((B, max_pos + 1, C, H, W), dtype=batch[0][0].dtype)
    goals = torch.zeros((B, max_pos, C, H, W), dtype=batch[0][1].dtype)
    actions = torch.zeros((B, max_pos, adim), dtype=batch[0][2].dtype)
    horizon = torch.ones((B, max_pos), dtype=torch.long)
    for i, (item_window, item_goals, item_actions, item_horizon, item_n_pos) in enumerate(batch):
        window[i, : item_n_pos + 1] = item_window
        goals[i, : item_n_pos] = item_goals
        actions[i, : item_n_pos] = item_actions
        horizon[i, : item_n_pos] = item_horizon
        if item_n_pos < max_pos:
            window[i, item_n_pos + 1:] = item_window[-1]
            goals[i, item_n_pos:] = item_goals[-1]
    return window, goals, actions, horizon, n_pos


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
                    help="max goal distance (horizon) in OBS-STEPS (frames are 1-per-obs-step, "
                         "post-frameskip; NOT divided by frameskip); h~U[1,H_max], episode-clamped. "
                         "DECOUPLED from the window length (see --context_len); does NOT affect "
                         "memory (goals are looked up per position, not loaded as a tail)")
    ap.add_argument("--context_len", type=int, default=5,
                    help="decision points per context window = the aggregator's max sequence length "
                         "at train (eval defaults its ctx_cap to this). Frames per item = "
                         "2*context_len+1, so batch x context_len sets GPU memory")
    ap.add_argument("--p_shared", type=float, default=0.0,
                    help="fraction of windows trained in SHARED-goal mode (one goal ahead of the "
                         "window, horizons counting down -- the structure eval runs); the rest use "
                         "per-position independent goals (the split's sampling). 0 = all random")
    ap.add_argument("--goal_close_bias", type=float, default=0.0,
                    help="skew the training goal-distance draw toward SMALL h (more close-to-goal "
                         "supervision): h=1+floor((H_max-1)*u^(1+bias)). 0=uniform U[1,H_max]. "
                         "Applied to train only; val stays uniform for a comparable metric.")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--embed_dim", type=int, default=192,
                    help="latent width feeding aggregator + heads (ViT-tiny cls is projected to this)")
    ap.add_argument("--encoder_size", type=str, default="tiny",
                    help="ViT backbone size: tiny | small | base — the main param-count lever")
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
    ap.add_argument("--dyn_no_goal", action="store_true",
                    help="drop z_goal from the DYNAMICS head -> a pure forward model f(z_t,a_t) "
                         "(the gc_head stays goal-conditioned). A goal-conditioned dynamics can "
                         "drift toward z_goal ignoring the action, flattening the planning cost "
                         "surface; the pure forward model is the honest world model for CEM/planning.")
    # dynamics-on-policy-action arm: feed the gc_head's PREDICTED action into the dynamics head
    # (a convex mix with the ground-truth action, weight alpha ramped by a schedule), closing the
    # train/rollout covariate gap. The action is still BC-supervised on ground truth, so it stays a
    # grounded action, not a latent-action model.
    ap.add_argument("--dyn_action_from_policy", action="store_true",
                    help="feed a mix of the policy's predicted action into the dynamics head "
                         "(alpha per --dyn_policy_schedule); off = dynamics on ground-truth actions")
    ap.add_argument("--dyn_policy_schedule", type=str, default="cosine",
                    help="alpha(epoch) schedule for the policy-action mix: const|linear|cosine|"
                         "sigmoid. const = full alpha from epoch 0 (least stable, 'from the start').")
    ap.add_argument("--dyn_policy_alpha_max", type=float, default=1.0,
                    help="max mix weight (1.0 = dynamics sees ONLY the policy action at the end)")
    ap.add_argument("--dyn_policy_ramp_frac", type=float, default=0.5,
                    help="fraction of total epochs over which alpha ramps 0->alpha_max "
                         "(non-const schedules); after that alpha stays at alpha_max")
    ap.add_argument("--dyn_policy_detach", action="store_true",
                    help="stop-gradient the policy action into dynamics (dynamics adapts to the "
                         "policy's actions, but the dyn loss never reshapes the BC policy). "
                         "default: gradients flow policy->dynamics (coupled 'unified' arm)")
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
    ap.add_argument("--frames_cache", type=str, default="auto",
                    help="prebuilt-cache dir | 'auto' (env LEWAM_CACHE_DIR or the default HDFS "
                         "preload_cache) | 'off'. HIT -> np.load the shared <tag>.frames.npy + "
                         "<tag>.aux.npz instead of decoding the h5 (~1x-frames RAM vs classic "
                         "preload). Ignored when --max_eps set (falls back to classic preload).")
    args = ap.parse_args()

    if args.ablate_dynamics:
        args.w_dyn = 0.0
    assert args.context_len >= 1 and args.H_max >= 1, "context_len and H_max must be >= 1"
    assert 0.0 <= args.p_shared <= 1.0, "p_shared must be in [0, 1]"

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
    # HARD RULE (owner 2026-07-24): never write checkpoints under $HOME -- big .pt in the shared
    # home fs filled it and crashed the box. Entry scripts pass --run_dir to /opt/tiger (local
    # scratch); warn LOUD if anything resolves under home so it can't happen silently.
    if os.path.realpath(run_dir).startswith(os.path.realpath(os.path.expanduser("~"))):
        print(f"[lewam-uni] WARN run_dir under $HOME ({run_dir}); checkpoints in home can fill the "
              f"shared fs -- pass --run_dir to local scratch (/opt/tiger/...) or HDFS", flush=True)
    max_eps = args.max_eps or None

    # ---- build model ----
    model = LeWAMUnified(encoder_size=args.encoder_size, embed_dim=args.embed_dim,
                         action_dim=action_block_dim,
                         hidden_dim=args.hidden_dim, img_size=args.img_size, dropout=0.1,
                         agg_depth=args.agg_depth, agg_heads=args.agg_heads,
                         agg_residual=args.agg_residual, agg_gate=args.agg_gate,
                         agg_action_cond=args.agg_action_cond,
                         dyn_goal_cond=not args.dyn_no_goal).to(device)
    sigreg = SIGReg().to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_agg = sum(p.numel() for p in model.aggregator.parameters())
    n_head = sum(p.numel() for p in model.gc_head.parameters())
    n_dyn = sum(p.numel() for p in model.dynamics.parameters())
    print(f"[lewam-uni] encoder={n_enc/1e6:.2f}M  agg={n_agg/1e6:.2f}M  gc_head={n_head/1e6:.2f}M  "
          f"dynamics={n_dyn/1e6:.2f}M  total={(n_enc+n_agg+n_head+n_dyn)/1e6:.2f}M  "
          f"agg_depth={args.agg_depth} action_cond={args.agg_action_cond}", flush=True)

    # ---- data ----
    Frames = None
    if args.frames_cache != "off" and not max_eps:
        _cdir = args.frames_cache if args.frames_cache != "auto" else os.environ.get(
            "LEWAM_CACHE_DIR",
            "/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/preload_cache")
        _stem = os.path.basename(args.dataset_name).replace(".h5", "")
        _tag = f"{_stem}_fs{frameskip}_i{args.img_size}"
        _fp, _ap = f"{_cdir}/{_stem}/{_tag}.frames.npy", f"{_cdir}/{_stem}/{_tag}.aux.npz"
        if not (os.path.isfile(_fp) and os.path.isfile(_ap)):
            _fp, _ap = f"{_cdir}/{_tag}.frames.npy", f"{_cdir}/{_tag}.aux.npz"
        if os.path.isfile(_fp) and os.path.isfile(_ap):
            _t0 = time.time()
            print(f"[lewam-uni] frames-cache HIT {_fp}", flush=True)
            Frames = torch.from_numpy(np.load(_fp))
            _aux = np.load(_ap)
            # The cache is built by the GC pipeline: A_flat is per-SAMPLE (one action block per
            # valid start), t_gidx/maxh are the same per-start arrays this script uses. Scatter
            # A_flat back to a per-FRAME tensor A_frame[t_gidx]=A_flat: every frame the window
            # dataset reads (frames[start:start+n_pos], all valid starts) is thereby set, so this
            # is exact -- only episode-last frames stay zero, and their action is never read.
            A_flat = torch.from_numpy(_aux["A_flat"])
            t_gidx = torch.from_numpy(_aux["t_gidx"])
            maxh = torch.from_numpy(_aux["maxh"])
            A_frame = torch.zeros((Frames.shape[0], A_flat.shape[1]), dtype=A_flat.dtype)
            A_frame[t_gidx] = A_flat
            print(f"[lewam-uni] cache loaded in {time.time()-_t0:.0f}s "
                  f"frames={tuple(Frames.shape)}", flush=True)
        else:
            print(f"[lewam-uni] frames-cache MISS ({_fp}) -> classic preload", flush=True)
    if Frames is None:
        Frames, A_frame, t_gidx, maxh = preload_flat(base, img_t, act_mean, act_std,
                                                     frameskip, max_eps=max_eps)
    n_starts = t_gidx.shape[0]
    print(f"[lewam-uni] frames={Frames.shape} starts={n_starts}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_starts, generator=g)
    n_val = int(round((1 - args.train_split) * n_starts))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[lewam-uni] train={train_idx.numel()} val={val_idx.numel()} "
          f"H_max={args.H_max} context_len={args.context_len} p_shared={args.p_shared} "
          f"action_block={action_block_dim}", flush=True)

    # ---- DataLoaders ----
    # Each item is a context window (~context_len decision points), so iterating EVERY start
    # supervises each decision point ~context_len times per epoch. Draw ~1x coverage per epoch
    # instead: n_starts / context_len windows. Re-randomized each epoch (RandomSampler), so over
    # many epochs all starts are still seen.
    train_ds = SeqTrajDataset(Frames, A_frame, t_gidx, maxh, train_idx,
                              args.H_max, args.context_len, args.p_shared, args.goal_close_bias)
    val_ds = SeqTrajDataset(Frames, A_frame, t_gidx, maxh, val_idx,
                            args.H_max, args.context_len, args.p_shared, 0.0)
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

    def dyn_alpha(epoch):
        """Mix weight for feeding the policy action into dynamics at this epoch (0 = ground-truth
        only). const holds alpha_max from epoch 0; the ramps go 0->alpha_max over ramp_frac of the
        run, then hold."""
        if not args.dyn_action_from_policy:
            return 0.0
        a_max = float(args.dyn_policy_alpha_max)
        if args.dyn_policy_schedule == "const":
            return a_max
        ramp = max(1, int(round(args.dyn_policy_ramp_frac * total_epochs)))
        p = min(1.0, epoch / ramp)
        if args.dyn_policy_schedule == "linear":
            f = p
        elif args.dyn_policy_schedule == "sigmoid":
            k = 10.0
            s0, s1 = 1.0 / (1.0 + math.exp(k * 0.5)), 1.0 / (1.0 + math.exp(-k * 0.5))
            f = (1.0 / (1.0 + math.exp(-k * (p - 0.5))) - s0) / (s1 - s0)
        else:  # cosine (default)
            f = 0.5 * (1.0 - math.cos(math.pi * p))
        return a_max * f

    # ---- save config ----
    cfg_out = dict(
        model="lewam_unified", z_dim=args.embed_dim, encoder_size=args.encoder_size,
        action_dim=action_block_dim,
        hidden_dim=args.hidden_dim, n_freqs=64, dropout=0.1,
        agg_depth=args.agg_depth, agg_heads=args.agg_heads,
        agg_residual=args.agg_residual, agg_gate=args.agg_gate,
        agg_action_cond=args.agg_action_cond, dyn_goal_cond=not args.dyn_no_goal,
        H_max=args.H_max, context_len=args.context_len, p_shared=args.p_shared,
        frameskip=frameskip, action_raw_dim=raw_adim,
        action_mean=act_mean, action_std=act_std,
        w_act=args.w_act, w_dyn=args.w_dyn, w_reg=args.w_reg, w_cyc=args.w_cyc,
        ablate_dynamics=args.ablate_dynamics,
        dyn_action_from_policy=args.dyn_action_from_policy,
        dyn_policy_schedule=args.dyn_policy_schedule,
        dyn_policy_alpha_max=args.dyn_policy_alpha_max,
        dyn_policy_ramp_frac=args.dyn_policy_ramp_frac,
        dyn_policy_detach=args.dyn_policy_detach,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr,
        dynamics_lr=args.dynamics_lr, agg_lr=args.agg_lr,
    )
    (run_dir / "lewam_unified_config.json").write_text(json.dumps(cfg_out, indent=2))
    durable_sync([run_dir / "lewam_unified_config.json"], args.ckpt_sync_dir)

    D = 192
    Hmax = float(args.H_max)

    def run_batch(window, goals, actions, horizon, n_pos, train, dyn_mix=0.0):
        """One sequence-parallel step over a batch of context windows.

        window  (B, max_pos+1, 3, H, W)  states + next-state targets, tail-padded
        goals   (B, max_pos, 3, H, W)    each position's goal frame (sampled in the dataset)
        actions (B, max_pos, adim)       action block taken AT each decision point
        horizon (B, max_pos)             distance to each position's goal, in prediction steps
        n_pos   (B,)                     valid decision points per item

        Window + goal frames go through the encoder in ONE call (shared BatchNorm statistics,
        like the split's cat-encode of its t/goal/next triple); only the window latents are
        aggregator inputs. Losses are means over valid decision points."""
        B = window.shape[0]
        max_pos = actions.shape[1]
        window = window.to(device, non_blocking=True)
        goals = goals.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True).float()
        horizon = horizon.to(device, non_blocking=True)
        n_pos = n_pos.to(device, non_blocking=True)

        n_window_frames = B * (max_pos + 1)
        frames_all = torch.cat([window.reshape(n_window_frames, *window.shape[2:]),
                                goals.reshape(B * max_pos, *goals.shape[2:])]).float()
        z_all = model.encode(frames_all)
        z_window = z_all[:n_window_frames].reshape(B, max_pos + 1, D)
        z_goal = z_all[n_window_frames:].reshape(B, max_pos, D).float()
        states = z_window[:, :max_pos].float()        # (B, max_pos, D)  z_p at each decision point
        next_tgt = z_window[:, 1:].float()            # (B, max_pos, D)  z_{p+1}, the dynamics target

        pos = torch.arange(max_pos, device=device).unsqueeze(0)         # (1, max_pos)
        valid = pos < n_pos.unsqueeze(1)                                # (B, max_pos)
        # SHARED-mode horizons can exceed H_max (early positions are farther from the window's
        # goal); clamp exactly like the eval countdown's min(steps, H_max)/H_max.
        h_norm = horizon.clamp(max=args.H_max).float() / Hmax

        a_prev = a_prev_mask = None
        if model.agg_action_cond:
            a_prev = torch.zeros_like(actions)
            a_prev[:, 1:] = actions[:, :-1]           # a_{p-1}; position 0 gets the null embedding
            a_prev_mask = valid & (pos >= 1)

        a_pred, z_pred = model.forward_seq(states, z_goal, h_norm, actions, a_prev, a_prev_mask,
                                           dyn_action_mix=(dyn_mix if train else 0.0),
                                           dyn_action_detach=args.dyn_policy_detach)

        loss_mask = valid.unsqueeze(-1).float()
        n_valid = valid.sum().clamp(min=1)
        loss_act = ((a_pred - actions) ** 2 * loss_mask).sum() / (n_valid * actions.shape[-1])
        loss_dyn = ((z_pred - next_tgt) ** 2 * loss_mask).sum() / (n_valid * D)  # no stop-grad (like split)
        loss_cyc = torch.zeros((), device=device)
        if train and args.w_cyc > 0:
            c = model.aggregate(states, a_prev, a_prev_mask)
            z_cyc = model.dynamics(c.reshape(B * max_pos, D), a_pred.reshape(B * max_pos, -1),
                                   z_goal.reshape(B * max_pos, D)).reshape(B, max_pos, D)
            loss_cyc = ((z_cyc - next_tgt.detach()) ** 2 * loss_mask).sum() / (n_valid * D)
        # SIGReg over the valid decision-point latents only -- the split regularizes z_t, not the
        # goal/target frames, and states IS the per-position z_t set here.
        loss_reg = sigreg(states[valid].unsqueeze(0)) if train else torch.zeros((), device=device)
        return loss_act, loss_dyn, loss_reg, loss_cyc, int(n_valid.item())

    # ---- training loop ----
    best_val = float("inf")

    for ep in range(args.epochs):
        t0 = time.time()
        alpha = dyn_alpha(ep)

        # ---- train ----
        model.train()
        tr_act, tr_dyn, tr_reg, tr_cyc, tr_count = 0.0, 0.0, 0.0, 0.0, 0
        for window, goals, actions, horizon, n_pos in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_act, loss_dyn, loss_reg, loss_cyc, nval = run_batch(
                    window, goals, actions, horizon, n_pos, train=True, dyn_mix=alpha)
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
            for window, goals, actions, horizon, n_pos in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_act, loss_dyn, _, _, nval = run_batch(
                        window, goals, actions, horizon, n_pos, train=False)
                va_act += loss_act.item() * nval
                va_dyn += loss_dyn.item() * nval
                va_count += nval

        tr_a = tr_act / max(tr_count, 1); tr_d = tr_dyn / max(tr_count, 1)
        tr_r = tr_reg / max(tr_count, 1); tr_c = tr_cyc / max(tr_count, 1)
        va_a = va_act / max(va_count, 1); va_d = va_dyn / max(va_count, 1)
        lrs = sched.get_last_lr()
        dt = time.time() - t0
        cyc_str = f"  cyc={tr_c:.5f}" if args.w_cyc > 0 else ""
        dyn_str = f"  a={alpha:.2f}" if args.dyn_action_from_policy else ""
        print(f"[lewam-uni] ep {ep+1}/{args.epochs}  "
              f"act={tr_a:.5f}/{va_a:.5f}  dyn={tr_d:.5f}/{va_d:.5f}  reg={tr_r:.5f}{cyc_str}{dyn_str}  "
              f"lr_enc={lrs[0]:.2e}  {dt:.1f}s", flush=True)

        # ---- save checkpoints ----
        # the full model state_dict (encoder./aggregator./gc_head./dynamics.[/gate_proj.]) loads
        # strict into a LeWAMUnified rebuilt from the config (gip.load_lewam_unified_model).
        full_sd = model.state_dict()
        torch.save(full_sd, run_dir / "lewam_unified_latest.pt")
        # sync latest.pt EVERY epoch (not just best.pt on improvement) so a killed/reclaimed pod
        # never loses more than one epoch of progress -- best.pt alone can freeze on an early
        # improvement epoch for the rest of the run, silently discarding everything trained after it.
        durable_sync([run_dir / "lewam_unified_config.json",
                      run_dir / "lewam_unified_latest.pt"], args.ckpt_sync_dir)

        combined_val = va_a + va_d
        if combined_val < best_val:
            best_val = combined_val
            torch.save(full_sd, run_dir / "lewam_unified_best.pt")
            durable_sync([run_dir / "lewam_unified_config.json",
                          run_dir / "lewam_unified_best.pt"], args.ckpt_sync_dir)

    print(f"[lewam-uni] DONE  best_val={best_val:.5f}  saved -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
