"""
Training script for LeWAM-Unified

Example:
>>> python scripts/train_lewam_unified.py --dataset_name reacher.h5 \
       --run_name reacher_lewam_unified --epochs 50 --H_max 50 --context_len 5 --p_shared 0.5 \
       --agg_depth 4 --ckpt_sync_dir /hdfs/.../reacher_lewam_unified
"""
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
# Data loading (identical to train_lewam_gc.py)                               
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
# Context-window Dataset
# Each item is one window of up to `context_len` decision points plus
# exactly the frames those decision points use:                                                  
#   window  frames [start .. start+n_pos]   (n_pos+1: each position's state AND its z_{p+1} target)
#   goals   one frame per position, at the sampled goal offsets (RANDOM or SHARED mode, see the
#           file header) -- looked up by index, so memory per item is ~2*context_len+1 frames
#           REGARDLESS of H_max
#   actions the action block taken AT each decision point
#   horizon each position's distance to its goal in prediction steps
# Windows are clamped at the episode end, never cross it. Each window is a fresh start (no prior
# history), like an eval episode / the eval adapter's sliding window.
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

    Args:
        n_pos (int): number of decision positions in this window.
        frames_left (int): obs-frames remaining to the episode's last frame from the window start
            (the clamp ceiling for goal_offsets).
        h_max (int): max horizon (obs-steps) to draw h from.
        p_shared (float): probability of SHARED mode vs. RANDOM mode (see above).
        close_bias (float): >0 skews h toward small values (see above); 0 = uniform U[1,h_max].

    Returns:
        tuple: (goal_offsets (n_pos,) long, horizon (n_pos,) long). horizon can exceed h_max in
        SHARED mode (earlier positions are farther); the trainer clamps h_norm at 1.0 exactly like
        the eval countdown's min(steps, H_max)/H_max.
    """
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
# Prefix (Fast-LeWM) dataset. Same anchor points as SeqTrajDataset, but the window extends prefix_H  #
# frames PAST the last anchor so every anchor has its world-model targets z_{t+1..t+H} in the window, #
# plus a per-anchor H-action prefix. `prefix_valid[t,k]` marks anchor t's real horizons (episode end).#
# --------------------------------------------------------------------------- #
class PrefixSeqDataset(Dataset):
    def __init__(self, frames, a_frame, t_gidx, maxh, indices, h_max, ctx_len, prefix_H,
                 p_shared, close_bias=0.0):
        self.frames = frames          # [N,3,H,W] fp16 CPU; all obs-frames
        self.a_frame = a_frame        # [N,adim] fp16; a_frame[t] = block taken AT frame t (0 at ep-last)
        self.t_gidx = t_gidx          # [M] episode-global frame index of each valid start
        self.maxh = maxh              # [M] frames from that start to the episode's last frame
        self.indices = indices        # [K] train/val subset of the M starts
        self.h_max = int(h_max)
        self.ctx_len = int(ctx_len)   # decision points (anchors) per window
        self.prefix_H = int(prefix_H) # world-model prefix horizon
        self.p_shared = float(p_shared)
        self.close_bias = float(close_bias)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        start = int(self.t_gidx[idx])
        frames_left = int(self.maxh[idx])                        # frames start..start+frames_left exist
        n_anchor = min(self.ctx_len, frames_left)                # anchors 0..n_anchor-1 (each >=1 future frame)
        prefix_H = self.prefix_H
        win_len = min(n_anchor + prefix_H, frames_left + 1)      # window frames: states + WM targets
        window = self.frames[start:start + win_len]              # (win_len, 3, H, W)
        goal_offsets, horizon = sample_goal_offsets(n_anchor, frames_left, self.h_max,
                                                    self.p_shared, self.close_bias)
        goals = self.frames[start + goal_offsets]                # (n_anchor, 3, H, W) per-position policy goal
        action_dim = self.a_frame.shape[1]
        action_prefix = torch.zeros(n_anchor, prefix_H, action_dim, dtype=self.a_frame.dtype)
        prefix_valid = torch.zeros(n_anchor, prefix_H, dtype=torch.bool)
        for t in range(n_anchor):
            # anchor t: horizon k=1..k_valid has target z_{t+k} in-window AND action a_{t+k-1} real
            k_valid = min(prefix_H, win_len - 1 - t)
            if k_valid > 0:
                action_prefix[t, :k_valid] = self.a_frame[start + t:start + t + k_valid]
                prefix_valid[t, :k_valid] = True
        return window, goals, action_prefix, horizon, prefix_valid, n_anchor


def collate_prefix(batch):
    """Pad a batch of prefix windows. window -> (B, win_max, C, H, W) tail-padded with each item's LAST
    REAL frame (BatchNorm safety); goals -> (B, anchor_max, C, H, W); action_prefix -> (B, anchor_max,
    prefix_H, adim) zero-pad; horizon -> (B, anchor_max) pad 1; prefix_valid -> (B, anchor_max, prefix_H)
    pad False; n_anchor -> (B,)."""
    n_anchor = torch.tensor([item[5] for item in batch], dtype=torch.long)
    anchor_max = int(n_anchor.max())
    win_max = max(item[0].shape[0] for item in batch)
    B = len(batch)
    C, H, W = batch[0][0].shape[1:]
    prefix_H, action_dim = batch[0][2].shape[1], batch[0][2].shape[2]
    window = torch.zeros((B, win_max, C, H, W), dtype=batch[0][0].dtype)
    goals = torch.zeros((B, anchor_max, C, H, W), dtype=batch[0][1].dtype)
    action_prefix = torch.zeros((B, anchor_max, prefix_H, action_dim), dtype=batch[0][2].dtype)
    horizon = torch.ones((B, anchor_max), dtype=torch.long)
    prefix_valid = torch.zeros((B, anchor_max, prefix_H), dtype=torch.bool)
    for i, (win, goal, act_pref, hz, pv, na) in enumerate(batch):
        win_len = win.shape[0]
        window[i, :win_len] = win
        if win_len < win_max:
            window[i, win_len:] = win[-1]                        # repeat last real frame (BatchNorm)
        goals[i, :na] = goal
        if na < anchor_max:
            goals[i, na:] = goal[-1]
        action_prefix[i, :na] = act_pref
        horizon[i, :na] = hz
        prefix_valid[i, :na] = pv
    return window, goals, action_prefix, horizon, prefix_valid, n_anchor


# --------------------------------------------------------------------------- #
# Main training                                                                #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    # train config
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--run_name", type=str, default="reacher_lewam_unified")
    ap.add_argument("--exp_tag", type=str, default="",
                    help="short experiment slug (e.g. 'noh', 'aggcat'). Appended to the checkpoint "
                         "directory AND to --ckpt_sync_dir, so two experiments that differ only by "
                         "a flag cannot silently overwrite each other's run. Recorded in the config.")
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--dataset_name", type=str, default="reacher.h5")
    ap.add_argument("--keys_to_load", type=str, default="pixels,action")
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--ckpt_sync_dir", type=str, default=None,
                    help="durable dir (e.g. an HDFS mount) to mirror config + best ckpt into on "
                            "each improvement, so losing the worker's ephemeral disk never costs the run.")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--prefetch_factor", type=int, default=3)
    ap.add_argument("--frames_cache", type=str, default="auto",
                    help="prebuilt-cache dir | 'auto' (env LEWAM_CACHE_DIR or the default HDFS "
                            "preload_cache) | 'off'. HIT -> np.load the shared <tag>.frames.npy + "
                            "<tag>.aux.npz instead of decoding the h5 (~1x-frames RAM vs classic "
                            "preload). Ignored when --max_eps set (falls back to classic preload).")
    ap.add_argument("--cache_mmap", action="store_true",
                    help="memory-map the cache .frames.npy (np.load mmap_mode='r') "
                            "instead of reading it fully into RAM. "
                            "Peak RSS drops at the cost of fuse page-in latency on random "
                            "reads (hidden by --num_workers; warm after epoch 1). Default load stays full-RAM.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128,
                    help="windows per batch (each carries up to context_len decision points, so the "
                         "effective decision-point batch is larger; raise/lower per GPU mem)")
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--dynamics_lr", type=float, default=3e-4)
    ap.add_argument("--agg_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--H_max", type=int, default=50,
                    help="max goal distance (horizon) in obs-steps (no frameskip); "
                         "each goal is at horizon h~U[1,H_max], episode-clamped. "
                         "Decoupled from the window length (see --context_len); does not affect "
                         "memory (goals are looked up per position, not loaded as a tail)")
    ap.add_argument("--ablate_horizon", action="store_true",
                    help="drop the horizon conditioning: h_norm is fed as 0 everywhere. "
                         "Hindsight h is the TRUE remaining distance to the relabelled "
                         "goal, so conditioning on it hands the policy privileged "
                         "information no deployment has (and under full-traj it saturates "
                         "to a constant anyway). Recorded in the run config so eval picks "
                         "it up automatically.")
    ap.add_argument("--context_len", type=int, default=5,
                    help="decision points per context window = the aggregator's max sequence length "
                         "at train (eval defaults its ctx_cap to this). Frames per item = "
                         "2*context_len+1, so batch * context_len determines GPU memory")
    ap.add_argument("--p_shared", type=float, default=0.0,
                    help="fraction of windows trained in SHARED-goal mode (one goal ahead of the "
                         "window, horizons counting down -- the structure eval runs); the rest use "
                         "per-position independent goals (the split's sampling). 0 = all random")
    ap.add_argument("--goal_close_bias", type=float, default=0.0,
                    help="skew the training goal-distance draw toward SMALL h (more close-to-goal "
                         "supervision): h=1+floor((H_max-1)*u^(1+bias)). 0=uniform U[1,H_max]. "
                         "Applied to train only; val stays uniform for a comparable metric.")
    # Restored: the 0497f88 cleanup commented these two flags out but left their three usage
    # sites (args.perturb_data at ~L707/709, args.perturb_ratio at ~L716/795), so ANY run died at
    # startup with AttributeError. Defaults keep the feature off, exactly as before the cleanup.
    ap.add_argument("--perturb_data", type=str, default="",
                    help="off-policy transition h5 (gen_offpolicy.py); '' = none. Mixed into the dynamics "
                         "loss only (DAgger-for-the-critic) -- the reactive/BC head stays on-policy.")
    ap.add_argument("--perturb_ratio", type=float, default=0.0,
                    help="dynamics-loss weight on off-policy transitions: loss_dyn=(1-r)*on + r*off. 0=baseline.")

    # model config
    ap.add_argument("--head_type", type=str, default="mse",
                    help="gc_head output: 'mse' (deterministic point, MSE loss), 'gmm' (K-component "
                         "diagonal-Gaussian mixture density net, mixture-NLL loss), or 'flow' "
                         "(rectified flow-matching action-chunk head). gmm/flow capture multimodal "
                         "actions and can be SAMPLED for policy-proposal planning.")
    ap.add_argument("--n_mix", type=int, default=5,
                    help="number of mixture components when --head_type gmm")
    ap.add_argument("--flow_H", type=int, default=1,
                    help="action-chunk length in blocks for --head_type flow; H=1 = single-block "
                         "drop-in (current data path), H>1 (chunking) needs (N,H,d) targets + mask")
    ap.add_argument("--embed_dim", type=int, default=192,
                    help="latent width feeding aggregator + heads (ViT-tiny cls is projected to this)")
    ap.add_argument("--hidden_dim", type=int, default=512,
                    help="hidden width for gc_head/dynamics/idm_head MLPs.")
    ap.add_argument("--encoder_size", type=str, default="tiny",
                    help="ViT backbone size: tiny | small | base | large")
    ap.add_argument("--encoder_backbone", type=str, default="scratch",
                    help="scratch = from-scratch ViT (SIGReg-trained); dinov3 = frozen pretrained "
                         "DINOv3 ViT-{encoder_size}/16")
    ap.add_argument("--encoder_ckpt", type=str, default="",
                    help="state_dict .pt for the pretrained backbone, loaded at train init only "
                         "(dinov3). At eval the frozen weights live in the full checkpoint.")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--agg_depth", type=int, default=4,
                    help="causal-transformer depth of the aggregator")
    ap.add_argument("--agg_heads", type=int, default=4)
    ap.add_argument("--agg_dim_head", type=int, default=0,
                    help="per-head dim of the aggregator attention; 0 = embed_dim//agg_heads.")
    ap.add_argument("--agg_mlp_dim", type=int, default=0,
                    help="aggregator FFN width; 0 = 4*embed_dim (baseline). >0 widens the FFN.")
    ap.add_argument("--agg_residual", action="store_true",
                    help="c = z + Aggr(z) with a zero-init correction (boots as the split, "
                         "identity at init); off (default) = c = Aggr(z), no residual")
    ap.add_argument("--agg_gate", action="store_true",
                    help="input-dependent sigmoid gate on the residual correction "
                         "(c = z + g(z)*Aggr(z); still boots as split). residual only.")
    ap.add_argument("--agg_action_cond", action="store_true",
                    help="condition each token z_tau (AdaLN) on the embedded previous action "
                         "a_{tau-1} (null-action at the sequence start); conditioning, not tokens")
    ap.add_argument("--dyn_action_embed_dim", type=int, default=128,
                        help="0 = raw-concat the action into the dynamics head (baseline). >0 embeds the "
                             "action (Linear->LayerNorm->GELU) to this width before concat, so it isn't "
                             "drowned by the z_dim latents (a stronger action pathway).")
    ap.add_argument("--dyn_goal_cond", action="store_true",
                    help="use z_goal in the dynamics head")
    # dynamics-on-policy-action arm: feed the gc_head's predicted action into the dynamics head
    # (a convex mix with the ground-truth action, weight alpha ramped by a schedule), closing the
    # train/rollout covariate gap. The action is still BC-supervised on ground truth, so it stays a
    # grounded action, not a latent-action model.
    ap.add_argument("--dyn_action_from_policy", action="store_true",
                    help="feed a mix of the policy's predicted action into the dynamics head "
                         "(alpha per --dyn_policy_schedule); off = dynamics on ground-truth actions")
    ap.add_argument("--dyn_policy_schedule", type=str, default="const",
                    help="alpha(epoch) schedule for the policy-action mix: const|linear|cosine|"
                         "sigmoid. const = full alpha from epoch 0")
    ap.add_argument("--dyn_policy_alpha_max", type=float, default=1.0,
                    help="max mix weight (1.0 = dynamics sees only the policy action at the end)")
    ap.add_argument("--dyn_policy_ramp_frac", type=float, default=0.5,
                    help="fraction of total epochs over which alpha ramps 0->alpha_max "
                         "(non-const schedules); after that alpha stays at alpha_max")
    ap.add_argument("--dyn_policy_detach", action="store_true",
                    help="stop-gradient the policy action into dynamics (dynamics adapts to the "
                         "policy's actions, but the dyn loss never reshapes the BC policy). "
                         "default: gradients flow policy->dynamics (coupled 'unified' arm)")
    ap.add_argument("--dyn_prefix", action="store_true",
                        help="Fast-LeWM action-prefix dynamics: predict all H future latents per anchor in "
                             "parallel from the aggregated context c_t (no autoregression, no compounding). "
                             "Replaces the single-step dynamics + rollout_k. Uses PrefixSeqDataset (window "
                             "extends prefix_H past the last anchor) + the dense prefix loss.")
    ap.add_argument("--prefix_H", type=int, default=5,
                    help="prefix horizon for --dyn_prefix (action blocks predicted per anchor); should "
                            "match the eval plan horizon cem_H. Window frames/item = ctx_len + prefix_H.")
    ap.add_argument("--prefix_depth", type=int, default=2,
                    help="causal-transformer depth of the action-prefix encoder")
    ap.add_argument("--prefix_heads", type=int, default=4,
                    help="attention heads of the action-prefix encoder")
    # loss
    ap.add_argument("--latent_h", type=str, default="", choices=["", "vq", "scalar"],
                    help="condition the GC head on a horizon inferred from (c_t, z_goal) instead of "
                         "the dataset's h. 'vq' quantizes it to --h_codes entries; 'scalar' predicts "
                         "h and reuses the sinusoidal path (the control that separates the "
                         "quantization from the h-supervision). The true h becomes a regression "
                         "target only, so eval never needs it.")
    ap.add_argument("--h_codes", type=int, default=16, help="codebook size for --latent_h vq")
    ap.add_argument("--h_code_dim", type=int, default=64, help="code width for --latent_h vq")
    ap.add_argument("--h_commit", type=float, default=0.25, help="VQ commitment weight")
    ap.add_argument("--h_pred_w", type=float, default=1.0,
                    help="weight on the h regression that grounds the code in real distance")
    ap.add_argument("--w_act", type=float, default=1.0)
    ap.add_argument("--w_dyn", type=float, default=1.0)
    ap.add_argument("--w_idm", type=float, default=0.0,
                    help="inverse-dynamics aux-loss weight: recover a_t from (z_t, predicted z_{t+1}) -> "
                         "forces the dynamics to be action-aware. 0 = off (baseline).")
    ap.add_argument("--w_acons", type=float, default=0.0,
                    help="action-consistency aux-loss weight: decode the next action from the "
                         "dynamics' own predicted latent -- a_hat_{t+1}=gc_head(z_hat_{t+1}, goal_{t+1}, "
                         "h_{t+1}) supervised on a_{t+1}. Couples dynamics->policy so z_hat_{t+1} must land "
                         "in a policy-decodable latent (the representation lever). Grad flows through both "
                         "the dynamics (predicted latent) and the head (no detach). 0 = off. Non-prefix "
                         "path only (run_batch).")
    ap.add_argument("--rollout_k", type=int, default=1,
                    help="latent-rollout depth for the dynamics loss. 1 = teacher "
                         "forcing only (baseline). 2 = one rollout step (feed the predicted z_{t+1} "
                         "back to predict z_{t+2}); K>2 rolls deeper. State-only rollout (actions/goals "
                         "stay ground truth) -> targets compounding error.")
    ap.add_argument("--w_rollout", type=float, default=1.0,
                    help="weight on the rollout term relative to the teacher-forced dyn term (both "
                         "scaled by w_dyn): L_dyn = w_dyn*(L_tf + w_rollout*L_rollout)")
    ap.add_argument("--w_reg", type=float, default=0.04,
                    help="SIGReg regularization weight")
    ap.add_argument("--w_cyc", type=float, default=0.0,
                    help="FDM-IDM consistency loss weight (0=without, 1.0=with)")
    ap.add_argument("--w_straight", type=float, default=0.0,
                    help="temporal-straightening loss weight: w*(1 - cos(v_t, v_{t+1})) over "
                         "consecutive latent velocities within each context window (v_t=z_{t+1}-z_t). "
                         "Straightens the latent dynamics -> better rollout/cost geometry/subgoal "
                         "interpolation. 0=off. Computed on the UNFLATTENED per-window latents (no "
                         "cross-sequence bleed); needs context_len>=3 (>=4 recommended).")
    ap.add_argument("--straight_target", type=str, default="z",
                    help="what to straighten: 'z' (encoder latents; traditional AND the "
                         "dynamics-target/cost space) or 'c' (aggregated context). z recommended.")
    ap.add_argument("--ablate_dynamics", action="store_true",
                        help="disable dynamics loss (w_dyn=0), train only a policy. "
                             "Behavior is idential to w_dyn=0")
    args = ap.parse_args()
    if args.latent_h and args.ablate_horizon:
        raise SystemExit("--latent_h and --ablate_horizon are different arms; pick one")

    if args.ablate_dynamics:
        args.w_dyn = 0.0
    assert args.context_len >= 1 and args.H_max >= 1, "context_len and H_max must be >= 1"
    assert 0.0 <= args.p_shared <= 1.0, "p_shared must be in [0, 1]"
    assert args.straight_target in ("z", "c"), "straight_target must be 'z' or 'c'"
    if args.w_straight > 0:
        assert args.context_len >= 3, "temporal straightening needs context_len>=3 (>=4 recommended)"

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

    # --exp_tag differentiates otherwise-identical runs. There is no resume path here and
    # run_dir.mkdir(exist_ok=True) writes straight into whatever is there, so a repeated run_name
    # silently overwrites the earlier experiment's weights and config.
    _tag = args.exp_tag.strip().strip("_")
    _run_name = f"{args.run_name}_{_tag}" if _tag else args.run_name
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), _run_name)
    if _tag and args.ckpt_sync_dir:
        # keep the remote layout in step with the local one
        args.ckpt_sync_dir = str(Path(args.ckpt_sync_dir) / _tag)
    if (run_dir / "lewam_unified_best.pt").exists():
        raise SystemExit(f"[lewam-uni] {run_dir} already holds a finished run "
                         f"(lewam_unified_best.pt). Pass a different --exp_tag / --run_name rather "
                         f"than overwriting it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if os.path.realpath(run_dir).startswith(os.path.realpath(os.path.expanduser("~"))):
        print(f"[lewam-uni] WARN run_dir under $HOME ({run_dir}); checkpoints in home can fill the "
              f"shared fs -- pass --run_dir to local scratch (/opt/tiger/...) or HDFS", flush=True)
    max_eps = args.max_eps or None

    # ---- build model ----
    model = LeWAMUnified(encoder_size=args.encoder_size, embed_dim=args.embed_dim,
                         action_dim=action_block_dim,
                         hidden_dim=args.hidden_dim, img_size=args.img_size, dropout=0.1,
                         agg_depth=args.agg_depth, agg_heads=args.agg_heads,
                         agg_dim_head=(args.agg_dim_head or None),
                         agg_mlp_dim=(args.agg_mlp_dim or None),
                         agg_residual=args.agg_residual, agg_gate=args.agg_gate,
                         agg_action_cond=args.agg_action_cond,
                         dyn_goal_cond=args.dyn_goal_cond,
                         dyn_action_embed_dim=args.dyn_action_embed_dim,
                         head_type=args.head_type, n_mix=args.n_mix, flow_H=args.flow_H,
                         encoder_backbone=args.encoder_backbone,
                         encoder_ckpt=(args.encoder_ckpt or None),
                         use_idm=(args.w_idm > 0),
                         use_prefix=args.dyn_prefix, prefix_H=args.prefix_H,
                         prefix_depth=args.prefix_depth, prefix_heads=args.prefix_heads,
                         latent_h=args.latent_h, h_codes=args.h_codes,
                         h_code_dim=args.h_code_dim, h_commit=args.h_commit,
                         h_pred_w=args.h_pred_w).to(device)
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
            print(f"[lewam-uni] frames-cache HIT {_fp}"
                  + (" (mmap: low-RAM, fuse-latency)" if args.cache_mmap else ""), flush=True)
            # mmap keeps large tensor on disk/fuse (kernel-evictable pages)
            Frames = torch.from_numpy(np.load(_fp, mmap_mode="r" if args.cache_mmap else None))
            _aux = np.load(_ap)
            # cache built by the GC pipeline: A_flat is per-SAMPLE (one action block per
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
    if args.dyn_prefix:
        train_ds = PrefixSeqDataset(Frames, A_frame, t_gidx, maxh, train_idx, args.H_max,
                                    args.context_len, args.prefix_H, args.p_shared, args.goal_close_bias)
        val_ds = PrefixSeqDataset(Frames, A_frame, t_gidx, maxh, val_idx, args.H_max,
                                  args.context_len, args.prefix_H, args.p_shared, 0.0)
    else:
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
        num_workers=args.num_workers, collate_fn=(collate_prefix if args.dyn_prefix else collate_pad),
    )
    if args.num_workers > 0:
        _loader_common.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    train_loader = DataLoader(train_ds, sampler=train_sampler, **_loader_common)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **_loader_common)
    print(f"[lewam-uni] loaders ready: workers={args.num_workers} prefetch={args.prefetch_factor} "
          f"pin_memory=True  trajectories/epoch: train={n_tr_ep} val={n_va_ep} (~1x coverage)", flush=True)

    # ---- optimizer: 4 param groups ----
    opt = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.encoder_lr},
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
        encoder_backbone=args.encoder_backbone,
        action_dim=action_block_dim,
        hidden_dim=args.hidden_dim, n_freqs=64, dropout=0.1,
        agg_depth=args.agg_depth, agg_heads=args.agg_heads,
        agg_dim_head=args.agg_dim_head, agg_mlp_dim=args.agg_mlp_dim,
        agg_residual=args.agg_residual, agg_gate=args.agg_gate,
        agg_action_cond=args.agg_action_cond, dyn_goal_cond=args.dyn_goal_cond,
        dyn_action_embed_dim=args.dyn_action_embed_dim,
        H_max=args.H_max, context_len=args.context_len, p_shared=args.p_shared,
        frameskip=frameskip, action_raw_dim=raw_adim,
        action_mean=act_mean, action_std=act_std,
        latent_h=args.latent_h, h_codes=args.h_codes, h_code_dim=args.h_code_dim,
        h_commit=args.h_commit, h_pred_w=args.h_pred_w,
        w_act=args.w_act, w_dyn=args.w_dyn, w_idm=args.w_idm, w_reg=args.w_reg, w_cyc=args.w_cyc,
        w_acons=args.w_acons,
        rollout_k=args.rollout_k, w_rollout=args.w_rollout,
        use_prefix=args.dyn_prefix, prefix_H=args.prefix_H, prefix_depth=args.prefix_depth,
        prefix_heads=args.prefix_heads,
        w_straight=args.w_straight, straight_target=args.straight_target,
        head_type=args.head_type, n_mix=args.n_mix, flow_H=args.flow_H,
        ablate_dynamics=args.ablate_dynamics,
        dyn_action_from_policy=args.dyn_action_from_policy,
        dyn_policy_schedule=args.dyn_policy_schedule,
        dyn_policy_alpha_max=args.dyn_policy_alpha_max,
        dyn_policy_ramp_frac=args.dyn_policy_ramp_frac,
        dyn_policy_detach=args.dyn_policy_detach,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr,
        dynamics_lr=args.dynamics_lr, agg_lr=args.agg_lr,
        ablate_horizon=args.ablate_horizon, exp_tag=_tag, run_name=_run_name,
    )
    (run_dir / "lewam_unified_config.json").write_text(json.dumps(cfg_out, indent=2))
    durable_sync([run_dir / "lewam_unified_config.json"], args.ckpt_sync_dir)

    D = args.embed_dim
    Hmax = float(args.H_max)

    # --- off-policy transitions for the DAgger-for-the-critic dynamics mix (optional) ---
    OP = None
    if args.perturb_data and args.perturb_ratio > 0:
        import h5py as _h5
        with _h5.File(args.perturb_data, "r") as _f:
            OP = dict(ctx=torch.from_numpy(_f["ctx_frames"][:]),   # (M,cap,224,224,3) uint8
                      goal=torch.from_numpy(_f["goal_frame"][:]),
                      nxt=torch.from_numpy(_f["next_frame"][:]),
                      act=torch.from_numpy(_f["action"][:]).float())  # (M,adim) z-scored
        _imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _istd  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        print(f"[lewam-uni] off-policy dynamics mix: {OP['ctx'].shape[0]} transitions r={args.perturb_ratio}", flush=True)

    def _enc_op(frames_u8):                                   # (N,224,224,3) uint8 -> (N,D), norm like the cache
        x = frames_u8.to(device, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0
        return model.encode((x - _imean) / _istd)

    def run_batch(window, goals, actions, horizon, n_pos, train, dyn_mix=0.0):
        """One sequence-parallel step over a batch of context windows.

        window  (B, max_pos+1, 3, H, W)  states + next-state targets, tail-padded
        goals   (B, max_pos, 3, H, W)    each position's goal frame (sampled in the dataset)
        actions (B, max_pos, adim)       action block taken AT each decision point
        horizon (B, max_pos)             distance to each position's goal, in prediction steps
        n_pos   (B,)                     valid decision points per item

        Window + goal frames go through the encoder in one call (shared BatchNorm statistics, like the
        split's cat-encode of its t/goal/next triple); only the window latents are aggregator inputs.
        Losses are means over valid decision points."""
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
        if args.ablate_horizon:
            h_norm = torch.zeros_like(h_norm)

        a_prev = a_prev_mask = None
        if model.agg_action_cond:
            a_prev = torch.zeros_like(actions)
            a_prev[:, 1:] = actions[:, :-1]           # a_{p-1}; position 0 gets the null embedding
            a_prev_mask = valid & (pos >= 1)

        # model forward
        a_out, z_pred, a_idm, loss_h = model.forward_seq(states, z_goal, h_norm, actions, a_prev, a_prev_mask,
                                                 dyn_action_mix=(dyn_mix if train else 0.0),
                                                 dyn_action_detach=args.dyn_policy_detach)
        
        loss_mask = valid.unsqueeze(-1).float()
        n_valid = valid.sum().clamp(min=1)

        # action loss (split by head type)
        if getattr(model.gc_head, "head_type", "") == "flow" and model.flow_H > 1:
            # action-CHUNK target: at position t the next flow_H blocks a_frame[t:t+H], masked past valid
            fh, adim = model.flow_H, actions.shape[-1]
            a_pad = torch.cat([actions, actions.new_zeros(B, fh - 1, adim)], dim=1)
            chunk = a_pad.unfold(1, fh, 1).permute(0, 1, 3, 2)                 # (B, max_pos, fh, adim)
            v_pad = torch.cat([valid, valid.new_zeros(B, fh - 1)], dim=1)
            cmask = v_pad.unfold(1, fh, 1)                                     # (B, max_pos, fh)
            aloss = model.gc_head.action_loss(a_out.reshape(B * max_pos, -1).float(),
                                              chunk.reshape(B * max_pos, fh, adim).float(),
                                              mask=cmask.reshape(B * max_pos, fh).float())
        else:
            aloss = model.gc_head.action_loss(a_out.reshape(B * max_pos, -1).float(),
                                              actions.reshape(B * max_pos, actions.shape[-1]).float())
        loss_act = (aloss * valid.reshape(-1).float()).sum() / n_valid

        # dynamics loss
        loss_dyn = ((z_pred - next_tgt) ** 2 * loss_mask).sum() / (n_valid * D)

        # DAgger for dynamics 
        if train and OP is not None:
            # mix an off-policy dynamics term (env-rendered (ctx,a',next) with a'
            # perturbed in z-scored space). Dynamics only -- the BC/action loss above stays on-policy, so
            # the reactive head is untouched. loss_dyn = (1-r)*on + r*off holds the dyn budget fixed.
            r = float(args.perturb_ratio); cap_op = OP["ctx"].shape[1]
            bso = min(32, OP["ctx"].shape[0])
            ix = torch.randint(0, OP["ctx"].shape[0], (bso,))
            with torch.no_grad():
                zc = _enc_op(OP["ctx"][ix].reshape(bso * cap_op, 224, 224, 3)).reshape(bso, cap_op, D)
                a_pv = a_pm = None
                if model.agg_action_cond:
                    a_pv = torch.zeros(bso, cap_op, actions.shape[-1], device=device)
                    a_pm = torch.zeros(bso, cap_op, dtype=torch.bool, device=device)
                c_op = model.aggregate(zc, a_pv, a_pm)[:, -1]
                zg_op = _enc_op(OP["goal"][ix]); zn_op = _enc_op(OP["nxt"][ix])
            z_pred_op = model.dynamics(c_op, OP["act"][ix].to(device), zg_op)   # grad -> dynamics head only
            loss_dyn_op = ((z_pred_op - zn_op) ** 2).mean()
            loss_dyn = (1.0 - r) * loss_dyn + r * loss_dyn_op

        # inverse-dynamics loss
        loss_idm = torch.zeros((), device=device)
        if a_idm is not None:
            loss_idm = ((a_idm - actions) ** 2 * loss_mask).sum() / (n_valid * a_idm.shape[-1])

        # rollout loss
        loss_rollout = torch.zeros((), device=device)
        if train and args.rollout_k > 1 and args.w_rollout > 0:
            loss_rollout = model.rollout_dyn(states, z_pred, next_tgt, z_goal, actions, n_pos,
                                             args.rollout_k, a_prev, a_prev_mask)

        # cycle consistency loss
        loss_cyc = torch.zeros((), device=device)
        if train and args.w_cyc > 0:
            c = model.aggregate(states, a_prev, a_prev_mask)
            a_pt = model.gc_head.point(a_out.reshape(B * max_pos, -1))   # policy action (mixture mean for gmm)
            z_cyc = model.dynamics(c.reshape(B * max_pos, D), a_pt,
                                   z_goal.reshape(B * max_pos, D)).reshape(B, max_pos, D)
            loss_cyc = ((z_cyc - next_tgt.detach()) ** 2 * loss_mask).sum() / (n_valid * D)

        # action-consistency lozz: the dynamics' OWN predicted latent must decode to the NEXT
        # action. Anchor t predicts z_hat_{t+1}=z_pred[:,t]; require gc_head(z_hat_{t+1}, goal_{t+1},
        # h_{t+1}) == a_{t+1}. Couples dynamics->policy -- z_hat_{t+1} is pushed into a policy-decodable
        # latent (grad flows through both the dynamics that made z_hat and the head that reads it, no
        # detach). Valid where both t and t+1 are real decision points (consecutive in-window anchors).
        loss_acons = torch.zeros((), device=device)
        if train and args.w_acons > 0 and max_pos >= 2:
            n_pair = B * (max_pos - 1)
            z_next = z_pred[:, :-1].reshape(n_pair, D)                    # z_hat_{t+1}, t=0..max_pos-2
            goal_next = z_goal[:, 1:].reshape(n_pair, D)                  # position t+1's goal
            h_next = h_norm[:, 1:].reshape(n_pair)                        # position t+1's horizon
            a_next = actions[:, 1:].reshape(n_pair, actions.shape[-1])    # a_{t+1}
            acons_out = model.gc_head(z_next, goal_next, h_next)
            acons_loss = model.gc_head.action_loss(acons_out.float(), a_next.float())
            pair_valid = (valid[:, 1:] & valid[:, :-1]).reshape(-1).float()   # t AND t+1 real
            loss_acons = (acons_loss * pair_valid).sum() / pair_valid.sum().clamp(min=1)

        # SIGReg over the valid decision-point latents only -- the split regularizes z_t, not the
        # goal/target frames, and states IS the per-position z_t set here.
        loss_reg = sigreg(states[valid].unsqueeze(0)) if train else torch.zeros((), device=device)

        # temporal straightening: w*(1 - cos(v_t, v_{t+1})) over consecutive latent velocities within
        # each window. Velocities never bleed across sequences. target z.
        #  Masked to real (non-pad) triples: cos(v_t, v_{t+1}) needs frames t, t+1, t+2 all real.
        loss_straight = torch.zeros((), device=device)
        if train and args.w_straight > 0:
            if args.straight_target == "c":
                seq = model.aggregate(states, a_prev, a_prev_mask)      # (B, max_pos, D)
                nreal = n_pos                                          # c: n_pos valid positions
            else:
                seq = z_window                                         # (B, max_pos+1, D): frames 0..n_pos real
                nreal = n_pos + 1
            v = F.normalize(seq.float()[:, 1:] - seq.float()[:, :-1], dim=-1, eps=1e-6)  # (B,T-1,D)
            csim = (v[:, 1:] * v[:, :-1]).sum(-1)                       # (B, T-2) cos(v_t, v_{t+1})
            tpos = torch.arange(csim.shape[1], device=device).unsqueeze(0)
            vmask = tpos < (nreal.unsqueeze(1) - 2)                     # real triple t,t+1,t+2
            loss_straight = ((1.0 - csim) * vmask.float()).sum() / vmask.sum().clamp(min=1)
        
        return (loss_act, loss_dyn, loss_reg, loss_cyc, loss_straight, loss_idm, loss_rollout,
                loss_acons, loss_h, int(n_valid.item()))

    def run_batch_prefix(window, goals, action_prefix, horizon, prefix_valid, n_anchor, train):
        """One Fast-LeWM prefix step. Encodes the window (anchor states + WM targets) + policy goals in
        one pass; the policy head trains on the anchor action a_t (prefix block 0), the world model on
        the dense prefix loss ‖ẑ_{t+k}−z_{t+k}‖² over the valid (anchor, horizon) pairs. Returns the same
        9-tuple as run_batch (cyc/straight/idm/rollout/acons are 0) so the train loop stays uniform."""
        B, win_max = window.shape[0], window.shape[1]
        anchor_max, prefix_H, action_dim = action_prefix.shape[1], action_prefix.shape[2], action_prefix.shape[3]
        window = window.to(device, non_blocking=True)
        goals = goals.to(device, non_blocking=True)
        action_prefix = action_prefix.to(device, non_blocking=True).float()
        horizon = horizon.to(device, non_blocking=True)
        prefix_valid = prefix_valid.to(device, non_blocking=True)
        n_anchor = n_anchor.to(device, non_blocking=True)
        n_win = B * win_max
        frames_all = torch.cat([window.reshape(n_win, *window.shape[2:]),
                                goals.reshape(B * anchor_max, *goals.shape[2:])]).float()
        z_all = model.encode(frames_all)
        z_window = z_all[:n_win].reshape(B, win_max, D).float()             # anchor states + WM targets
        z_goal = z_all[n_win:].reshape(B, anchor_max, D).float()           # per-position policy goal
        state_seq = z_window[:, :anchor_max].float()                       # anchors 0..anchor_max-1
        pos = torch.arange(anchor_max, device=device).unsqueeze(0)         # (1, anchor_max)
        anchor_valid = pos < n_anchor.unsqueeze(1)                         # (B, anchor_max)
        horizon_norm = horizon.clamp(max=args.H_max).float() / Hmax
        a_prev = a_prev_mask = None
        if model.agg_action_cond:
            anchor_action = action_prefix[:, :, 0]                         # a_t at each anchor
            a_prev = torch.zeros_like(anchor_action); a_prev[:, 1:] = anchor_action[:, :-1]
            a_prev_mask = anchor_valid & (pos >= 1)
        action_out, pred_seq, loss_h = model.forward_seq_prefix(state_seq, z_goal, horizon_norm,
                                                                action_prefix, a_prev, a_prev_mask)
        # policy BC loss on the anchor action (prefix block 0), masked to valid anchors
        anchor_action = action_prefix[:, :, 0]                            # (B, anchor_max, adim)
        aloss = model.gc_head.action_loss(action_out.reshape(B * anchor_max, -1).float(),
                                          anchor_action.reshape(B * anchor_max, action_dim).float())
        n_valid_anchor = anchor_valid.sum().clamp(min=1)
        loss_act = (aloss * anchor_valid.reshape(-1).float()).sum() / n_valid_anchor
        # dense prefix WM loss: target z_{t+k} = z_window[:, t+k]  (clamp OOB -> masked out by prefix_valid)
        horizon_k = torch.arange(1, prefix_H + 1, device=device)          # (prefix_H,)
        tgt_idx = (pos.reshape(-1, 1) + horizon_k.reshape(1, -1)).clamp(max=win_max - 1)  # (anchor_max, prefix_H)
        target_seq = z_window[:, tgt_idx]                                 # (B, anchor_max, prefix_H, D)
        prefix_mask = prefix_valid.float().unsqueeze(-1)                  # (B, anchor_max, prefix_H, 1)
        n_prefix = prefix_valid.sum().clamp(min=1)
        loss_dyn = ((pred_seq - target_seq) ** 2 * prefix_mask).sum() / (n_prefix * D)
        loss_reg = sigreg(state_seq[anchor_valid].unsqueeze(0)) if train else torch.zeros((), device=device)
        zero = torch.zeros((), device=device)
        return (loss_act, loss_dyn, loss_reg, zero, zero, zero, zero, zero, loss_h,
                int(n_valid_anchor.item()))

    # ---- training loop ----
    best_val = float("inf")

    for ep in range(args.epochs):
        t0 = time.time()
        alpha = dyn_alpha(ep)

        # ---- train ----
        model.train()
        tr_hsum = 0.0
        tr_act, tr_dyn, tr_reg, tr_cyc, tr_str, tr_idm, tr_roll, tr_acons, tr_count = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        for batch in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if args.dyn_prefix:
                    loss_act, loss_dyn, loss_reg, loss_cyc, loss_str, loss_idm, loss_roll, loss_acons, loss_h, nval = \
                        run_batch_prefix(*batch, train=True)
                else:
                    loss_act, loss_dyn, loss_reg, loss_cyc, loss_str, loss_idm, loss_roll, loss_acons, loss_h, nval = run_batch(
                        *batch, train=True, dyn_mix=alpha)
                # L_dyn = w_dyn*(L_tf + w_rollout*L_rollout): the rollout rides the same dyn weight
                loss = (args.w_act * loss_act + args.w_dyn * loss_dyn
                        + args.w_dyn * args.w_rollout * loss_roll
                        + args.w_reg * loss_reg + args.w_cyc * loss_cyc
                        + args.w_straight * loss_str + args.w_idm * loss_idm
                        + args.w_acons * loss_acons + loss_h)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_act += loss_act.item() * nval
            tr_hsum += float(loss_h) * nval
            tr_dyn += loss_dyn.item() * nval
            tr_reg += loss_reg.item() * nval
            tr_cyc += loss_cyc.item() * nval
            tr_str += loss_str.item() * nval
            tr_idm += loss_idm.item() * nval
            tr_roll += loss_roll.item() * nval
            tr_acons += loss_acons.item() * nval
            tr_count += nval
        sched.step()

        # ---- val ----
        model.eval()
        tr_h = tr_hsum / max(tr_count, 1)
        va_act, va_dyn, va_count = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if args.dyn_prefix:
                        loss_act, loss_dyn, _, _, _, _, _, _, _, nval = run_batch_prefix(*batch, train=False)
                    else:
                        loss_act, loss_dyn, _, _, _, _, _, _, _, nval = run_batch(*batch, train=False)
                va_act += loss_act.item() * nval
                va_dyn += loss_dyn.item() * nval
                va_count += nval

        tr_a = tr_act / max(tr_count, 1); tr_d = tr_dyn / max(tr_count, 1)
        tr_r = tr_reg / max(tr_count, 1); tr_c = tr_cyc / max(tr_count, 1)
        tr_s = tr_str / max(tr_count, 1); tr_i = tr_idm / max(tr_count, 1)
        tr_ro = tr_roll / max(tr_count, 1); tr_ac = tr_acons / max(tr_count, 1)
        va_a = va_act / max(va_count, 1); va_d = va_dyn / max(va_count, 1)

        lrs = sched.get_last_lr()
        dt = time.time() - t0

        cyc_str = f"  cyc={tr_c:.5f}" if args.w_cyc > 0 else ""
        dyn_str = f"  a={alpha:.2f}" if args.dyn_action_from_policy else ""
        str_str = f"  straight={tr_s:.4f}(cos~{1-tr_s:.3f})" if args.w_straight > 0 else ""
        idm_str = f"  idm={tr_i:.5f}" if args.w_idm > 0 else ""
        roll_str = f"  roll{args.rollout_k}={tr_ro:.5f}" if args.rollout_k > 1 else ""
        acons_str = f"  acons={tr_ac:.5f}" if args.w_acons > 0 else ""
        cs = model.gc_head.code_stats() if hasattr(model.gc_head, "code_stats") else {}
        h_str = (f"  h={tr_h:.5f}" if args.latent_h else "") + (
            f" ppl={cs['h_ppl']:.1f}/{args.h_codes} live={cs['h_live']}" if cs else "")
        print(f"[lewam-uni] ep {ep+1}/{args.epochs}  "
              f"act={tr_a:.5f}/{va_a:.5f}  dyn={tr_d:.5f}/{va_d:.5f}  reg={tr_r:.5f}{cyc_str}{dyn_str}{str_str}{idm_str}{roll_str}{acons_str}{h_str}  "
              f"lr_enc={lrs[0]:.2e}  {dt:.1f}s", flush=True)

        # ---- save checkpoints ----
        full_sd = model.state_dict()
        torch.save(full_sd, run_dir / "lewam_unified_latest.pt")
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
