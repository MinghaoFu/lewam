import argparse
import copy
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

torch.backends.cudnn.benchmark = True  # fixed 224x224 input, let cuDNN pick the conv algo
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from train_crossattn import load_cache, RawContextDataset
from train_lewam_unified import durable_sync, _IMG_MEAN, _IMG_STD
from lewam.models.jointflow import build_model
from lewam.models.twinflow import build_model as build_twinflow
from lewam.models.motflow import build_model as build_motflow
from lewam.models.module import SIGReg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="tool_hang.h5")
    ap.add_argument("--run_name", default="jointflow")
    ap.add_argument("--exp_tag", default="")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--ckpt_sync_dir", default="")
    ap.add_argument("--frames_cache", default="auto")
    ap.add_argument("--cache_mmap", action="store_true")
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--views", default="pixels",
                    help="comma list of the h5 image columns to train on (cameras); each needs its own raw "
                         "cache from make_preload_cache.py --pixels_key; the model gets num_views = len")
    ap.add_argument("--goal_views", default="",
                    help="goal-conditioned models: the cameras whose goal frames condition the policy, one goal "
                         "token each: a comma list of columns from --views, 'all', or empty = the first view")
    ap.add_argument("--encoder_checkpoint_chunks", type=int, default=0,
                    help="run the encoder in this many chunks with activation checkpointing (memory for "
                         "many cameras at a large batch; 0 = one pass)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="one uniform rate for encoder + flow transformer (UWM precedent)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    # per-module learning rates (LeWAM-unified's mechanism: encoder 1e-4 / heads 3e-4 / dynamics
    # 3e-4). None = --lr, so existing recipes are unchanged. Groups by parameter name: the
    # encoder ("encoder." / "policy.encoder."), the dynamics trunk ("state_flow.", twinflow) and
    # everything else (the policy trunk + readout, plus the IDM head).
    ap.add_argument("--encoder_lr", type=float, default=None)
    ap.add_argument("--policy_lr", type=float, default=None)
    ap.add_argument("--dynamics_lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fp32", action="store_true",
                    help="disable the bf16 autocast the reference trainers use (train+val "
                         "forward, backward in fp32) -- the jointflow campaign pre-2026-08-23 "
                         "ran fp32; see docs/RECIPES.md")
    # model
    ap.add_argument("--encoder_backbone", default="resnet18dp")
    ap.add_argument("--encoder_size", default="tiny")
    ap.add_argument("--z_dim", type=int, default=384)
    ap.add_argument("--proj_hidden", type=int, default=768,
                    help="encoder projector hidden width (64-d keypoints -> proj_hidden -> z_dim)")
    ap.add_argument("--d_model", type=int, default=384,
                    help="flow transformer width; = z_dim so state/memory tokens are never compressed")
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n_flow_steps", type=int, default=8)
    ap.add_argument("--num_actions_pred", type=int, default=5)
    ap.add_argument("--num_states_pred", type=int, default=1)
    ap.add_argument("--model", default="jointflow", choices=["jointflow", "twinflow", "motflow"],
                    help="jointflow = one flow over [action tokens; state token]; twinflow = the "
                         "same action flow with no state slot + a separate state flow conditioned "
                         "on the CLEAN action chunk (lewam.models.twinflow)")
    ap.add_argument("--mot_state_head", default="flow", choices=["flow", "mse"],
                    help="motflow: objective on the next-state tokens (flow = rectified flow on a "
                         "noisy token; mse = regression from a learned query, clean-action-conditioned)")
    ap.add_argument("--mot_action_head", default="flow", choices=["flow", "mse"],
                    help="motflow: objective on the action chunk (flow = rectified flow, sampled; mse = "
                         "one-pass regression from a zero input at tau 1, deterministic: the policy returns "
                         "the conditional MEAN action -- for random-policy data such as reacher)")
    ap.add_argument("--state_detach", action="store_true",
                    help="twinflow: stop-grad the encoder for the state branch (its history memory "
                         "and target), so the state flow cannot collapse or reshape the encoder")
    ap.add_argument("--state_depth", type=int, default=0,
                    help="twinflow: depth of the state trunk (0 = same as --depth)")
    ap.add_argument("--policy_history_len", type=int, default=2,
                    help="raw-consecutive obs frames the model uses as history")
    ap.add_argument("--steps_per_epoch", type=int, default=0,
                    help="cap train/val batches per epoch (0 = one pass over the decision points)")
    ap.add_argument("--sample_replacement", action="store_true",
                    help="sample training windows WITH replacement (num_samples = dataset size). Default "
                         "False = shuffle, each window once per epoch (the LeWM protocol / fixed behavior). "
                         "Set only to reproduce a pre-fix baseline's sampler in a controlled ablation.")
    ap.add_argument("--actions_attend_states", type=int, default=1,
                    help="1: action tokens can attend to jointly predicted state tokens; "
                         "0: mask action tokens from attending the (noisy) state tokens")
    ap.add_argument("--split_tau", action="store_true",
                    help="draw tau_action and tau_state independently instead of one shared tau")
    ap.add_argument("--tau_cond", default="per_modality", choices=["per_modality", "summed"],
                    help="per_modality: each modality modulated by its own flow-time; "
                         "summed: one blended cond for all tokens")
    # goal reaching
    ap.add_argument("--goal_conditioning", action="store_true",
                    help="goal + horizon condition the action readout (GCHeadMSE). With "
                         "--goal_conditioning alone this is GR: the raw cache sampled at stride-1 "
                         "starts (LeWM protocol); add --goal_terminal for the TC terminal-goal mode")
    ap.add_argument("--H_max", type=int, default=50,
                    help="horizon cap in obs-steps; h_norm = min(h, H_max)/H_max")
    ap.add_argument("--goal_cond", choices=["token", "head"], default="token",
                    help="motflow only: where goal + horizon enter -- 'token' (goal token in the "
                         "trunk, horizon AdaLN on the noisy action tokens) or 'head' (jointflow's "
                         "GCHeadMSE readout: the trunk never sees goal or horizon)")
    ap.add_argument("--goal_terminal", action="store_true",
                    help="TC goal convention: goal = terminal/success frame, h fixed at 0 (the raw "
                         "dataset supplies both). Selects JointFlowDataset over the GR sampled-goal "
                         "dataset, and is recorded in the config for the eval adapter. Requires "
                         "--goal_conditioning.")
    ap.add_argument("--p_drop_goal", type=float, default=0.0,
                    help="per-row goal dropout to the learned null goal (0 = every sample "
                         "keeps its goal)")
    ap.add_argument("--detach_goal_grad", action="store_true",
                    help="stop-gradient z_goal into the encoder: the goal frame is still "
                         "encoded and conditions the policy/dynamics, but the encoder cannot "
                         "be shaped by the goal pathway (tests whether the goal 'hacks' the "
                         "representation into shortcuts). Requires --goal_conditioning.")
    # anti-collapse
    ap.add_argument("--zstd_floor", type=float, default=0.0,
                    help="collapse guard: after epoch 5, abort (write run_dir/collapse_killed) when the val "
                         "per-dim std of the state latents drops below this (0 = off; reacher mse-noreg "
                         "collapsed to 1e-4 and the NaN-only guard let it run to the end)")
    ap.add_argument("--w_reg", type=float, default=0.0,
                    help="SIGReg anti-collapse weight on the encoder latent")
    ap.add_argument("--w_idm", type=float, default=0.0,
                    help="IDM aux weight (idm05 mechanism): MLP recovers the first action "
                         "block from the online (z_t, z_{t+fs}); shapes the encoder toward "
                         "action-aware latents. 0 = off.")
    ap.add_argument("--w_act", type=float, default=1.0,
                    help="policy (action-flow) loss weight")
    ap.add_argument("--w_dyn", type=float, default=1.0,
                    help="dynamics (state-flow) loss weight; 0 = policy-only ablation")
    ap.add_argument("--state_ema_target", action="store_true",
                    help="state flow targets from a momentum copy of the encoder, layernormed and "
                         "stop-gradded")
    ap.add_argument("--state_ema_base", type=float, default=0.998,
                    help="base momentum for the target encoder, linearly annealed to 1.0")
    ap.add_argument("--tau_alpha", type=float, default=1.0,
                    help="training tau ~ Beta(alpha,1) (U^(1/alpha)); >1 biases toward the clean "
                         "end tau=1; 1 = uniform")
    ap.add_argument("--tau_alpha_state", type=float, default=0.0,
                    help="state-branch tau alpha (needs --split_tau); 0 = same as --tau_alpha")
    ap.add_argument("--state_target_norm", action="store_true",
                    help="state flow in per-dim standardized latent coords (running EMA stats in "
                         "model buffers; ONLINE target keeps its gradient); sample/inpaint "
                         "de-normalize. Incompatible with --state_ema_target.")
    ap.add_argument("--state_residual", action="store_true",
                    help="state flow denoises delta = z[t+q*fs] - z[t] instead of z[t+q*fs]; "
                         "sample/inpaint add z_t back (incompatible with --state_ema_target)")
    ap.add_argument("--state_prior", default="gauss", choices=["gauss", "prev"],
                    help="MoT flow state head: x_0 = N(0,I) (gauss) or x_0 = z_t (prev): the flow "
                         "learns the displacement z[t+fs] - z[t] along a straight path and the ODE "
                         "starts at z_t (owner 2026-08-29, after Action-to-Action Flow Matching)")
    ap.add_argument("--state_prior_sigma", type=float, default=0.0,
                    help="with --state_prior prev: x_0 = z_t + sigma * rms(z_t) * eps (0 = deterministic prior)")
    ap.add_argument("--state_param", default="v", choices=["v", "x"],
                    help="MoT flow state head output: velocity (v) or JiT-style clean-state prediction (x): "
                         "the flow loss and the sampler go through v = (z_hat - x_tau) / (1 - tau)")
    ap.add_argument("--state_x_eps", type=float, default=0.05,
                    help="--state_param x: clamp of (1 - tau) in the reparameterization (bounds the 1/(1-tau)^2 weight)")
    ap.add_argument("--sep_policy_state", action="store_true",
                    help="MoT: the action branch attends only P(z) copies of the history (and a "
                         "projected goal); the policy's gradient reaches z through the projection "
                         "only, the dynamics owns z raw (owner design 2026-09-01)")
    ap.add_argument("--policy_proj_rank", type=int, default=0,
                    help="with --sep_policy_state: bottleneck the policy view to a rank-r "
                         "LoRA-style factorization P = U V (output stays z_dim); 0 = full linear")
    ap.add_argument("--sigreg_pertime", action=argparse.BooleanOptionalAction, default=True,
                    help="SIGReg per latent group (each history slot, each state target, goal as "
                         "context), losses averaged; --no-sigreg_pertime = the original pooled "
                         "cat(z_t, state_target) batch")
    ap.add_argument("--sigreg_proj_dim", type=int, default=-1,
                    help="SIGReg sees a learned z_dim -> N linear of each latent (identity-init "
                         "when square; colleague recipe). -1 = z_dim (default), 0 = no projection")
    ap.add_argument("--state_tau_logit", type=float, nargs=2, default=None, metavar=("MU", "SIGMA"),
                    help="MoT flow state head: draw tau as sigmoid(N(MU, SIGMA)) (JiT: -0.8 0.8) instead of uniform")
    ap.add_argument("--grad_probe_every", type=int, default=0,
                    help="every N steps, log encoder gradient geometry (P / D-input / D-target / "
                         "SIGReg norms, cosines, EMA cosines, Adam-preconditioned) to grad_probe.jsonl; 0 = off")
    ap.add_argument("--grad_probe_ema", type=int, default=64,
                    help="EMA window (in probe steps) for the gradient-vector averages")
    ap.add_argument("--pcgrad", default="off", choices=["off", "sym", "protect_p", "match_s"],
                    help="gradient surgery between the task losses (P=action, D=state, S=sigreg) "
                         "over the full parameter vector: sym = original PCGrad (every task "
                         "projected away from each conflicting other, random order); "
                         "protect_p = only D/S are projected away from P. Per-epoch conflict "
                         "rates and cosines are logged. Costs one backward per task.")
    return ap.parse_args()


class MultiViewFrames:
    """The frame arrays of several cameras behind one index: `frames[i]` is (views, 3, H, W) and
    `frames[a:b]` is (b - a, views, 3, H, W), so the datasets index it exactly like one camera's array."""

    def __init__(self, views):
        self.views = views
        self.shape = (views[0].shape[0], len(views), *views[0].shape[1:])
        self.dtype = views[0].dtype
        assert all(v.shape == views[0].shape for v in views), "every camera's cache has the same frames"

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return torch.stack([v[index] for v in self.views], dim=-4)


def load_cache_views(args):
    """The raw cache of every camera in args.views (the first one also carries the aux arrays);
    a camera other than `pixels` sits at the tag suffix .<column> make_preload_cache.py writes."""
    from train_crossattn import load_cache
    views = [v for v in args.views.split(",") if v]
    if views == ["pixels"]:
        return load_cache(args)
    cdir = args.frames_cache if args.frames_cache != "auto" else os.environ.get(
        "LEWAM_CACHE_DIR", "/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/preload_cache")
    stem = os.path.basename(args.dataset_name).replace(".h5", "")
    tags = [f"{stem}_fs{args.frameskip}_i{args.img_size}_raw" + ("" if v == "pixels" else f".{v}") for v in views]
    frames = [torch.from_numpy(np.load(f"{cdir}/{stem}/{tag}.frames.npy",
                                       mmap_mode="r" if args.cache_mmap else None)) for tag in tags]
    aux = np.load(f"{cdir}/{stem}/{tags[0]}.aux.npz")
    a_frame = torch.from_numpy(aux["A_flat"])
    assert a_frame.shape[0] == frames[0].shape[0], "raw cache A_flat must be frame-aligned"
    return (MultiViewFrames(frames), a_frame, torch.from_numpy(aux["t_gidx"]), torch.from_numpy(aux["ep_base"]),
            torch.from_numpy(aux["frames_to_terminal"]), (aux["act_mean"].tolist(), aux["act_std"].tolist()))


def read_cache_pixel_norm(args):
    """The stored pixel scale of this run's cache (make_preload_cache's meta). Returns
    (pixel_norm, mean, std); (None, None, None) if the meta predates the field (caller falls
    back to a per-frame range check)."""
    cdir = args.frames_cache if args.frames_cache != "auto" else os.environ.get(
        "LEWAM_CACHE_DIR", "/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/preload_cache")
    stem = os.path.basename(args.dataset_name).replace(".h5", "")
    tag = f"{stem}_fs{args.frameskip}_i{args.img_size}_raw"
    try:
        m = json.load(open(f"{cdir}/{stem}/{tag}.meta.json"))
    except Exception:
        return None, None, None
    return m.get("pixel_norm"), m.get("pixel_norm_mean"), m.get("pixel_norm_std")


class JointFlowGRDataset(torch.utils.data.Dataset):
    """Goal-reaching on the RAW cache with LeWM start sampling: every raw timestep is a decision
    point, so starts advance by 1 raw step (the fs-strided cache advanced them by one fs-anchor =
    `frameskip` raw steps -- the bug this fixes). From a raw start p the window keeps the fs-spaced
    GR architecture unchanged: history = frames at p, p-fs, ... (left-padded with the oldest real
    frame at the episode start); actions = the next `num_actions` raw actions; state targets at
    p+q*fs; goal at p+h*fs with h ~ U[1, H_max] fs-steps clamped to the tail, h_norm =
    min(h, H_max)/H_max. (The removed fs-strided cache advanced starts by a whole fs-anchor =
    frameskip raw steps; this samples every raw start -- the fix.)"""

    def __init__(self, frames, a_frame, t_gidx, ep_base, frames_to_terminal, indices,
                 history_len, num_actions, num_states, frameskip, h_max):
        self.frames = frames
        self.a_frame = a_frame
        self.t_gidx = t_gidx
        self.ep_base = ep_base
        self.frames_to_terminal = frames_to_terminal
        self.indices = indices
        self.history_len = history_len
        self.num_actions = num_actions
        self.num_states = num_states
        self.frameskip = frameskip
        self.h_max = h_max

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        p = int(self.t_gidx[idx])
        ep0 = int(self.ep_base[idx])
        ttl = int(self.frames_to_terminal[idx])
        hl, fs = self.history_len, self.frameskip

        # history: fs-spaced frames ending at p (p, p-fs, ...), oldest real frame padding the start
        offs = [p - k * fs for k in range(hl - 1, -1, -1)]   # oldest..newest; newest = p
        k = sum(o >= ep0 for o in offs)                      # how many land inside this episode (>=1)
        recent = torch.stack([self.frames[o] for o in offs[hl - k:]])
        history = torch.empty((hl, *recent.shape[1:]), dtype=recent.dtype)
        history[hl - k:] = recent
        if k < hl:
            history[:hl - k] = recent[0]
        history_pad = torch.ones(hl, dtype=torch.bool)
        history_pad[hl - k:] = False

        # actions: the next num_actions raw actions (frame-aligned, already z-scored in the cache)
        avail = min(self.num_actions, ttl)
        target = torch.zeros((self.num_actions, self.a_frame.shape[1]), dtype=torch.float32)
        target[:avail] = self.a_frame[p:p + avail].float()
        target_valid = torch.zeros(self.num_actions, dtype=torch.float32)
        target_valid[:avail] = 1.0

        # state targets: fs-spaced boundaries p + q*fs, clamped to the terminal frame
        if self.num_states:
            state_frames = torch.stack([self.frames[p + min(q * fs, ttl)]
                                        for q in range(1, self.num_states + 1)])
            state_valid = torch.tensor([1.0 if q * fs <= ttl else 0.0
                                        for q in range(1, self.num_states + 1)])
        else:
            state_frames = torch.zeros((0, *self.frames.shape[1:]), dtype=self.frames.dtype)
            state_valid = torch.zeros(0)

        # goal: h fs-steps ahead, h ~ U[1, H_max] clamped to the fs-steps remaining (>=1 by the filter)
        mh = max(1, ttl // fs)
        h = min(int(torch.randint(1, self.h_max + 1, (1,)).item()), mh)
        goal_frame = self.frames[p + min(h * fs, ttl)]
        h_norm = torch.tensor(min(h, self.h_max) / self.h_max, dtype=torch.float32)
        return history, history_pad, target, target_valid, state_frames, state_valid, \
            goal_frame, h_norm


class JointFlowDataset(RawContextDataset):
    """RawContextDataset plus the block-boundary state-target frames: for q in 1..num_states the
    frame at p + q*frameskip, clamped to the terminal, valid while the boundary is inside the
    episode. The inherited goal/next_frame outputs are dropped."""

    def __init__(self, frames, a_frame, t_gidx, ep_base, frames_to_terminal, indices,
                 history_len, n_tokens, frameskip, num_states):
        super().__init__(frames, a_frame, t_gidx, ep_base, frames_to_terminal, indices,
                         history_len, n_tokens, frameskip)
        self.num_states = num_states

    def __getitem__(self, i):
        history_frames, history_pad, target, target_valid, goal_frame, _next_frame, _dyn_valid = \
            super().__getitem__(i)
        idx = int(self.indices[i])
        p = int(self.t_gidx[idx])
        ttl = int(self.frames_to_terminal[idx])
        fs, n_states = self.frameskip, self.num_states
        if n_states:
            state_frames = torch.stack([self.frames[p + min(q * fs, ttl)]
                                        for q in range(1, n_states + 1)])
            state_valid = torch.tensor([1.0 if q * fs <= ttl else 0.0
                                        for q in range(1, n_states + 1)])
        else:
            state_frames = torch.zeros((0, *self.frames.shape[1:]), dtype=self.frames.dtype)
            state_valid = torch.zeros(0)
        # goal_frame is the parent's terminal frame (== the success frame on a success-patched
        # aux); run_batch consumes it only under --goal_conditioning. Horizon carries no
        # signal on this path, so h_norm is the constant 0.
        return history_frames, history_pad, target, target_valid, state_frames, state_valid, \
            goal_frame, torch.tensor(0.0, dtype=torch.float32)


def build_param_groups(model, idm_head, args):
    """AdamW param groups: encoder / dynamics (twinflow state trunk) / policy (+ IDM head)."""
    enc_lr = args.lr if args.encoder_lr is None else args.encoder_lr
    pol_lr = args.lr if args.policy_lr is None else args.policy_lr
    dyn_lr = args.lr if args.dynamics_lr is None else args.dynamics_lr
    enc, dyn, pol = [], [], []
    for name, prm in model.named_parameters():
        if hasattr(model, "param_group_of"):
            g = model.param_group_of(name)
        elif name.startswith("encoder.") or name.startswith("policy.encoder."):
            g = "encoder"
        elif name.startswith("state_flow."):
            g = "dynamics"
        else:
            g = "policy"
        {"encoder": enc, "dynamics": dyn, "policy": pol}[g].append(prm)
    if idm_head is not None:
        pol += list(idm_head.parameters())
    groups = [dict(params=enc, lr=enc_lr, name="encoder", n=sum(p.numel() for p in enc)),
              dict(params=pol, lr=pol_lr, name="policy", n=sum(p.numel() for p in pol))]
    if dyn:
        groups.append(dict(params=dyn, lr=dyn_lr, name="dynamics", n=sum(p.numel() for p in dyn)))
    assert sum(g["n"] for g in groups) == sum(p.numel() for p in model.parameters()) + \
        (sum(p.numel() for p in idm_head.parameters()) if idm_head is not None else 0)
    return groups


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    need = args.num_states_pred * args.frameskip
    if args.num_actions_pred < need:
        print(f"[jointflow] WARN num_actions_pred {args.num_actions_pred} < num_states_pred*frameskip "
              f"{need}; padding action horizon to {need}", flush=True)
        args.num_actions_pred = need

    tag = args.exp_tag.strip("_")
    run_dir = Path(args.run_dir) if args.run_dir else Path(f"runs/{args.run_name}_{tag}" if tag else args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    assert not (args.detach_goal_grad and not args.goal_conditioning), \
        "--detach_goal_grad requires --goal_conditioning"
    if args.goal_terminal:
        assert args.goal_conditioning, "--goal_terminal is the TC goal mode: needs --goal_conditioning"
    # GR (--goal_conditioning without --goal_terminal) reads the raw cache and samples starts at raw
    # stride 1 (LeWM protocol, JointFlowGRDataset); TC uses the terminal goal (JointFlowDataset).
    views = [v for v in args.views.split(",") if v]
    # goal views: which cameras' goal frames condition the policy (one goal token each); default the first
    goal_views = (views if args.goal_views == "all"
                  else [v for v in args.goal_views.split(",") if v] or views[:1])
    assert all(v in views for v in goal_views), f"--goal_views {goal_views} must be among --views {views}"
    goal_view_index = [views.index(v) for v in goal_views]
    frames, a_frame, t_gidx, ep_base, frames_to_terminal, action_stats = load_cache_views(args)
    raw_adim = a_frame.shape[1]
    n_starts = t_gidx.shape[0]
    _cache_norm, _cache_nmean, _cache_nstd = read_cache_pixel_norm(args)
    print(f"[jointflow] cache pixel_norm={_cache_norm} -> target ImageNet", flush=True)
    print(f"[jointflow] frames={tuple(frames.shape)} starts={n_starts} raw_dim={raw_adim} "
          f"layout=(a{args.num_actions_pred},s{args.num_states_pred},fs{args.frameskip}) "
          f"actions_attend_states={bool(args.actions_attend_states)} split_tau={args.split_tau} "
          f"goal_cond={args.goal_conditioning} goal_terminal={args.goal_terminal}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_starts, generator=gen)
    perm = perm[frames_to_terminal[perm] >= args.frameskip]
    n_val = int(round((1 - args.train_split) * perm.numel()))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers:
        loader_args.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)

    def make_loader(idx):
        if args.goal_conditioning and not args.goal_terminal:   # GR: raw cache, stride-1 starts (LeWM)
            ds = JointFlowGRDataset(frames, a_frame, t_gidx, ep_base, frames_to_terminal, idx,
                                    args.policy_history_len, args.num_actions_pred,
                                    args.num_states_pred, args.frameskip, args.H_max)
        else:                                                    # TC (terminal goal) / plain
            ds = JointFlowDataset(frames, a_frame, t_gidx, ep_base, frames_to_terminal, idx,
                                  args.policy_history_len, args.num_actions_pred, args.frameskip,
                                  args.num_states_pred)
        n = max(args.batch_size, int(idx.numel()))
        if args.steps_per_epoch:
            n = min(n, args.steps_per_epoch * args.batch_size)
        # Default: shuffle window indices, each once per epoch (replacement=False) -- the LeWM protocol
        # and the DP baseline (train_dp.py). --sample_replacement opts back into the old with-replacement
        # sampler (~63% coverage/epoch) only to reproduce a pre-fix baseline in a controlled ablation.
        return DataLoader(ds, sampler=RandomSampler(ds, replacement=args.sample_replacement, num_samples=n), **loader_args)
    train_loader, val_loader = make_loader(train_idx), make_loader(val_idx)

    cfg = dict(fs=args.frameskip, action_raw_dim=raw_adim, img_size=args.img_size,
               encoder_size=args.encoder_size, encoder_backbone=args.encoder_backbone,
               encoder_ckpt=None, z_dim=args.z_dim, proj_hidden=args.proj_hidden,
               d_model=args.d_model, n_heads=args.n_heads,
               depth=args.depth, dropout=args.dropout, n_flow_steps=args.n_flow_steps,
               num_actions_pred=args.num_actions_pred, num_states_pred=args.num_states_pred,
               policy_history_len=args.policy_history_len,
               actions_attend_states=bool(args.actions_attend_states), split_tau=args.split_tau,
               goal_conditioning=args.goal_conditioning, goal_cond=args.goal_cond, tau_cond=args.tau_cond,
               state_residual=bool(args.state_residual), tau_alpha=float(args.tau_alpha),
               tau_alpha_state=float(args.tau_alpha_state),
               state_target_norm=bool(args.state_target_norm), model=args.model,
               state_detach=bool(args.state_detach), state_depth=int(args.state_depth),
               state_ctx_actions=int(args.frameskip * args.num_states_pred),
               action_head=args.mot_action_head,
               state_head=args.mot_state_head, state_prior=args.state_prior,
               state_prior_sigma=float(args.state_prior_sigma), state_param=args.state_param,
               state_x_eps=float(args.state_x_eps),
               state_tau_logit=(tuple(args.state_tau_logit) if args.state_tau_logit else None),
               sep_policy_state=bool(args.sep_policy_state),
               policy_proj_rank=int(args.policy_proj_rank),
               num_views=len(views), views=views, goal_views=goal_views, goal_view_index=goal_view_index,
               sigreg_pertime=bool(args.sigreg_pertime),
               # no reg -> no projection module (keeps noreg checkpoints free of dead params)
               sigreg_proj_dim=(0 if args.w_reg == 0
                                else (args.z_dim if args.sigreg_proj_dim < 0
                                      else args.sigreg_proj_dim)))
    builder = {"jointflow": build_model, "twinflow": build_twinflow, "motflow": build_motflow}[args.model]
    model = builder(cfg).to(device)
    action_mean, action_std = action_stats
    dumped = {**cfg, **vars(args), "action_mean": action_mean, "action_std": action_std,
              "input_norm": "imagenet", "input_norm_mean": _IMG_MEAN.view(-1).tolist(),
              "input_norm_std": _IMG_STD.view(-1).tolist()}
    # vars(args) carries the RAW -1 sentinel; the loader must see the resolved width or it
    # rebuilds without the projection module and the state-dict assert fires
    dumped["sigreg_proj_dim"] = cfg["sigreg_proj_dim"]
    dumped["views"] = views          # the lists, not the comma strings vars(args) carries
    dumped["goal_views"], dumped["goal_view_index"] = goal_views, goal_view_index
    (run_dir / "jointflow_config.json").write_text(json.dumps(dumped, indent=1))
    n_params = sum(p.numel() for p in model.parameters())
    assert not (args.tau_alpha_state and not args.split_tau), \
        "--tau_alpha_state only takes effect with --split_tau (tied tau cannot bias one branch)"
    assert not (args.state_target_norm and args.state_ema_target), \
        "--state_target_norm is the online-target normalization; the ema path has its own"
    assert args.state_prior == "gauss" or (args.model == "motflow" and args.mot_state_head == "flow"), \
        "--state_prior prev is implemented for the MoT flow state head"
    assert args.state_param == "v" or (args.model == "motflow" and args.mot_state_head == "flow"), \
        "--state_param x is implemented for the MoT flow state head"
    assert args.state_tau_logit is None or (args.model == "motflow" and args.mot_state_head == "flow"), \
        "--state_tau_logit is a MoT flow-state-head option"
    assert not (args.state_prior == "prev" and args.state_residual), \
        "--state_prior prev and --state_residual both anchor the flow at z_t; pick one"
    assert not (args.state_residual and args.state_ema_target), \
        "--state_residual mixes raw z_t into the target; incompatible with the layernormed ema path"
    if args.model == "twinflow":
        assert not args.split_tau, "twinflow's branches have independent taus by construction; drop --split_tau"
        assert not (args.state_target_norm and not args.state_detach), \
            "twinflow --state_target_norm needs --state_detach (the 1/sd runaway needs a gradient path into the encoder)"
    elif args.model == "motflow":
        assert not (args.split_tau or args.state_target_norm or args.state_ema_target or args.state_detach
                    or args.state_depth), "motflow: split_tau/state_target_norm/state_ema_target/state_detach/state_depth do not apply"
    else:
        assert not (args.state_detach or args.state_depth), "--state_detach/--state_depth are twinflow flags"
    assert not args.sep_policy_state or args.model == "motflow", \
        "--sep_policy_state is a MoT option"
    assert args.w_reg == 0 or args.sigreg_proj_dim == 0 or args.model == "motflow", \
        "projected SIGReg stores its projection on the MoT model"
    print(f"[jointflow] model={args.model}{' detach' if args.state_detach else ''} "
          f"params={n_params/1e6:.2f}M  w_reg={args.w_reg} "
          f"state_target={'ema' if args.state_ema_target else 'online'}"
          f"{' residual' if args.state_residual else ''}", flush=True)

    idm_head = None
    if args.w_idm > 0:
        # IDM aux (idm05 mechanism): recover the FIRST action block from (z_t, z_{t+fs})
        # online latents, so the gradient shapes the encoder toward action-aware features.
        assert args.num_states_pred >= 1, "--w_idm needs a state slot (num_states_pred >= 1)"
        idm_head = nn.Sequential(nn.Linear(2 * args.z_dim, 512), nn.SiLU(),
                                 nn.Linear(512, args.frameskip * raw_adim)).to(device)
    groups = build_param_groups(model, idm_head, args)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    print("[jointflow] lr groups: " + ", ".join(f"{g['name']} lr={g['lr']:.2e} n={g['n']}" for g in groups),
          flush=True)

    def lr_scale(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        p = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    tgt_encoder = None
    if args.state_ema_target:
        tgt_encoder = copy.deepcopy(model.encoder).to(device)
        for p in tgt_encoder.parameters():
            p.requires_grad_(False)
        tgt_encoder.eval()

    start_epoch, best_val = 0, float("inf")
    full_path = run_dir / "jointflow_full.pt"
    if args.resume and full_path.exists():
        state = torch.load(full_path, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"]); start_epoch = state["epoch"] + 1
        best_val = state["best_val"]
        if idm_head is not None and "idm_head" in state:
            idm_head.load_state_dict(state["idm_head"])
        print(f"[jointflow] resumed at epoch {start_epoch} best_val={best_val:.5f}", flush=True)

    mean = _IMG_MEAN.to(device); std = _IMG_STD.to(device)   # this trainer's target input norm (ImageNet)
    _cache_mean = torch.tensor(_cache_nmean).view(1, 3, 1, 1).to(device).float() if _cache_nmean else None
    _cache_std = torch.tensor(_cache_nstd).view(1, 3, 1, 1).to(device).float() if _cache_nstd else None
    sigreg = SIGReg().to(device)

    def sigreg_loss(z_history, state_target, z_goal):
        """The applied anti-collapse loss; probe_measure reuses it so the counterfactual
        always matches the applied form."""
        zd = args.z_dim
        proj = model.sigreg_proj if getattr(model, "sigreg_proj_dim", 0) > 0 else (lambda x: x)
        if not args.sigreg_pertime:
            zs = torch.cat([z_history[:, -1], state_target.reshape(-1, zd)])
            return sigreg(proj(zs).unsqueeze(0))
        groups = [z_history[:, k] for k in range(z_history.shape[1])]
        groups += [state_target[:, q] for q in range(state_target.shape[1])]
        if z_goal is not None:            # goal grouped in as policy context (owner 2026-09-01); one group per goal view
            groups += [z_goal] if z_goal.dim() == 2 else [z_goal[:, g] for g in range(z_goal.shape[1])]
        return torch.stack([sigreg(proj(g).unsqueeze(0)) for g in groups]).mean()
    probe = None
    if args.grad_probe_every > 0:
        from lewam.models.grad_probe import GradProbe
        probe = GradProbe(model.encoder, run_dir / "grad_probe.jsonl",
                          every=args.grad_probe_every, ema_steps=args.grad_probe_ema,
                          run_info=dict(run=Path(args.ckpt_sync_dir or str(run_dir)).name,
                                        model=args.model, state_head=args.mot_state_head,
                                        w_reg=args.w_reg, dataset=args.dataset_name,
                                        seed=args.seed, lr=args.lr, epochs=args.epochs))
        probe.optimizer = opt
    n_states = args.num_states_pred
    # bf16 autocast around the forward (train AND val), backward in fp32 -- exactly the
    # reference trainers' setup (train_lewam_gc.py:372, train_lewam_unified.py:1106)
    use_amp = (device == "cuda") and not args.fp32
    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_amp \
        else nullcontext
    print(f"[jointflow] runtime: amp={'bf16' if use_amp else 'fp32'} "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}", flush=True)

    def run_batch(batch, train=True, return_probe=False):
        # both datasets return the same 8-tuple; the goal/h are consumed only under
        # --goal_conditioning (GR: sampled offset goal + countdown h; TC: terminal goal + h=0)
        (history_frames, history_pad, action_target, action_valid, state_frames,
         state_valid, goal_frames, h_norm) = batch
        if args.goal_conditioning:
            goal_frames = goal_frames.to(device, non_blocking=True)
            h_norm = h_norm.to(device, non_blocking=True)
        else:
            goal_frames = h_norm = None
        history_frames = history_frames.to(device, non_blocking=True)
        history_pad = history_pad.to(device, non_blocking=True)
        action_target = action_target.to(device, non_blocking=True).float()
        action_valid = action_valid.to(device, non_blocking=True)
        state_frames = state_frames.to(device, non_blocking=True)
        state_valid = state_valid.to(device, non_blocking=True)
        B, n_history = history_frames.shape[0], history_frames.shape[1]
        # Recover cache pixels to [0,1] per their recorded pixel_norm (from meta), then apply this
        # trainer's target scale (mean/std). Explicit + metadata-driven -- no dtype/range guessing.
        def norm(x):
            if _cache_norm == "none":
                unit = x / 255.0
            elif _cache_norm == "unit":
                unit = x
            elif _cache_mean is not None:                     # imagenet / custom: undo the cache's own mean/std
                unit = x * _cache_std + _cache_mean
            else:                                             # meta predates pixel_norm: safe per-frame range fallback
                raw = x.flatten(1).amax(1).view(-1, 1, 1, 1) > 4.0
                unit = torch.where(raw, x / 255.0, x * std + mean)
            return (unit - mean) / std
        # multi-view samples carry a camera axis after the frame axis: (B, frames, views, 3, H, W); every
        # frame of every camera goes through the encoder once, and the latents are laid out view-major
        # (camera 0's frames, then camera 1's, ...) as the model expects
        n_views = history_frames.shape[2] if history_frames.dim() == 6 else 1
        image_shape = history_frames.shape[-3:]

        pix = [history_frames.reshape(B * n_history * n_views, *image_shape),
               state_frames.reshape(B * n_states * n_views, *image_shape)]
        n_goal = len(goal_view_index)
        if args.goal_conditioning:
            # only the goal views' frames are encoded: (B, views, 3, H, W) from a multi-camera sample,
            # (B, 3, H, W) from a single camera
            goal_sel = goal_frames[:, goal_view_index] if goal_frames.dim() == 5 else goal_frames[:, None]
            pix.append(goal_sel.reshape(B * n_goal, *image_shape))
        pixels = norm(torch.cat(pix).float())
        if args.encoder_checkpoint_chunks > 1 and train:
            from torch.utils.checkpoint import checkpoint
            z = torch.cat([checkpoint(model.encode, chunk, use_reentrant=False)
                           for chunk in pixels.chunk(args.encoder_checkpoint_chunks)])
        else:
            z = model.encode(pixels)

        def view_major(latents, n_frames):
            """(B * n_frames * views, z) encoder output -> (B, views * n_frames, z), camera 0 first."""
            return latents.reshape(B, n_frames, n_views, args.z_dim).transpose(1, 2).reshape(B, n_views * n_frames, args.z_dim)

        z_history = view_major(z[:B * n_history * n_views], n_history)
        z_state_online = view_major(z[B * n_history * n_views:B * (n_history + n_states) * n_views], n_states)
        if n_views > 1:
            state_valid = state_valid.repeat(1, n_views)
        z_goal = goal_keep = None
        if args.goal_conditioning:
            z_goal = z[B * (n_history + n_states) * n_views:].reshape(B, n_goal, args.z_dim)
            if n_goal == 1:
                z_goal = z_goal[:, 0]                    # one goal view: the (B, z) latent the model always took
            if args.detach_goal_grad:
                # goal still conditions the policy/dynamics, but its gradient cannot reshape
                # the encoder (also removes the goal group from SIGReg's encoder push, since
                # a detached group contributes no encoder gradient)
                z_goal = z_goal.detach()
            if train and args.p_drop_goal > 0:
                goal_keep = torch.rand(B, device=device) >= args.p_drop_goal
        if tgt_encoder is not None:
            with torch.no_grad():
                z_target_raw = tgt_encoder(
                    norm(state_frames.reshape(B * n_states * n_views, *image_shape).float()))
            state_target = view_major(F.layer_norm(z_target_raw, (args.z_dim,)), n_states)
        else:
            state_target = z_state_online          # the encoder learns from being the target
        if args.state_residual:
            # owner design 2026-08-26: the flow denoises the CHANGE delta = z[t+q*fs] - z[t];
            # sample/inpaint add z_t back, so imagination stays anchored to the current state.
            # Encoder collapse now forces the flow toward a point mass at delta=0 -- visible
            # directly as the sampled ||delta|| (legible, unlike z-space collapse).
            state_target = state_target - (model._last_frames(z_history) if n_views > 1 else z_history[:, -1:])

        loss_action, loss_state = model.loss(z_history, history_pad, action_target, action_valid,
                                             state_target, state_valid,
                                             z_goal=z_goal, h_norm=h_norm, goal_keep=goal_keep)
        loss = args.w_act * loss_action + args.w_dyn * loss_state
        parts = {"P": loss_action, "D": loss_state}      # per-task losses for --pcgrad
        loss_terms = {"act": loss_action.item(), "state": loss_state.item()}
        if idm_head is not None:                   # camera 0's newest frame and first predicted state
            idm_in = torch.cat([z_history[:, n_history - 1], z_state_online[:, 0]], dim=-1)
            loss_idm = F.mse_loss(idm_head(idm_in),
                                  action_target[:, :args.frameskip].reshape(B, -1))
            loss = loss + args.w_idm * loss_idm
            parts["I"] = args.w_idm * loss_idm
            loss_terms["idm"] = loss_idm.item()
        if n_states:
            # collapse telemetry: per-dim std of the online state latents (collapse -> ~0)
            loss_terms["zstd"] = z_state_online.reshape(-1, args.z_dim).std(0).mean().item()
        if args.w_reg > 0:
            loss_reg = sigreg_loss(z_history, state_target, z_goal)
            loss = loss + args.w_reg * loss_reg
            parts["S"] = args.w_reg * loss_reg
            loss_terms["reg"] = loss_reg.item()
        extra = []
        if return_probe:
            extra.append(dict(z_history=z_history, history_pad=history_pad,
                              action_target=action_target, action_valid=action_valid,
                              state_target=state_target, state_valid=state_valid,
                              z_goal=z_goal, h_norm=h_norm, goal_keep=goal_keep, B=B))
        if args.pcgrad != "off":
            extra.append(parts)
        return (loss, loss_terms, B, *extra)

    def probe_measure(pc, loss_terms, gstep, epoch):
        """Encoder gradient geometry on the current batch (owner design 2026-08-30). Runs in
        fp32 outside the amp context; the forked, re-seeded RNG makes the two model.loss calls
        draw IDENTICAL tau/noise, so g_D(target-detached) is the exact input-path component, and
        SIGReg's projections are drawn once per measurement."""
        seed = 10_000_019 + gstep
        devs = [torch.device(device)] if str(device).startswith("cuda") else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(seed)
            if devs:
                torch.cuda.manual_seed_all(seed)
            la1, ld1 = model.loss(pc["z_history"], pc["history_pad"], pc["action_target"],
                                  pc["action_valid"], pc["state_target"], pc["state_valid"],
                                  z_goal=pc["z_goal"], h_norm=pc["h_norm"], goal_keep=pc["goal_keep"])
            torch.manual_seed(seed)
            if devs:
                torch.cuda.manual_seed_all(seed)
            la2, ld2 = model.loss(pc["z_history"], pc["history_pad"], pc["action_target"],
                                  pc["action_valid"], pc["state_target"].detach(), pc["state_valid"],
                                  z_goal=pc["z_goal"], h_norm=pc["h_norm"], goal_keep=pc["goal_keep"])
            loss_sig = sigreg_loss(pc["z_history"], pc["state_target"], pc["z_goal"])
        probe.measure(la1, ld1, ld2, loss_sig, gstep,
                      scalars=dict({k: v for k, v in loss_terms.items()}, lam_S=args.w_reg,
                                   lr=sched.get_last_lr()[0], epoch=epoch))

    def accumulate(store, loss_terms, n):
        for k, v in loss_terms.items():
            store[k] = store.get(k, 0.0) + v * n

    def fmt(store, n):
        return " ".join(f"{k}={store[k]/max(n,1):.5f}"
                        for k in ("act", "state", "reg", "zstd") if k in store)

    pcg = None
    if args.pcgrad != "off":
        from lewam.models.pcgrad import PCGrad
        pcg = PCGrad(model.parameters(), mode=args.pcgrad,
                     match_params=list(model.encoder.parameters()) if args.pcgrad == "match_s" else None)
        print(f"[jointflow] PCGrad mode={args.pcgrad} over {len(pcg.params)} param tensors", flush=True)
    gstep = start_epoch * max(1, len(train_loader))
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        mom_now = args.state_ema_base + (1.0 - args.state_ema_base) * (epoch / max(1, args.epochs))
        model.train()
        train_stats, train_n = {}, 0.0
        for batch in train_loader:
            do_probe = probe is not None and gstep % args.grad_probe_every == 0
            with amp_ctx():
                out = run_batch(batch, return_probe=do_probe)
            loss, loss_terms, n = out[:3]
            parts = out[-1] if args.pcgrad != "off" else None
            if do_probe:
                probe_measure(out[3], loss_terms, gstep, epoch)
            gstep += 1
            opt.zero_grad(set_to_none=True)
            if pcg is not None:
                pcg.backward(parts)      # per-task backwards + surgery -> .grad
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if tgt_encoder is not None:
                with torch.no_grad():
                    for p_target, p_online in zip(tgt_encoder.parameters(),
                                                  model.encoder.parameters()):
                        p_target.mul_(mom_now).add_(p_online.detach(), alpha=1.0 - mom_now)
                    for b_target, b_online in zip(tgt_encoder.buffers(), model.encoder.buffers()):
                        b_target.copy_(b_online)
            accumulate(train_stats, loss_terms, n); train_n += n
        sched.step()

        model.eval()
        val_stats, val_n = {}, 0.0
        with torch.no_grad():
            for batch in val_loader:
                with amp_ctx():
                    out = run_batch(batch, train=False)
                loss, loss_terms, n = out[:3]     # --pcgrad appends the per-loss parts (4-tuple)
                accumulate(val_stats, loss_terms, n); val_n += n
        val_act = val_stats.get("act", 0.0) / max(val_n, 1)   # best-checkpoint metric = val action loss
        pcg_msg = ""
        if pcg is not None:
            st = pcg.stats()
            pcg_msg = "  pcgrad[" + " ".join(f"{k}={v:.3f}" for k, v in sorted(st.items())) + "]"
            pcg.reset_stats()
        print(f"[jointflow] ep {epoch+1}/{args.epochs}  train[{fmt(train_stats,train_n)}]  "
              f"val[{fmt(val_stats,val_n)}]  lr={sched.get_last_lr()[0]:.2e}  "
              f"{time.time()-t0:.1f}s{pcg_msg}", flush=True)

        val_zstd = val_stats.get("zstd", float("inf")) / max(val_n, 1)
        if args.zstd_floor > 0 and epoch + 1 >= 5 and val_zstd < args.zstd_floor:
            (run_dir / "collapse_killed").write_text(f"epoch {epoch+1} val zstd {val_zstd:.6f} < floor {args.zstd_floor}\n")
            print(f"[jointflow] COLLAPSE_KILL zstd={val_zstd:.6f} < {args.zstd_floor} at ep {epoch+1}", flush=True)
            break

        torch.save(model.state_dict(), run_dir / "jointflow_latest.pt")
        full_state = dict(model=model.state_dict(), optimizer=opt.state_dict(),
                          scheduler=sched.state_dict(), epoch=epoch, best_val=best_val)
        if idm_head is not None:
            full_state["idm_head"] = idm_head.state_dict()
        torch.save(full_state, full_path)
        files = [run_dir / "jointflow_config.json", run_dir / "jointflow_latest.pt", full_path]
        if probe is not None and (run_dir / "grad_probe.jsonl").exists():
            files.append(run_dir / "grad_probe.jsonl")
        if val_act < best_val:
            best_val = val_act
            torch.save(model.state_dict(), run_dir / "jointflow_best.pt")
            files.append(run_dir / "jointflow_best.pt")
        if args.ckpt_sync_dir:
            durable_sync(files, args.ckpt_sync_dir)

    print(f"[jointflow] DONE best_val={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
