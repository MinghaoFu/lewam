# LeWAM-Seq trainer: one causal Transformer over interleaved [z_t, e_t] state/action
# tokens (lewam.models.lewam_seq.LeWAMSeq). State-tokens read out actions (IDM/BC),
# action-tokens read out next latents (FDM). Supports CEM (get_cost) and policy
# rollout (get_action) at eval time.
#
# Losses (all ablatable via weights):
#   L = w_act * action_loss(a_pred, a)                 # BC over all window positions
#     + w_dyn * MSE(z_pred[:, :-1], z[:, 1:])          # FDM, in-window targets (no stop-grad)
#     + w_reg * SIGReg(z)                              # anti-collapse (needed: dyn loss collapses z)
#     + w_rollout * rollout_loss                       # K-step autoregressive roll vs GT latents
#
# Design choices (see the design discussion):
#   - full num_frames windows so every positional-embedding slot is trained (no PE
#     extrapolation at inference; matches the sliding rollout window).
#   - PER-POSITION horizon: each frame conditions on its own remaining distance to the goal.
#   - goal-dropout p: 0 = always goal-conditioned, 1 = pure BC/FDM, in between = mixed.
#     Dropped samples get a zeroed goal latent + zero horizon (goal-agnostic via zero-fill).
#   - action head is MSE for now (baseline parity); --action_head reserved for gmm/diffusion.
#   - rollout loss is end-to-end (SIGReg-only, no stop-grad); --rollout_stopgrad parked for later.
#   - --stream reads frames from the h5 on demand (no preload -> no OOM on capped nodes).
#
#   python scripts/train_lewam_seq.py --dataset_name ogbench/cube_single_expert.h5 \
#       --run_name cube_lewam_seq --epochs 50 --num_frames 8 --stream
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True

import stable_worldmodel as swm

from lewam.utils import get_img_preprocessor, get_column_normalizer
from lewam.models.module import SIGReg
from lewam.models.lewam_seq import LeWAMSeq


# --------------------------------------------------------------------------- #
# Collapse metrics on latents z [N, D] (does dyn/action prediction avoid collapse?)  #
#   z_std     mean per-dim std           -> 0 on full collapse                   #
#   eff_rank  participation ratio of cov -> 1 on rank-1 collapse (D if full)     #
#   cos       mean off-diagonal cosine   -> 1 on directional collapse            #
#   sigreg    SIGReg statistic           -> comparable across w_reg settings     #
# --------------------------------------------------------------------------- #
def collapse_metrics(Z, sigreg=None, device="cpu"):
    Z = Z.float()
    N, D = Z.shape
    z_std = Z.std(dim=0).mean().item()
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = (Zc.t() @ Zc) / max(N - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    denom = (eig * eig).sum().item()
    eff_rank = (eig.sum().item() ** 2 / denom) if denom > 0 else 0.0
    Zn = Z / Z.norm(dim=1, keepdim=True).clamp(min=1e-6)
    G = Zn @ Zn.t()
    cos_off = ((G.sum() - G.diag().sum()) / max(N * (N - 1), 1)).item()
    out = {"z_std": z_std, "eff_rank": eff_rank, "eff_rank_frac": eff_rank / D, "cos_offdiag": cos_off}
    if sigreg is not None:
        with torch.no_grad():
            out["sigreg"] = sigreg(Z.to(device).unsqueeze(0)).item()
    return out


# --------------------------------------------------------------------------- #
# Sequence index: every full num_frames window that also has room for a goal    #
# frame. Actions (tiny) are z-scored and kept in RAM; frames are read from the  #
# h5 on demand (StreamSeqDataset) or preloaded (PreloadSeqDataset).             #
# --------------------------------------------------------------------------- #
def _episode_actions(raw_act, n_obs, frameskip, raw_adim, act_mean_t, act_std_t):
    a = raw_act[:n_obs * frameskip].reshape(n_obs, frameskip, raw_adim)
    a = ((a - act_mean_t) / act_std_t).reshape(n_obs, frameskip * raw_adim)
    return a


def build_seq_index(base, num_frames, frameskip, act_mean, act_std, raw_adim, max_eps=None):
    """Read only actions + episode lengths from the h5; enumerate valid window starts.
    A window is [start .. start+num_frames-1] (num_frames obs-frames); it is valid if
    a goal frame can sit at start+num_frames-1+h for h>=1, i.e. start <= n_fr-1-num_frames."""
    import h5py
    act_mean_t = torch.tensor(act_mean, dtype=torch.float32)
    act_std_t = torch.tensor(act_std, dtype=torch.float32)
    lengths = np.asarray(base.lengths)
    n_eps = len(lengths) if max_eps is None else min(max_eps, len(lengths))
    with h5py.File(base.h5_path, "r", swmr=True) as hf:
        act_all = torch.from_numpy(np.asarray(hf["action"][:]))
        offsets = (np.asarray(hf["ep_offset"][:]).astype(np.int64) if "ep_offset" in hf
                   else np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64))
    acts_by_ep, nfr_by_ep, index = [], [], []
    t0 = time.time()
    for ep in range(n_eps):
        L = int(lengths[ep]); off = int(offsets[ep]); n_obs = L // frameskip
        a = _episode_actions(act_all[off:off + n_obs * frameskip], n_obs, frameskip, raw_adim,
                             act_mean_t, act_std_t).half()
        n_fr = min((L + frameskip - 1) // frameskip, n_obs + 1)
        acts_by_ep.append(a); nfr_by_ep.append(n_fr)
        for start in range(0, n_fr - num_frames):        # need >=1 frame after the window for a goal
            index.append((ep, start))
    print(f"[lewam-seq] index: {len(index)} windows (nf={num_frames}) over {n_eps} eps "
          f"({time.time()-t0:.0f}s)", flush=True)
    return acts_by_ep, nfr_by_ep, index


class SeqDataset(Dataset):
    """One full-length window + a goal frame + per-position horizon. Frames come from
    a per-episode preloaded list (preload) or the h5 on demand (stream). Goal is at
    start+num_frames-1+h, h~U[1,H_max] clamped to the episode; dropped with prob
    goal_dropout_p (use_goal=0 -> trainer zeros the goal latent + horizon)."""

    def __init__(self, acts_by_ep, nfr_by_ep, index, indices, num_frames, h_max,
                 goal_dropout_p, frames_by_ep=None, base=None, img_t=None, frameskip=None):
        self.acts = acts_by_ep
        self.nfr = nfr_by_ep
        self.index = index
        self.indices = indices
        self.nf = int(num_frames)
        self.h_max = int(h_max)
        self.gdrop = float(goal_dropout_p)
        self.frames_by_ep = frames_by_ep          # preload path (list of [n_fr,3,H,W] fp16)
        self.base = base; self.img_t = img_t       # stream path
        self.fs = int(frameskip) if frameskip else None

    def __len__(self):
        return self.indices.numel()

    def _frames(self, ep, s, e):
        """obs-frames [s, e) of episode ep -> (e-s, 3, H, W) fp16."""
        if self.frames_by_ep is not None:
            return self.frames_by_ep[ep][s:e]
        r0, r1 = s * self.fs, (e - 1) * self.fs + 1
        pix = self.base._load_slice(ep, r0, r1)["pixels"]  # strided by fs -> obs-frames s..e-1
        if not torch.is_tensor(pix):
            pix = torch.as_tensor(np.asarray(pix))
        return self.img_t({"pixels": pix})["pixels"].float().half()

    def _frame(self, ep, t):
        return self._frames(ep, t, t + 1)[0]

    def __getitem__(self, i):
        ep, start = self.index[int(self.indices[i])]
        nf, n_fr = self.nf, self.nfr[ep]
        # goal at start+nf-1+h, clamp to the last frame
        h = int(torch.randint(1, self.h_max + 1, (1,)).item())
        g = min(start + nf - 1 + h, n_fr - 1)
        window = self._frames(ep, start, start + nf)              # (nf,3,H,W)
        goal = self._frame(ep, g)                                 # (3,H,W)
        actions = self.acts[ep][start:start + nf]                 # (nf, adim)
        pos = torch.arange(start, start + nf, dtype=torch.float32)
        h_norm = ((g - pos) / self.h_max).clamp(0.0, 1.0)         # per-position horizon (nf,)
        use_goal = 1.0 if torch.rand(1).item() >= self.gdrop else 0.0
        return window, actions, goal, h_norm, torch.tensor(use_goal, dtype=torch.float32)


def preload_seq_frames(base, img_t, frameskip, max_eps=None):
    """Preload per-episode obs-frames (fp16). Kept as a list (no giant torch.cat)."""
    n_eps = len(base.lengths) if max_eps is None else min(max_eps, len(base.lengths))
    frames, t0 = [], time.time()
    for ep in range(n_eps):
        L = int(base.lengths[ep])
        pix = base._load_slice(ep, 0, L)["pixels"]
        if not torch.is_tensor(pix):
            pix = torch.as_tensor(np.asarray(pix))
        frames.append(img_t({"pixels": pix})["pixels"].float().half())
        if (ep + 1) % 500 == 0:
            print(f"[lewam-seq] preload {ep+1}/{n_eps} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[lewam-seq] preload DONE {n_eps} eps in {time.time()-t0:.0f}s", flush=True)
    return frames


# --------------------------------------------------------------------------- #
# Rollout loss: from z[:, :1], roll `steps` predictor steps feeding the model's  #
# own predicted latent back (closed-loop dynamics). Actions are the model's own  #
# a_pred (closed_loop) or ground truth (open-loop). MSE vs the GT in-window       #
# latents z[:, 1:steps+1]. End-to-end (no stop-grad) by default.                 #
# --------------------------------------------------------------------------- #
def rollout_loss(model, z, e, h_norm, z_goal, steps, closed_loop, stopgrad):
    z_roll = z[:, :1]
    e_roll = e[:, :0]
    tgt = z[:, 1:steps + 1]
    if stopgrad:
        tgt = tgt.detach()
    losses = []
    for k in range(steps):
        if closed_loop:
            pending = model._tokenize_pending(z_roll, e_roll)
            h_act = model.predictor(pending)[:, 0::2][:, -1:]        # (B,1,D)
            hk = h_norm[:, k:k + 1] if h_norm is not None else h_act.new_zeros(h_act.size(0), 1)
            a_k = model._apply_action_head(h_act, hk, z_goal)[:, -1]  # (B, adim)
            e_k = model.encode_actions(a_k.unsqueeze(1))             # (B,1,D)
        else:
            e_k = e[:, k:k + 1]                                       # GT action
        e_roll = torch.cat([e_roll, e_k], dim=1)
        z_next = model._dynamics_step(z_roll, e_roll, z_goal)         # (B,1,D) predict z_{k+1}
        losses.append(F.mse_loss(z_next[:, 0], tgt[:, k]))
        z_roll = torch.cat([z_roll, z_next], dim=1)                   # feed predicted z back
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=3e-4, help="predictor + heads + adapters lr")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_frames", type=int, default=8, help="window length = inference context")
    ap.add_argument("--H_max", type=int, default=50)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--mlp_dim", type=int, default=384)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--run_name", type=str, default="cube_lewam_seq")
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--dataset_name", type=str, default="ogbench/cube_single_expert.h5")
    ap.add_argument("--keys_to_load", type=str, default="pixels,action,observation")
    ap.add_argument("--warmup_epochs", type=int, default=5)
    # loss weights
    ap.add_argument("--w_act", type=float, default=1.0)
    ap.add_argument("--w_dyn", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=0.04, help="SIGReg (0 = ablate; latent will collapse)")
    ap.add_argument("--w_rollout", type=float, default=0.0, help="K-step rollout loss (0 = off)")
    ap.add_argument("--rollout_steps", type=int, default=3)
    ap.add_argument("--rollout_closed_loop", action="store_true",
                    help="feed the action head's own prediction (else ground-truth actions)")
    ap.add_argument("--rollout_stopgrad", action="store_true",
                    help="stop-grad the rollout target latents (parked; default end-to-end)")
    # goal / head
    ap.add_argument("--goal_dropout_p", type=float, default=0.5,
                    help="0 = always goal-conditioned, 1 = pure BC/FDM, in between = mixed")
    ap.add_argument("--action_head", type=str, default="mse", choices=["mse", "gmm", "diffusion"])
    # data pipeline
    ap.add_argument("--stream", action="store_true", help="read frames from h5 on demand (no preload)")
    ap.add_argument("--collapse_n", type=int, default=8192)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    args = ap.parse_args()

    if args.action_head != "mse":
        raise NotImplementedError(
            "action_head=gmm/diffusion needs LeWAMSeq.ActionHead split into intention "
            "embedding + pluggable decoder (module.GMMHead/DiffusionHead exist). MSE for now.")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dataset ----
    _ktl = [k.strip() for k in args.keys_to_load.split(",") if k.strip()]
    _ktc = [k for k in _ktl if k != "pixels"]
    base = swm.data.load_dataset(args.dataset_name, transform=None, cache_dir=None, num_steps=4,
                                 frameskip=5, keys_to_load=_ktl, keys_to_cache=_ktc)
    frameskip = int(base.frameskip)
    raw_adim = int(base.get_dim("action"))
    action_block_dim = raw_adim * frameskip

    act_norm = get_column_normalizer(base, "action", "action")
    _zn = act_norm.lambd
    act_mean = _zn.mean.squeeze(0).cpu().numpy().tolist()
    act_std = _zn.std.squeeze(0).cpu().numpy().tolist()
    img_t = get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size)

    run_dir = Path(args.run_dir) if args.run_dir else Path(
        swm.data.utils.get_cache_dir(sub_folder="checkpoints"), args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    max_eps = args.max_eps or None

    # ---- model ----
    model = LeWAMSeq(act_dim=action_block_dim, img_size=args.img_size, embed_dim=192,
                     n_layers=args.n_layers, n_heads=args.n_heads, mlp_dim=args.mlp_dim,
                     num_frames=args.num_frames, head_hidden=args.head_hidden).to(device)
    sigreg = SIGReg().to(device)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[lewam-seq] model={n_tot/1e6:.2f}M  nf={args.num_frames} action_block={action_block_dim}",
          flush=True)

    # ---- data ----
    acts_by_ep, nfr_by_ep, index = build_seq_index(
        base, args.num_frames, frameskip, act_mean, act_std, raw_adim, max_eps=max_eps)
    frames_by_ep = None if args.stream else preload_seq_frames(base, img_t, frameskip, max_eps=max_eps)
    n = len(index)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(round((1 - args.train_split) * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[lewam-seq] windows train={train_idx.numel()} val={val_idx.numel()} "
          f"mode={'stream' if args.stream else 'preload'}", flush=True)

    def make_ds(idx):
        return SeqDataset(acts_by_ep, nfr_by_ep, index, idx, args.num_frames, args.H_max,
                          args.goal_dropout_p, frames_by_ep=frames_by_ep,
                          base=(base if args.stream else None),
                          img_t=(img_t if args.stream else None), frameskip=frameskip)
    lc = dict(batch_size=args.batch_size, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    if args.num_workers > 0:
        lc.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    train_loader = DataLoader(make_ds(train_idx), shuffle=True, **lc)
    val_loader = DataLoader(make_ds(val_idx), shuffle=False, **lc)

    # ---- optimizer: ViT encoder at encoder_lr, everything else at lr ----
    enc_params = list(model.state_encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    rest_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": rest_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    warmup, total = args.warmup_epochs, args.epochs
    def lr_lambda(ep):
        if ep < warmup:
            return max(ep / max(warmup, 1), 1e-2)
        p = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    (run_dir / "lewam_seq_config.json").write_text(json.dumps(dict(
        model="lewam_seq", embed_dim=192, action_dim=action_block_dim, num_frames=args.num_frames,
        H_max=args.H_max, frameskip=frameskip, action_raw_dim=raw_adim,
        action_mean=act_mean, action_std=act_std, n_layers=args.n_layers, n_heads=args.n_heads,
        mlp_dim=args.mlp_dim, head_hidden=args.head_hidden, goal_dropout_p=args.goal_dropout_p,
        w_act=args.w_act, w_dyn=args.w_dyn, w_reg=args.w_reg, w_rollout=args.w_rollout,
        rollout_steps=args.rollout_steps, rollout_closed_loop=args.rollout_closed_loop,
        rollout_stopgrad=args.rollout_stopgrad, action_head=args.action_head,
        encoder_lr=args.encoder_lr, lr=args.lr,
    ), indent=2))

    def run_batch(batch, train):
        window, actions, goal, h_norm, use_goal = [x.to(device, non_blocking=True) for x in batch]
        window = window.float(); actions = actions.float(); goal = goal.float()
        h_norm = h_norm.float(); use_goal = use_goal.float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_goal = model.state_encoder(goal) * use_goal[:, None]          # zero goal latent if dropped
            h_norm = h_norm * use_goal[:, None]                            # zero horizon if dropped
            a_pred, z_pred, z, e = model(window, actions, h_norm, z_goal, return_z=True)
            l_act = F.mse_loss(a_pred, actions)
            l_dyn = F.mse_loss(z_pred[:, :-1], z[:, 1:])                    # in-window FDM targets
            l_reg = sigreg(z.reshape(-1, z.size(-1)).unsqueeze(0))          # (1, B*nf, D)
            l_roll = z.new_zeros(())
            if args.w_rollout > 0 and args.rollout_steps > 0:
                k = min(args.rollout_steps, args.num_frames - 1)
                l_roll = rollout_loss(model, z, e, h_norm, z_goal, k,
                                      args.rollout_closed_loop, args.rollout_stopgrad)
            loss = args.w_act * l_act + args.w_dyn * l_dyn + args.w_reg * l_reg + args.w_rollout * l_roll
        return loss, l_act, l_dyn, l_reg, l_roll, z

    best_val, collapse_hist = float("inf"), []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tr = np.zeros(4); tr_n = 0  # sums of [act, dyn, reg, roll]
        for batch in train_loader:
            loss, la, ld, lr_, lro, _ = run_batch(batch, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += [la.item(), ld.item(), lr_.item(), float(lro)]; tr_n += 1
        sched.step()
        tr_a, tr_d, tr_r, tr_roll = (tr / max(tr_n, 1)).tolist()

        # ---- val + collapse ----
        model.eval()
        va = np.zeros(2); va_n = 0; z_buf = []
        with torch.no_grad():
            for batch in val_loader:
                _, la, ld, _, _, z = run_batch(batch, False)
                va += [la.item(), ld.item()]; va_n += 1
                if args.collapse_n and sum(b.shape[0] for b in z_buf) < args.collapse_n:
                    z_buf.append(z.reshape(-1, z.size(-1)).float().cpu())
        va_a, va_d = (va / max(va_n, 1)).tolist()
        cm = {}
        if z_buf:
            cm = collapse_metrics(torch.cat(z_buf)[:args.collapse_n], sigreg=sigreg, device=device)
            collapse_hist.append({"epoch": ep + 1, "val_act": va_a, "val_dyn": va_d, **cm})
            (run_dir / "collapse_metrics.json").write_text(json.dumps(collapse_hist, indent=2))

        col = (f"  [collapse] z_std={cm['z_std']:.4f} eff_rank={cm['eff_rank']:.1f}"
               f"/{cm['eff_rank_frac']*100:.0f}% cos={cm['cos_offdiag']:.3f} "
               f"sigreg={cm.get('sigreg', float('nan')):.2f}" if cm else "")
        roll_s = f"  roll={tr_roll:.5f}" if args.w_rollout > 0 else ""
        print(f"[lewam-seq] ep {ep+1}/{args.epochs}  act={tr_a:.5f}/{va_a:.5f}  "
              f"dyn={tr_d:.5f}/{va_d:.5f}  reg={tr_r:.4f}{roll_s}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s{col}", flush=True)

        torch.save(model.state_dict(), run_dir / "lewam_seq_latest.pt")
        cval = va_a + va_d
        if cval < best_val:
            best_val = cval
            torch.save(model.state_dict(), run_dir / "lewam_seq_best.pt")

    print(f"[lewam-seq] DONE best_val={best_val:.5f} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
