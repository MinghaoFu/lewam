"""Train the cross-attention LeWAM (docs/wip/CROSSATTN_DESIGN.md). Reuses the raw-anchor cache and
SeqTrajDataset from train_lewam_unified; the model owns the conditioning, so the loop just encodes,
weight-gates the two losses, and steps."""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from train_lewam_unified import durable_sync, _IMG_MEAN, _IMG_STD
from lewam.models.lewam_crossattn import build_model
from lewam.models.module import SIGReg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="tool_hang.h5")
    ap.add_argument("--run_name", default="crossattn")
    ap.add_argument("--exp_tag", default="")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--ckpt_sync_dir", default="")
    ap.add_argument("--frames_cache", default="auto")
    ap.add_argument("--cache_mmap", action="store_true")
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--context_len", type=int, default=3)
    ap.add_argument("--H_max", type=int, default=50)
    ap.add_argument("--p_terminal_goal", type=float, default=1.0)
    ap.add_argument("--p_shared", type=float, default=0.0)
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--ablate_horizon", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    # model
    ap.add_argument("--head", default="flow", choices=["flow", "dp"])
    ap.add_argument("--encoder_backbone", default="resnet18sp")
    ap.add_argument("--encoder_size", default="tiny")
    ap.add_argument("--z_dim", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--policy_depth", type=int, default=6)
    ap.add_argument("--dyn_depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n_flow_steps", type=int, default=8)
    ap.add_argument("--num_chunks", type=int, default=2)
    ap.add_argument("--policy_history_len", type=int, default=3)
    ap.add_argument("--dyn_history_len", type=int, default=3)
    ap.add_argument("--goal_conditioning", type=int, default=1)
    ap.add_argument("--dyn_goal_cond", action="store_true")
    ap.add_argument("--w_act", type=float, default=1.0)
    ap.add_argument("--w_dyn", type=float, default=0.0)
    ap.add_argument("--w_reg", type=float, default=0.04,
                    help="SIGReg anti-collapse weight on the encoder latent (unified default 0.04)")
    ap.add_argument("--ema", action="store_true",
                    help="track EMA weights (DP schedule) and eval/checkpoint them")
    ap.add_argument("--dyn_ema_target", action="store_true",
                    help="use a momentum (EMA) copy of the encoder, stop-gradded, as the dynamics "
                         "target (BYOL-style) to slow the latent collapsing onto the easy dyn task")
    ap.add_argument("--dyn_ema_base", type=float, default=0.998,
                    help="base momentum for the dynamics EMA target encoder, linearly annealed to "
                         "1.0 over training (V-JEPA schedule; I-JEPA uses 0.996)")
    return ap.parse_args()


class Ema:
    """Exponential moving average of model PARAMETERS, matching Diffusion Policy's EMAModel:
    decay ramps as 1 - (1 + step)^(-power), capped at max_value. Only parameters are averaged;
    buffers (attention masks, BatchNorm running stats) are left to the live model. Averaging a
    fixed -inf attention-mask buffer would nan on the first update (decay 0 -> mul_(0) -> -inf*0)."""

    def __init__(self, model, inv_gamma=1.0, power=0.75, max_value=0.9999):
        self.inv_gamma, self.power, self.max_value = inv_gamma, power, max_value
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.named_parameters()}

    def _decay(self):
        s = max(0, self.step - 1)
        if s <= 0:
            return 0.0
        return min(self.max_value, 1.0 - (1.0 + s / self.inv_gamma) ** (-self.power))

    @torch.no_grad()
    def update(self, model):
        d = self._decay()
        for k, v in model.named_parameters():
            self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
        self.step += 1

    def merged_state(self, model):
        """A full state_dict = the live model's buffers + the EMA parameters, for load_state_dict."""
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in self.shadow.items():
            sd[k] = v.clone()
        return sd

    def state(self):
        return {"step": self.step, "shadow": self.shadow}

    def load(self, state):
        self.step = state["step"]
        dev = next(iter(self.shadow.values())).device
        self.shadow = {k: v.to(dev) for k, v in state["shadow"].items()}


def load_cache(args):
    cdir = args.frames_cache if args.frames_cache != "auto" else os.environ.get(
        "LEWAM_CACHE_DIR", "/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/preload_cache")
    stem = os.path.basename(args.dataset_name).replace(".h5", "")
    tag = f"{stem}_fs{args.frameskip}_i{args.img_size}_raw"
    frames_path = f"{cdir}/{stem}/{tag}.frames.npy"
    aux_path = f"{cdir}/{stem}/{tag}.aux.npz"
    frames = torch.from_numpy(np.load(frames_path, mmap_mode="r" if args.cache_mmap else None))
    aux = np.load(aux_path)
    a_frame = torch.from_numpy(aux["A_flat"])
    t_gidx = torch.from_numpy(aux["t_gidx"])
    ep_base = torch.from_numpy(aux["ep_base"])
    frames_to_terminal = torch.from_numpy(aux["frames_to_terminal"])
    action_stats = (aux["act_mean"].tolist(), aux["act_std"].tolist())
    assert a_frame.shape[0] == frames.shape[0], "raw cache A_flat must be frame-aligned"
    return frames, a_frame, t_gidx, ep_base, frames_to_terminal, action_stats


class RawContextDataset(Dataset):
    """One decision point per sample: the last `history_len` CONSECUTIVE raw frames as policy context
    (instantaneous state, DP's obs structure), the next `n_tokens` raw actions as target, the terminal
    frame as goal. Decoupled from the frameskip world model, which the policy does not use here."""

    def __init__(self, frames, a_frame, t_gidx, ep_base, frames_to_terminal, indices,
                 history_len, n_tokens, frameskip):
        self.frames = frames
        self.a_frame = a_frame
        self.t_gidx = t_gidx
        self.ep_base = ep_base
        self.frames_to_terminal = frames_to_terminal
        self.indices = indices
        self.history_len = history_len
        self.n_tokens = n_tokens
        self.frameskip = frameskip

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, i):
        idx = int(self.indices[i])
        p = int(self.t_gidx[idx])
        ep0 = int(self.ep_base[idx])
        ttl = int(self.frames_to_terminal[idx])
        hl, n = self.history_len, self.n_tokens

        recent = self.frames[max(ep0, p - hl + 1):p + 1]
        k = recent.shape[0]
        context = torch.empty((hl, *recent.shape[1:]), dtype=recent.dtype)
        context[hl - k:] = recent                            # right-aligned, most recent last
        if k < hl:
            # repeat-edge pad (masked in attention): zero frames through the encoder would
            # corrupt its projector BatchNorm batch stats (see train_lewam_unified collate_pad).
            # NOTE: crossattn_bc board numbers predate this fix (they trained with zero-pad).
            context[:hl - k] = recent[0]
        ctx_pad = torch.ones(hl, dtype=torch.bool)
        ctx_pad[hl - k:] = False

        avail = min(n, ttl)
        raw = self.a_frame[p:p + avail].float()
        target = torch.zeros((n, raw.shape[1]), dtype=torch.float32)
        target[:avail] = raw
        target_valid = torch.zeros(n, dtype=torch.float32)
        target_valid[:avail] = 1.0

        goal = self.frames[p + ttl]

        # dynamics target: the next decision-point frame (frameskip ahead). Its transition action
        # block is target[:frameskip]. Invalid (masked) when the next point is past the terminal.
        fs = self.frameskip
        dyn_valid = torch.tensor(1.0 if fs <= ttl else 0.0, dtype=torch.float32)
        next_frame = self.frames[min(p + fs, p + ttl)]
        return context, ctx_pad, target, target_valid, goal, next_frame, dyn_valid


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = args.exp_tag.strip("_")
    run_dir = Path(args.run_dir) if args.run_dir else Path(f"runs/{args.run_name}_{tag}" if tag else args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames, a_frame, t_gidx, ep_base, frames_to_terminal, action_stats = load_cache(args)
    n_starts = t_gidx.shape[0]
    n_tokens = args.frameskip * args.num_chunks
    print(f"[crossattn] frames={tuple(frames.shape)} starts={n_starts} raw_dim={a_frame.shape[1]} "
          f"policy_ctx=raw-consecutive({args.policy_history_len})", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_starts, generator=gen)
    perm = perm[frames_to_terminal[perm] >= args.frameskip]
    n_val = int(round((1 - args.train_split) * perm.numel()))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers:
        loader_args.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)

    def make_loader(idx):
        ds = RawContextDataset(frames, a_frame, t_gidx, ep_base, frames_to_terminal, idx,
                               args.policy_history_len, n_tokens, args.frameskip)
        # one decision point per sample; one epoch = one pass over the decision points (DP-aligned at
        # batch_size 64), unlike the old windowed-budget heuristic that divided by context_len.
        n = max(args.batch_size, int(idx.numel()))
        return DataLoader(ds, sampler=RandomSampler(ds, replacement=True, num_samples=n), **loader_args)
    train_loader, val_loader = make_loader(train_idx), make_loader(val_idx)

    cfg = dict(fs=args.frameskip, action_raw_dim=int(a_frame.shape[1]), img_size=args.img_size,
               encoder_size=args.encoder_size, encoder_backbone=args.encoder_backbone,
               encoder_ckpt=None, z_dim=args.z_dim, d_model=args.d_model, n_heads=args.n_heads,
               policy_depth=args.policy_depth, dyn_depth=args.dyn_depth, dropout=args.dropout,
               n_flow_steps=args.n_flow_steps, num_chunks=args.num_chunks,
               policy_history_len=args.policy_history_len, dyn_history_len=args.dyn_history_len,
               goal_conditioning=bool(args.goal_conditioning), dyn_goal_cond=args.dyn_goal_cond,
               head=args.head)
    model = build_model(cfg).to(device)
    action_mean, action_std = action_stats
    dumped = {**cfg, **vars(args), "action_mean": action_mean, "action_std": action_std,
              "frameskip": args.frameskip, "action_dim": cfg["fs"] * cfg["action_raw_dim"]}
    (run_dir / "crossattn_config.json").write_text(json.dumps(dumped, indent=1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[crossattn] params={n_params/1e6:.2f}M  w_act={args.w_act} w_dyn={args.w_dyn}", flush=True)

    enc_ids = {id(p) for p in model.encoder.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in enc_ids], "lr": args.encoder_lr},
        {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": args.lr},
    ], weight_decay=args.weight_decay)

    def lr_scale(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        p = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    ema = Ema(model) if args.ema else None
    tgt_encoder = None
    if args.dyn_ema_target:
        tgt_encoder = copy.deepcopy(model.encoder).to(device)
        for p in tgt_encoder.parameters():
            p.requires_grad_(False)
        tgt_encoder.eval()
    start_epoch, best_val = 0, float("inf")
    full_path = run_dir / "crossattn_full.pt"
    if args.resume and full_path.exists():
        state = torch.load(full_path, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"]); start_epoch = state["epoch"] + 1
        best_val = state["best_val"]
        if ema is not None and state.get("ema") is not None:
            ema.load(state["ema"])
        print(f"[crossattn] resumed at epoch {start_epoch} best_val={best_val:.5f}", flush=True)

    mean = _IMG_MEAN.to(device); std = _IMG_STD.to(device)
    sigreg = SIGReg().to(device)
    fs = args.frameskip

    def run_batch(batch):
        context, ctx_pad, target, target_valid, goal, next_frame, dyn_valid = batch
        context = context.to(device, non_blocking=True)
        ctx_pad = ctx_pad.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        target_valid = target_valid.to(device, non_blocking=True)
        goal = goal.to(device, non_blocking=True)
        next_frame = next_frame.to(device, non_blocking=True)
        dyn_valid = dyn_valid.to(device, non_blocking=True)
        B, hl = context.shape[0], context.shape[1]
        was_uint8 = context.dtype == torch.uint8
        norm = lambda x: (x / 255.0 - mean) / std if was_uint8 else x
        online = norm(torch.cat([context.reshape(B * hl, *context.shape[2:]),
                                 goal.reshape(B, *goal.shape[1:])]).float())
        z = model.encode(online)
        z_ctx = z[:B * hl].reshape(B, hl, args.z_dim)
        z_goal = z[B * hl:].reshape(B, args.z_dim)
        horizon = torch.zeros(B, device=device)          # ablate_horizon
        item_valid = torch.ones(B, device=device)

        loss_act = model.policy_head.loss(z_ctx, ctx_pad, z_goal, model.goal_conditioning,
                                          target, target_valid, horizon, item_valid)
        loss = args.w_act * loss_act
        comps = {"act": loss_act.item()}

        if args.w_dyn > 0 or args.w_reg > 0:
            # next-state latent z_{t+fs}: from the momentum encoder (stop-gradded) when enabled,
            # else the online encoder (the encoder learns from being the target).
            nf = norm(next_frame.reshape(B, *next_frame.shape[1:]).float())
            z_next = tgt_encoder(nf).detach() if tgt_encoder is not None else model.encode(nf)
            z_t = z_ctx[:, -1]                                        # (B, z_dim)
            if args.w_dyn > 0:
                # forward model on z_t only (dyn_history_len=1): z_t + first action chunk -> z_{t+fs}.
                dyn_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
                loss_dyn = model.dynamics_head.loss(z_t, z_t.unsqueeze(1), dyn_pad,
                                                    target[:, :fs, :], z_next, None, False, dyn_valid)
                loss = loss + args.w_dyn * loss_dyn
                comps["dyn"] = loss_dyn.item()
            if args.w_reg > 0:
                z_states = torch.cat([z_t, z_next], dim=0).unsqueeze(0)   # (1, 2B, z_dim)
                loss_reg = sigreg(z_states)
                loss = loss + args.w_reg * loss_reg
                comps["reg"] = loss_reg.item()

        return loss, comps, B

    def accumulate(store, comps, n):
        for k, v in comps.items():
            store[k] = store.get(k, 0.0) + v * n

    def fmt(store, n):
        return " ".join(f"{k}={store[k]/max(n,1):.5f}" for k in ("act", "dyn", "reg") if k in store)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        mom_now = args.dyn_ema_base + (1.0 - args.dyn_ema_base) * (epoch / max(1, args.epochs))
        model.train()
        tr, tr_n = {}, 0.0
        for batch in train_loader:
            loss, comps, n = run_batch(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)
            if tgt_encoder is not None:
                with torch.no_grad():
                    d = mom_now
                    for pt, ps in zip(tgt_encoder.parameters(), model.encoder.parameters()):
                        pt.mul_(d).add_(ps.detach(), alpha=1.0 - d)
                    for bt, bs in zip(tgt_encoder.buffers(), model.encoder.buffers()):
                        bt.copy_(bs)
            accumulate(tr, comps, n); tr_n += n
        sched.step()

        # val (and the checkpoints the eval harness reads) run on the EMA weights when enabled;
        # the raw weights are restored afterwards so training continues on them.
        raw_sd = None
        if ema is not None:
            raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.merged_state(model), strict=True)
        model.eval()
        va, va_n = {}, 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, comps, n = run_batch(batch)
                accumulate(va, comps, n); va_n += n
        va_a = va.get("act", 0.0) / max(va_n, 1)      # best-checkpoint metric = val action loss
        print(f"[crossattn] ep {epoch+1}/{args.epochs}  train[{fmt(tr,tr_n)}]  val[{fmt(va,va_n)}]  "
              f"lr={sched.get_last_lr()[1]:.2e}  {time.time()-t0:.1f}s", flush=True)

        torch.save(model.state_dict(), run_dir / "crossattn_latest.pt")
        torch.save(dict(model=(raw_sd if raw_sd is not None else model.state_dict()),
                        optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                        epoch=epoch, best_val=best_val,
                        ema=(ema.state() if ema is not None else None)),
                   full_path)
        files = [run_dir / "crossattn_config.json", run_dir / "crossattn_latest.pt", full_path]
        if va_a < best_val:
            best_val = va_a
            torch.save(model.state_dict(), run_dir / "crossattn_best.pt")
            files.append(run_dir / "crossattn_best.pt")
        if raw_sd is not None:
            model.load_state_dict(raw_sd, strict=True)
        if args.ckpt_sync_dir:
            durable_sync(files, args.ckpt_sync_dir)

    print(f"[crossattn] DONE best_val={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
