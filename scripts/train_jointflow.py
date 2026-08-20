import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from train_crossattn import load_cache, RawContextDataset
from train_lewam_unified import durable_sync, _IMG_MEAN, _IMG_STD
from lewam.models.jointflow import build_model
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
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="one uniform rate for encoder + flow transformer (UWM precedent)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
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
    ap.add_argument("--policy_history_len", type=int, default=2,
                    help="raw-consecutive obs frames the model uses as history")
    ap.add_argument("--steps_per_epoch", type=int, default=0,
                    help="cap train/val batches per epoch (0 = one pass over the decision points)")
    ap.add_argument("--actions_attend_states", type=int, default=1,
                    help="1: action tokens can attend to jointly predicted state tokens; "
                         "0: mask action tokens from attending the (noisy) state tokens")
    ap.add_argument("--split_tau", action="store_true",
                    help="draw tau_action and tau_state independently instead of one shared tau")
    ap.add_argument("--tau_cond", default="per_modality", choices=["per_modality", "summed"],
                    help="per_modality: each modality modulated by its own flow-time; "
                         "summed: one blended cond for all tokens")
    # anti-collapse
    ap.add_argument("--w_reg", type=float, default=0.0,
                    help="SIGReg anti-collapse weight on the encoder latent")
    ap.add_argument("--state_ema_target", action="store_true",
                    help="state flow targets from a momentum copy of the encoder, layernormed and "
                         "stop-gradded")
    ap.add_argument("--state_ema_base", type=float, default=0.998,
                    help="base momentum for the target encoder, linearly annealed to 1.0")
    return ap.parse_args()


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
        history_frames, history_pad, target, target_valid, _goal, _next_frame, _dyn_valid = \
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
        return history_frames, history_pad, target, target_valid, state_frames, state_valid


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

    frames, a_frame, t_gidx, ep_base, frames_to_terminal, action_stats = load_cache(args)
    n_starts = t_gidx.shape[0]
    print(f"[jointflow] frames={tuple(frames.shape)} starts={n_starts} raw_dim={a_frame.shape[1]} "
          f"layout=(a{args.num_actions_pred},s{args.num_states_pred},fs{args.frameskip}) "
          f"actions_attend_states={bool(args.actions_attend_states)} split_tau={args.split_tau}",
          flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_starts, generator=gen)
    perm = perm[frames_to_terminal[perm] >= args.frameskip]
    n_val = int(round((1 - args.train_split) * perm.numel()))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers:
        loader_args.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)

    def make_loader(idx):
        ds = JointFlowDataset(frames, a_frame, t_gidx, ep_base, frames_to_terminal, idx,
                              args.policy_history_len, args.num_actions_pred, args.frameskip,
                              args.num_states_pred)
        n = max(args.batch_size, int(idx.numel()))
        if args.steps_per_epoch:
            n = min(n, args.steps_per_epoch * args.batch_size)
        return DataLoader(ds, sampler=RandomSampler(ds, replacement=True, num_samples=n), **loader_args)
    train_loader, val_loader = make_loader(train_idx), make_loader(val_idx)

    cfg = dict(fs=args.frameskip, action_raw_dim=int(a_frame.shape[1]), img_size=args.img_size,
               encoder_size=args.encoder_size, encoder_backbone=args.encoder_backbone,
               encoder_ckpt=None, z_dim=args.z_dim, proj_hidden=args.proj_hidden,
               d_model=args.d_model, n_heads=args.n_heads,
               depth=args.depth, dropout=args.dropout, n_flow_steps=args.n_flow_steps,
               num_actions_pred=args.num_actions_pred, num_states_pred=args.num_states_pred,
               policy_history_len=args.policy_history_len,
               actions_attend_states=bool(args.actions_attend_states), split_tau=args.split_tau,
               tau_cond=args.tau_cond)
    model = build_model(cfg).to(device)
    action_mean, action_std = action_stats
    dumped = {**cfg, **vars(args), "action_mean": action_mean, "action_std": action_std}
    (run_dir / "jointflow_config.json").write_text(json.dumps(dumped, indent=1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[jointflow] params={n_params/1e6:.2f}M  w_reg={args.w_reg} "
          f"state_target={'ema' if args.state_ema_target else 'online'}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        print(f"[jointflow] resumed at epoch {start_epoch} best_val={best_val:.5f}", flush=True)

    mean = _IMG_MEAN.to(device); std = _IMG_STD.to(device)
    sigreg = SIGReg().to(device)
    n_states = args.num_states_pred

    def run_batch(batch):
        history_frames, history_pad, action_target, action_valid, state_frames, state_valid = batch
        history_frames = history_frames.to(device, non_blocking=True)
        history_pad = history_pad.to(device, non_blocking=True)
        action_target = action_target.to(device, non_blocking=True).float()
        action_valid = action_valid.to(device, non_blocking=True)
        state_frames = state_frames.to(device, non_blocking=True)
        state_valid = state_valid.to(device, non_blocking=True)
        B, n_history = history_frames.shape[0], history_frames.shape[1]
        was_uint8 = history_frames.dtype == torch.uint8
        norm = lambda x: (x / 255.0 - mean) / std if was_uint8 else x

        pixels = norm(torch.cat([
            history_frames.reshape(B * n_history, *history_frames.shape[2:]),
            state_frames.reshape(B * n_states, *state_frames.shape[2:])]).float())
        z = model.encode(pixels)
        z_history = z[:B * n_history].reshape(B, n_history, args.z_dim)
        z_state_online = z[B * n_history:].reshape(B, n_states, args.z_dim)
        if tgt_encoder is not None:
            with torch.no_grad():
                z_target_raw = tgt_encoder(
                    norm(state_frames.reshape(B * n_states, *state_frames.shape[2:]).float()))
            state_target = F.layer_norm(z_target_raw, (args.z_dim,)).reshape(B, n_states, args.z_dim)
        else:
            state_target = z_state_online          # the encoder learns from being the target

        loss_action, loss_state = model.loss(z_history, history_pad, action_target, action_valid,
                                             state_target, state_valid)
        loss = loss_action + loss_state
        loss_terms = {"act": loss_action.item(), "state": loss_state.item()}
        if n_states:
            # collapse telemetry: per-dim std of the online state latents (collapse -> ~0)
            loss_terms["zstd"] = z_state_online.reshape(-1, args.z_dim).std(0).mean().item()
        if args.w_reg > 0:
            z_states = torch.cat([z_history[:, -1],
                                  state_target.reshape(B * n_states, args.z_dim)]).unsqueeze(0)
            loss_reg = sigreg(z_states)
            loss = loss + args.w_reg * loss_reg
            loss_terms["reg"] = loss_reg.item()
        return loss, loss_terms, B

    def accumulate(store, loss_terms, n):
        for k, v in loss_terms.items():
            store[k] = store.get(k, 0.0) + v * n

    def fmt(store, n):
        return " ".join(f"{k}={store[k]/max(n,1):.5f}"
                        for k in ("act", "state", "reg", "zstd") if k in store)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        mom_now = args.state_ema_base + (1.0 - args.state_ema_base) * (epoch / max(1, args.epochs))
        model.train()
        train_stats, train_n = {}, 0.0
        for batch in train_loader:
            loss, loss_terms, n = run_batch(batch)
            opt.zero_grad(set_to_none=True)
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
                loss, loss_terms, n = run_batch(batch)
                accumulate(val_stats, loss_terms, n); val_n += n
        val_act = val_stats.get("act", 0.0) / max(val_n, 1)   # best-checkpoint metric = val action loss
        print(f"[jointflow] ep {epoch+1}/{args.epochs}  train[{fmt(train_stats,train_n)}]  "
              f"val[{fmt(val_stats,val_n)}]  lr={sched.get_last_lr()[0]:.2e}  "
              f"{time.time()-t0:.1f}s", flush=True)

        torch.save(model.state_dict(), run_dir / "jointflow_latest.pt")
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        scheduler=sched.state_dict(), epoch=epoch, best_val=best_val),
                   full_path)
        files = [run_dir / "jointflow_config.json", run_dir / "jointflow_latest.pt", full_path]
        if val_act < best_val:
            best_val = val_act
            torch.save(model.state_dict(), run_dir / "jointflow_best.pt")
            files.append(run_dir / "jointflow_best.pt")
        if args.ckpt_sync_dir:
            durable_sync(files, args.ckpt_sync_dir)

    print(f"[jointflow] DONE best_val={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
