"""Train DP-C (diffusion_policy's DiffusionUnetHybridImagePolicy) on a LeWAM preload cache with the LeWAM
trainer's data, so a DP-C row is comparable to a LeWAM row up to the architecture: the same cameras and
frames, one window per frame per epoch, a validation split of windows drawn like the LeWAM trainer's, and the
same epoch count.

Windows follow diffusion_policy's rule. For an episode of L frames the window starts are
s = -1 ... L - horizon + n_action_steps - 1 (L - 7 windows at the published 16 / 8 / 2): the observation is
the frames at s ... s + n_obs_steps - 1 (frame 0 repeated before the episode start), the target is the
horizon recorded actions from s (the last action repeated past the episode end). At eval the policy executes
the n_action_steps actions from s + n_obs_steps - 1, so the last observed frame is the current step.

Loss, optimizer and schedule are diffusion_policy's: the denoising MSE over the whole chunk, AdamW 1e-4
(0.95, 0.999) with weight decay 1e-6, cosine with 500 warmup steps stepped per batch, no EMA (the raw weights
are saved and evaluated). The model and normalizer rules live in lewam.models.dp_policy.
Checkpoints: dp_best.pt (lowest validation loss), dp_latest.pt (the last epoch's weights), dp_full.pt (weights
plus optimizer, schedule and RNG for resume), dp_config.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from train_jointflow import load_cache_views
from train_lewam_unified import durable_sync
from lewam.models import dp_policy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True, help="cache stem, e.g. tool_hang.h5")
    ap.add_argument("--frames_cache", default="auto")
    ap.add_argument("--views", default="pixels", help="h5 image columns, first = scene camera")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frameskip", type=int, default=5, help="cache tag only; windows are raw frames")
    ap.add_argument("--cache_mmap", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n_obs_steps", type=int, default=2)
    ap.add_argument("--n_action_steps", type=int, default=8)
    ap.add_argument("--crop", type=int, default=0, help="0 = 0.9 of the image size (202 at 224)")
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="cap (smoke tests); 0 = one pass")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt_sync_dir", default="")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def episode_table(t_gidx, ep_base, frames_to_terminal):
    """(first frame, length) of every episode, from the cache's per-anchor episode start and frames left."""
    start = ep_base.numpy()
    last = (t_gidx + frames_to_terminal).numpy()
    episodes = sorted({(int(s), int(e) - int(s) + 1) for s, e in zip(start, last)})
    ends = [s + n for s, n in episodes]
    assert all(ends[i] == episodes[i + 1][0] for i in range(len(episodes) - 1)), "episodes must tile the cache"
    return episodes


def window_indices(episodes, horizon, n_obs_steps, n_action_steps):
    """diffusion_policy's windows over every episode: frame indices of the observation and of the action
    chunk, both clamped into the episode (front padding repeats the first frame, back padding the last action)."""
    obs_offsets = np.arange(n_obs_steps)
    action_offsets = np.arange(horizon)
    obs_index, action_index = [], []
    for first, length in episodes:
        starts = np.arange(-(n_obs_steps - 1), length - horizon + n_action_steps)
        obs_index.append(first + np.clip(starts[:, None] + obs_offsets, 0, length - 1))
        action_index.append(first + np.clip(starts[:, None] + action_offsets, 0, length - 1))
    return torch.from_numpy(np.concatenate(obs_index)), torch.from_numpy(np.concatenate(action_index))


class WindowDataset(Dataset):
    """One window: the observation frames (n_obs_steps, cameras, 3, H, W) uint8 and the action chunk
    (horizon, action_dim) in raw units."""

    def __init__(self, frames, actions, obs_index, action_index, rows):
        self.frames, self.actions = frames, actions
        self.obs_index, self.action_index, self.rows = obs_index, action_index, rows

    def __len__(self):
        return self.rows.numel()

    def __getitem__(self, i):
        row = int(self.rows[i])
        obs = self.frames[self.obs_index[row]]
        if obs.dim() == 4:                       # one camera: (n_obs, 3, H, W) -> (n_obs, 1, 3, H, W)
            obs = obs.unsqueeze(1)
        return obs, self.actions[self.action_index[row]]


def to_batch(obs, actions, keys, device):
    """diffusion_policy's batch: one (B, n_obs, 3, H, W) float image tensor in [0, 1] per camera."""
    obs = obs.to(device, non_blocking=True)
    return {"obs": {key: obs[:, :, cam].float().div_(255.0) for cam, key in enumerate(keys)},
            "action": actions.to(device, non_blocking=True)}


def mean_loss(policy, loader, keys, device):
    policy.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for obs, actions in loader:
            total += policy.compute_loss(to_batch(obs, actions, keys, device)).item() * obs.shape[0]
            count += obs.shape[0]
    policy.train()
    return total / max(count, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames, a_frame, t_gidx, ep_base, frames_to_terminal, (act_mean, act_std) = load_cache_views(args)
    actions = a_frame.float() * torch.tensor(act_std) + torch.tensor(act_mean)     # raw units, per frame
    columns = [column for column in args.views.split(",") if column]
    episodes = episode_table(t_gidx, ep_base, frames_to_terminal)
    obs_index, action_index = window_indices(episodes, args.horizon, args.n_obs_steps, args.n_action_steps)
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(obs_index.shape[0], generator=generator)
    n_val = int(round((1 - args.train_split) * perm.numel()))
    val_rows, train_rows = perm[:n_val], perm[n_val:]

    config = dp_policy.make_config(columns, args.img_size, actions.shape[1], crop=args.crop or None,
                                   horizon=args.horizon, n_obs_steps=args.n_obs_steps,
                                   n_action_steps=args.n_action_steps)
    policy = dp_policy.build_policy(config)
    normalizer_rule = dp_policy.fit_normalizer(policy, actions.numpy(), config)
    config.update(dataset_name=args.dataset_name, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  warmup_steps=args.warmup_steps, train_split=args.train_split, seed=args.seed,
                  n_windows=int(obs_index.shape[0]), n_train=int(train_rows.numel()), n_val=int(n_val))
    dp_policy.save_config(run_dir, config)
    policy.to(device).train()
    keys = dp_policy.camera_keys(columns)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[dp] frames={tuple(frames.shape)} episodes={len(episodes)} windows={obs_index.shape[0]} "
          f"train={train_rows.numel()} val={n_val} action_dim={actions.shape[1]} cameras={keys} "
          f"chunk={args.horizon}/{args.n_action_steps}/{args.n_obs_steps} crop={config['crop']} "
          f"action_normalizer={normalizer_rule} params={n_params / 1e6:.2f}M", flush=True)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers:
        loader_args.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)

    def make_loader(rows):
        dataset = WindowDataset(frames, actions, obs_index, action_index, rows)
        n = int(rows.numel())
        if args.steps_per_epoch:
            n = min(n, args.steps_per_epoch * args.batch_size)
        return DataLoader(dataset, sampler=RandomSampler(dataset, num_samples=n), **loader_args)
    train_loader, val_loader = make_loader(train_rows), make_loader(val_rows)

    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8,
                                  weight_decay=1e-6)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                              num_training_steps=len(train_loader) * args.epochs)

    start_epoch, best_val = 0, float("inf")
    full_path = run_dir / "dp_full.pt"
    if args.resume and full_path.exists():
        state = torch.load(full_path, map_location=device)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_val = state["epoch"] + 1, state["best_val"]
        torch.set_rng_state(state["rng"])
        print(f"[dp] resumed at epoch {start_epoch} best_val={best_val:.5f}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        total, count = 0.0, 0
        for obs, actions_chunk in train_loader:
            loss = policy.compute_loss(to_batch(obs, actions_chunk, keys, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * obs.shape[0]
            count += obs.shape[0]
        train_loss = total / max(count, 1)
        val_loss = mean_loss(policy, val_loader, keys, device)
        print(f"[dp] ep {epoch + 1}/{args.epochs}  train[loss={train_loss:.5f}]  val[loss={val_loss:.5f}]  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {time.time() - started:.1f}s", flush=True)

        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        weights = dict(model=policy.state_dict())
        torch.save(weights, run_dir / "dp_latest.pt")
        torch.save(dict(weights, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        epoch=epoch, best_val=best_val, rng=torch.get_rng_state()), full_path)
        files = [run_dir / "dp_config.json", run_dir / "dp_latest.pt", full_path]
        if improved:
            torch.save(dict(model=policy.state_dict()), run_dir / "dp_best.pt")
            files.append(run_dir / "dp_best.pt")
        if args.ckpt_sync_dir:
            durable_sync(files, args.ckpt_sync_dir)
    print(f"[dp] DONE best_val={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
