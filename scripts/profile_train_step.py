"""Profile one jointflow training configuration for real: where does the ~370s epoch go?

Loads a RAM-resident SLICE of the u8 pusht cache (no 67G preload), then runs the exact
training step (normalize -> encode -> flow loss -> backward -> clip -> AdamW) through the
standard DataLoader (workers/pin/prefetch as in the jobs), and reports:
  1. wall decomposition: time blocked on next(loader) vs synchronized GPU-step time;
  2. torch.profiler key_averages by CUDA time and by CPU time (top ops);
  3. nvidia-smi utilization sampled during the loop (if pynvml is available).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from train_jointflow import JointFlowGRDataset
from train_lewam_unified import _IMG_MEAN, _IMG_STD
from lewam.models.jointflow import build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="u8 cache dir for pusht")
    ap.add_argument("--stem", default="pusht_expert_train")
    ap.add_argument("--encoder_backbone", default="resnet18dp")
    ap.add_argument("--encoder_size", default="tiny")
    ap.add_argument("--z_dim", type=int, default=192)
    ap.add_argument("--n_heads", type=int, default=3)
    ap.add_argument("--proj_hidden", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--keep_rows", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default="/tmp/profile_report.txt")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    tag = f"{args.stem}_fs5_i224"
    src = np.load(f"{args.cache_dir}/{args.stem}/{tag}.frames.npy", mmap_mode="r")
    n_keep = min(args.keep_rows, src.shape[0])
    frames = torch.from_numpy(np.array(src[:n_keep]))          # RAM slice, uint8
    aux = np.load(f"{args.cache_dir}/{args.stem}/{tag}.aux.npz")
    a_block = torch.from_numpy(aux["A_flat"])
    t_gidx = torch.from_numpy(aux["t_gidx"]).long()
    maxh = torch.from_numpy(aux["maxh"]).long()
    ep_base = torch.from_numpy(aux["ep_base"]).long()
    h_max = 50
    ok = (t_gidx + h_max + 1 < n_keep) & (maxh >= 1)
    idx = torch.nonzero(ok).flatten()
    print(f"[profile] rows={n_keep} usable_anchors={idx.numel()}", flush=True)

    ds = JointFlowGRDataset(frames, a_block, t_gidx, maxh, ep_base, idx,
                            2, 10, 1, 5, h_max)
    loader = DataLoader(ds, sampler=RandomSampler(ds, replacement=True, num_samples=10 ** 9),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, prefetch_factor=4, persistent_workers=True)

    cfg = dict(fs=5, action_raw_dim=2, img_size=224, encoder_size=args.encoder_size,
               encoder_backbone=args.encoder_backbone, encoder_ckpt=None, z_dim=args.z_dim,
               proj_hidden=args.proj_hidden, d_model=args.z_dim, n_heads=args.n_heads,
               depth=8, dropout=0.1, n_flow_steps=8, num_actions_pred=10, num_states_pred=1,
               policy_history_len=2, actions_attend_states=True, split_tau=False,
               goal_conditioning=True, tau_cond="per_modality")
    model = build_model(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    mean = _IMG_MEAN.to(device)
    std = _IMG_STD.to(device)

    def step(batch):
        (hist, pad, target, tvalid, sf, sv, goal, h_norm) = batch
        hist = hist.to(device, non_blocking=True)
        pad = pad.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        tvalid = tvalid.to(device, non_blocking=True)
        sf = sf.to(device, non_blocking=True)
        sv = sv.to(device, non_blocking=True)
        goal = goal.to(device, non_blocking=True)
        h_norm = h_norm.to(device, non_blocking=True)
        B, nh = hist.shape[0], hist.shape[1]
        pix = torch.cat([hist.reshape(B * nh, *hist.shape[2:]),
                         sf.reshape(B * 1, *sf.shape[2:]), goal]).float()
        pix = (pix / 255.0 - mean) / std
        z = model.encode(pix)
        z_hist = z[:B * nh].reshape(B, nh, args.z_dim)
        z_state = z[B * nh:B * (nh + 1)].reshape(B, 1, args.z_dim)
        z_goal = z[B * (nh + 1):]
        la, ls = model.loss(z_hist, pad, target, tvalid, z_state, sv,
                            z_goal=z_goal, h_norm=h_norm)
        loss = la + ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    it = iter(loader)
    for _ in range(12):                                        # warmup: workers, cudnn, allocator
        step(next(it))
    torch.cuda.synchronize()

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        utils = []
    except Exception:
        h, utils = None, None

    t_wait = t_step = 0.0
    for _ in range(args.steps):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()
        step(batch)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_wait += t1 - t0
        t_step += t2 - t1
        if h is not None:
            utils.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)

    lines = [f"WALL loader_wait={t_wait:.2f}s gpu_step={t_step:.2f}s over {args.steps} steps "
             f"({(t_wait + t_step) / args.steps * 1000:.0f} ms/step; "
             f"wait {t_wait / (t_wait + t_step) * 100:.0f}%)"]
    if utils:
        lines.append(f"GPU_UTIL mean={np.mean(utils):.0f}% min={min(utils)}% max={max(utils)}%")

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(12):
            step(next(it))
        torch.cuda.synchronize()
    lines.append("== by CUDA time ==")
    lines.append(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22))
    lines.append("== by CPU time ==")
    lines.append(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))
    report = "\n".join(lines)
    print(report, flush=True)
    Path(args.out).write_text(report)


if __name__ == "__main__":
    main()
