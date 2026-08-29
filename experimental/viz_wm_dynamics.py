"""Visualize a MoT world model's imagined latents against ground truth (owner/Minghao 2026-08-29;
adapted from train_decoder_unified.py + viz_rollout_unified.py).

Given k anchor frames of one episode (fs raw steps apart) and the (k-1) action blocks that
produced them, the dynamics imagines the episode OPEN-LOOP: the history starts from the first
policy_history_len real latents, each block's clean actions go through predict_state, and the
imagined latent slides into the history. A small transformer decoder (LeWMVisDecoder), trained
on this checkpoint's own encoder latents, renders three rows per episode:

    GT frames | decode(z_GT)  (decoder-quality reference) | decode(z_imagined)

with the per-column relative latent error ||z_hat - z|| / RMS(z - mean) and cosine similarity.
Flow state heads are sampled under a fixed seed (their imagination is a draw; MSE heads are
deterministic).

  python viz_wm_dynamics.py --run_name <ckpt name under $STABLEWM_HOME/checkpoints> --h5 <dsn> \\
      --out_dir <dir> [--k 10 --eps 3,77,1234 --dec_epochs 12 --dec_max_samples 20000 --dec_path <pt>]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import stable_worldmodel.data.formats.hdf5  # noqa: F401
import stable_worldmodel as swm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from decoder import LeWMVisDecoder  # noqa: E402
import lewam.models.gip as gip  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def norm(px):
    return (px - IMAGENET_MEAN.to(px.device)) / IMAGENET_STD.to(px.device)


def amp(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def build_decoder(z_dim, device):
    return LeWMVisDecoder(cls_dim=z_dim, hidden_dim=256, depth=6, heads=8, dim_head=32,
                          mlp_ratio=4.0, img_size=224, patch_size=16, out_channels=3, dropout=0.0).to(device)


def train_decoder(model, a, z_dim, device, out_dir):
    """The unified trainer's loop, single latent z = encode(frame)."""
    ds = swm.data.load_dataset(a.h5, format="hdf5", keys_to_load=["pixels"], frameskip=a.fs, num_steps=1)
    if a.dec_max_samples and a.dec_max_samples < len(ds):
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[: a.dec_max_samples].tolist()
        ds = torch.utils.data.Subset(ds, idx)
    n = len(ds)
    n_val = max(64, n // 20)
    tr, va = random_split(ds, [n - n_val, n_val], generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr, batch_size=a.dec_batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=a.dec_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    dec = build_decoder(z_dim, device)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.dec_lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, t0 = 1e9, time.time()
    path = out_dir / f"decoder_z_{a.run_name}.pt"
    for ep in range(a.dec_epochs):
        dec.train()
        for batch in tl:
            px = batch["pixels"].float().to(device) / 255.0
            frame = px[:, 0] if px.ndim == 5 else px
            with torch.no_grad(), amp(device):
                z = model.encode(norm(frame))
            opt.zero_grad()
            with amp(device):
                loss = F.mse_loss(dec(z.float()), frame)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        dec.eval()
        vm = []
        with torch.no_grad(), amp(device):
            for batch in vl:
                px = batch["pixels"].float().to(device) / 255.0
                frame = px[:, 0] if px.ndim == 5 else px
                z = model.encode(norm(frame))
                vm.append(F.mse_loss(dec(z.float()), frame).item())
        v = sum(vm) / len(vm)
        print(f"[dec ep {ep + 1}/{a.dec_epochs}] val_mse={v:.4f} t={time.time() - t0:.0f}s", flush=True)
        if v < best:
            best = v
            torch.save({"state_dict": dec.state_dict(), "val_mse": v, "run_name": a.run_name}, path)
    dec.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return dec.eval(), best


@torch.no_grad()
def imagine_episode(model, cfg, px, act_raw, device, seed=777):
    """px (k, 3, 224, 224) [0,1]; act_raw (k, fs*adim): block i produced frame i+1. Returns
    (z_gt (k, D), z_hat (k, D) with the first L = policy_history_len columns copied from GT)."""
    k = px.shape[0]
    fs = int(cfg["fs"]) if "fs" in cfg else int(cfg["frameskip"])
    adim = len(cfg["action_mean"])
    L = int(cfg["policy_history_len"])
    amean = torch.tensor(cfg["action_mean"], dtype=torch.float32, device=device)
    astd = torch.tensor(cfg["action_std"], dtype=torch.float32, device=device)
    with amp(device):
        z_gt = model.encode(norm(px.to(device)))                                  # (k, D)
    z_gt = z_gt.float()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hist = [z_gt[i] for i in range(L)]
    z_hat = [z_gt[i].clone() for i in range(L)]
    S, D = model.num_states, model.z_dim
    for i in range(L - 1, k - 1):
        blocks = act_raw[i].reshape(fs, adim).to(device)
        bz = ((blocks - amean) / astd).unsqueeze(0)                               # (1, fs, adim)
        h = torch.stack(hist[-L:], 0).unsqueeze(0)                                # (1, L, D)
        pad = torch.zeros(1, L, dtype=torch.bool, device=device)
        noise = torch.randn(1, S, D, generator=gen).to(device)                    # flow heads: fixed draw
        with amp(device):
            z_next = model.predict_state(h, pad, bz, noise_state=noise)
        z_next = z_next.float()[0, 0]
        z_hat.append(z_next)
        hist.append(z_next)                                                       # OPEN loop: imagined feeds back
    return z_gt, torch.stack(z_hat, 0)


def render(px, z_gt, z_hat, dec, cfg, run_name, ep, dec_mse, out_dir, device):
    k = px.shape[0]
    L = int(cfg["policy_history_len"])
    with torch.no_grad(), amp(device):
        rec_gt = dec(z_gt.to(device)).clamp(0, 1).float().cpu()
        rec_hat = dec(z_hat.to(device)).clamp(0, 1).float().cpu()
    rel = (z_hat - z_gt).norm(dim=-1) / max(z_gt.norm(dim=-1).mean().item(), 1e-8)
    cos = F.cosine_similarity(z_hat, z_gt, dim=-1)
    rows = [("GT frames", px.cpu()), ("decode(z_GT)", rec_gt), ("decode(z_imagined)", rec_hat)]
    fig, axes = plt.subplots(3, k, figsize=(k * 1.35, 3 * 1.55), squeeze=False)
    for r, (label, imgs) in enumerate(rows):
        for c in range(k):
            ax = axes[r][c]
            ax.imshow(imgs[c].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t+{c}·fs", fontsize=7)
            if r == 2:
                tag = "ctx" if c < L else f"{rel[c]:.2f}/{cos[c]:.2f}"
                ax.set_xlabel(tag, fontsize=6)
            if c == 0:
                ax.set_ylabel(label + ("" if r == 0 else f"\nval mse={dec_mse:.4f}" if r == 1 else "\nrel err/cos"),
                              fontsize=7)
    head = cfg.get("state_head", cfg.get("mot_state_head", "?"))
    fig.suptitle(f"{run_name} (state head {head}, w_reg {cfg.get('w_reg', '?')}) ep {ep}: "
                 f"open-loop imagination vs ground truth ({k - L} imagined blocks)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = Path(out_dir) / f"wmviz_{run_name}_ep{ep}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[viz] ep {ep}: mean rel err {rel[L:].mean():.3f}, mean cos {cos[L:].mean():.3f} -> {png}", flush=True)


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, cfg = gip.load_jointflow_model(a.run_name)
    model = model.eval().to(device)
    model.requires_grad_(False)
    a.fs = int(cfg.get("frameskip", cfg.get("fs", 5)))
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_dim = int(cfg["z_dim"])
    if a.dec_path and Path(a.dec_path).exists():
        dec = build_decoder(z_dim, device)
        ck = torch.load(a.dec_path, map_location="cpu")
        dec.load_state_dict(ck["state_dict"])
        dec.eval()
        dec_mse = float(ck.get("val_mse", float("nan")))
        print(f"[dec] reused {a.dec_path} (val mse {dec_mse:.4f})", flush=True)
    else:
        dec, dec_mse = train_decoder(model, a, z_dim, device, out_dir)
    ds = swm.data.load_dataset(a.h5, format="hdf5", keys_to_load=["pixels", "action"],
                               frameskip=a.fs, num_steps=a.k)
    for ep in [int(e) for e in a.eps.split(",") if e]:
        item = ds[min(ep, len(ds) - 1)]
        px = item["pixels"].float() / 255.0
        z_gt, z_hat = imagine_episode(model, cfg, px, item["action"].float(), device, seed=a.seed)
        render(px, z_gt.cpu(), z_hat.cpu(), dec, cfg, a.run_name, ep, dec_mse, out_dir, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", type=int, default=10, help="anchor frames per episode (fs raw steps apart)")
    p.add_argument("--eps", default="3,77,1234", help="comma window indices into the fs-strided dataset")
    p.add_argument("--dec_epochs", type=int, default=12)
    p.add_argument("--dec_max_samples", type=int, default=20000)
    p.add_argument("--dec_batch", type=int, default=32)
    p.add_argument("--dec_lr", type=float, default=3e-4)
    p.add_argument("--dec_path", default="", help="reuse a trained decoder_z_<run>.pt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=777)
    main(p.parse_args())
