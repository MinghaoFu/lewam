"""Visualize a MoT world model's imagined latents against ground truth (owner/Minghao 2026-08-29;
adapted from the owner's viz_train_decoder.py sample + this repo's train_decoder_unified.py).

Given k anchor frames of one episode (fs raw steps apart) and the (k-1) action blocks that
produced them, the dynamics imagines the episode AUTOREGRESSIVELY (open-loop): the history starts
from the first policy_history_len real latents, each block's clean actions go through
predict_state, and the imagined latent slides into the history -- real frames are never
re-encoded past the context. A small transformer decoder (LeWMVisDecoder), trained on this
checkpoint's own encoder latents AND (pred_decode_coef, as in the owner's sample) on
dynamics-rolled latents so imagined latents decode on-manifold, renders per episode:

    GT frames | decode(z_GT)  (decoder-quality reference) | decode(z_imagined)

with per-column relative latent error ||z_hat - z|| / mean||z|| and cosine similarity, plus
metrics.json: recon PSNR, imagined PSNR per step, latent MSE per imagined step and the
copy-last-latent baseline (does the dynamics beat freezing z?). Flow state heads are sampled
under a fixed seed (their imagination is a draw; MSE heads are deterministic).

  python viz_wm_dynamics.py --run_name <ckpt name under $STABLEWM_HOME/checkpoints> --h5 <dsn> \\
      --out_dir <dir> [--k 10 --eps 3,77,1234 --dec_epochs 20 --dec_max_samples 20000 \\
       --pred_decode_coef 1.0 --dec_window 4 --dec_path <pt> --video]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def psnr(a, b):
    mse = (a.clamp(0, 1) - b.clamp(0, 1)).pow(2).mean().item()
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def build_decoder(z_dim, device):
    return LeWMVisDecoder(cls_dim=z_dim, hidden_dim=256, depth=6, heads=8, dim_head=32,
                          mlp_ratio=4.0, img_size=224, patch_size=16, out_channels=3, dropout=0.0).to(device)


def z_score_blocks(act_raw, cfg, device):
    """(T, fs*adim) raw -> (T, fs, adim) z-scored."""
    adim = len(cfg["action_mean"])
    amean = torch.tensor(cfg["action_mean"], dtype=torch.float32, device=device)
    astd = torch.tensor(cfg["action_std"], dtype=torch.float32, device=device)
    b = act_raw.to(device).reshape(act_raw.shape[0], -1, adim)
    return (b - amean) / astd


@torch.no_grad()
def roll_latents(model, cfg, z_gt, blocks_z, L, gen=None):
    """Autoregressive imagination for a batch: z_gt (B, T, D) real latents, blocks_z
    (B, T-1, fs, adim) with block i producing frame i+1. History = the first L real latents;
    every predict_state output feeds the next step. Returns (B, T, D) with columns < L = GT."""
    B, T, D = z_gt.shape
    S = model.num_states
    hist = [z_gt[:, i] for i in range(L)]
    out = [z_gt[:, i] for i in range(L)]
    for i in range(L - 1, T - 1):
        h = torch.stack(hist[-L:], 1)
        pad = torch.zeros(B, L, dtype=torch.bool, device=z_gt.device)
        noise = None
        if gen is not None:
            noise = torch.randn(B, S, D, generator=gen).to(z_gt.device)
        z_next = model.predict_state(h, pad, blocks_z[:, i], noise_state=noise).float()[:, 0]
        out.append(z_next)
        hist.append(z_next)
    return torch.stack(out, 1)


def train_decoder(model, a, cfg, device, out_dir):
    """The owner's scheme: reconstruction on encoder latents + pred_decode_coef * reconstruction
    on dynamics-rolled latents (both detached -- only the decoder learns)."""
    L = int(cfg["policy_history_len"])
    W = max(a.dec_window, L + 1)
    ds = swm.data.load_dataset(a.h5, format="hdf5", keys_to_load=["pixels", "action"],
                               frameskip=a.fs, num_steps=W)
    if a.dec_max_samples and a.dec_max_samples < len(ds):
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[: a.dec_max_samples].tolist()
        ds = torch.utils.data.Subset(ds, idx)
    n = len(ds)
    n_val = max(64, n // 20)
    tr, va = random_split(ds, [n - n_val, n_val], generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr, batch_size=a.dec_batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=a.dec_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    z_dim = int(cfg["z_dim"])
    dec = build_decoder(z_dim, device)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.dec_lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, t0 = 1e9, time.time()
    path = out_dir / f"decoder_z_{a.run_name}.pt"
    print(f"[dec] window {W} (history {L}), pred_decode_coef {a.pred_decode_coef}, "
          f"train {n - n_val} val {n_val}", flush=True)

    def latents(batch):
        px = batch["pixels"].float().to(device) / 255.0                      # (B, W, 3, 224, 224)
        B, T = px.shape[:2]
        with torch.no_grad(), amp(device):
            z = model.encode(norm(px.reshape(B * T, *px.shape[2:]))).reshape(B, T, -1).float()
        rolled = None
        if a.pred_decode_coef > 0:
            blocks = z_score_blocks(batch["action"].float().reshape(-1, batch["action"].shape[-1]), cfg, device)
            blocks = blocks.reshape(B, T, *blocks.shape[1:])
            with torch.no_grad():
                rolled = roll_latents(model, cfg, z, blocks, L)
        return px, z, rolled

    for ep in range(a.dec_epochs):
        dec.train()
        for batch in tl:
            px, z, rolled = latents(batch)
            B, T = px.shape[:2]
            opt.zero_grad()
            with amp(device):
                recon = dec(z.reshape(B * T, -1)).reshape(B, T, 3, 224, 224)
                loss = F.mse_loss(recon, px)
                if rolled is not None:
                    im = dec(rolled[:, L:].reshape(B * (T - L), -1)).reshape(B, T - L, 3, 224, 224)
                    loss = loss + a.pred_decode_coef * F.mse_loss(im, px[:, L:])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        dec.eval()
        vm = []
        with torch.no_grad(), amp(device):
            for batch in vl:
                px, z, _ = latents({**batch, "action": batch["action"]})
                B, T = px.shape[:2]
                recon = dec(z.reshape(B * T, -1)).reshape(B, T, 3, 224, 224)
                vm.append(F.mse_loss(recon, px).item())
        v = sum(vm) / len(vm)
        print(f"[dec ep {ep + 1}/{a.dec_epochs}] val_mse={v:.4f} t={time.time() - t0:.0f}s", flush=True)
        if v < best:
            best = v
            torch.save({"state_dict": dec.state_dict(), "val_mse": v, "run_name": a.run_name}, path)
    dec.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return dec.eval(), best


def render(px, z_gt, z_hat, dec, cfg, run_name, ep, dec_mse, out_dir, device, video=False):
    k = px.shape[0]
    L = int(cfg["policy_history_len"])
    with torch.no_grad(), amp(device):
        rec_gt = dec(z_gt.to(device)).clamp(0, 1).float().cpu()
        rec_hat = dec(z_hat.to(device)).clamp(0, 1).float().cpu()
    rel = (z_hat - z_gt).norm(dim=-1) / max(z_gt.norm(dim=-1).mean().item(), 1e-8)
    cos = F.cosine_similarity(z_hat, z_gt, dim=-1)
    lat_mse = (z_hat[L:] - z_gt[L:]).pow(2).mean(-1)
    copy_mse = (z_gt[L - 1].unsqueeze(0).expand_as(z_gt[L:]) - z_gt[L:]).pow(2).mean(-1)
    m = {
        "episode": ep,
        "recon_psnr": float(np.mean([psnr(rec_gt[i], px[i]) for i in range(k)])),
        "imagined_psnr_per_step": [psnr(rec_hat[i], px[i]) for i in range(L, k)],
        "latent_mse_per_step": [round(float(v), 6) for v in lat_mse],
        "copy_last_mse_per_step": [round(float(v), 6) for v in copy_mse],
        "rel_err_per_step": [round(float(v), 4) for v in rel[L:]],
        "cos_per_step": [round(float(v), 4) for v in cos[L:]],
        "dec_val_mse": dec_mse,
    }
    print(f"[viz] ep {ep}: recon PSNR {m['recon_psnr']:.2f} dB; latent MSE per step "
          + " ".join(f"{v:.4f}" for v in lat_mse) + "; copy-last baseline "
          + " ".join(f"{v:.4f}" for v in copy_mse), flush=True)
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
                ax.set_xlabel("ctx" if c < L else f"{rel[c]:.2f}/{cos[c]:.2f}", fontsize=6)
            if c == 0:
                ax.set_ylabel(label + ("" if r == 0 else f"\nval mse={dec_mse:.4f}" if r == 1 else "\nrel err/cos"),
                              fontsize=7)
    head = cfg.get("state_head", cfg.get("mot_state_head", "?"))
    fig.suptitle(f"{run_name} (state head {head}, w_reg {cfg.get('w_reg', '?')}) ep {ep}: "
                 f"autoregressive imagination vs ground truth ({k - L} imagined blocks)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = Path(out_dir) / f"wmviz_{run_name}_ep{ep}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[viz] -> {png}", flush=True)
    if video:
        try:
            import imageio.v2 as imageio
            u8 = [(torch.cat([px[t], rec_gt[t], rec_hat[t]], dim=2).permute(1, 2, 0).numpy() * 255)
                  .round().astype(np.uint8) for t in range(k)]
            imageio.mimwrite(Path(out_dir) / f"wmviz_{run_name}_ep{ep}.mp4", u8, fps=2)
            print("[viz] mp4 written (GT | recon | imagined)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[viz] video skipped: {type(e).__name__}: {e}", flush=True)
    return m


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, cfg = gip.load_jointflow_model(a.run_name)
    model = model.eval().to(device)
    model.requires_grad_(False)
    a.fs = int(cfg.get("frameskip", cfg.get("fs", 5)))
    L = int(cfg["policy_history_len"])
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.dec_path and Path(a.dec_path).exists():
        dec = build_decoder(int(cfg["z_dim"]), device)
        ck = torch.load(a.dec_path, map_location="cpu")
        dec.load_state_dict(ck["state_dict"])
        dec.eval()
        dec_mse = float(ck.get("val_mse", float("nan")))
        print(f"[dec] reused {a.dec_path} (val mse {dec_mse:.4f})", flush=True)
    else:
        dec, dec_mse = train_decoder(model, a, cfg, device, out_dir)
    ds = swm.data.load_dataset(a.h5, format="hdf5", keys_to_load=["pixels", "action"],
                               frameskip=a.fs, num_steps=a.k)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    metrics = []
    for ep in [int(e) for e in a.eps.split(",") if e]:
        item = ds[min(ep, len(ds) - 1)]
        px = item["pixels"].float() / 255.0
        with amp(device):
            z_gt = model.encode(norm(px.to(device))).float()
        blocks = z_score_blocks(item["action"].float(), cfg, device).unsqueeze(0)
        z_hat = roll_latents(model, cfg, z_gt.unsqueeze(0), blocks, L, gen=gen)[0]
        metrics.append(render(px, z_gt.cpu(), z_hat.cpu(), dec, cfg, a.run_name, ep, dec_mse,
                              out_dir, device, video=a.video))
    json.dump({"run_name": a.run_name, "k": a.k, "history": L, "episodes": metrics},
              open(out_dir / f"metrics_{a.run_name}.json", "w"), indent=1)
    print(f"[done] metrics_{a.run_name}.json -> {out_dir}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", type=int, default=10, help="anchor frames per episode (fs raw steps apart)")
    p.add_argument("--eps", default="3,77,1234", help="comma window indices into the fs-strided dataset")
    p.add_argument("--dec_epochs", type=int, default=20)
    p.add_argument("--dec_max_samples", type=int, default=20000)
    p.add_argument("--dec_batch", type=int, default=16)
    p.add_argument("--dec_lr", type=float, default=3e-4)
    p.add_argument("--dec_window", type=int, default=4, help="decoder-training window (history + rolled frames)")
    p.add_argument("--pred_decode_coef", type=float, default=1.0,
                   help="also fit the decoder on dynamics-rolled latents (0 disables; owner's sample)")
    p.add_argument("--dec_path", default="", help="reuse a trained decoder_z_<run>.pt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--video", action="store_true", help="also write an mp4 per episode (GT | recon | imagined)")
    main(p.parse_args())
