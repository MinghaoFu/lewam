"""LeWM visualization decoder v3 — paper-faithful (App. D + Fig 8).

Differences from v2:
  - Decodes RAW CLS (encoder.last_hidden_state[:, 0]) — NOT projector(CLS).
    Per Sec 5.1: "decoder trained to reconstruct pixel observations from a
    single latent embedding (192 dim)". Per App. D: "decode the [CLS] token
    embedding (192 dim) from the last encoder layer".
  - Trains for --max_steps (default 200,000) to match Fig 8.
  - Bigger decoder: hidden=512, depth=10.
  - Saves intermediate ckpts at fixed step thresholds (for Fig 8-style progression).
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro')
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro/scripts')
import stable_worldmodel.data.formats.hdf5  # noqa
import stable_worldmodel as swm
from stable_pretraining.backbone.utils import vit_hf

from decoder import LeWMVisDecoder


def load_frozen_encoder(weights_path: str, device):
    enc = vit_hf(size='tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    sd_full = torch.load(weights_path, map_location='cpu', weights_only=False)
    enc_sd = {k[len('encoder.'):]: v for k, v in sd_full.items() if k.startswith('encoder.')}
    enc.load_state_dict(enc_sd, strict=True)
    print(f'[encoder load] OK ({len(enc_sd)} keys) from {weights_path}', flush=True)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def main(args):
    device = torch.device('cuda')
    torch.manual_seed(0)
    print(f'[v3] device={device}, h5={args.h5}, weights={args.weights}, target_steps={args.max_steps}', flush=True)

    enc = load_frozen_encoder(args.weights, device)
    dec = LeWMVisDecoder(
        cls_dim=192, hidden_dim=args.hidden_dim, depth=args.depth,
        heads=args.heads, dim_head=args.dim_head, mlp_ratio=4.0,
        img_size=224, patch_size=16, out_channels=3, dropout=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f'[decoder] params: {n_params/1e6:.2f} M  (hidden={args.hidden_dim}, depth={args.depth})', flush=True)

    ds = swm.data.load_dataset(
        args.h5, format='hdf5', keys_to_load=['pixels'],
        frameskip=1, num_steps=1,
    )
    n_val = max(64, len(ds) // 20)
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    print(f'[dataset] total={len(ds)}, train={len(train_ds)}, val={len(val_ds)}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler('cuda')

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = out_dir / Path(args.out).stem
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot step thresholds (paper Fig 8 milestones)
    snapshot_steps = [10_000, 20_000, 40_000, 100_000, 200_000]
    next_snap_idx = 0

    step = 0
    best_val = float('inf')
    epoch = 0
    t0 = time.time()
    epoch_train_losses = []

    while step < args.max_steps:
        dec.train()
        for batch in train_loader:
            if step >= args.max_steps: break
            pixels = batch['pixels'].to(device, non_blocking=True).float() / 255.0
            pixels = pixels.squeeze(1)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    cls = enc(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                recon = dec(cls)
                loss = F.mse_loss(recon, pixels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            epoch_train_losses.append(loss.item())
            step += 1

            # Snapshot at milestones
            if next_snap_idx < len(snapshot_steps) and step >= snapshot_steps[next_snap_idx]:
                snap_path = snap_dir / f'step_{snapshot_steps[next_snap_idx]:07d}.pt'
                torch.save({'state_dict': dec.state_dict(), 'args': vars(args), 'step': step}, snap_path)
                print(f'[snap] step={step} saved {snap_path}', flush=True)
                next_snap_idx += 1

        # End of epoch — validate
        epoch += 1
        dec.eval()
        vlosses = []
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in val_loader:
                pixels = batch['pixels'].to(device, non_blocking=True).float() / 255.0
                pixels = pixels.squeeze(1)
                cls = enc(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                recon = dec(cls)
                vlosses.append(F.mse_loss(recon, pixels).item())
        train_loss = sum(epoch_train_losses) / max(1, len(epoch_train_losses))
        val_loss = sum(vlosses) / max(1, len(vlosses))
        elapsed = time.time() - t0
        rate = step / elapsed if elapsed > 0 else 0
        print(f'[ep {epoch:>4d} step {step:>6d}/{args.max_steps}] train_mse={train_loss:.5f} val_mse={val_loss:.5f}  t={elapsed:.0f}s ({rate:.1f} step/s)', flush=True)
        epoch_train_losses = []
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'state_dict': dec.state_dict(), 'args': vars(args), 'val_mse': val_loss, 'step': step, 'epoch': epoch}, args.out)

    print(f'[done] best val_mse={best_val:.5f}, saved to {args.out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max_steps', type=int, default=200_000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--depth', type=int, default=10)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dim_head', type=int, default=64)
    main(p.parse_args())
