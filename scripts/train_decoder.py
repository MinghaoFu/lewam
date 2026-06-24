"""Train the LeWM visualization decoder on frozen lift encoder.

Loads workspace/le-wm-repro/.stable-wm/checkpoints/lewm/weights_epoch_99.pt (lift)
plus the same dataset config used in training (lift.h5), then trains the
decoder against ground-truth pixels with MSE loss.

Saves to <STABLEWM_HOME>/decoders/lewm_lift_vis_decoder.pt.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Make local repo importable
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro')
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro/scripts')
import stable_worldmodel.data.formats.hdf5  # noqa
import stable_worldmodel as swm
from stable_pretraining.backbone.utils import vit_hf

from decoder import LeWMVisDecoder


def load_frozen_encoder(weights_path: str, device):
    """Build a stable-pretraining ViT-tiny matching LeWM training, load weights, freeze."""
    enc = vit_hf(size='tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    sd_full = torch.load(weights_path, map_location='cpu', weights_only=False)
    enc_sd = {k[len('encoder.'):]: v for k, v in sd_full.items() if k.startswith('encoder.')}
    enc.load_state_dict(enc_sd, strict=True)
    print(f'[encoder load] OK ({len(enc_sd)} keys)')
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def main(args):
    device = torch.device('cuda')
    torch.manual_seed(0)
    print(f'[train_decoder] device={device}, h5={args.h5}, ckpt={args.ckpt}')

    enc = load_frozen_encoder(args.ckpt, device)

    # Quick forward check: 1 random image → encoder → CLS shape
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        out = enc(dummy, interpolate_pos_encoding=True)
        cls_dim = out.last_hidden_state[:, 0].shape[-1]
        print(f'[encoder check] CLS dim={cls_dim}')

    # Decoder
    dec = LeWMVisDecoder(
        cls_dim=cls_dim, hidden_dim=args.hidden_dim, depth=args.depth,
        heads=args.heads, dim_head=args.dim_head, mlp_ratio=4.0,
        img_size=224, patch_size=16, out_channels=3, dropout=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f'[decoder] params: {n_params/1e6:.2f} M')

    # Dataset — frameskip 1, num_steps 1 (decode per-frame; no temporal needed)
    ds = swm.data.load_dataset(
        args.h5, format='hdf5', keys_to_load=['pixels'],
        frameskip=1, num_steps=1,
    )
    n = len(ds)
    n_val = max(64, n // 20)
    n_train = n - n_val
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    print(f'[dataset] total={n}, train={n_train}, val={n_val}')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    best_val = float('inf')
    for epoch in range(args.epochs):
        dec.train()
        losses = []
        for batch in train_loader:
            pixels = batch['pixels'].to(device, non_blocking=True).float() / 255.0  # (B, T=1, 3, 224, 224)
            pixels = pixels.squeeze(1)  # (B, 3, 224, 224)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    cls = enc(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]  # (B, 192)
                recon = dec(cls)  # (B, 3, 224, 224)
                loss = F.mse_loss(recon, pixels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        # Val
        dec.eval()
        vlosses = []
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in val_loader:
                pixels = batch['pixels'].to(device, non_blocking=True).float() / 255.0
                pixels = pixels.squeeze(1)
                cls = enc(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                recon = dec(cls)
                vlosses.append(F.mse_loss(recon, pixels).item())
        train_loss = sum(losses) / len(losses)
        val_loss = sum(vlosses) / len(vlosses)
        elapsed = time.time() - t0
        print(f'[ep {epoch+1}/{args.epochs}] train_mse={train_loss:.4f} val_mse={val_loss:.4f} t={elapsed:.1f}s', flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'state_dict': dec.state_dict(), 'args': vars(args), 'val_mse': val_loss, 'epoch': epoch+1}, args.out)

    print(f'[done] best val_mse={best_val:.4f}, saved to {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dim_head', type=int, default=32)
    main(p.parse_args())
