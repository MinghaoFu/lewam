"""Train LeWM visualization decoder on projector-output embeddings (true predictor space).

Loads the full JEPA model (encoder + projector + predictor + action_encoder + pred_proj)
from a clean lift weights file. Decoder takes the *projected* embedding (which is what
both prediction targets and predictor outputs live in) → 224x224 RGB.

Saves to <STABLEWM_HOME>/decoders/<name>_vis_decoder.pt.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import stable_worldmodel.data.formats.hdf5  # noqa
import stable_worldmodel as swm
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from decoder import LeWMVisDecoder
import lewam.models.jepa as jepa, lewam.models.module as module  # JEPA imports

EMBED_DIM = 192  # ViT-tiny + lewm.yaml setting


def build_jepa(lift_weights: str, action_input_dim: int, device):
    """Build a JEPA matching config/train/model/lewm.yaml and load lift weights."""
    cfg_str = f"""
_target_: jepa.JEPA
encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: 224
  pretrained: false
  use_mask_token: false
predictor:
  _target_: module.ARPredictor
  num_frames: 3
  input_dim: {EMBED_DIM}
  hidden_dim: {EMBED_DIM}
  output_dim: {EMBED_DIM}
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0
action_encoder:
  _target_: module.Embedder
  input_dim: {action_input_dim}
  emb_dim: {EMBED_DIM}
projector:
  _target_: module.MLP
  input_dim: {EMBED_DIM}
  output_dim: {EMBED_DIM}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
pred_proj:
  _target_: module.MLP
  input_dim: {EMBED_DIM}
  output_dim: {EMBED_DIM}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
"""
    cfg = OmegaConf.create(cfg_str)
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(lift_weights, map_location='cpu', weights_only=False)
    res = model.load_state_dict(sd, strict=False)
    print(f'[jepa load] strict=False missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    if res.missing_keys[:3]: print('  missing:', res.missing_keys[:3])
    if res.unexpected_keys[:3]: print('  unexpected:', res.unexpected_keys[:3])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def encode_projected(model, pixels):
    """pixels: (B, 3, 224, 224) float in [0,1]. Returns (B, EMBED_DIM) — after projector."""
    # encoder.last_hidden_state[:, 0] = CLS token
    cls = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    # JEPA.encode internally does: emb = projector(pixels_emb) → reshape (B,T,D)
    # We do the same path manually
    proj = model.projector(cls)
    return proj


def main(args):
    device = torch.device('cuda')
    torch.manual_seed(0)
    print(f'[train_decoder_v2] device={device}, h5={args.h5}, weights={args.weights}')

    jepa_model = build_jepa(args.weights, action_input_dim=args.action_input_dim, device=device)

    dec = LeWMVisDecoder(
        cls_dim=EMBED_DIM, hidden_dim=args.hidden_dim, depth=args.depth,
        heads=args.heads, dim_head=args.dim_head, mlp_ratio=4.0,
        img_size=224, patch_size=16, out_channels=3, dropout=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f'[decoder] params: {n_params/1e6:.2f} M')

    ds = swm.data.load_dataset(
        args.h5, format='hdf5', keys_to_load=['pixels'],
        frameskip=1, num_steps=1,
    )
    n_val = max(64, len(ds) // 20)
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    print(f'[dataset] total={len(ds)}, train={len(train_ds)}, val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_val = float('inf')
    for epoch in range(args.epochs):
        dec.train()
        losses = []
        for batch in train_loader:
            pixels = batch['pixels'].to(device, non_blocking=True).float() / 255.0
            pixels = pixels.squeeze(1)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    proj = encode_projected(jepa_model, pixels)  # (B, EMBED_DIM)
                recon = dec(proj)
                loss = F.mse_loss(recon, pixels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        dec.eval()
        vlosses = []
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in val_loader:
                pixels = batch['pixels'].to(device, non_blocking=True).float() / 255.0
                pixels = pixels.squeeze(1)
                proj = encode_projected(jepa_model, pixels)
                recon = dec(proj)
                vlosses.append(F.mse_loss(recon, pixels).item())
        train_loss = sum(losses) / len(losses)
        val_loss = sum(vlosses) / len(vlosses)
        print(f'[ep {epoch+1}/{args.epochs}] train_mse={train_loss:.4f} val_mse={val_loss:.4f} t={time.time()-t0:.1f}s', flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'state_dict': dec.state_dict(), 'args': vars(args), 'val_mse': val_loss, 'epoch': epoch+1}, args.out)
    print(f'[done] best val_mse={best_val:.4f}, saved to {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--weights', required=True, help='clean lift weights (encoder + projector + …)')
    p.add_argument('--action_input_dim', type=int, default=35, help='frameskip * action_dim (lift: 5*7=35)')
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--hidden_dim', type=int, default=384)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dim_head', type=int, default=48)
    main(p.parse_args())
