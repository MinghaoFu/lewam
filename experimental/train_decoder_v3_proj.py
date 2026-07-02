"""v3 decoder trained on projector(CLS) instead of raw CLS — for rollout decoding.

Same architecture + 200K steps as v3 raw-CLS decoder. The only difference is the
training-time encoder pathway: pixels → encoder → CLS → projector → fed to decoder.
This way, the decoder sees inputs from the SAME distribution as predictor outputs.
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
from omegaconf import OmegaConf
import lewam.models.jepa as jepa, lewam.models.module as module
from decoder import LeWMVisDecoder


def build_jepa(weights, action_input_dim, device):
    cfg_str = f"""
_target_: jepa.JEPA
encoder: {{_target_: stable_pretraining.backbone.utils.vit_hf, size: tiny, patch_size: 14, image_size: 224, pretrained: false, use_mask_token: false}}
predictor: {{_target_: module.ARPredictor, num_frames: 3, input_dim: 192, hidden_dim: 192, output_dim: 192, depth: 6, heads: 16, mlp_dim: 2048, dim_head: 64, dropout: 0.1, emb_dropout: 0.0}}
action_encoder: {{_target_: module.Embedder, input_dim: {action_input_dim}, emb_dim: 192}}
projector: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
pred_proj: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
"""
    cfg = OmegaConf.create(cfg_str)
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(weights, map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad = False
    return model


def main(args):
    device = torch.device('cuda')
    torch.manual_seed(0)
    print(f'[v3-proj] {args.h5}, weights={args.weights}, target_steps={args.max_steps}', flush=True)

    jepa_model = build_jepa(args.weights, args.action_input_dim, device)
    dec = LeWMVisDecoder(cls_dim=192, hidden_dim=512, depth=10, heads=8, dim_head=64,
                        mlp_ratio=4.0, img_size=224, patch_size=16, out_channels=3).to(device)
    print(f'[decoder] params: {sum(p.numel() for p in dec.parameters() if p.requires_grad)/1e6:.2f} M', flush=True)

    ds = swm.data.load_dataset(args.h5, format='hdf5', keys_to_load=['pixels'], frameskip=1, num_steps=1)
    n_val = max(64, len(ds) // 20)
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    opt = torch.optim.AdamW(dec.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=3e-6)
    scaler = torch.amp.GradScaler('cuda')

    snap_dir = Path(args.out).parent / Path(args.out).stem
    snap_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    snapshot_steps = [10_000, 20_000, 40_000, 100_000, 200_000]
    next_snap_idx = 0

    step = epoch = 0
    best_val = float('inf')
    t0 = time.time()
    while step < args.max_steps:
        dec.train()
        ep_losses = []
        for batch in train_loader:
            if step >= args.max_steps: break
            pixels = batch['pixels'].to(device, non_blocking=True).float() / 255.0
            pixels = pixels.squeeze(1)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    cls = jepa_model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                    proj = jepa_model.projector(cls)  # projector-space input
                recon = dec(proj)
                loss = F.mse_loss(recon, pixels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ep_losses.append(loss.item())
            step += 1
            if next_snap_idx < len(snapshot_steps) and step >= snapshot_steps[next_snap_idx]:
                torch.save({'state_dict': dec.state_dict(), 'args': {'hidden_dim':512,'depth':10,'heads':8,'dim_head':64}, 'step': step},
                           snap_dir / f'step_{snapshot_steps[next_snap_idx]:07d}.pt')
                next_snap_idx += 1
        epoch += 1
        dec.eval()
        vlosses = []
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in val_loader:
                pixels = batch['pixels'].to(device, non_blocking=True).float() / 255.0
                pixels = pixels.squeeze(1)
                cls = jepa_model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                proj = jepa_model.projector(cls)
                vlosses.append(F.mse_loss(dec(proj), pixels).item())
        tl = sum(ep_losses)/max(1,len(ep_losses)); vl = sum(vlosses)/max(1,len(vlosses))
        rate = step/(time.time()-t0)
        print(f'[ep {epoch:>4d} step {step:>6d}/{args.max_steps}] train_mse={tl:.5f} val_mse={vl:.5f}  ({rate:.1f} step/s)', flush=True)
        if vl < best_val:
            best_val = vl
            torch.save({'state_dict': dec.state_dict(), 'args': {'hidden_dim':512,'depth':10,'heads':8,'dim_head':64}, 'val_mse': vl, 'step': step, 'epoch': epoch}, args.out)
    print(f'[done] best val_mse={best_val:.5f}, saved to {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--action_input_dim', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max_steps', type=int, default=200_000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    main(p.parse_args())
