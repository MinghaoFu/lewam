"""Generate paper-faithful GT vs encoder→decoder reconstruction videos using v3 CLS decoder.

Two outputs per env:
  - <env>_recon.mp4   — episode replay, GT | decoder(encode(GT_t)) side by side
  - <env>_progression.mp4  — paper Fig 8 style: GT plus decoder snapshots at 10K/20K/40K/100K/200K steps

Plus an HTML index page.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import h5py
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro')
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro/scripts')
import stable_worldmodel.data.formats.hdf5  # noqa
from stable_pretraining.backbone.utils import vit_hf
from decoder import LeWMVisDecoder


def load_encoder(weights_path, device):
    enc = vit_hf(size='tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    sd = torch.load(weights_path, map_location='cpu', weights_only=False)
    enc_sd = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
    enc.load_state_dict(enc_sd, strict=True)
    enc.eval().to(device)
    for p in enc.parameters(): p.requires_grad = False
    return enc


def load_decoder(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck['args']
    dec = LeWMVisDecoder(cls_dim=192, hidden_dim=a['hidden_dim'], depth=a['depth'],
                        heads=a['heads'], dim_head=a['dim_head'], mlp_ratio=4.0,
                        img_size=224, patch_size=16, out_channels=3).to(device).eval()
    dec.load_state_dict(ck['state_dict'])
    return dec, ck.get('val_mse', None), ck.get('step', a.get('max_steps', None))


@torch.no_grad()
def render_recon_video(enc, dec, h5_path, out_path, ep_idx, fps=4, frame_stride=2):
    f = h5py.File(h5_path, 'r', swmr=True)
    ep_len = f['ep_len'][:]
    ep_offset = f['ep_offset'][:]
    T = int(ep_len[ep_idx]); off = int(ep_offset[ep_idx])
    idxs = np.arange(0, T, frame_stride)
    pix = f['pixels'][off + idxs]  # (T', 224, 224, 3) uint8
    f.close()

    device = next(enc.parameters()).device
    pix_t = torch.from_numpy(pix).permute(0, 3, 1, 2).float().to(device) / 255.0
    cls = enc(pix_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    recon = dec(cls).clamp(0, 1)
    recon_np = (recon.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    H_img, W_img = 224, 224
    label_h = 20
    pad = 4
    out_w = W_img * 2 + pad
    out_h = H_img + label_h

    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=9,
                                ffmpeg_log_level='error', macro_block_size=1)
    for t in range(len(idxs)):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        canvas[label_h:, 0:W_img] = pix[t]
        canvas[label_h:, W_img + pad:] = recon_np[t]
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try: font = ImageFont.load_default()
        except: font = None
        d.text((4, 3), f'GT', fill=(0, 0, 0), font=font)
        d.text((W_img + pad + 4, 3), f'encoder→decoder (frame {int(idxs[t])}/{T-1})', fill=(0, 0, 0), font=font)
        writer.append_data(np.array(im))
    writer.close()


@torch.no_grad()
def render_progression_video(enc, ckpt_template_dir, h5_path, out_path, ep_idx, fps=2, frame_stride=4):
    """Paper Fig 8 style: GT + decoder@10K/20K/40K/100K/200K snapshots, one row per snapshot."""
    snapshot_steps = [10_000, 20_000, 40_000, 100_000, 200_000]
    device = next(enc.parameters()).device
    decoders = {}
    for s in snapshot_steps:
        p = Path(ckpt_template_dir) / f'step_{s:07d}.pt'
        if p.exists():
            d, _, _ = load_decoder(str(p), device)
            decoders[s] = d

    f = h5py.File(h5_path, 'r', swmr=True)
    T = int(f['ep_len'][ep_idx]); off = int(f['ep_offset'][ep_idx])
    idxs = np.arange(0, T, frame_stride)
    pix = f['pixels'][off + idxs]
    f.close()

    pix_t = torch.from_numpy(pix).permute(0, 3, 1, 2).float().to(device) / 255.0
    cls = enc(pix_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]

    recons = {}
    for s, d in decoders.items():
        recons[s] = (d(cls).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    H_img, W_img = 224, 224
    label_h = 20
    pad = 4
    cols = 1 + len(decoders)  # GT + each snapshot
    out_w = W_img * cols + pad * (cols - 1)
    out_h = H_img + label_h

    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=9,
                                ffmpeg_log_level='error', macro_block_size=1)
    for t in range(len(idxs)):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        canvas[label_h:, 0:W_img] = pix[t]
        for i, s in enumerate(sorted(decoders.keys())):
            x0 = W_img * (i + 1) + pad * (i + 1)
            canvas[label_h:, x0:x0 + W_img] = recons[s][t]
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try: font = ImageFont.load_default()
        except: font = None
        d.text((4, 3), 'GT', fill=(0, 0, 0), font=font)
        for i, s in enumerate(sorted(decoders.keys())):
            x0 = W_img * (i + 1) + pad * (i + 1)
            d.text((x0 + 4, 3), f'{s//1000}k steps', fill=(0, 0, 0), font=font)
        writer.append_data(np.array(im))
    writer.close()


def main(args):
    device = torch.device('cuda')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    SWMHOME = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')

    envs = args.envs
    cards = []
    for env in envs:
        h5 = SWMHOME / args.h5_map[env]
        weights = SWMHOME / 'decoders' / f'{env}_lewm_weights.pt'
        decoder = SWMHOME / 'decoders' / f'{env}_cls_decoder_v3.pt'
        snap_dir = SWMHOME / 'decoders' / f'{env}_cls_decoder_v3'

        if not (h5.exists() and weights.exists() and decoder.exists()):
            print(f'[viz] skip {env}: missing files'); continue

        enc = load_encoder(str(weights), device)
        dec, val_mse, step = load_decoder(str(decoder), device)
        recon_path = out_dir / f'{env}_recon.mp4'
        render_recon_video(enc, dec, str(h5), recon_path, ep_idx=args.ep_idx_map.get(env, 0),
                           fps=args.fps, frame_stride=args.frame_stride)
        print(f'[viz] {env}: recon → {recon_path.name}  (val_mse={val_mse:.5f})')

        prog_path = out_dir / f'{env}_progression.mp4'
        if snap_dir.exists():
            render_progression_video(enc, str(snap_dir), str(h5), prog_path,
                                     ep_idx=args.ep_idx_map.get(env, 0), fps=args.fps,
                                     frame_stride=args.frame_stride * 2)
            print(f'[viz] {env}: progression → {prog_path.name}')
        cards.append((env, val_mse, recon_path.name, prog_path.name if snap_dir.exists() else None))

    # HTML index
    html_path = out_dir / 'index.html'
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LeWM v3 decoders — paper-faithful 200K-step reconstruction</title>
<style>
  body { font-family: -apple-system,BlinkMacSystemFont,sans-serif; background:#0d1117; color:#c9d1d9; padding:24px; margin:0; }
  h1 { color:#58a6ff; margin:0 0 4px; }
  .sub { color:#8b949e; margin-bottom:24px; font-size:13px; }
  h2 { color:#58a6ff; font-size:14px; text-transform:uppercase; letter-spacing:.04em; margin:32px 0 12px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:24px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }
  video { width:100%; image-rendering:pixelated; background:#000; }
  .legend { font-size:13px; color:#c9d1d9; padding:0 0 8px; }
  .stats { font-size:11px; color:#8b949e; font-family:ui-monospace,monospace; padding:4px 0 10px; }
  code { background:rgba(255,255,255,.08); padding:1px 6px; border-radius:3px; font-size:12px; }
</style></head><body>
<h1>LeWM v3 decoders — 200K steps (paper-faithful, App. D + Fig 8)</h1>
<div class="sub">
  Raw CLS (192-D) → cross-attention transformer decoder (32M params, hidden 512, depth 10) trained 200K steps with MSE in [0,1] pixel space. <br>
  Each env's encoder comes from its specific Lightning ckpt dir (no shared-file overwrites). Snapshots saved at 10k/20k/40k/100k/200k matching paper Fig 8.
</div>

<h2>Reconstruction: GT vs encoder→decoder (per env)</h2>
<div class="grid">
"""
    for env, val_mse, recon, _ in cards:
        html += f'<div class="card"><div class="legend">{env}</div><div class="stats">best val_mse = <code>{val_mse:.5f}</code></div><video src="{recon}" controls autoplay loop muted></video></div>\n'

    html += '</div><h2>Decoder progression (Fig 8 reproduction): GT | 10k | 20k | 40k | 100k | 200k steps</h2><div class="grid">\n'
    for env, val_mse, _, prog in cards:
        if prog is None: continue
        html += f'<div class="card"><div class="legend">{env}</div><div class="stats">left → right: ground truth, then decoder snapshots at 10k, 20k, 40k, 100k, 200k training steps</div><video src="{prog}" controls autoplay loop muted></video></div>\n'

    html += '</div></body></html>'
    html_path.write_text(html)
    print(f'[viz] {html_path}')


H5_MAP = {
    'lift': 'lift.h5', 'can': 'can.h5', 'square': 'square.h5',
    'rope': 'rope.h5', 'granular': 'granular.h5',
    'tworoom': 'tworoom.h5', 'pusht': 'pusht_expert_train.h5', 'dmc': 'reacher.h5',
}
EP_IDX = {'lift': 7, 'can': 7, 'square': 7, 'rope': 23, 'granular': 23,
          'tworoom': 5, 'pusht': 0, 'dmc': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--envs', nargs='+', default=['lift', 'can', 'square', 'rope', 'granular'])
    p.add_argument('--ep_idx', type=int, default=None)
    p.add_argument('--fps', type=int, default=4)
    p.add_argument('--frame_stride', type=int, default=2)
    args = p.parse_args()
    args.h5_map = H5_MAP
    args.ep_idx_map = EP_IDX if args.ep_idx is None else {e: args.ep_idx for e in args.envs}
    main(args)
