"""Re-render the per-env reconstruction videos using the explicit 200K snapshot
(matching the rightmost column of the progression videos)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro')
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro/scripts')

import torch, numpy as np, h5py, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import stable_worldmodel.data.formats.hdf5  # noqa
from stable_pretraining.backbone.utils import vit_hf
from decoder import LeWMVisDecoder

SWMHOME = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')
ENVS_H5 = {'lift': 'lift.h5', 'can': 'can.h5', 'square': 'square.h5',
           'rope': 'rope.h5', 'granular': 'granular.h5'}
EP_IDX = {'lift': 7, 'can': 7, 'square': 7, 'rope': 23, 'granular': 23}


def load_encoder(weights_path, device):
    enc = vit_hf(size='tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    sd = torch.load(weights_path, map_location='cpu', weights_only=False)
    enc.load_state_dict({k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}, strict=True)
    enc.eval().to(device)
    for p in enc.parameters(): p.requires_grad = False
    return enc


def load_dec(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck['args']
    dec = LeWMVisDecoder(cls_dim=192, hidden_dim=a['hidden_dim'], depth=a['depth'],
                        heads=a['heads'], dim_head=a['dim_head'], mlp_ratio=4.0,
                        img_size=224, patch_size=16, out_channels=3).to(device).eval()
    dec.load_state_dict(ck['state_dict'])
    return dec, ck.get('step', 'n/a'), ck.get('val_mse', None)


@torch.no_grad()
def render(env, out_dir, fps=4, stride=2):
    device = torch.device('cuda')
    enc = load_encoder(SWMHOME / 'decoders' / f'{env}_lewm_weights.pt', device)
    # Explicit 200K snapshot
    dec, step, val_mse = load_dec(SWMHOME / 'decoders' / f'{env}_cls_decoder_v3' / 'step_0200000.pt', device)
    print(f'[{env}] using step={step} val_mse={val_mse}')

    f = h5py.File(SWMHOME / ENVS_H5[env], 'r', swmr=True)
    ep = EP_IDX[env]
    T = int(f['ep_len'][ep]); off = int(f['ep_offset'][ep])
    idxs = np.arange(0, T, stride)
    pix = f['pixels'][off + idxs]
    f.close()
    pix_t = torch.from_numpy(pix).permute(0, 3, 1, 2).float().to(device) / 255.0
    cls = enc(pix_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    recon_np = (dec(cls).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    H_img = W_img = 224; lab = 20; pad = 4
    out_w, out_h = W_img * 2 + pad, H_img + lab
    out = Path(out_dir) / f'{env}_recon.mp4'
    w = imageio.get_writer(str(out), fps=fps, codec='libx264', quality=9,
                           ffmpeg_log_level='error', macro_block_size=1)
    for t in range(len(idxs)):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        canvas[lab:, 0:W_img] = pix[t]
        canvas[lab:, W_img + pad:] = recon_np[t]
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try: font = ImageFont.load_default()
        except: font = None
        d.text((4, 3), 'GT', fill=(0,0,0), font=font)
        d.text((W_img + pad + 4, 3), f'encoder→decoder @ 200K (frame {int(idxs[t])}/{T-1})', fill=(0,0,0), font=font)
        w.append_data(np.array(im))
    w.close()
    print(f'  → {out}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--envs', nargs='+', default=list(ENVS_H5.keys()))
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for env in args.envs:
        render(env, args.out_dir)
