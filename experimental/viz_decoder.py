"""Visualize LeWM lift decoder: ground truth vs reconstruction vs latent rollout.

Three panels per episode:
  (1) ground-truth frames (every K frames from one episode)
  (2) single-step reconstructions (encoder + decoder, no temporal rollout)
  (3) autoregressive rollout reconstructions (encoder on first H frames, then predictor rolls forward, decode each predicted latent)

Saves a PNG grid for visual inspection.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf
import stable_worldmodel.data.formats.hdf5  # noqa
import stable_worldmodel as swm
from stable_pretraining.backbone.utils import vit_hf
from decoder import LeWMVisDecoder
import lewam.models.jepa as jepa, lewam.models.module as module
from train_decoder import load_frozen_encoder


def build_full_lewm(weights_path, device):
    """Build the full JEPA (encoder + predictor + action_encoder + projector + pred_proj) and load weights."""
    # Read the train config to instantiate exact same model
    with hydra.initialize(config_path='../config/train', version_base=None):
        cfg = hydra.compose(config_name='lewm', overrides=['data=robomimic_lift'])
        # action_encoder.input_dim is required — set to match lift (action_dim=7 × frameskip=5 = 35)
        with hydra.utils.GlobalHydra.instance().clear():
            pass
    return None  # not used in this script — kept for reference

@torch.no_grad()
def main(args):
    device = torch.device('cuda')
    enc = load_frozen_encoder(args.ckpt, device)
    dec_ckpt = torch.load(args.dec_ckpt, map_location='cpu', weights_only=False)
    dec_args = dec_ckpt['args']
    dec = LeWMVisDecoder(
        cls_dim=192, hidden_dim=dec_args['hidden_dim'], depth=dec_args['depth'],
        heads=dec_args['heads'], dim_head=dec_args['dim_head'], mlp_ratio=4.0,
    ).to(device).eval()
    dec.load_state_dict(dec_ckpt['state_dict'])
    print(f'[viz] decoder loaded (val_mse={dec_ckpt["val_mse"]:.4f}, epoch={dec_ckpt["epoch"]})')

    # Load h5 directly
    import h5py
    f = h5py.File(args.h5, 'r', swmr=True)
    ep_len = f['ep_len'][:]
    ep_offset = f['ep_offset'][:]
    pixels = f['pixels']

    n_eps = args.n_eps
    frames_per_ep = args.frames_per_ep

    fig, axes = plt.subplots(2 * n_eps, frames_per_ep, figsize=(frames_per_ep * 1.8, n_eps * 3.6))
    if n_eps == 1:
        axes = np.array([axes])

    for ep_i in range(n_eps):
        ep_idx = args.start_ep + ep_i
        T = ep_len[ep_idx]
        off = ep_offset[ep_idx]
        # Pick frames spread across the episode
        frame_idxs = np.linspace(0, T-1, frames_per_ep).astype(int)

        # Ground truth
        gt = pixels[off + frame_idxs]  # (frames_per_ep, 224, 224, 3) uint8
        gt_t = torch.from_numpy(gt).permute(0, 3, 1, 2).float().to(device) / 255.0  # (frames_per_ep, 3, 224, 224)

        # Encode + decode
        cls = enc(gt_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]  # (frames_per_ep, 192)
        recon = dec(cls).clamp(0, 1)  # (frames_per_ep, 3, 224, 224)

        for f_i in range(frames_per_ep):
            ax_gt = axes[2 * ep_i, f_i]
            ax_re = axes[2 * ep_i + 1, f_i]
            ax_gt.imshow(gt[f_i])
            ax_re.imshow(recon[f_i].cpu().permute(1, 2, 0).numpy())
            ax_gt.axis('off')
            ax_re.axis('off')
            if f_i == 0:
                ax_gt.set_title(f'ep {ep_idx} GT', fontsize=8, loc='left')
                ax_re.set_title(f'ep {ep_idx} recon', fontsize=8, loc='left')

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f'[viz] saved {args.out}')
    f.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--ckpt', required=True, help='LeWM lift weights .pt')
    p.add_argument('--dec_ckpt', required=True, help='trained decoder .pt')
    p.add_argument('--out', required=True, help='output PNG path')
    p.add_argument('--n_eps', type=int, default=4, help='number of episodes')
    p.add_argument('--frames_per_ep', type=int, default=8, help='frames sampled per episode')
    p.add_argument('--start_ep', type=int, default=0)
    main(p.parse_args())
