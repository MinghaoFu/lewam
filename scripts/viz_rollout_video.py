"""Build LeWM autoregressive rollout videos for visualization.

For each episode:
  - encode first H=3 frames → projector → ctx_emb (in predictor space)
  - autoregressively roll predictor forward K steps (using GT actions per step)
  - decode every embedding (H real + K predicted) → 224x224 RGB
  - render alongside ground-truth pixels as side-by-side mp4

Outputs an HTML report embedding the mp4s.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import h5py
import torch
import imageio.v2 as imageio

sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro')
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro/scripts')
import stable_worldmodel.data.formats.hdf5  # noqa
import stable_worldmodel as swm

from decoder import LeWMVisDecoder
from train_decoder_v2 import build_jepa, encode_projected, EMBED_DIM

H = 3  # num_hist (matches LeWM lewm.yaml)
FRAMESKIP = 5


def stack_actions(actions_flat: np.ndarray, frameskip: int, t_start: int, num_steps: int) -> np.ndarray:
    """Build (num_steps, action_dim * frameskip) by stacking frameskip consecutive raw actions per step."""
    action_dim = actions_flat.shape[1]
    out = np.zeros((num_steps, action_dim * frameskip), dtype=actions_flat.dtype)
    for i in range(num_steps):
        for k in range(frameskip):
            idx = t_start + i * frameskip + k
            if idx < actions_flat.shape[0]:
                out[i, k * action_dim:(k + 1) * action_dim] = actions_flat[idx]
    return out


@torch.no_grad()
def rollout_episode(jepa_model, decoder, pixels, actions, K, device):
    """Run encoder(H frames) + predictor(K steps) + decoder; return (H+K, 3, 224, 224) decoded + GT."""
    # pixels: (T_total, 224, 224, 3) uint8 numpy
    # actions: (T_total, action_dim_stacked) float32 (already frameskip-stacked)
    T = H + K
    pix_t = torch.from_numpy(pixels[:T]).permute(0, 3, 1, 2).float().to(device) / 255.0  # (T, 3, 224, 224)
    acts_t = torch.from_numpy(actions[:T]).float().to(device)  # (T, action_dim_stacked)

    # Encode ALL frames once (we'll use first H as context, others as comparison)
    cls = jepa_model.encoder(pix_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]  # (T, 192)
    proj_all = jepa_model.projector(cls)  # (T, 192) — projected embeddings for ALL frames

    # Encoder-only path: decoder reconstructs from REAL encoded embeddings
    recon_enc = decoder(proj_all).clamp(0, 1)  # (T, 3, 224, 224)

    # Rollout: start with first H projected embeddings, predict next K
    emb = proj_all[:H].unsqueeze(0).clone()  # (1, H, 192)
    act_emb_all = jepa_model.action_encoder(acts_t.unsqueeze(0))  # (1, T, action_emb_dim)

    for k in range(K):
        emb_window = emb[:, -H:]  # (1, H, 192)
        # For step k, predictor consumes (emb_window, act_window) where act_window aligns
        act_window = act_emb_all[:, k:k + H]
        pred = jepa_model.predict(emb_window, act_window)  # (1, H, 192) via predictor + pred_proj
        next_emb = pred[:, -1:, :]  # (1, 1, 192)
        emb = torch.cat([emb, next_emb], dim=1)  # (1, k+H+1, 192)

    rollout_proj = emb[0]  # (H+K, 192) — first H are real ctx, next K are predicted
    recon_roll = decoder(rollout_proj).clamp(0, 1)  # (T, 3, 224, 224)

    return pix_t, recon_enc, recon_roll


def make_mp4(gt: torch.Tensor, recon_enc: torch.Tensor, recon_roll: torch.Tensor, out_path: str, fps: int = 6):
    """Side-by-side GT | encoder-recon | rollout, with a label strip."""
    T, _, H_img, W_img = gt.shape
    pad = 4
    label_h = 18

    out_w = W_img * 3 + pad * 2
    out_h = H_img + label_h

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8,
                                 ffmpeg_log_level='error', macro_block_size=1)

    for t in range(T):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        gt_im = (gt[t].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        enc_im = (recon_enc[t].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        roll_im = (recon_roll[t].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        canvas[label_h:label_h + H_img, 0:W_img] = gt_im
        canvas[label_h:label_h + H_img, W_img + pad:W_img * 2 + pad] = enc_im
        canvas[label_h:label_h + H_img, 2 * (W_img + pad):3 * W_img + 2 * pad] = roll_im
        # Top label using PIL
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        is_pred = t >= H
        d.text((4, 3), 'GT', fill=(0, 0, 0), font=font)
        d.text((W_img + pad + 4, 3), 'encoder→decoder', fill=(0, 0, 0), font=font)
        roll_label = f't={t}  (' + ('rollout pred' if is_pred else 'real context') + ')'
        d.text((2 * (W_img + pad) + 4, 3), roll_label, fill=((180, 0, 0) if is_pred else (0, 0, 0)), font=font)
        writer.append_data(np.array(im))
    writer.close()


def main(args):
    device = torch.device('cuda')
    print(f'[viz_rollout] loading model + decoder')

    jepa = build_jepa(args.weights, action_input_dim=args.action_input_dim, device=device)
    dec_ck = torch.load(args.dec_ckpt, map_location='cpu', weights_only=False)
    dargs = dec_ck['args']
    dec = LeWMVisDecoder(
        cls_dim=EMBED_DIM, hidden_dim=dargs['hidden_dim'], depth=dargs['depth'],
        heads=dargs['heads'], dim_head=dargs['dim_head'], mlp_ratio=4.0,
    ).to(device).eval()
    dec.load_state_dict(dec_ck['state_dict'])
    print(f'[viz_rollout] decoder val_mse={dec_ck["val_mse"]:.4f} (ep {dec_ck["epoch"]})')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(args.h5, 'r', swmr=True)
    ep_len = f['ep_len'][:]
    ep_offset = f['ep_offset'][:]
    raw_actions = f['action'][:]  # (N_total, 7) — un-stacked per-frame actions
    pixels_all = f['pixels']

    mp4_paths = []
    for ep_idx in args.episodes:
        T_ep = int(ep_len[ep_idx])
        off = int(ep_offset[ep_idx])
        # K such that (H+K-1)*FRAMESKIP <= T_ep-1  →  K <= (T_ep-1)/FRAMESKIP - H + 1
        K = min(args.K, max(1, (T_ep - 1) // FRAMESKIP - H + 1))
        T_used = H + K

        # Sample frames at frameskip intervals (matches what predictor was trained on)
        frame_idxs = np.arange(T_used) * FRAMESKIP  # (T_used,)
        pix = pixels_all[off + frame_idxs]  # (T_used, 224, 224, 3) uint8

        # Stack actions: for each "predictor step", concatenate FRAMESKIP raw actions
        # raw_actions[off + i*frameskip : off + (i+1)*frameskip] all combined
        eps_raw_actions = raw_actions[off:off + T_ep]
        stacked = stack_actions(eps_raw_actions, FRAMESKIP, 0, T_used)

        gt_t, recon_enc, recon_roll = rollout_episode(jepa, dec, pix, stacked, K, device)
        mp4 = out_dir / f'lift_ep{ep_idx:03d}_rollout.mp4'
        make_mp4(gt_t, recon_enc, recon_roll, str(mp4), fps=args.fps)
        print(f'[viz_rollout] ep {ep_idx}: T_used={T_used} ({H} ctx + {K} rollout), saved {mp4}')
        mp4_paths.append(mp4.name)

    f.close()

    # HTML wrapper
    html_path = out_dir / 'lift_rollout.html'
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LeWM lift — decoder + rollout</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,sans-serif; background:#0d1117; color:#c9d1d9; padding:24px; margin:0; }}
  h1 {{ color:#58a6ff; margin:0 0 4px; }}
  .sub {{ color:#8b949e; margin-bottom:24px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(540px,1fr)); gap:24px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }}
  video {{ width:100%; image-rendering:pixelated; background:#000; }}
  .legend {{ font-size:12px; color:#8b949e; padding:6px 0 10px; }}
  code {{ background:rgba(255,255,255,.08); padding:1px 6px; border-radius:3px; font-size:12px; }}
</style></head><body>
<h1>LeWM lift — visualization decoder + autoregressive rollout</h1>
<div class="sub">
  Lift trained 100 ep · decoder trained {dec_ck['epoch']} ep on projector-output embeddings, val_mse=<code>{dec_ck['val_mse']:.4f}</code> · per paper App. D.<br>
  Each video shows three panels: <b>GT</b> (input pixels) · <b>encoder→decoder</b> (single-step reconstruction at each frame) · <b>rollout</b> (first {H} frames as context in red-free, then predictor rolls forward in latent space and the decoder visualizes each predicted latent — t≥{H} is what the world model imagines).
</div>
<div class="grid">
"""
    for ep_idx, mp4 in zip(args.episodes, mp4_paths):
        html += f'  <div class="card"><div class="legend">episode {ep_idx}</div><video src="{mp4}" controls autoplay loop muted></video></div>\n'
    html += "</div></body></html>"
    html_path.write_text(html)
    print(f'[viz_rollout] wrote {html_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--dec_ckpt', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--episodes', type=int, nargs='+', default=[0, 7, 23, 50, 100, 150])
    p.add_argument('--K', type=int, default=8, help='rollout steps (predictor steps, each = frameskip frames)')
    p.add_argument('--action_input_dim', type=int, default=35)
    p.add_argument('--fps', type=int, default=4)
    main(p.parse_args())
