"""Two-section visualization: (1) reconstruction, (2) autoregressive rollout.

(1) Reconstruction: per frame, decode(encoder(GT_t)) — what info encoder retained
(2) Rollout: encode first H frames → projector → predictor rolls forward K steps → decode each predicted emb

Both use the v3 raw-CLS decoder (paper App. D, 200K-step snapshot).
For rollout, the predictor outputs live in projector-CLS space and we feed them to
the raw-CLS decoder anyway (same as paper Fig 7 setup).
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
import hydra
from omegaconf import OmegaConf
import jepa, module  # noqa: needed for hydra to find local module

SWMHOME = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')
ENVS_H5 = {'lift': 'lift.h5', 'can': 'can.h5', 'square': 'square.h5',
           'rope': 'rope.h5', 'granular': 'granular.h5'}
EP_IDX = {'lift': 7, 'can': 7, 'square': 7, 'rope': 23, 'granular': 23}
ENV_ACTION_DIM = {'lift': 35, 'can': 35, 'square': 35, 'rope': 20, 'granular': 20}  # frameskip*action
H = 3       # num_hist (matches lewm.yaml)
FRAMESKIP = 5  # frameskip for paper envs; for rope/granular dataset frameskip is also 5 in our configs


def build_jepa(weights_path, action_input_dim, device):
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
  num_frames: {H}
  input_dim: 192
  hidden_dim: 192
  output_dim: 192
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0
action_encoder:
  _target_: module.Embedder
  input_dim: {action_input_dim}
  emb_dim: 192
projector:
  _target_: module.MLP
  input_dim: 192
  output_dim: 192
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
pred_proj:
  _target_: module.MLP
  input_dim: 192
  output_dim: 192
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
"""
    cfg = OmegaConf.create(cfg_str)
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(weights_path, map_location='cpu', weights_only=False)
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys[:3]:
        print(f'  [warn] missing keys (first 3): {res.missing_keys[:3]}')
    if res.unexpected_keys[:3]:
        print(f'  [warn] unexpected keys (first 3): {res.unexpected_keys[:3]}')
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad = False
    return model


def load_dec_200k(env, device):
    p = SWMHOME / 'decoders' / f'{env}_cls_decoder_v3' / 'step_0200000.pt'
    ck = torch.load(p, map_location='cpu', weights_only=False)
    a = ck['args']
    dec = LeWMVisDecoder(cls_dim=192, hidden_dim=a['hidden_dim'], depth=a['depth'],
                        heads=a['heads'], dim_head=a['dim_head'], mlp_ratio=4.0,
                        img_size=224, patch_size=16, out_channels=3).to(device).eval()
    dec.load_state_dict(ck['state_dict'])
    return dec


def stack_actions(actions_flat: np.ndarray, frameskip: int, t_start: int, num_steps: int) -> np.ndarray:
    ad = actions_flat.shape[1]
    out = np.zeros((num_steps, ad * frameskip), dtype=actions_flat.dtype)
    for i in range(num_steps):
        for k in range(frameskip):
            idx = t_start + i * frameskip + k
            if idx < actions_flat.shape[0]:
                out[i, k * ad:(k + 1) * ad] = actions_flat[idx]
    return out


@torch.no_grad()
def make_recon_mp4(model, decoder, h5_path, out_path, ep_idx, fps=4, frame_stride=2):
    """Section 1: GT | encoder→decoder, per frame."""
    f = h5py.File(h5_path, 'r', swmr=True)
    T = int(f['ep_len'][ep_idx]); off = int(f['ep_offset'][ep_idx])
    idxs = np.arange(0, T, frame_stride)
    pix = f['pixels'][off + idxs]
    f.close()

    device = next(model.parameters()).device
    pix_t = torch.from_numpy(pix).permute(0, 3, 1, 2).float().to(device) / 255.0
    cls = model.encoder(pix_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    recon = decoder(cls).clamp(0, 1)
    recon_np = (recon.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    H_img = W_img = 224; lab = 22; pad = 4
    out_w, out_h = W_img * 2 + pad, H_img + lab
    w = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=9,
                           ffmpeg_log_level='error', macro_block_size=1)
    for t in range(len(idxs)):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        canvas[lab:, 0:W_img] = pix[t]
        canvas[lab:, W_img + pad:] = recon_np[t]
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try: font = ImageFont.load_default()
        except: font = None
        d.text((4, 4), 'GT', fill=(0,0,0), font=font)
        d.text((W_img + pad + 4, 4), f'reconstruct: dec(encoder(GT_t))  [frame {int(idxs[t])}/{T-1}]', fill=(0,0,0), font=font)
        w.append_data(np.array(im))
    w.close()


@torch.no_grad()
def make_rollout_mp4(model, decoder, h5_path, out_path, ep_idx, K, raw_actions, fps=4):
    """Section 2: GT | encoder→decoder (single-step at each t) | rollout decode (predicted emb at each t).

    For t < H: rollout panel shows real context (encoder→decoder, same as middle).
    For t >= H: rollout panel shows decode(predictor's predicted emb at step t)."""
    f = h5py.File(h5_path, 'r', swmr=True)
    T_ep = int(f['ep_len'][ep_idx]); off = int(f['ep_offset'][ep_idx])
    K = min(K, max(1, (T_ep - 1) // FRAMESKIP - H + 1))
    T_used = H + K
    frame_idxs = np.arange(T_used) * FRAMESKIP
    pix = f['pixels'][off + frame_idxs]
    eps_actions = raw_actions[off:off + T_ep]
    stacked = stack_actions(eps_actions, FRAMESKIP, 0, T_used)
    f.close()

    device = next(model.parameters()).device
    pix_t = torch.from_numpy(pix).permute(0, 3, 1, 2).float().to(device) / 255.0  # (T_used, 3, 224, 224)
    acts_t = torch.from_numpy(stacked).float().to(device)  # (T_used, action_dim_stacked)

    # Encoder on all frames → real CLS  (and projector(CLS) for predictor pathway)
    cls_all = model.encoder(pix_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]  # (T_used, 192)
    proj_all = model.projector(cls_all)  # (T_used, 192) — projector-space

    # Single-step reconstruction: dec(raw CLS) per frame
    recon_enc = decoder(cls_all).clamp(0, 1)  # (T_used, 3, 224, 224)

    # Autoregressive rollout in projector-CLS space
    emb = proj_all[:H].unsqueeze(0).clone()  # (1, H, 192)
    act_emb_all = model.action_encoder(acts_t.unsqueeze(0))  # (1, T_used, action_emb)
    for k in range(K):
        emb_window = emb[:, -H:]
        act_window = act_emb_all[:, k:k + H]
        pred = model.predict(emb_window, act_window)  # (1, H, 192)
        emb = torch.cat([emb, pred[:, -1:, :]], dim=1)
    rollout_proj = emb[0]  # (T_used, 192) — first H are real ctx (= proj_all[:H]), next K are predicted

    # Decode rollout embeddings with the raw-CLS-trained decoder (OOD for predicted emb, but
    # matches paper Fig 7's setup of decoding predictor outputs with the single decoder).
    recon_roll = decoder(rollout_proj).clamp(0, 1)  # (T_used, 3, 224, 224)

    gt_np = (pix_t.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    enc_np = (recon_enc.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    roll_np = (recon_roll.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    H_img = W_img = 224; lab = 22; pad = 4
    cols = 3
    out_w = W_img * cols + pad * (cols - 1)
    out_h = H_img + lab
    w = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=9,
                           ffmpeg_log_level='error', macro_block_size=1)
    for t in range(T_used):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        canvas[lab:, 0:W_img] = gt_np[t]
        canvas[lab:, W_img + pad:2 * W_img + pad] = enc_np[t]
        canvas[lab:, 2 * (W_img + pad):] = roll_np[t]
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try: font = ImageFont.load_default()
        except: font = None
        d.text((4, 4), f'GT  t={t}', fill=(0,0,0), font=font)
        d.text((W_img + pad + 4, 4), 'reconstruct: dec(enc(GT_t))', fill=(0,0,0), font=font)
        is_pred = t >= H
        roll_label = f'rollout: dec(pred_t)' + ('  [context]' if not is_pred else '  [imagined]')
        d.text((2 * (W_img + pad) + 4, 4), roll_label, fill=((180, 0, 0) if is_pred else (0, 0, 0)), font=font)
        w.append_data(np.array(im))
    w.close()


def main(args):
    device = torch.device('cuda')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    for env in args.envs:
        if env not in ENVS_H5:
            print(f'[skip] {env} (no h5 mapping)'); continue
        h5 = SWMHOME / ENVS_H5[env]
        weights = SWMHOME / 'decoders' / f'{env}_lewm_weights.pt'
        if not (h5.exists() and weights.exists()):
            print(f'[skip] {env}: files missing'); continue
        print(f'[{env}] loading…')
        model = build_jepa(str(weights), ENV_ACTION_DIM[env], device)
        decoder = load_dec_200k(env, device)

        recon_path = out_dir / f'{env}_recon.mp4'
        make_recon_mp4(model, decoder, str(h5), recon_path, EP_IDX[env], fps=args.fps, frame_stride=args.frame_stride)
        print(f'  recon → {recon_path.name}')

        # rollout — read raw actions once per env
        with h5py.File(h5, 'r', swmr=True) as f: raw_actions = f['action'][:]
        rollout_path = out_dir / f'{env}_rollout.mp4'
        make_rollout_mp4(model, decoder, str(h5), rollout_path, EP_IDX[env], K=args.K, raw_actions=raw_actions, fps=args.fps)
        print(f'  rollout → {rollout_path.name}')
        cards.append((env, recon_path.name, rollout_path.name))

    # HTML
    html_path = out_dir / 'index.html'
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LeWM decoders — reconstruction + rollout</title>
<style>
  body { font-family: -apple-system,BlinkMacSystemFont,sans-serif; background:#0d1117; color:#c9d1d9; padding:24px; margin:0; }
  h1 { color:#58a6ff; margin:0 0 4px; }
  .sub { color:#8b949e; margin-bottom:24px; font-size:13px; max-width:1100px; }
  h2 { color:#58a6ff; font-size:14px; text-transform:uppercase; letter-spacing:.04em; margin:32px 0 12px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(580px,1fr)); gap:24px; }
  .grid-wide { display:grid; grid-template-columns:repeat(auto-fit,minmax(820px,1fr)); gap:24px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }
  video { width:100%; image-rendering:pixelated; background:#000; }
  .legend { font-size:13px; color:#c9d1d9; padding:0 0 8px; }
  .what { font-size:12px; color:#8b949e; padding:6px 0 10px; line-height:1.5; }
  code { background:rgba(255,255,255,.08); padding:1px 6px; border-radius:3px; font-size:12px; }
</style></head><body>
<h1>LeWM v3 decoders — reconstruction vs. rollout</h1>
<div class="sub">
  Both panels use the same v3 raw-CLS decoder at the 200k-step snapshot (paper App. D, hidden=512, depth=10, 32 M params).
  <b>Reconstruction</b> tests the encoder: each frame is encoded to its 192-D CLS and decoded back.
  <b>Rollout</b> tests the world model: the first 3 frames bootstrap the predictor, then the predictor rolls forward in latent space for ~K steps using ground-truth actions, and we decode each predicted latent (paper Fig 7 setup).
  Predictor outputs live in projector-CLS space (after the BN MLP); we feed them to the raw-CLS-trained decoder anyway — slightly out-of-distribution, same as the paper.
</div>

<h2>(1) Reconstruction: GT vs dec(encoder(GT_t))</h2>
<div class="what">Tests what each GT frame's 192-D CLS retains. No prediction involved.</div>
<div class="grid">
"""
    for env, rec, _ in cards:
        html += f'<div class="card"><div class="legend">{env}</div><video src="{rec}" controls autoplay loop muted></video></div>\n'
    html += '</div><h2>(2) Rollout: 3-frame context → predictor → decode every step</h2>\n'
    html += '<div class="what">Panels left→right: <b>GT</b> · <b>reconstruction</b> dec(enc(GT_t)) at each t · <b>rollout</b> dec(predictor_t). For t&lt;3 (context), the rollout panel just shows the real encoded frame; for t≥3 it shows what the world model imagines.</div>\n'
    html += '<div class="grid-wide">\n'
    for env, _, roll in cards:
        html += f'<div class="card"><div class="legend">{env}</div><video src="{roll}" controls autoplay loop muted></video></div>\n'

    html += '</div></body></html>'
    html_path.write_text(html)
    print(f'[done] {html_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--envs', nargs='+', default=['lift', 'can', 'square', 'rope', 'granular'])
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--fps', type=int, default=4)
    p.add_argument('--frame_stride', type=int, default=2)
    main(p.parse_args())
