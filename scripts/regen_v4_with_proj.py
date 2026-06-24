"""Regenerate v4 rollout panel using v3 (raw CLS) for reconstruction + v3-proj for rollout."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro')
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro/scripts')
import numpy as np, h5py, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import hydra
from omegaconf import OmegaConf
import stable_worldmodel.data.formats.hdf5  # noqa
import jepa, module
from decoder import LeWMVisDecoder

SWMHOME = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')
ENVS_H5 = {'lift': 'lift.h5', 'can': 'can.h5', 'square': 'square.h5',
           'rope': 'rope.h5', 'granular': 'granular.h5'}
EP_IDX = {'lift': 7, 'can': 7, 'square': 7, 'rope': 23, 'granular': 23}
ENV_AD = {'lift': 35, 'can': 35, 'square': 35, 'rope': 20, 'granular': 20}
H = 3; FRAMESKIP = 5


def build_jepa(weights, ad, device):
    cfg = OmegaConf.create(f"""
_target_: jepa.JEPA
encoder: {{_target_: stable_pretraining.backbone.utils.vit_hf, size: tiny, patch_size: 14, image_size: 224, pretrained: false, use_mask_token: false}}
predictor: {{_target_: module.ARPredictor, num_frames: 3, input_dim: 192, hidden_dim: 192, output_dim: 192, depth: 6, heads: 16, mlp_dim: 2048, dim_head: 64, dropout: 0.1, emb_dropout: 0.0}}
action_encoder: {{_target_: module.Embedder, input_dim: {ad}, emb_dim: 192}}
projector: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
pred_proj: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
""")
    m = hydra.utils.instantiate(cfg)
    sd = torch.load(weights, map_location='cpu', weights_only=False)
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    for p in m.parameters(): p.requires_grad = False
    return m


def load_dec(p, device):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    a = ck['args']
    d = LeWMVisDecoder(cls_dim=192, hidden_dim=a['hidden_dim'], depth=a['depth'],
                      heads=a['heads'], dim_head=a['dim_head'], mlp_ratio=4.0,
                      img_size=224, patch_size=16, out_channels=3).to(device).eval()
    d.load_state_dict(ck['state_dict'])
    return d


def stack_actions(a, fs, ts, n):
    ad = a.shape[1]
    out = np.zeros((n, ad * fs), dtype=a.dtype)
    for i in range(n):
        for k in range(fs):
            j = ts + i * fs + k
            if j < a.shape[0]: out[i, k*ad:(k+1)*ad] = a[j]
    return out


@torch.no_grad()
def run(env, out_dir, K=8, fps=4):
    device = torch.device('cuda')
    model = build_jepa(SWMHOME / 'decoders' / f'{env}_lewm_weights.pt', ENV_AD[env], device)
    # raw CLS decoder for reconstruction; proj decoder for rollout
    dec_recon = load_dec(SWMHOME / 'decoders' / f'{env}_cls_decoder_v3' / 'step_0200000.pt', device)
    dec_roll  = load_dec(SWMHOME / 'decoders' / f'{env}_proj_decoder_v3' / 'step_0200000.pt', device)
    print(f'[{env}] loaded raw + proj decoders @ 200K')

    f = h5py.File(SWMHOME / ENVS_H5[env], 'r', swmr=True)
    T_ep = int(f['ep_len'][EP_IDX[env]]); off = int(f['ep_offset'][EP_IDX[env]])
    K_use = min(K, max(1, (T_ep - 1) // FRAMESKIP - H + 1))
    T_used = H + K_use
    frame_idxs = np.arange(T_used) * FRAMESKIP
    pix = f['pixels'][off + frame_idxs]
    raw_a = f['action'][off:off + T_ep]
    f.close()
    stacked = stack_actions(raw_a, FRAMESKIP, 0, T_used)

    pix_t = torch.from_numpy(pix).permute(0, 3, 1, 2).float().to(device) / 255.0
    acts_t = torch.from_numpy(stacked).float().to(device)

    cls_all = model.encoder(pix_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    proj_all = model.projector(cls_all)

    # Reconstruction column: dec_recon(raw CLS_t)
    recon = dec_recon(cls_all).clamp(0, 1)

    # Rollout: roll predictor in projector-space, decode with proj decoder
    emb = proj_all[:H].unsqueeze(0).clone()
    act_emb_all = model.action_encoder(acts_t.unsqueeze(0))
    for k in range(K_use):
        emb_w = emb[:, -H:]
        a_w = act_emb_all[:, k:k + H]
        pred = model.predict(emb_w, a_w)
        emb = torch.cat([emb, pred[:, -1:, :]], dim=1)
    rollout_proj = emb[0]  # (T_used, 192) projector-space
    roll = dec_roll(rollout_proj).clamp(0, 1)

    gt_np = (pix_t.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    rec_np = (recon.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    roll_np = (roll.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    H_img = W_img = 224; lab = 22; pad = 4
    out_w = W_img * 3 + pad * 2
    out_h = H_img + lab
    out_path = Path(out_dir) / f'{env}_rollout.mp4'
    w = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=9,
                           ffmpeg_log_level='error', macro_block_size=1)
    for t in range(T_used):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        canvas[lab:, 0:W_img] = gt_np[t]
        canvas[lab:, W_img + pad:2*W_img + pad] = rec_np[t]
        canvas[lab:, 2*(W_img + pad):] = roll_np[t]
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try: font = ImageFont.load_default()
        except: font = None
        d.text((4, 4), f'GT  t={t}', fill=(0,0,0), font=font)
        d.text((W_img + pad + 4, 4), 'reconstruct: dec_raw(enc(GT_t))', fill=(0,0,0), font=font)
        is_pred = t >= H
        roll_label = 'rollout: dec_proj(pred_t)' + ('  [imagined]' if is_pred else '  [context]')
        d.text((2*(W_img + pad) + 4, 4), roll_label, fill=((180, 0, 0) if is_pred else (0, 0, 0)), font=font)
        w.append_data(np.array(im))
    w.close()
    print(f'  → {out_path.name}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--envs', nargs='+', default=list(ENVS_H5.keys()))
    p.add_argument('--K', type=int, default=8)
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for env in args.envs:
        run(env, args.out_dir, K=args.K)
