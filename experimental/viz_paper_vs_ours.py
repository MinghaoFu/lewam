"""Side-by-side comparison: HF (paper) vs OUR reproduction, per paper env.
Renders 5-column video per env: GT | HF recon | HF rollout | OUR recon | OUR rollout."""
import sys
from pathlib import Path
import hdf5plugin  # noqa
import numpy as np, h5py, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import hydra
from omegaconf import OmegaConf
import lewam.models.jepa as jepa, lewam.models.module as module
from decoder import LeWMVisDecoder

SWMHOME = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')
ENVS_H5 = {'pusht': 'pusht_expert_train.h5', 'tworoom': 'tworoom.h5', 'reacher': 'reacher.h5'}
EP_IDX = {'pusht': 0, 'tworoom': 0, 'reacher': 0}
H, FRAMESKIP = 3, 5


def build_jepa(weights, device, ad=10):
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


def stack_actions(a, fs, n):
    ad = a.shape[1]
    out = np.zeros((n, ad * fs), dtype=a.dtype)
    for i in range(n):
        for k in range(fs):
            j = i * fs + k
            if j < a.shape[0]: out[i, k*ad:(k+1)*ad] = a[j]
    return out


def pick_dec_path(env, variant, kind):
    """variant in {'hf','ours'}, kind in {'cls','proj'}"""
    snap = SWMHOME / 'decoders' / f'{env}_{variant}_{kind}_decoder_v3' / 'step_0200000.pt'
    best = SWMHOME / 'decoders' / f'{env}_{variant}_{kind}_decoder_v3.pt'
    return snap if snap.exists() else best


@torch.no_grad()
def encode_and_rollout(model, pix_t, acts_t, K_use, dec_recon, dec_roll):
    cls = model.encoder(pix_t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    proj = model.projector(cls)
    recon = dec_recon(cls).clamp(0, 1)
    emb = proj[:H].unsqueeze(0).clone()
    act_emb_all = model.action_encoder(acts_t.unsqueeze(0))
    for k in range(K_use):
        pred = model.predict(emb[:, -H:], act_emb_all[:, k:k + H])
        emb = torch.cat([emb, pred[:, -1:, :]], dim=1)
    roll = dec_roll(emb[0]).clamp(0, 1)
    return recon, roll


@torch.no_grad()
def run(env, out_dir, K=8, fps=4):
    device = torch.device('cuda')
    # HF
    m_hf = build_jepa(SWMHOME / 'decoders' / f'{env}_hf_lewm_weights.pt', device)
    d_hf_r = load_dec(pick_dec_path(env, 'hf', 'cls'), device)
    d_hf_p = load_dec(pick_dec_path(env, 'hf', 'proj'), device)
    # ours
    m_ours = build_jepa(SWMHOME / 'decoders' / f'{env}_ours_lewm_weights.pt', device)
    d_ours_r = load_dec(pick_dec_path(env, 'ours', 'cls'), device)
    d_ours_p = load_dec(pick_dec_path(env, 'ours', 'proj'), device)
    print(f'[{env}] HF + ours models + 4 decoders loaded')

    f = h5py.File(SWMHOME / ENVS_H5[env], 'r', swmr=True)
    T_ep = int(f['ep_len'][EP_IDX[env]]); off = int(f['ep_offset'][EP_IDX[env]])
    K_use = min(K, max(1, (T_ep - 1) // FRAMESKIP - H + 1))
    T_used = H + K_use
    frame_idxs = np.clip(np.arange(T_used) * FRAMESKIP, 0, T_ep - 1)
    pix = f['pixels'][off + frame_idxs]
    raw_a = f['action'][off:off + T_ep]
    f.close()
    stacked = stack_actions(raw_a, FRAMESKIP, T_used)

    pix_t = torch.from_numpy(pix).permute(0, 3, 1, 2).float().to(device) / 255.0
    acts_t = torch.from_numpy(stacked).float().to(device)

    rec_hf, roll_hf = encode_and_rollout(m_hf, pix_t, acts_t, K_use, d_hf_r, d_hf_p)
    rec_ours, roll_ours = encode_and_rollout(m_ours, pix_t, acts_t, K_use, d_ours_r, d_ours_p)

    np_fn = lambda x: (x.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    gt = np_fn(pix_t); rh = np_fn(rec_hf); roh = np_fn(roll_hf); ro = np_fn(rec_ours); roo = np_fn(roll_ours)

    H_img = W_img = 224; lab = 22; pad = 4
    out_w = W_img * 5 + pad * 4
    out_h = H_img + lab
    out_path = Path(out_dir) / f'{env}_compare.mp4'
    w = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=9,
                           ffmpeg_log_level='error', macro_block_size=1)
    for t in range(T_used):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        for i, im_t in enumerate([gt[t], rh[t], roh[t], ro[t], roo[t]]):
            x0 = i * (W_img + pad)
            canvas[lab:, x0:x0 + W_img] = im_t
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try: font = ImageFont.load_default()
        except: font = None
        labels = [f'GT t={t}', 'HF recon', 'HF rollout', 'OUR recon', 'OUR rollout']
        for i, lab_t in enumerate(labels):
            x0 = i * (W_img + pad)
            color = (180, 0, 0) if (t >= H and 'rollout' in lab_t.lower()) else (0, 0, 0)
            d.text((x0 + 4, 4), lab_t, fill=color, font=font)
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
    # HTML
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Paper-env reproduction gap (HF vs ours)</title>
<style>
  body { font-family: -apple-system,BlinkMacSystemFont,sans-serif; background:#0d1117; color:#c9d1d9; padding:24px; margin:0; }
  h1 { color:#58a6ff; margin:0 0 4px; }
  .sub { color:#8b949e; margin-bottom:24px; font-size:13px; max-width:1400px; }
  .grid { display:grid; grid-template-columns:1fr; gap:24px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }
  video { width:100%; image-rendering:pixelated; background:#000; }
  .legend { font-size:14px; color:#c9d1d9; padding:0 0 8px; font-weight:bold; }
  code { background:rgba(255,255,255,.08); padding:1px 6px; border-radius:3px; font-size:12px; }
</style></head><body>
<h1>Paper-env reproduction — HF reference vs. ours, side-by-side</h1>
<div class="sub">
  Five panels per row: <b>GT</b> · <b>HF recon</b> · <b>HF rollout</b> · <b>OUR recon</b> · <b>OUR rollout</b>. <br>
  HF = author's HuggingFace ckpts (paper reference, fully trained). <br>
  OUR = our reproduction at current training state (tworoom ep 99, pusht ep 35, reacher ep 34 of planned 100). <br>
  Both pipelines use the same dataset and the same decoder recipe (32M-param cross-attn transformer, 200K steps).<br>
  <b>Gap interpretation</b>: differences between HF rollout and OUR rollout are the reproduction gap — should shrink to ≈0 once our LeWM training completes 100 epochs.
</div>
<div class="grid">"""
    for env in args.envs:
        html += f'<div class="card"><div class="legend">{env}</div><video src="{env}_compare.mp4" controls autoplay loop muted></video></div>\n'
    html += '</div></body></html>'
    Path(args.out_dir, 'index.html').write_text(html)
    print(f'[done] {Path(args.out_dir) / "index.html"}')
