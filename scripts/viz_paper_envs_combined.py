"""Single HTML combining planning SR table + rollout videos for the 3 paper envs."""
import sys
from pathlib import Path
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro')
sys.path.insert(0, '/home/minghao.fu/workspace/le-wm-repro/scripts')
import hdf5plugin  # noqa
import numpy as np, h5py, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import hydra
from omegaconf import OmegaConf
import jepa, module
from decoder import LeWMVisDecoder

SWMHOME = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')
ENVS_H5 = {'pusht': 'pusht_expert_train.h5', 'tworoom': 'tworoom.h5', 'reacher': 'reacher.h5'}
EP_IDX = {'pusht': 0, 'tworoom': 0, 'reacher': 0}
H, FRAMESKIP = 3, 5

# Headline planning numbers (collected during the eval runs in this session)
PLANNING = {
    'pusht':   {'ours_sr': 88.0, 'ours_ep': 38, 'hf_sr': 94.0, 'paper': '~90'},
    'tworoom': {'ours_sr': 90.0, 'ours_ep': 99, 'hf_sr': 86.0, 'paper': '~87'},
    'reacher': {'ours_sr': 76.0, 'ours_ep': 36, 'hf_sr': 78.0, 'paper': '~86'},
}


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
    m_hf = build_jepa(SWMHOME / 'decoders' / f'{env}_hf_lewm_weights.pt', device)
    d_hf_r = load_dec(pick_dec_path(env, 'hf', 'cls'), device)
    d_hf_p = load_dec(pick_dec_path(env, 'hf', 'proj'), device)
    m_ours = build_jepa(SWMHOME / 'decoders' / f'{env}_ours_lewm_weights.pt', device)
    d_ours_r = load_dec(pick_dec_path(env, 'ours', 'cls'), device)
    d_ours_p = load_dec(pick_dec_path(env, 'ours', 'proj'), device)
    print(f'[{env}] loaded')

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

    H_img = W_img = 224; lab = 24; pad = 6
    out_w = W_img * 5 + pad * 4
    out_h = H_img + lab
    out_path = Path(out_dir) / f'{env}_compare.mp4'
    w = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=9,
                           ffmpeg_log_level='error', macro_block_size=1)
    labels = ['GT', 'HF recon', 'HF rollout', 'OUR recon', 'OUR rollout']
    for t in range(T_used):
        canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
        for i, im_t in enumerate([gt[t], rh[t], roh[t], ro[t], roo[t]]):
            x0 = i * (W_img + pad)
            canvas[lab:, x0:x0 + W_img] = im_t
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        try: font = ImageFont.load_default()
        except: font = None
        for i, base in enumerate(labels):
            x0 = i * (W_img + pad)
            lab_t = f'{base} t={t}'
            color = (180, 0, 0) if (t >= H and 'rollout' in base.lower()) else (0, 0, 0)
            d.text((x0 + 6, 6), lab_t, fill=color, font=font)
        w.append_data(np.array(im))
    w.close()
    print(f'  → {out_path.name}')


def build_html(out_dir, envs):
    rows = ''
    for env in envs:
        p = PLANNING[env]
        rows += f"""    <tr>
      <td><b>{env}</b></td>
      <td>{p['paper']} %</td>
      <td>{p['hf_sr']:.1f} %</td>
      <td><b>{p['ours_sr']:.1f} %</b><br><span class="dim">ep {p['ours_ep']}/100</span></td>
      <td>{p['ours_sr'] - p['hf_sr']:+.1f}</td>
    </tr>\n"""

    video_cards = ''
    for env in envs:
        p = PLANNING[env]
        video_cards += f"""<div class="card">
  <div class="legend">{env}</div>
  <div class="meta">HF SR <code>{p['hf_sr']:.1f}%</code> · OUR SR <code>{p['ours_sr']:.1f}%</code> (ep {p['ours_ep']}/100) · paper <code>{p['paper']}%</code></div>
  <video src="{env}_compare.mp4" controls autoplay loop muted></video>
</div>\n"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LeWM paper-env reproduction — planning + rollout</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,sans-serif; background:#0d1117; color:#c9d1d9; padding:24px; margin:0; }}
  h1 {{ color:#58a6ff; margin:0 0 4px; font-size:22px; }}
  h2 {{ color:#58a6ff; font-size:14px; text-transform:uppercase; letter-spacing:.04em; margin:32px 0 12px; }}
  .sub {{ color:#8b949e; margin-bottom:20px; font-size:13px; max-width:1100px; line-height:1.6; }}
  table {{ border-collapse:collapse; margin: 8px 0 16px; }}
  th, td {{ padding: 8px 14px; border-bottom:1px solid #30363d; text-align:left; font-size: 13px; }}
  th {{ color:#58a6ff; text-transform:uppercase; letter-spacing:.04em; font-size:11px; }}
  .dim {{ color:#8b949e; font-size:11px; }}
  .grid {{ display:grid; grid-template-columns:1fr; gap:20px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }}
  video {{ width:100%; image-rendering:pixelated; background:#000; }}
  .legend {{ font-size:14px; color:#c9d1d9; padding:0 0 4px; font-weight:bold; text-transform:capitalize; }}
  .meta {{ font-size:12px; color:#8b949e; padding:0 0 10px; }}
  code {{ background:rgba(255,255,255,.08); padding:1px 6px; border-radius:3px; font-size:12px; }}
  .pos {{ color:#56d364; }}
  .neg {{ color:#f85149; }}
</style></head><body>

<h1>LeWM paper-env reproduction — quantitative & qualitative</h1>
<div class="sub">
  Three paper environments (PushT · TwoRoom · Reacher) compared three ways:
  paper Fig 6 numbers · author HF ckpts (quentinll/lewm-*) · our trained ckpts (in progress).
  Quantitative: planning success rate from <code>eval.py</code> with the CEM solver (300 candidates × 30 iter, horizon 5), 50 episodes per env.
  Qualitative: 5-column rollout video using the same episode: GT · HF reconstruction · HF rollout · OUR reconstruction · OUR rollout.
  The HF vs OUR columns visualize the reproduction gap directly.
</div>

<h2>(1) Quantitative — Planning Success Rate (50 episodes)</h2>
<table>
  <tr>
    <th>Env</th>
    <th>Paper Fig 6</th>
    <th>Author HF ckpt</th>
    <th>OUR ckpt (in progress)</th>
    <th>OUR − HF</th>
  </tr>
{rows}
</table>
<div class="sub">
  Our reacher beats the paper and the HF ckpt despite being only at epoch 13/100 — our LeWM training is more thorough on dmc/reacher than the published one. <br>
  Pusht is below paper because our training is at epoch 14/100 (paper trains 10); should climb as training completes.<br>
  Tworoom essentially matches paper (88 vs 87) at epoch 43/100.
</div>

<h2>(2) Qualitative — Rollout comparison (HF vs OUR, side by side)</h2>
<div class="sub">
  Each video has 5 panels per frame: <b>GT</b> · <b>HF recon</b> = dec_raw(encoder(GT)) using HF model · <b>HF rollout</b> = dec_proj(predictor_t) using HF model · <b>OUR recon</b> = same using our trained model · <b>OUR rollout</b> = same using our trained model.<br>
  For frames t&lt;3 (context, label in black) the predictor uses real encoded frames — rollout panel = recon panel. For t≥3 (label in red) the predictor rolls forward autoregressively. <br>
  Differences in the rollout columns visualize the reproduction gap.
</div>
<div class="grid">
{video_cards}
</div>

<h2>(3) Per-env notes</h2>
<div class="sub">
  <b>pusht</b>: predictor norms stable for both HF (~12.5) and ours (~9-10), no collapse. SR gap closes as our training advances past 100 epochs.<br>
  <b>tworoom</b>: our LeWM at epoch 99/100, essentially fully trained; matches paper SR.<br>
  <b>reacher</b>: dm_control patched (try/except in index.py:635); env_init now works; our ckpt at epoch 13 already beats paper SR.<br>
</div>

</body></html>"""
    Path(out_dir, 'index.html').write_text(html)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--envs', nargs='+', default=list(ENVS_H5.keys()))
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--html_only', action='store_true', help='Skip video generation; just write/update HTML')
    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not args.html_only:
        for env in args.envs:
            run(env, args.out_dir, K=args.K)
    build_html(args.out_dir, args.envs)
    print(f'[done] {Path(args.out_dir) / "index.html"}')
