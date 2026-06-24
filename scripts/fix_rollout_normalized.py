"""Re-render rollout videos with CORRECT action normalization (z-score per dataset stats),
matching how eval.py/the policy feeds actions to the predictor.
recon panel unchanged (no actions); only the rollout panel is affected.
"""
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

SWM = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')
H, FS = 3, 5
# env -> (h5, raw action dim, ep_idx, lewm-weights tag, decoder tag)
CFG = {
  'lift':     ('lift.h5', 7, 7),
  'can':      ('can.h5', 7, 7),
  'square':   ('square.h5', 7, 7),
  'rope':     ('rope.h5', 4, 23),
  'granular': ('granular.h5', 4, 23),
}

def build(weights, ad, device):
    cfg = OmegaConf.create(f"""
_target_: jepa.JEPA
encoder: {{_target_: stable_pretraining.backbone.utils.vit_hf, size: tiny, patch_size: 14, image_size: 224, pretrained: false, use_mask_token: false}}
predictor: {{_target_: module.ARPredictor, num_frames: 3, input_dim: 192, hidden_dim: 192, output_dim: 192, depth: 6, heads: 16, mlp_dim: 2048, dim_head: 64, dropout: 0.1, emb_dropout: 0.0}}
action_encoder: {{_target_: module.Embedder, input_dim: {ad*FS}, emb_dim: 192}}
projector: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
pred_proj: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
""")
    m = hydra.utils.instantiate(cfg)
    m.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False), strict=False)
    m.eval().to(device)
    for p in m.parameters(): p.requires_grad=False
    return m

def load_dec(p, device):
    ck = torch.load(p, map_location='cpu', weights_only=False); a=ck['args']
    d = LeWMVisDecoder(cls_dim=192, hidden_dim=a['hidden_dim'], depth=a['depth'], heads=a['heads'],
                       dim_head=a['dim_head'], mlp_ratio=4.0, img_size=224, patch_size=16, out_channels=3).to(device).eval()
    d.load_state_dict(ck['state_dict']); return d

@torch.no_grad()
def render(env, out_path, K=8, fps=4, normalize=True):
    device = torch.device('cuda')
    h5, ad, ep = CFG[env]
    m = build(SWM/'decoders'/f'{env}_lewm_weights.pt', ad, device)
    dec_recon = load_dec(SWM/'decoders'/f'{env}_cls_decoder_v3'/'step_0200000.pt', device)
    dec_roll  = load_dec(SWM/'decoders'/f'{env}_proj_decoder_v3'/'step_0200000.pt', device)

    f = h5py.File(SWM/h5,'r',swmr=True)
    # action normalizer stats (per-frame, raw action dim) — same as eval.py StandardScaler
    alla = f['action'][:]
    amean = alla.mean(0); astd = alla.std(0); astd[astd<1e-6]=1.0
    T_ep=int(f['ep_len'][ep]); off=int(f['ep_offset'][ep])
    K=min(K, max(1,(T_ep-1)//FS - H + 1)); T=H+K
    idxs=np.clip(np.arange(T)*FS,0,T_ep-1)
    pix=f['pixels'][off+idxs]; rawA=f['action'][off:off+T_ep]; f.close()

    # normalize per frame then stack FS
    A = rawA.astype(np.float32)
    if normalize: A = (A-amean)/astd
    stacked=np.zeros((T, ad*FS), dtype=np.float32)
    for i in range(T):
        for k in range(FS):
            j=i*FS+k
            if j<A.shape[0]: stacked[i,k*ad:(k+1)*ad]=A[j]

    pix_t=torch.from_numpy(pix).permute(0,3,1,2).float().to(device)/255.0
    acts_t=torch.from_numpy(stacked).float().to(device)
    cls=m.encoder(pix_t, interpolate_pos_encoding=True).last_hidden_state[:,0]
    proj=m.projector(cls)
    recon=dec_recon(cls).clamp(0,1)
    emb=proj[:H].unsqueeze(0).clone()
    ae=m.action_encoder(acts_t.unsqueeze(0))
    for k in range(K):
        pr=m.predict(emb[:,-H:], ae[:,k:k+H]); emb=torch.cat([emb,pr[:,-1:,:]],dim=1)
    roll=dec_roll(emb[0]).clamp(0,1)
    npf=lambda x:(x.cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8)
    gt,rc,ro=npf(pix_t),npf(recon),npf(roll)
    W=224; lab=22; pad=4; ow=W*3+pad*2; oh=W+lab
    w=imageio.get_writer(str(out_path),fps=fps,codec='libx264',quality=9,ffmpeg_log_level='error',macro_block_size=1)
    for t in range(T):
        c=np.full((oh,ow,3),245,np.uint8)
        c[lab:,0:W]=gt[t]; c[lab:,W+pad:2*W+pad]=rc[t]; c[lab:,2*(W+pad):]=ro[t]
        im=Image.fromarray(c); d=ImageDraw.Draw(im)
        try: fnt=ImageFont.load_default()
        except: fnt=None
        d.text((4,4),f'GT t={t}',fill=(0,0,0),font=fnt)
        d.text((W+pad+4,4),'reconstruction',fill=(0,0,0),font=fnt)
        ispred=t>=H
        d.text((2*(W+pad)+4,4),'rollout'+(' [imagined]' if ispred else ' [context]'),fill=((180,0,0) if ispred else (0,0,0)),font=fnt)
        w.append_data(np.array(im))
    w.close()
    print(f'  → {out_path}  (norm={normalize})')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--envs',nargs='+',default=list(CFG)); p.add_argument('--out_dir',required=True)
    a=p.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    for e in a.envs: render(e, Path(a.out_dir)/f'{e}_rollout.mp4')
