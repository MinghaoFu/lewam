"""Final rollout render with FIXED (predicted-latent-trained) decoders, all 8 envs, OUR ckpts.
GT | reconstruction(real cls decoder) | rollout(fixed pred decoder). Eval mode, normalized actions."""
import sys
from pathlib import Path
import hdf5plugin, numpy as np, h5py, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import hydra
from omegaconf import OmegaConf
import lewam.models.jepa as jepa, lewam.models.module as module
from decoder import LeWMVisDecoder
SWM=Path('/mnt/data_nvme1/minghao.fu/.stable-wm'); H,FS=3,5
# env -> (h5, weights_tag, cls_decoder_tag, ad, K)
CFG={
 'pusht':   ('pusht_expert_train.h5','pusht_ours','pusht_ours',2,8),
 'tworoom': ('tworoom.h5','tworoom_ours','tworoom_ours',2,8),
 'reacher': ('reacher.h5','reacher_ours','reacher_ours',2,8),
 'can':     ('can.h5','can','can',7,8),
 'square':  ('square.h5','square','square',7,8),
 'lift':    ('lift.h5','lift','lift',7,4),
 'rope':    ('rope.h5','rope','rope',4,1),
 'granular':('granular.h5','granular','granular',4,1),
}
def build(wt,ad,dev):
    cfg=OmegaConf.create(f'''
_target_: jepa.JEPA
encoder: {{_target_: stable_pretraining.backbone.utils.vit_hf, size: tiny, patch_size: 14, image_size: 224, pretrained: false, use_mask_token: false}}
predictor: {{_target_: module.ARPredictor, num_frames: 3, input_dim: 192, hidden_dim: 192, output_dim: 192, depth: 6, heads: 16, mlp_dim: 2048, dim_head: 64, dropout: 0.1, emb_dropout: 0.0}}
action_encoder: {{_target_: module.Embedder, input_dim: {ad*FS}, emb_dim: 192}}
projector: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
pred_proj: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
''')
    m=hydra.utils.instantiate(cfg); m.load_state_dict(torch.load(SWM/'decoders'/f'{wt}_lewm_weights.pt',map_location='cpu',weights_only=False),strict=False)
    m.eval().to(dev)
    for p in m.parameters(): p.requires_grad=False
    return m
def ld(path,dev):
    ck=torch.load(path,map_location='cpu',weights_only=False); a=ck['args']
    d=LeWMVisDecoder(cls_dim=192,hidden_dim=a['hidden_dim'],depth=a['depth'],heads=a['heads'],dim_head=a['dim_head'],mlp_ratio=4.0,img_size=224,patch_size=16,out_channels=3).to(dev).eval()
    d.load_state_dict(ck['state_dict']); return d
@torch.no_grad()
def render(env,out,fps=4):
    dev=torch.device('cuda'); h5,wt,ct,ad,K=CFG[env]
    m=build(wt,ad,dev)
    dr=ld(SWM/'decoders'/f'{ct}_cls_decoder_v3'/'step_0200000.pt',dev)   # recon: real cls decoder
    pred_snap=SWM/'decoders'/f'{env}_fixed_pred_decoder'/'step_0020000.pt'
    if not pred_snap.exists():
        snaps=sorted((SWM/'decoders'/f'{env}_fixed_pred_decoder').glob('step_*.pt'))
        pred_snap=snaps[-1]
    dp=ld(pred_snap,dev)                                                 # rollout: fixed pred decoder
    ep=7 if env in('pusht','tworoom','reacher','lift','can','square') else 23
    f=h5py.File(SWM/h5,'r',swmr=True); alla=f['action'][:]; am=alla.mean(0); ast=alla.std(0); ast[ast<1e-6]=1
    if ep>=len(f['ep_len']): ep=0
    T_ep=int(f['ep_len'][ep]); off=int(f['ep_offset'][ep]); K=min(K,max(1,(T_ep-1)//FS-H+1)); T=H+K
    idx=np.clip(np.arange(T)*FS,0,T_ep-1); pix=f['pixels'][off+idx]; rawA=f['action'][off:off+T_ep]; f.close()
    A=((rawA-am)/ast).astype(np.float32); st=np.zeros((T,ad*FS),np.float32)
    for i in range(T):
        for k in range(FS):
            if i*FS+k<A.shape[0]: st[i,k*ad:(k+1)*ad]=A[i*FS+k]
    pt=torch.from_numpy(pix).permute(0,3,1,2).float().to(dev)/255.; at=torch.from_numpy(st).float().to(dev)
    cls=m.encoder(pt,interpolate_pos_encoding=True).last_hidden_state[:,0]; proj=m.projector(cls)
    rc=dr(cls).clamp(0,1); emb=proj[:H].unsqueeze(0).clone(); ae=m.action_encoder(at.unsqueeze(0))
    for k in range(K):
        pr=m.predict(emb[:,-H:],ae[:,k:k+H]); emb=torch.cat([emb,pr[:,-1:,:]],1)
    ro=dp(emb[0]).clamp(0,1)
    npf=lambda x:(x.cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8); g,r,o=npf(pt),npf(rc),npf(ro)
    W=224;lab=22;pad=4;ow=W*3+pad*2;oh=W+lab
    w=imageio.get_writer(str(out),fps=fps,codec='libx264',quality=9,ffmpeg_log_level='error',macro_block_size=1)
    for t in range(T):
        c=np.full((oh,ow,3),245,np.uint8); c[lab:,:W]=g[t]; c[lab:,W+pad:2*W+pad]=r[t]; c[lab:,2*(W+pad):]=o[t]
        im=Image.fromarray(c); d=ImageDraw.Draw(im); fnt=ImageFont.load_default()
        d.text((4,4),f'GT t={t}',fill=(0,0,0),font=fnt); d.text((W+pad+4,4),'reconstruction',fill=(0,0,0),font=fnt)
        d.text((2*(W+pad)+4,4),'rollout'+(' [imagined]' if t>=H else ' [context]'),fill=((180,0,0) if t>=H else (0,0,0)),font=fnt)
        w.append_data(np.array(im))
    w.close(); print('  →',out.name)
if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('--out_dir',required=True); p.add_argument('--envs',nargs='+',default=list(CFG)); a=p.parse_args()
    Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    for e in a.envs:
        try: render(e,Path(a.out_dir)/f'rollout_{e}.mp4')
        except Exception as ex: print(f'  {e} FAIL: {ex}')
