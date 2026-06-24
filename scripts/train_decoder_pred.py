"""Decoder trainer that matches ROLLOUT-TIME input distribution: trains the proj decoder on
BOTH real projector(cls) latents AND multi-step autoregressive predictor latents (all depths),
each paired with the co-located GT frame. Mirrors TC-WM's decoder_recon_loss_pred + _reconstructed.
Encoder/projector/predictor/pred_proj frozen; only the decoder is trained.
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0,'/home/minghao.fu/workspace/le-wm-repro'); sys.path.insert(0,'/home/minghao.fu/workspace/le-wm-repro/scripts')
import hdf5plugin, numpy as np, h5py, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import OmegaConf
import jepa, module
from decoder import LeWMVisDecoder

SWM = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')

def build_jepa(weights, ad, device):
    cfg = OmegaConf.create(f"""
_target_: jepa.JEPA
encoder: {{_target_: stable_pretraining.backbone.utils.vit_hf, size: tiny, patch_size: 14, image_size: 224, pretrained: false, use_mask_token: false}}
predictor: {{_target_: module.ARPredictor, num_frames: 3, input_dim: 192, hidden_dim: 192, output_dim: 192, depth: 6, heads: 16, mlp_dim: 2048, dim_head: 64, dropout: 0.1, emb_dropout: 0.0}}
action_encoder: {{_target_: module.Embedder, input_dim: {ad*5}, emb_dim: 192}}
projector: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
pred_proj: {{_target_: module.MLP, input_dim: 192, output_dim: 192, hidden_dim: 2048, norm_fn: {{_target_: torch.nn.BatchNorm1d, _partial_: true}}}}
""")
    m = hydra.utils.instantiate(cfg)
    m.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False), strict=False)
    m.eval().to(device)
    for p in m.parameters(): p.requires_grad=False
    return m

class ClipDS(Dataset):
    """Returns clips of T frames at frameskip spacing + frameskip-stacked normalized actions."""
    def __init__(self, h5, T, ad, frameskip=5):
        self.h5=h5; self.T=T; self.ad=ad; self.fs=frameskip
        with h5py.File(h5,'r',swmr=True) as f:
            self.ep_len=f['ep_len'][:]; self.ep_off=f['ep_offset'][:]
            a=f['action'][:]; self.amean=a.mean(0); self.astd=a.std(0); self.astd[self.astd<1e-6]=1.0
        # valid (ep, start): need frames [s .. s+(T-1)*fs] and actions [s .. s+(T-1)*fs+(fs-1)]
        # all within the episode -> last valid s = L - ((T-1)*fs + fs). Skip episodes too short.
        self.index=[]
        need=(T-1)*frameskip + frameskip
        for L,O in zip(self.ep_len, self.ep_off):
            last_s=int(L) - need
            for s in range(0, last_s+1):       # empty if episode shorter than one clip
                self.index.append((int(O), int(s)))
        assert len(self.index)>0, f'no clips fit (T={T}, fs={frameskip}); episodes too short'
        self._f=None
    def __len__(self): return len(self.index)
    def _file(self):
        if self._f is None: self._f=h5py.File(self.h5,'r',swmr=True)
        return self._f
    def __getitem__(self, i):
        off,s=self.index[i]; f=self._file()
        fidx=[s+t*self.fs for t in range(self.T)]
        pix=f['pixels'][[off+j for j in fidx]]            # (T,224,224,3) uint8
        # read the contiguous raw-action span once, then stack frameskip per step
        span=(self.T-1)*self.fs + self.fs
        raw=f['action'][off+s:off+s+span]                 # (span, ad)
        rawn=(raw-self.amean)/self.astd
        st=np.zeros((self.T, self.ad*self.fs), np.float32)
        for t in range(self.T):
            for k in range(self.fs):
                j=t*self.fs+k
                if j<rawn.shape[0]: st[t,k*self.ad:(k+1)*self.ad]=rawn[j]
        return torch.from_numpy(pix).permute(0,3,1,2).float()/255.0, torch.from_numpy(st)

@torch.no_grad()
def latents(m, pixels, acts, H, K):
    # pixels (B,T,3,224,224) in [0,1]; acts (B,T,ad*fs)
    B,T=pixels.shape[:2]
    info={'pixels':pixels,'action':acts}
    m.encode(info)
    proj=info['emb']            # (B,T,192) real projector(cls)
    ae=info['act_emb']          # (B,T,192)
    # autoregressive rollout latents
    emb=proj[:,:H].clone()
    for k in range(K):
        pr=m.predict(emb[:,-H:], ae[:,k:k+H])
        emb=torch.cat([emb, pr[:,-1:]],dim=1)
    pred=emb[:,H:]              # (B,K,192) predicted latents at depths 1..K
    return proj, pred

def main(a):
    dev=torch.device('cuda'); H,K=a.H,a.K; T=H+K
    m=build_jepa(SWM/'decoders'/f'{a.weights_tag}_lewm_weights.pt', a.ad, dev)
    dec=LeWMVisDecoder(cls_dim=192,hidden_dim=512,depth=10,heads=8,dim_head=64,mlp_ratio=4.0,img_size=224,patch_size=16,out_channels=3).to(dev)
    print(f'[decoder] {sum(p.numel() for p in dec.parameters())/1e6:.1f}M params', flush=True)
    ds=ClipDS(str(SWM/a.h5), T, a.ad); print(f'[data] {len(ds)} clips (T={T})', flush=True)
    dl=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True)
    opt=torch.optim.AdamW(dec.parameters(),lr=3e-4,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.max_steps, eta_min=3e-6)
    scaler=torch.amp.GradScaler('cuda')
    outdir=SWM/'decoders'/f'{a.out_tag}_pred_decoder'; outdir.mkdir(parents=True,exist_ok=True)
    snaps={10000,40000,100000,a.max_steps}
    step=0; t0=time.time(); best=1e9; printed=False
    while step<a.max_steps:
        for pix,act in dl:
            pix=pix.to(dev,non_blocking=True); act=act.to(dev,non_blocking=True)
            if not printed:
                print(f'[shape] pix={tuple(pix.shape)} act={tuple(act.shape)}', flush=True); printed=True
            proj,pred=latents(m,pix,act,H,K)            # (B,T,192),(B,K,192)
            # targets
            gt_all=pix.reshape(-1,3,224,224)            # for real branch (all T)
            gt_pred=pix[:,H:].reshape(-1,3,224,224)     # for predicted branch (K)
            opt.zero_grad()
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                rec_real=dec(proj.reshape(-1,192))          # NO clamp in loss (clamp kills gradients)
                rec_pred=dec(pred.reshape(-1,192))
                loss=F.mse_loss(rec_real,gt_all)+F.mse_loss(rec_pred,gt_pred)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            step+=1
            if step%200==0:
                with torch.no_grad():
                    lp=F.mse_loss(rec_pred.float(),gt_pred).item(); lr=F.mse_loss(rec_real.float(),gt_all).item()
                print(f'[step {step}/{a.max_steps}] real_mse={lr:.4f} pred_mse={lp:.4f} t={time.time()-t0:.0f}s', flush=True)
            if step in snaps:
                torch.save({'state_dict':dec.state_dict(),'args':{'hidden_dim':512,'depth':10,'heads':8,'dim_head':64},'step':step}, outdir/f'step_{step:07d}.pt')
                print(f'  saved snapshot {step}', flush=True)
            if step>=a.max_steps: break
    print(f'[done] {outdir}', flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--h5',required=True); p.add_argument('--weights_tag',required=True); p.add_argument('--out_tag',required=True)
    p.add_argument('--ad',type=int,required=True); p.add_argument('--H',type=int,default=3); p.add_argument('--K',type=int,default=8)
    p.add_argument('--batch',type=int,default=16); p.add_argument('--workers',type=int,default=6); p.add_argument('--max_steps',type=int,default=40000)
    main(p.parse_args())
