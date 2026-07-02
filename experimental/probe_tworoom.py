#!/usr/bin/env python
"""S4 scratch-probe R2 for two-room. Frozen CONVERGED scratch base WM (canonical
vit-tiny-192, from-scratch SIGReg = tworoom_ours_lewm_weights.pt, config from
tworoom_lewm_base). Freeze encoder -> CLS latent -> Ridge(alpha=10) -> ground-truth
sim-state. tworoom h5 has NO 'state' key; the privileged full sim-state is
'observation' (10-d). Report held-out R2 over 3 split-seeds {42,0,1}.
"""
import json, os
import hdf5plugin  # register Blosc/zstd codec
import numpy as np, h5py, torch
from omegaconf import OmegaConf
import hydra
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

CKPT = "/mnt/minghao_data/.stable-wm/checkpoints"
DEC  = "/mnt/minghao_data/.stable-wm/decoders"
DATASETS = "/mnt/minghao_data/.stable-wm/datasets"
CONFIG_RUN = "tworoom_lewm_base"                 # canonical 192-d scratch config
WEIGHTS = f"{DEC}/tworoom_ours_lewm_weights.pt"  # converged scratch base
IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

def load_encoder():
    cfg = OmegaConf.create(json.loads(open(f"{CKPT}/{CONFIG_RUN}/config.json").read()))
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(WEIGHTS, map_location="cpu")
    res = model.load_state_dict(sd, strict=False)
    enc = model.encoder.eval().cuda()
    return enc, len(res.missing_keys), len(res.unexpected_keys)

@torch.no_grad()
def encode_frames(enc, pixels_uint8, bs=128):
    out=[]
    for i in range(0,len(pixels_uint8),bs):
        x = torch.from_numpy(pixels_uint8[i:i+bs]).float().div_(255.0)
        x = x.permute(0,3,1,2)
        x = (x - IMAGENET_MEAN)/IMAGENET_STD
        x = x.cuda()
        z = enc(x, interpolate_pos_encoding=True).last_hidden_state[:,0]
        out.append(z.float().cpu().numpy())
    return np.concatenate(out,0)

def main():
    with h5py.File(f"{DATASETS}/tworoom.h5","r") as f:
        N = f["pixels"].shape[0]
        idx = np.arange(0, N, max(1, N//6000))[:6000]   # ~6k state-diverse frames
        idx = np.sort(idx)  # h5py fancy-index requires increasing order
        # read only the sampled frames (dataset is ~138GB uncompressed; never load all)
        pixels = np.empty((len(idx),)+f["pixels"].shape[1:], dtype=f["pixels"].dtype)
        for j in range(0,len(idx),512):
            sl=idx[j:j+512]
            pixels[j:j+len(sl)] = f["pixels"][list(sl)]
        observation = f["observation"][list(idx)].astype(np.float32)  # 10-d full sim-state
        pos = np.concatenate([f["pos_agent"][list(idx)], f["pos_target"][list(idx)]],1).astype(np.float32) # 4-d geom
        proprio = f["proprio"][list(idx)].astype(np.float32)          # 2-d
    print(f"[tworoom] N_frames={len(idx)} obs_dim={observation.shape[1]} pos_dim={pos.shape[1]} proprio_dim={proprio.shape[1]}")
    enc, miss, unexp = load_encoder()
    print(f"load: missing={miss} unexpected={unexp}")
    Z = encode_frames(enc, pixels)
    zstd = float(Z.std(0).mean())
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    erank = float((s.sum()**2)/(s**2).sum())
    print(f"latent: dim={Z.shape[1]} zstd={zstd:.3f} erank={erank:.1f}/{Z.shape[1]}")
    targets = [("observation(full)", observation), ("pos(agent+target)", pos), ("proprio", proprio)]
    results = {name: [] for name,_ in targets}
    for seed in [42,0,1]:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(idx))
        ntr = int(0.8*len(idx))
        tr, te = perm[:ntr], perm[ntr:]
        line=[f"seed={seed}"]
        for name,Y in targets:
            mu,sd = Z[tr].mean(0), Z[tr].std(0)+1e-6
            rgr = Ridge(alpha=10.0).fit((Z[tr]-mu)/sd, Y[tr])
            r2 = r2_score(Y[te], rgr.predict((Z[te]-mu)/sd), multioutput="variance_weighted")
            results[name].append(r2); line.append(f"{name}={r2:.4f}")
        print("  "+"  ".join(line))
    print("=== MEANS (3 split-seeds {42,0,1}) ===")
    for name,_ in targets:
        arr=results[name]
        print(f"  {name:20s} {np.mean(arr):.4f}  per-seed={['%.4f'%x for x in arr]}")
    # headline = full observation state R2
    hn="observation(full)"
    print(f"HEADLINE_R2 {np.mean(results[hn]):.4f} PER_SEED {results[hn]}")

if __name__=="__main__":
    main()
