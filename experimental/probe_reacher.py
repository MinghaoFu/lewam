#!/usr/bin/env python
"""Frozen-encoder linear probe for the reacher scratch base WM (reacher_lewm_base).
Mirrors frozen_probe.py's recipe: freeze encoder, encode held-out frames, fit
Ridge(alpha=10) on z-scored CLS latent -> ground-truth sim-state, held-out
variance_weighted R² (last 20% held out). Privileged sim-state =
[qpos(2), qvel(2), finger_pos(2), target_pos(2)] (full physical state; object=target).
Also reports the standard 6-D `observation` for reference.
"""
import json, os
import numpy as np, h5py, torch
import hdf5plugin  # register Blosc/zstd codec for compressed pixels  # noqa
from omegaconf import OmegaConf
import hydra
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

CKPT = "/mnt/minghao_data/.stable-wm/checkpoints"
DATASETS = "/mnt/minghao_data/.stable-wm/datasets"
RUN = "reacher_lewm_base"; EP = 0
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load_encoder(run, epoch):
    cfg = OmegaConf.create(json.loads(open(f"{CKPT}/{run}/config.json").read()))
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(f"{CKPT}/{run}/weights_epoch_{epoch}.pt", map_location="cpu")
    res = model.load_state_dict(sd, strict=False)
    enc = model.encoder.eval().cuda()
    return enc, len(res.missing_keys), len(res.unexpected_keys)

@torch.no_grad()
def encode_idx(enc, dset, idx, bs=128):
    out = []
    for i in range(0, len(idx), bs):
        chunk = idx[i:i+bs]
        px = dset[np.sort(chunk)]  # h5 needs increasing order; idx is already a sorted stride
        x = torch.from_numpy(px).float().div_(255.0)
        x = x.permute(0, 3, 1, 2)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = x.cuda()
        z = enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0)

def main():
    with h5py.File(f"{DATASETS}/reacher.h5", "r") as f:
        N = f["pixels"].shape[0]
        idx = np.arange(0, N, max(1, N // 6000))[:6000]   # sorted stride, state-diverse, reproducible
        qpos = f["qpos"][:][idx].astype(np.float32)
        qvel = f["qvel"][:][idx].astype(np.float32)
        finger = f["finger_pos"][:][idx].astype(np.float32)
        target = f["target_pos"][:][idx].astype(np.float32)
        obs = f["observation"][:][idx].astype(np.float32)
        enc, miss, unexp = load_encoder(RUN, EP)
        Z = encode_idx(enc, f["pixels"], idx)
    del enc; torch.cuda.empty_cache()
    state = np.concatenate([qpos, qvel, finger, target], axis=1)  # 8-D privileged
    ntr = int(0.8 * len(idx))
    print(f"[reacher] N_frames={len(idx)} state_dim={state.shape[1]} obs_dim={obs.shape[1]} train={ntr}")
    zstd = float(Z.std(0).mean())
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    erank = float((s.sum()**2) / (s**2).sum())
    Ztr, Zte = Z[:ntr], Z[ntr:]
    for name, Y in [("state(qpos,qvel,finger,target;8d)", state), ("observation(6d)", obs)]:
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
        rgr = Ridge(alpha=10.0).fit((Ztr-mu)/sd, Y[:ntr])
        r2 = r2_score(Y[ntr:], rgr.predict((Zte-mu)/sd), multioutput="variance_weighted")
        print(f"[reacher] {RUN}/ep{EP}  miss={miss} unexp={unexp}  latent -> {name}: R2 = {r2:.4f}  zstd={zstd:.3f} erank={erank:.1f}/{Z.shape[1]}")

if __name__ == "__main__":
    main()
