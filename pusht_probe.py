#!/usr/bin/env python
"""S4 scratch-probe: pusht. Frozen CLS latent (pusht_lewm_base, the scratch LeWM
base, vit-tiny patch14 224, pretrained=False, epoch 0) -> GT sim-state (pusht
'state', 7-D), held-out Ridge(alpha=10) R². Matches §6/cube_probe recipe:
EPISODE-LEVEL 80/20 split (no shared episodes), ImageNet-normalized pixels,
variance_weighted R². 3 split seeds {42,0,1} (probe is fast/no-train; SR-style
N=50 does not apply to an R^2 probe). Reports per-seed + mean linear R2 (+ RBF)."""
import json, os
import numpy as np, h5py, torch, hdf5plugin
from omegaconf import OmegaConf
import hydra
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

CKPT = "/mnt/minghao_data/.stable-wm/checkpoints"
H5 = "/mnt/minghao_data/.stable-wm/datasets/pusht_expert_train.h5"
RUN, EP = "pusht_lewm_base", 0
IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SEEDS = [42, 0, 1]


def load_encoder(run, ep):
    cfg = OmegaConf.create(json.loads(open(f"{CKPT}/{run}/config.json").read()))
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(f"{CKPT}/{run}/weights_epoch_{ep}.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    enc_missing = [k for k in missing if k.startswith("encoder.")]
    assert not enc_missing, f"encoder weights missing: {enc_missing[:5]}"
    return model.encoder.eval().cuda()


@torch.no_grad()
def encode(enc, px, bs=128):
    out = []
    for i in range(0, len(px), bs):
        x = torch.from_numpy(px[i:i+bs]).float().div_(255.0).permute(0, 3, 1, 2)
        x = ((x - IMEAN) / ISTD).cuda()
        z = enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    with h5py.File(H5, "r") as f:
        N = f["pixels"].shape[0]
        idx = np.arange(0, N, max(1, N // 6000))[:6000]
        idx = np.sort(np.unique(idx))
        px = f["pixels"][idx]
        state = f["state"][idx].astype(np.float32)
        ep = f["episode_idx"][idx]
    enc = load_encoder(RUN, EP)
    Z = encode(enc, px)
    del enc; torch.cuda.empty_cache()
    print(f"[pusht] RUN={RUN} ep={EP} N={len(idx)} latent={Z.shape[1]} state_dim={state.shape[1]}", flush=True)
    uniq_ep = np.unique(ep)
    print(f"[pusht] {len(uniq_ep)} episodes sampled; episode-level 80/20 split", flush=True)
    lin, rbf = [], []
    print(f"{'seed':>5s} {'ntr':>6s} {'nte':>6s} {'overlap':>8s} {'R2_lin':>10s} {'R2_rbf':>10s}", flush=True)
    for s in SEEDS:
        rng = np.random.default_rng(s)
        ep_order = rng.permutation(uniq_ep)
        n_tr_ep = int(0.8 * len(ep_order))
        tr_e = set(ep_order[:n_tr_ep].tolist())
        te_e = set(ep_order[n_tr_ep:].tolist())
        tr_mask = np.array([e in tr_e for e in ep])
        tr_i = np.where(tr_mask)[0]
        te_i = np.where(~tr_mask)[0]
        ov = len(tr_e & te_e)
        Ztr, Zte = Z[tr_i], Z[te_i]
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
        Xtr, Xte = (Ztr-mu)/sd, (Zte-mu)/sd
        Ytr, Yte = state[tr_i], state[te_i]
        r2l = r2_score(Yte, Ridge(alpha=10.0).fit(Xtr, Ytr).predict(Xte), multioutput="variance_weighted")
        krr = KernelRidge(kernel="rbf", alpha=10.0, gamma=1.0/Xtr.shape[1]).fit(Xtr, Ytr)
        r2r = r2_score(Yte, krr.predict(Xte), multioutput="variance_weighted")
        lin.append(r2l); rbf.append(r2r)
        print(f"{s:5d} {len(tr_i):6d} {len(te_i):6d} {ov:8d} {r2l:10.4f} {r2r:10.4f}", flush=True)
    print(f"[pusht] MEAN R2_lin = {np.mean(lin):.4f}  (per-seed {[round(x,4) for x in lin]})", flush=True)
    print(f"[pusht] MEAN R2_rbf = {np.mean(rbf):.4f}  (per-seed {[round(x,4) for x in rbf]})", flush=True)


if __name__ == "__main__":
    main()
