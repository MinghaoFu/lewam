#!/usr/bin/env python
"""Frozen-encoder linear probe (the DINOv2 ablation's PRIMARY axis — measures latent
quality directly, independent of whether CEM can plan; answers "does DINOv2-init give
a more state-decodable latent" without needing planning to work).

For each arm's CONVERGED base WM: freeze the encoder, encode held-out frames, fit a
Ridge probe latent->ground-truth sim state, report held-out R². Clean comparison =
dinov2_pretrained vs dinov2_scratch (same 384-d space). Canonical (192-d) is a
dimension-confounded reference, reported but not the arbiter.

Run (from the repo dir, minghao.fu env):  python frozen_probe.py <task>
"""
import sys, json, os
import numpy as np, h5py, torch
import hdf5plugin  # register Blosc/zstd codec for compressed pixels
from omegaconf import OmegaConf
import hydra
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

CKPT = "/mnt/minghao_data/.stable-wm/checkpoints"
DATASETS = "/mnt/minghao_data/.stable-wm/datasets"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# task -> {arm_label: checkpoint_dir} ; resumed tool_hang dinov2_scratch lives in _r70
ARMS = {
    "lift": {
        "dinov2_scratch(384)":   ("lift_lewm_dinov2_scratch", 100),
        "dinov2_pretrained(384)":("lift_lewm_dinov2_pretrained", 100),
    },
    "tool_hang": {
        "canonical(192)":        ("tool_hang_lewm_scratch", 100),
        "dinov2_scratch(384)":   ("tool_hang_lewm_dinov2_scratch_r70", 30),
        "dinov2_pretrained(384)":("tool_hang_lewm_dinov2_pretrained", 100),
    },
    "transport": {
        "canonical(192)":        ("transport_lewm_scratch", 100),
        "dinov2_scratch(384)":   ("transport_lewm_dinov2_scratch", 100),
        "dinov2_pretrained(384)":("transport_lewm_dinov2_pretrained", 100),
    },
    "can": {
        "dinov2_scratch(384)":   ("can_lewm_dinov2_scratch", 100),
        "dinov2_pretrained(384)":("can_lewm_dinov2_pretrained", 100),
    },
    "square": {
        "dinov2_scratch(384)":   ("square_lewm_dinov2_scratch", 100),
        "dinov2_pretrained(384)":("square_lewm_dinov2_pretrained", 100),
    },
}


def load_encoder(run_name, epoch):
    cfg = OmegaConf.create(json.loads(open(f"{CKPT}/{run_name}/config.json").read()))
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(f"{CKPT}/{run_name}/weights_epoch_{epoch}.pt", map_location="cpu")
    res = model.load_state_dict(sd, strict=False)
    enc = model.encoder.eval().cuda()
    return enc, len(res.missing_keys), len(res.unexpected_keys)


@torch.no_grad()
def encode_frames(enc, pixels_uint8, bs=128):
    out = []
    for i in range(0, len(pixels_uint8), bs):
        x = torch.from_numpy(pixels_uint8[i:i+bs]).float().div_(255.0)      # (B,H,W,3) 0-1
        x = x.permute(0, 3, 1, 2)                                            # (B,3,H,W)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = x.cuda()
        z = enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]    # CLS
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0)


def main(task):
    # sample held-out frames: take a fixed stride across the dataset (state-diverse, reproducible)
    with h5py.File(f"{DATASETS}/{task}.h5", "r") as f:
        N = f["state"].shape[0]
        idx = np.arange(0, N, max(1, N // 6000))[:6000]      # ~6k frames
        pixels = f["pixels"][:][idx]                          # load then index (h5 fancy-index slow)
        state = f["state"][:][idx].astype(np.float32)
        proprio = f["proprio"][:][idx].astype(np.float32)
    # train/test split (last 20% held out)
    ntr = int(0.8 * len(idx))
    print(f"[{task}] N_frames={len(idx)} state_dim={state.shape[1]} proprio_dim={proprio.shape[1]} train={ntr}")
    print(f"{'arm':24s} {'miss/unexp':10s} {'state_R2':>9s} {'proprio_R2':>11s}")
    for arm, (run, ep) in ARMS[task].items():
        if not os.path.exists(f"{CKPT}/{run}/weights_epoch_{ep}.pt"):
            print(f"{arm:24s}  MISSING {run}/weights_epoch_{ep}.pt"); continue
        enc, miss, unexp = load_encoder(run, ep)
        Z = encode_frames(enc, pixels)
        del enc; torch.cuda.empty_cache()
        # collapse diagnostics: mean per-dim std + effective rank (participation ratio of singular values)
        zstd = float(Z.std(0).mean())
        s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        erank = float((s.sum()**2) / (s**2).sum())          # ~D if full-rank, ~1 if collapsed
        Ztr, Zte = Z[:ntr], Z[ntr:]
        r2s = {}
        for name, Y in [("state", state), ("proprio", proprio)]:
            mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
            rgr = Ridge(alpha=10.0).fit((Ztr-mu)/sd, Y[:ntr])
            r2s[name] = r2_score(Y[ntr:], rgr.predict((Zte-mu)/sd), multioutput="variance_weighted")
        print(f"{arm:24s} {miss:3d}/{unexp:<6d} {r2s['state']:9.4f} {r2s['proprio']:11.4f}  zstd={zstd:.3f} erank={erank:.1f}/{Z.shape[1]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tool_hang")
