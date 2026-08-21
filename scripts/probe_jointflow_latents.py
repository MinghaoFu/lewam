"""First-pass probes on jointflow encoder latents: geometry (spectrum, effective rank) plus
ridge probes for object poses and inverse dynamics. Compares checkpoints trained under
different anti-collapse regimes on the same frames.

    L = z-scored latents; probes are closed-form ridge on a train/test split.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

FACTORS_TOOLHANG = "stand:10:13,frame:17:20,tool:24:27"


def parse_factors(spec):
    """"name:lo:hi,..." -> {name: (lo, hi)}. A name ending in _angle with a single column
    is probed as (sin, cos) of that column (wraparound-safe)."""
    out = {}
    for part in spec.split(","):
        name, lo, hi = part.split(":")
        out[name] = (int(lo), int(hi))
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dirs", required=True,
                    help="comma-separated run dirs, each with jointflow_best.pt + jointflow_config.json")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--n_frames", type=int, default=20000)
    ap.add_argument("--idm_gap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--factors", default=FACTORS_TOOLHANG,
                    help='state-column blocks to probe, "name:lo:hi,...". pusht: '
                         '"agent:0:2,block:2:4,block_angle:4:5"')
    ap.add_argument("--pixel_scale", default="norm", choices=["norm", "raw255"],
                    help="encoder input scale: norm = /255+ImageNet (fp16-cache-era ckpts); "
                         "raw255 = raw 0-255 floats (uint8-cache-era ckpts, pixel-scale bug)")
    ap.add_argument("--out", default="/tmp/probe_latents.json")
    return ap.parse_args()


def ridge_r2(z_train, y_train, z_test, y_test, lam=1e-3):
    """Per-column R^2 of closed-form ridge, averaged over the target block."""
    zm, zs = z_train.mean(0), z_train.std(0) + 1e-8
    xtr = (z_train - zm) / zs
    xte = (z_test - zm) / zs
    xtr = np.concatenate([xtr, np.ones((len(xtr), 1))], 1)
    xte = np.concatenate([xte, np.ones((len(xte), 1))], 1)
    a = xtr.T @ xtr + lam * len(xtr) * np.eye(xtr.shape[1])
    w = np.linalg.solve(a, xtr.T @ y_train)
    pred = xte @ w
    ss_res = ((y_test - pred) ** 2).sum(0)
    ss_tot = ((y_test - y_test.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def geometry(z):
    cov = np.cov(z.T)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)[::-1]
    p = eig / (eig.sum() + 1e-12)
    erank = float(np.exp(-(p * np.log(p + 1e-20)).sum()))   # Roy-Vetterli effective rank
    csum = np.cumsum(p)
    return {"rank_90pct": int(np.searchsorted(csum, 0.90) + 1),
            "rank_99pct": int(np.searchsorted(csum, 0.99) + 1),
            "effective_rank": erank,
            "spectrum_normalized": [float(x) for x in p]}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    f = h5py.File(args.h5, "r")
    n_total = f["pixels"].shape[0]
    ep_key = "ep_idx" if "ep_idx" in f else "episode_idx"
    ep_idx = np.asarray(f[ep_key][:])
    # frame sample: only rows whose t+gap stays inside the same episode, so every row
    # also serves as an IDM pair start
    ok = np.zeros(n_total, dtype=bool)
    ok[:-args.idm_gap] = ep_idx[:-args.idm_gap] == ep_idx[args.idm_gap:]
    idx = rng.choice(np.flatnonzero(ok), size=min(args.n_frames, int(ok.sum())), replace=False)
    idx = np.sort(idx)

    state = np.asarray(f["state"][:])[idx].astype(np.float64)
    action = np.asarray(f["action"][:]).astype(np.float64)
    proprio = np.asarray(f["proprio"][:])[idx].astype(np.float64)
    idm_y = np.stack([action[i:i + args.idm_gap].reshape(-1) for i in idx])

    n = len(idx)
    split = int(0.8 * n)
    perm = rng.permutation(n)
    tr, te = perm[:split], perm[split:]

    # read every needed frame ONCE into RAM (~6GB for 40k frames); h5 fancy reads are far
    # too slow to repeat per checkpoint
    all_rows = np.unique(np.concatenate([idx, idx + args.idm_gap]))
    row_pos = {r: i for i, r in enumerate(all_rows)}
    print(f"[probe] caching {len(all_rows)} frames to RAM", flush=True)
    px_cache = np.empty((len(all_rows), 224, 224, 3), dtype=np.uint8)
    for b0 in range(0, len(all_rows), 2048):
        rb = all_rows[b0:b0 + 2048]
        span = int(rb[-1]) - int(rb[0]) + 1
        if span <= 4 * len(rb):        # dense rows: one span read
            px_cache[b0:b0 + len(rb)] = f["pixels"][rb[0]:rb[-1] + 1][rb - rb[0]]
        else:                          # sparse rows (big h5): sorted fancy read, no span blowup
            px_cache[b0:b0 + len(rb)] = f["pixels"][rb.tolist()]
    print("[probe] frame cache ready", flush=True)

    def encode_all(model, rows):
        pos = np.array([row_pos[r] for r in rows])
        zs = []
        with torch.no_grad():
            for b0 in range(0, len(pos), args.batch_size):
                pb = pos[b0:b0 + args.batch_size]
                px = torch.from_numpy(px_cache[pb]).permute(0, 3, 1, 2).float()
                if args.pixel_scale == "norm":
                    px = (px / 255.0 - _IMG_MEAN) / _IMG_STD
                px = px.to(device)
                zs.append(model.encode(px).cpu().numpy())
        return np.concatenate(zs).astype(np.float64)

    from lewam.models.jointflow import build_model

    def load_probe_model(run_dir, device):
        """A jointflow run dir (jointflow_config.json) loads the full model; a lewam_unified
        run dir (lewam_unified_config.json) loads ONLY its VisionEncoder submodule, so a
        reference encoder (e.g. the pusht idm05 SOTA) probes on the same frames."""
        jf_cfg = run_dir / "jointflow_config.json"
        if jf_cfg.exists():
            cfg = json.loads(jf_cfg.read_text())
            model = build_model(cfg)
            model.load_state_dict(torch.load(run_dir / "jointflow_best.pt", map_location="cpu"),
                                  strict=True)
            return model.to(device).eval()
        from types import SimpleNamespace
        from lewam.models.module import VisionEncoder
        cfg = json.loads((run_dir / "lewam_unified_config.json").read_text())
        enc = VisionEncoder(img_size=224, size=cfg.get("encoder_size", "tiny"),
                            output_dim=int(cfg["z_dim"]),
                            backbone=cfg.get("encoder_backbone", "scratch"))
        sd = torch.load(run_dir / "lewam_unified_best.pt", map_location="cpu")
        enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items()
                             if k.startswith("encoder.")}, strict=True)
        return SimpleNamespace(encode=enc.to(device).eval())

    results = {}
    for run_dir in args.ckpt_dirs.split(","):
        run_dir = Path(run_dir)
        name = run_dir.name if run_dir.name else run_dir.parent.name
        model = load_probe_model(run_dir, device)

        z = encode_all(model, idx)
        z_next = encode_all(model, idx + args.idm_gap)

        res = {"geometry": geometry(z)}
        for fac, (lo, hi) in parse_factors(args.factors).items():
            y = state[:, lo:hi]
            if fac.endswith("_angle") and hi - lo == 1:
                y = np.concatenate([np.sin(y), np.cos(y)], 1)
            res[f"probe_{fac}_r2"] = ridge_r2(z[tr], y[tr], z[te], y[te])
        if proprio.std(0).max() > 1e-6:
            res["probe_proprio_r2"] = ridge_r2(z[tr], proprio[tr], z[te], proprio[te])
        zz = np.concatenate([z, z_next], 1)
        res["probe_idm_r2"] = ridge_r2(zz[tr], idm_y[tr], zz[te], idm_y[te])
        res["probe_idm_from_zt_only_r2"] = ridge_r2(z[tr], idm_y[tr], z[te], idm_y[te])

        results[name] = res
        print(f"[probe] {name}: {json.dumps(res)}", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"[probe] wrote {args.out}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, res in results.items():
            ax.plot(np.arange(1, len(res["geometry"]["spectrum_normalized"]) + 1),
                    res["geometry"]["spectrum_normalized"], label=name)
        ax.set_yscale("log")
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("normalized eigenvalue")
        ax.set_title("latent covariance spectra (normalized)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = str(Path(args.out).with_suffix(".png"))
        fig.savefig(png, dpi=150)
        print(f"[probe] wrote {png}", flush=True)
    except Exception as ex:
        print(f"[probe] spectrum plot skipped: {ex}", flush=True)


if __name__ == "__main__":
    main()
