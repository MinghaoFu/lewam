# Offline collapse-metric probe: load a trained lewam_gc checkpoint's encoder, encode a
# random sample of val frames, and report the same collapse metrics the trainer logs
# (z_std / effective rank / off-diagonal cosine / SIGReg statistic). Used to get the
# converged collapse metrics of the SIGReg-enabled reproduction runs (w_reg=0.04) for a
# direct contrast against the no-SIGReg ablation. Checkpoints are per-run (latest/best),
# not per-epoch, so this is the endpoint value, not a trajectory.
import argparse, json, os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import stable_worldmodel as swm
from train_lewam_gc import build_encoder, collapse_metrics, SIGReg
from lewam.utils import get_img_preprocessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="lewam_gc_*.pt (has encoder.* keys)")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--keys_to_load", default="pixels,action,observation")
    ap.add_argument("--collapse_n", type=int, default=8192)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ktl = [k.strip() for k in a.keys_to_load.split(",") if k.strip()]
    base = swm.data.load_dataset(a.dataset_name, transform=None, cache_dir=None, num_steps=4,
                                 frameskip=5, keys_to_load=ktl,
                                 keys_to_cache=[k for k in ktl if k != "pixels"])
    fs = int(base.frameskip); adim = int(base.get_dim("action"))
    img_t = get_img_preprocessor(source="pixels", target="pixels", img_size=a.img_size)

    lewm = build_encoder(embed_dim=192, img_size=a.img_size, action_block_dim=adim * fs).to(dev)
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    r = lewm.load_state_dict(enc, strict=False)
    print(f"[collapse_eval] {a.tag} loaded encoder: {len(enc)} keys "
          f"missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}", flush=True)
    assert not r.unexpected_keys, r.unexpected_keys[:5]
    lewm.eval()
    sigreg = SIGReg().to(dev)

    lengths = np.asarray(base.lengths)
    zs, got, buf = [], 0, []
    with torch.no_grad():
        while got < a.collapse_n:
            ep = int(np.random.randint(len(lengths))); L = int(lengths[ep]); nobs = L // fs
            if nobs < 1:
                continue
            t = int(np.random.randint(nobs))
            pix = base._load_slice(ep, t * fs, t * fs + 1)["pixels"]
            if not torch.is_tensor(pix):
                pix = torch.as_tensor(np.asarray(pix))
            buf.append(img_t({"pixels": pix})["pixels"].float()[0])  # [3,H,W]
            if len(buf) >= 256:
                batch = torch.stack(buf).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z = lewm.encode({"pixels": batch.unsqueeze(1)})["emb"][:, 0]
                zs.append(z.float().cpu()); got += z.shape[0]; buf = []
    Z = torch.cat(zs)[:a.collapse_n]
    cm = collapse_metrics(Z, sigreg=sigreg, device=dev)
    print("COLLAPSE_RESULT " + json.dumps({"tag": a.tag, "checkpoint": a.checkpoint, **cm}), flush=True)


if __name__ == "__main__":
    main()
