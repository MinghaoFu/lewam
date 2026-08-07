"""Materialize train_lewam_gc's preload as a reusable cache on a2f.

Streams episodes (RAM = one episode, no torch.cat peak) into an np.memmap holding the
EXACT tensor preload_frames+flatten_for_training would build (same img_t, same fp16,
same obs-granularity keep rule, same index columns), then verifies K random episodes
byte-for-byte against the reference per-episode transform.

Outputs under --out (default $A2F/preload_cache):
  <stem>_fs<FS>_i<IMG>.frames.npy   [N,3,IMG,IMG] float16 (np format, mmap-able)
  <stem>_fs<FS>_i<IMG>.aux.npz      A_flat/t_gidx/maxh/ep_base + act_mean/act_std
  <stem>_fs<FS>_i<IMG>.meta.json    provenance + counts
Write to local /tmp first, then cp to --out (never in-place writes on fuse).
"""
import os, sys, json, time, argparse, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lewam_gc as TLG  # noqa: F401  -- same-dir sibling; kept importable for its helpers
from lewam.utils import get_img_preprocessor, get_column_normalizer
import stable_worldmodel as swm
from stable_worldmodel.data.utils import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--h5", required=True)
p.add_argument("--out", required=True)
p.add_argument("--img_size", type=int, default=224)
p.add_argument("--frameskip", type=int, default=5)
p.add_argument("--verify_eps", type=int, default=3)
args = p.parse_args()

stem = os.path.basename(args.h5).replace(".h5", "")
tag = f"{stem}_fs{args.frameskip}_i{args.img_size}"
tmp = f"/tmp/preload_cache/{tag}"
os.makedirs(tmp, exist_ok=True)
os.makedirs(args.out, exist_ok=True)

# EXACT same construction as train_lewam_gc main()
_ktl = ["pixels", "action"]
base = swm.data.load_dataset(args.h5, transform=None, cache_dir=None, num_steps=4,
                             frameskip=args.frameskip, keys_to_load=_ktl,
                             keys_to_cache=[k for k in _ktl if k != "pixels"],
                             format="hdf5")
assert int(base.frameskip) == args.frameskip

act_norm = get_column_normalizer(base, "action", "action")
_zn = act_norm.lambd
act_mean = _zn.mean.squeeze(0).cpu().numpy().tolist()
act_std = _zn.std.squeeze(0).cpu().numpy().tolist()
img_t = get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size)
am = torch.tensor(act_mean, dtype=torch.float32)
astd = torch.tensor(act_std, dtype=torch.float32)
FS = args.frameskip
n_eps = len(base.lengths)
print(f"[cache] {stem}: {n_eps} episodes, act_dim={len(act_mean)}", flush=True)

def episode_tensors(ep):
    """EXACT mirror of preload_frames' per-episode body."""
    L = int(base.lengths[ep])
    sl = base._load_slice(ep, 0, L)
    pix, raw_act = sl["pixels"], sl["action"]
    if not torch.is_tensor(pix):
        pix = torch.as_tensor(np.asarray(pix))
    raw_act = raw_act if torch.is_tensor(raw_act) else torch.as_tensor(np.asarray(raw_act))
    pp = img_t({"pixels": pix})["pixels"].float()
    n_obs = L // FS
    a = raw_act[:n_obs * FS].reshape(n_obs, FS, raw_act.shape[1])
    a = (a - am) / astd
    a = a.reshape(n_obs, FS * raw_act.shape[1]).half()
    n_keep = min(pp.shape[0], n_obs + 1)
    return pp[:n_keep].half(), a

# single streaming pass: raw bin (true counts recorded as we go), then finalize .npy
t0 = time.time()
IMG = args.img_size
row_bytes = 3 * IMG * IMG * 2
bin_path = f"{tmp}/{tag}.frames.bin"
keeps = []
A_l, tg_l, mh_l, eb_l = [], [], [], []
off = 0
with open(bin_path, "wb") as fb:
    for ep in range(n_eps):
        pp, a = episode_tensors(ep)
        k = pp.shape[0]
        keeps.append(k)
        fb.write(pp.numpy().tobytes())
        n_valid = min(a.shape[0], k - 1)
        last = k - 1
        for t in range(n_valid):
            tg_l.append(off + t); mh_l.append(last - t); eb_l.append(off)
            A_l.append(a[t])
        off += k
        if (ep + 1) % 500 == 0:
            print(f"[cache] {ep+1}/{n_eps} ({time.time()-t0:.0f}s)", flush=True)
N = off
print(f"[cache] streamed N={N} frames in {time.time()-t0:.0f}s", flush=True)

frames_path = f"{tmp}/{tag}.frames.npy"
fr = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.float16,
                               shape=(N, 3, IMG, IMG))
t1 = time.time()
CH = 4096
raw = np.memmap(bin_path, dtype=np.float16, mode="r", shape=(N, 3, IMG, IMG))
for i in range(0, N, CH):
    fr[i:i + CH] = raw[i:i + CH]
fr.flush(); del fr; del raw
os.remove(bin_path)
print(f"[cache] finalized npy in {time.time()-t1:.0f}s", flush=True)
A_flat = torch.stack(A_l).numpy() if A_l else np.zeros((0, 1), np.float16)
np.savez(f"{tmp}/{tag}.aux.npz",
         A_flat=A_flat, t_gidx=np.asarray(tg_l, np.int64),
         maxh=np.asarray(mh_l, np.int64), ep_base=np.asarray(eb_l, np.int64),
         act_mean=np.asarray(act_mean), act_std=np.asarray(act_std))
json.dump({"source_h5": args.h5, "img_size": IMG, "frameskip": FS,
           "n_frames": int(N), "n_samples": len(tg_l), "n_episodes": n_eps,
           "built": time.strftime("%F %T")},
          open(f"{tmp}/{tag}.meta.json", "w"), indent=1)
print(f"[cache] built: N={N} samples={len(tg_l)}", flush=True)

# verify K random episodes byte-for-byte
fr = np.load(frames_path, mmap_mode="r")
rng = np.random.default_rng(0)
offs = np.cumsum([0] + keeps[:-1])
for ep in rng.choice(n_eps, size=min(args.verify_eps, n_eps), replace=False):
    pp, _ = episode_tensors(int(ep))
    got = fr[offs[ep]:offs[ep] + keeps[ep]]
    assert np.array_equal(got, pp.numpy()), f"VERIFY FAIL ep{ep}"
print(f"[cache] VERIFY_OK ({args.verify_eps} episodes byte-identical)", flush=True)
del fr

for f in os.listdir(tmp):
    src = f"{tmp}/{f}"; dst = f"{args.out}/{f}"
    if os.path.exists(dst):
        os.remove(dst) if os.path.isfile(dst) else shutil.rmtree(dst)
    t1 = time.time()
    shutil.copy2(src, dst) if os.path.isfile(src) else shutil.copytree(src, dst)
    print(f"[cache] copied {f} -> {args.out} ({time.time()-t1:.0f}s)", flush=True)
shutil.rmtree(tmp)
print("[cache] DONE", flush=True)
