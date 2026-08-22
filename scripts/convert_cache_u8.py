"""Convert a normalized-fp16 preload cache to raw uint8 (4x smaller, page-cacheable).
Inverse of the trainer's uint8 normalize branch, so the round trip is exact up to the
original quantization. Writes under a NEW cache root with the same stem/tag; aux npz
copied unchanged. Generalized from the pusht one-off (convert_pusht_u8)."""

import argparse
import os
import shutil

import numpy as np
import numpy.lib.format as npy_fmt

p = argparse.ArgumentParser()
p.add_argument("--src_dir", required=True, help="fp16 cache dir (contains <tag>.frames.npy)")
p.add_argument("--dst_dir", required=True, help="u8 cache dir to create")
p.add_argument("--tag", required=True, help="cache tag, e.g. cube_single_sterm_fs5_i224_raw")
p.add_argument("--chunk", type=int, default=2048)
args = p.parse_args()

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

os.makedirs(args.dst_dir, exist_ok=True)
src = np.load(f"{args.src_dir}/{args.tag}.frames.npy", mmap_mode="r")
n = src.shape[0]
# fuse forbids write-mmap (EOPNOTSUPP): stream the npy sequentially instead
with open(f"{args.dst_dir}/{args.tag}.frames.npy", "wb") as out:
    npy_fmt.write_array_header_2_0(
        out, {"descr": "|u1", "fortran_order": False, "shape": tuple(src.shape)})
    for i in range(0, n, args.chunk):
        f = np.asarray(src[i:i + args.chunk], np.float32)
        x = np.clip(np.rint((f * STD + MEAN) * 255.0), 0, 255).astype(np.uint8)
        out.write(x.tobytes())
        if (i // args.chunk) % 20 == 0:
            print(f"[u8] {i}/{n}", flush=True)
shutil.copyfile(f"{args.src_dir}/{args.tag}.aux.npz", f"{args.dst_dir}/{args.tag}.aux.npz")
if os.path.exists(f"{args.src_dir}/{args.tag}.meta.json"):
    shutil.copyfile(f"{args.src_dir}/{args.tag}.meta.json", f"{args.dst_dir}/{args.tag}.meta.json")

# round-trip check on a random slab: u8 -> normalize must match the fp16 source
dst = np.load(f"{args.dst_dir}/{args.tag}.frames.npy", mmap_mode="r")
idx = np.random.default_rng(0).integers(0, n, 4)
err = 0.0
for i in idx:
    back = (dst[i].astype(np.float32) / 255.0 - MEAN[0]) / STD[0]
    err = max(err, float(np.abs(back - np.asarray(src[i], np.float32)).max()))
print(f"[u8] roundtrip max_abs_err={err:.6f} (expect < 0.02)", flush=True)
assert err < 0.02
print("[u8] DONE", flush=True)
