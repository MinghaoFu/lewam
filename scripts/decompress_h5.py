"""Write an uncompressed copy of an h5's image columns, read by the trainers under --data_source decomp.

The h5 stores frames gzip-compressed, so reading it directly can be much slower due to decompressing every call.
This copy does that work once at the cost of disk space: N x 3 x H x W bytes per column (2-11x the h5's size on disk). 
See docs/DATA.md for when we recommend building it.
"""

import argparse
import json
import os
import shutil
import h5py
import numpy as np

from lewam.train.datasets import H5Frames

ROWS_PER_READ = 1024


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_path", required=True, help="the h5 dataset to decompress")
    argparser.add_argument("--decomp_dir", required=True,
                           help="output dir; pass this same value to the trainer")
    argparser.add_argument("--views", default="pixels", help="comma separated list of the subset of h5 " \
                          "image columns to decompress")
    return argparser.parse_args()


def check_disk(frames: list[H5Frames], out_dir: str) -> None:
    """Check disk space and stop writing if the copies can't fit."""
    needed = sum(int(np.prod(f.shape)) for f in frames)
    free = shutil.disk_usage(out_dir).free
    print(f"[decompress] {', '.join(f'{f.img_column} {tuple(f.shape)}' for f in frames)} -> "
          f"{needed / 1e9:.1f} GB in {out_dir} ({free / 1e9:.1f} GB free)", flush=True)
    assert needed < free, f"not enough disk: need {needed / 1e9:.1f} GB, {free / 1e9:.1f} GB free"


def decompress_column(source: h5py.Dataset, frames: H5Frames, out_path: str) -> None:
    """Write one column channels-first as a `.npy`"""
    partial = out_path + ".partial"
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=tuple(frames.shape))
    for i in range(0, frames.shape[0], ROWS_PER_READ):
        rows = source[i:i + ROWS_PER_READ]
        out[i:i + len(rows)] = rows.transpose(0, 3, 1, 2) if frames.channels_last else rows
        print(f"[decompress] {frames.img_column}: {i + len(rows)}/{frames.shape[0]}", end="\r", flush=True)
    out.flush()
    del out
    os.replace(partial, out_path)
    print(f"\n[decompress] wrote {out_path}", flush=True)


def write_meta(out_dir: str, dataset_path: str, frames: list[H5Frames]) -> None:
    """Record the source h5 and each column's shape, keeping columns written by earlier runs."""
    meta_path = os.path.join(out_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {"views": {}}
    meta["source"] = os.path.abspath(dataset_path)
    meta["views"].update({f.img_column: list(f.shape) for f in frames})
    json.dump(meta, open(meta_path, "w"), indent=1)


def main():
    args = parse_args()

    stem = os.path.splitext(os.path.basename(args.dataset_path))[0]
    out_dir = os.path.join(args.decomp_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    frames = [H5Frames(args.dataset_path, v) for v in args.views.split(",") if v]
    # remove partial files from earlier runs
    for column_frames in frames:
        partial = os.path.join(out_dir, f"{column_frames.img_column}.npy.partial")
        if os.path.isfile(partial):
            os.remove(partial)

    check_disk(frames, out_dir)
    with h5py.File(args.dataset_path, "r") as f:
        for column_frames in frames:
            decompress_column(f[column_frames.img_column], column_frames,
                              os.path.join(out_dir, f"{column_frames.img_column}.npy"))
            
    write_meta(out_dir, args.dataset_path, frames)
    print("[decompress] done", flush=True)


if __name__ == "__main__":
    main()
