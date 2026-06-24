"""Convert robomimic native image_384_v15.hdf5 → LeWM flat HDF5.

Per-demo groups in source → one flat h5 with ep_len/ep_offset and concatenated
pixels (downscaled to 224x224), action, proprio, state arrays.

Usage:
    python convert_robomimic_h5.py <task>
    python convert_robomimic_h5.py lift
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import h5py
import hdf5plugin
import numpy as np
import cv2

PROPRIO_KEYS = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel']
TARGET_HW = 224
ROBOMIMIC_ROOT = Path('/var/lib/docker/data/minghao_home/robomimic')
DEST_ROOT = Path('/mnt/minghao_data/.stable-wm')


def demo_sort_key(name: str) -> int:
    # demo_0, demo_1, demo_10 → 0, 1, 10 (numeric, not lexicographic)
    return int(name.split('_', 1)[1])


def downscale_uint8(imgs: np.ndarray, size: int) -> np.ndarray:
    # imgs: (T, H, W, 3) uint8. Use INTER_AREA for downscale (best quality).
    out = np.empty((imgs.shape[0], size, size, 3), dtype=np.uint8)
    for t in range(imgs.shape[0]):
        out[t] = cv2.resize(imgs[t], (size, size), interpolation=cv2.INTER_AREA)
    return out


def main(task: str):
    src = ROBOMIMIC_ROOT / task / 'ph' / 'image_384_v15.hdf5'
    dst = DEST_ROOT / f'{task}.h5'
    assert src.exists(), f'source not found: {src}'
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    print(f'[convert] src: {src}\n[convert] dst: {dst}')
    t0 = time.time()

    with h5py.File(src, 'r', swmr=True) as f_in:
        demos = sorted(f_in['data'].keys(), key=demo_sort_key)
        print(f'[convert] {len(demos)} demos; sample sizes: {[int(f_in[f"data/{d}"].attrs["num_samples"]) for d in demos[:5]]}')

        # First pass: compute per-demo lengths, total N, and probe action / proprio dims
        ep_len = np.array([int(f_in[f'data/{d}'].attrs['num_samples']) for d in demos], dtype=np.int32)
        N = int(ep_len.sum())
        action_dim = f_in[f'data/{demos[0]}/actions'].shape[1]
        state_dim = f_in[f'data/{demos[0]}/states'].shape[1]
        proprio_dim = sum(f_in[f'data/{demos[0]}/obs/{k}'].shape[1] for k in PROPRIO_KEYS)
        print(f'[convert] N={N}, action_dim={action_dim}, state_dim={state_dim}, proprio_dim={proprio_dim}')

        # Open destination with compressed pixels
        with h5py.File(dst, 'w') as f_out:
            ds_pixels = f_out.create_dataset(
                'pixels', shape=(N, TARGET_HW, TARGET_HW, 3), dtype=np.uint8,
                chunks=(1, TARGET_HW, TARGET_HW, 3),
                **hdf5plugin.Blosc(cname='zstd', clevel=3),
            )
            ds_action = f_out.create_dataset('action', shape=(N, action_dim), dtype=np.float32)
            ds_state = f_out.create_dataset('state', shape=(N, state_dim), dtype=np.float32)
            ds_proprio = f_out.create_dataset('proprio', shape=(N, proprio_dim), dtype=np.float32)
            f_out.create_dataset('ep_len', data=ep_len)
            f_out.create_dataset('ep_offset', data=np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64))

            # Second pass: stream each demo, write to flat arrays
            offset = 0
            for i, d in enumerate(demos):
                T = int(ep_len[i])
                src_demo = f_in[f'data/{d}']
                imgs_384 = src_demo['obs/agentview_image'][:]  # (T, 384, 384, 3)
                ds_pixels[offset:offset+T] = downscale_uint8(imgs_384, TARGET_HW)
                ds_action[offset:offset+T] = src_demo['actions'][:].astype(np.float32)
                ds_state[offset:offset+T] = src_demo['states'][:].astype(np.float32)
                pp = np.concatenate([src_demo[f'obs/{k}'][:] for k in PROPRIO_KEYS], axis=1).astype(np.float32)
                ds_proprio[offset:offset+T] = pp
                offset += T
                if (i+1) % 20 == 0 or i == len(demos)-1:
                    print(f'  [{i+1}/{len(demos)}] ep_len={T}  offset={offset}/{N}  t={time.time()-t0:.1f}s')

    print(f'[convert] done: {dst} ({dst.stat().st_size / 2**30:.2f} GB) in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='lift, can, square, transport, tool_hang')
    args = parser.parse_args()
    main(args.task)
