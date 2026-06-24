"""Convert DINO-WM deformable (rope / granular) → LeWM flat HDF5.

Source layout inside `deformable_combined.zip`:
  deformable/<obj>/states.pth        (1000, 20, N_particles, 4) float32
  deformable/<obj>/actions.pth       (1000, 20, 4) float64
  deformable/<obj>/<idx:06d>/obses.pth   (20, 224, 224, 3) float32

We skip `state` in the LeWM h5 (particle field is huge and not needed for training;
planning eval would need a different state representation anyway). Proprio is
a dummy 1-D zero (matches TCWM's DeformDataset convention).

Usage: python convert_deformable_h5.py rope
       python convert_deformable_h5.py granular
"""
from __future__ import annotations
import argparse, io, time, zipfile
from pathlib import Path
import h5py, hdf5plugin, numpy as np, torch

ZIP_PATH = Path('/mnt/data_nvme1/minghao.fu/scratch/deformable_combined.zip')
DEST_ROOT = Path('/mnt/data_nvme1/minghao.fu/.stable-wm')
TARGET_HW = 224


def main(obj: str):
    dst = DEST_ROOT / f'{obj}.h5'
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    print(f'[convert] obj={obj}, dst={dst}')
    t0 = time.time()

    zf = zipfile.ZipFile(ZIP_PATH, 'r')
    actions = torch.load(io.BytesIO(zf.read(f'deformable/{obj}/actions.pth')), weights_only=False).numpy().astype(np.float32)
    n_eps, T, action_dim = actions.shape
    print(f'[convert] {n_eps} episodes, T={T}, action_dim={action_dim}')

    # Probe one obses to settle pixel value range
    sample = torch.load(io.BytesIO(zf.read(f'deformable/{obj}/000000/obses.pth')), weights_only=False).numpy()
    print(f'[convert] obses sample shape={sample.shape} dtype={sample.dtype} min={sample.min():.3f} max={sample.max():.3f}')
    # Decide cast factor based on observed max
    if sample.max() <= 1.5:
        pixel_scale = 255.0
    else:
        pixel_scale = 1.0
    print(f'[convert] pixel_scale={pixel_scale}')

    N = n_eps * T
    ep_len = np.full(n_eps, T, dtype=np.int32)
    ep_offset = (np.arange(n_eps) * T).astype(np.int64)
    proprio_dim = 1  # dummy

    with h5py.File(dst, 'w') as f_out:
        ds_pixels = f_out.create_dataset(
            'pixels', shape=(N, TARGET_HW, TARGET_HW, 3), dtype=np.uint8,
            chunks=(1, TARGET_HW, TARGET_HW, 3),
            **hdf5plugin.Blosc(cname='zstd', clevel=3),
        )
        f_out.create_dataset('action', data=actions.reshape(N, action_dim))
        f_out.create_dataset('proprio', data=np.zeros((N, proprio_dim), dtype=np.float32))
        f_out.create_dataset('ep_len', data=ep_len)
        f_out.create_dataset('ep_offset', data=ep_offset)

        offset = 0
        for i in range(n_eps):
            obses = torch.load(io.BytesIO(zf.read(f'deformable/{obj}/{i:06d}/obses.pth')), weights_only=False).numpy()
            # obses: (T, 224, 224, 3) float32. Cast → uint8.
            obses_u8 = np.clip(obses * pixel_scale, 0, 255).astype(np.uint8)
            ds_pixels[offset:offset+T] = obses_u8
            offset += T
            if (i+1) % 50 == 0 or i == n_eps-1:
                print(f'  [{i+1}/{n_eps}] offset={offset}/{N}  t={time.time()-t0:.1f}s')

    zf.close()
    print(f'[convert] done: {dst} ({dst.stat().st_size / 2**30:.2f} GB) in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('obj', choices=['rope', 'granular'])
    main(p.parse_args().obj)
