"""Materialize train_lewam_gc's preload as a reusable cache on a2f.

Outputs under --out (default $A2F/preload_cache):
  <stem>_fs<FS>_i<IMG>.frames.npy   [N,3,IMG,IMG] float16 (np format, mmap-able)
  <stem>_fs<FS>_i<IMG>.aux.npz      A_flat/t_gidx/maxh/ep_base/frames_to_terminal
                                    + act_mean/act_std
  <stem>_fs<FS>_i<IMG>.meta.json    provenance + counts + terminal_state rule
Write to local /tmp first, then cp to --out (never in-place writes on fuse).
"""
import argparse
import json
import os
import shutil
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lewam_gc as TLG
from lewam.utils import get_img_preprocessor, get_column_normalizer
import stable_worldmodel as swm

p = argparse.ArgumentParser()
p.add_argument("--h5", required=True)
p.add_argument("--out", required=True)
p.add_argument("--img_size", type=int, default=224)
p.add_argument("--frameskip", type=int, default=5)
p.add_argument("--verify_eps", type=int, default=3)
p.add_argument("--terminal_state", choices=["last", "success"], default="last",
               help="terminal frame policy: 'last' or first successful frame")
p.add_argument("--success_key", default="success",
               help="per-row bool column read for --terminal_state success")
p.add_argument("--patch_terminal", action="store_true",
               help="patch terminal metadata in an existing cache")
p.add_argument("--max_eps", type=int, default=0, help="cap episodes (0 = all); smoke/debug only")
p.add_argument("--u8", action="store_true",
               help="store fs-strided frames as raw uint8 (inverse of the ImageNet normalization, "
                    "the exact convert_cache_u8 semantics) and write the .npy straight to --out: "
                    "half the bytes, no fp16 intermediate, no second local copy")
p.add_argument("--anchor_rate", choices=["obs", "raw"], default="obs",
               help="anchor every frameskip-th frame, or every frame in raw mode")
p.add_argument("--pixels_key", default="pixels",
               help="the h5 image column to cache; a camera other than `pixels` gets its own frames file "
                    "(tag suffix .<key>) beside the same aux, for multi-view training")
args = p.parse_args()
raw_mode = args.anchor_rate == "raw"

stem = os.path.basename(args.h5).replace(".h5", "")
tag = f"{stem}_fs{args.frameskip}_i{args.img_size}" + ("_raw" if raw_mode else "")
if args.pixels_key != "pixels":
    tag = f"{tag}.{args.pixels_key}"
tmp = f"/tmp/preload_cache/{tag}"
os.makedirs(tmp, exist_ok=True)
os.makedirs(args.out, exist_ok=True)


def terminal_obs_per_episode(n_eps, keeps_per_ep):
    """Return the terminal observation index for each episode."""
    if args.terminal_state == "last":
        return np.asarray([n_frames - 1 for n_frames in keeps_per_ep], np.int64), 0
    import h5py
    with h5py.File(args.h5, "r") as f:
        success = np.asarray(f[args.success_key][:]).reshape(-1).astype(bool)
        ep_offset = np.asarray(f["ep_offset"][:]).reshape(-1)
        ep_len = np.asarray(f["ep_len"][:]).reshape(-1)
    starts = np.unique(ep_offset)
    assert starts.size == n_eps, f"h5 has {starts.size} episodes, cache/loader has {n_eps}"
    lens = np.asarray([int(ep_len[np.searchsorted(ep_offset, s)]) for s in starts], np.int64)
    stride = 1 if raw_mode else args.frameskip
    terminal_obs = np.empty(n_eps, np.int64)
    n_episodes_without_success = 0
    for i, (s, episode_length) in enumerate(zip(starts, lens)):
        n_frames = int(keeps_per_ep[i])
        n_kept = episode_length if raw_mode else episode_length // args.frameskip
        # The cache may contain one extra frame for the terminal state.
        assert n_kept <= n_frames <= n_kept + 1, \
            f"ep{i}: cache keeps {n_frames} frames but h5 ep_len {episode_length} implies {n_kept}..{n_kept + 1}"
        hits = np.flatnonzero(success[s:s + episode_length:stride][:n_frames])
        if hits.size == 0:
            n_episodes_without_success += 1
            terminal_obs[i] = n_frames - 1
        else:
            terminal_obs[i] = int(hits[0])
    return terminal_obs, n_episodes_without_success


def verify_success_order(n_verify=5):
    """Check that HDF5 and SWM agree on episode ordering."""
    sw = swm.data.load_dataset(args.h5, transform=None, cache_dir=None, num_steps=4,
                               frameskip=args.frameskip, keys_to_load=[args.success_key],
                               keys_to_cache=[args.success_key], format="hdf5")
    import h5py
    with h5py.File(args.h5, "r") as f:
        success = np.asarray(f[args.success_key][:]).reshape(-1).astype(bool)
        ep_offset = np.asarray(f["ep_offset"][:]).reshape(-1)
    starts = np.unique(ep_offset)
    rng = np.random.default_rng(0)
    for ep in rng.choice(len(sw.lengths), size=min(n_verify, len(sw.lengths)), replace=False):
        episode_length = int(sw.lengths[ep])
        via_swm = np.asarray(sw._load_slice(int(ep), 0, episode_length)[args.success_key]).reshape(-1).astype(bool)
        # SWM returns non-pixel keys at observation rate.
        via_h5 = success[starts[ep]:starts[ep] + episode_length:args.frameskip]
        assert np.array_equal(via_swm, via_h5[:via_swm.size]), \
            f"ep{ep}: success column differs between swm and raw-h5 episode order"
    print(f"[cache] success-order VERIFY_OK ({n_verify} episodes)", flush=True)


if args.patch_terminal:
    aux_path = f"{args.out}/{stem}/{tag}.aux.npz"
    if not os.path.isfile(aux_path):
        aux_path = f"{args.out}/{tag}.aux.npz"
    assert os.path.isfile(aux_path), f"no aux.npz for {tag} under {args.out}"
    aux = dict(np.load(aux_path))
    t_gidx, maxh, ep_base = aux["t_gidx"], aux["maxh"], aux["ep_base"]
    # Recover episode lengths from the first anchor in each episode.
    bases, first_row = np.unique(ep_base, return_index=True)
    order = np.argsort(first_row)
    bases, first_row = bases[order], first_row[order]
    assert (np.diff(bases) > 0).all(), "aux ep_base not in ascending stream order"
    keeps_per_ep = maxh[first_row] + (t_gidx[first_row] - ep_base[first_row]) + 1
    n_eps = bases.size
    if args.terminal_state == "success":
        try:
            verify_success_order()
        except AssertionError:
            raise
        except Exception as e:                     # loader can't do a success-only load
            print(f"[cache] WARN success-order verification skipped ({e!r}); relying on the "
                  f"per-episode length check only", flush=True)
    terminal_obs, n_episodes_without_success = terminal_obs_per_episode(n_eps, keeps_per_ep)
    frames_to_terminal = terminal_obs[np.searchsorted(bases, ep_base)] - (t_gidx - ep_base)
    assert frames_to_terminal.shape == maxh.shape
    assert (frames_to_terminal <= maxh).all(), "terminal past the episode's last frame"
    if args.terminal_state == "last":
        assert (frames_to_terminal == maxh).all(), "'last' must reproduce maxh exactly"
    aux["frames_to_terminal"] = frames_to_terminal.astype(np.int64)
    np.savez(f"{tmp}/{tag}.aux.npz", **aux)
    meta_path = aux_path.replace(".aux.npz", ".meta.json")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    meta.update(terminal_state=args.terminal_state,
                success_key=(args.success_key if args.terminal_state == "success" else None),
                n_eps_no_success=n_episodes_without_success,
                terminal_patched=time.strftime("%F %T"))
    json.dump(meta, open(f"{tmp}/{tag}.meta.json", "w"), indent=1)
    n_term_anchor = int((frames_to_terminal >= 1).sum())
    print(f"[cache] patch {tag}: eps={n_eps} no_success={n_episodes_without_success} "
          f"anchors {n_term_anchor}/{len(t_gidx)} pre-terminal "
          f"median frames_to_terminal={int(np.median(frames_to_terminal))}", flush=True)
    for src, dst in ((f"{tmp}/{tag}.aux.npz", aux_path), (f"{tmp}/{tag}.meta.json", meta_path)):
        shutil.copy2(src, dst + ".new")
        os.replace(dst + ".new", dst)   # readers see old-or-new aux, never a half-written one
        print(f"[cache] patched {dst}", flush=True)
    shutil.rmtree(tmp)
    print("[cache] PATCH DONE", flush=True)
    sys.exit(0)

# Raw mode loads every pixel frame; frameskip remains the semantic stride used by the cache.
load_keys = [args.pixels_key, "action"]
load_frameskip = 1 if raw_mode else args.frameskip
base = swm.data.load_dataset(args.h5, transform=None, cache_dir=None, num_steps=4,
                             frameskip=load_frameskip, keys_to_load=load_keys,
                             keys_to_cache=[n_frames for n_frames in load_keys if n_frames != args.pixels_key],
                             format="hdf5")
assert int(base.frameskip) == load_frameskip

act_norm = get_column_normalizer(base, "action", "action")
action_normalizer = act_norm.lambd
act_mean = action_normalizer.mean.squeeze(0).cpu().numpy().tolist()
act_std = action_normalizer.std.squeeze(0).cpu().numpy().tolist()
img_t = get_img_preprocessor(source=args.pixels_key, target="pixels", img_size=args.img_size)
_U8_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_U8_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
action_mean = torch.tensor(act_mean, dtype=torch.float32)
action_std = torch.tensor(act_std, dtype=torch.float32)
frameskip = args.frameskip
n_eps = len(base.lengths)
if args.max_eps > 0:
    n_eps = min(n_eps, args.max_eps)
    print(f"[cache] --max_eps: capping to {n_eps} episodes", flush=True)
print(f"[cache] {stem}: {n_eps} episodes, act_dim={len(act_mean)}", flush=True)

def episode_tensors(ep):
    """Load and preprocess one episode."""
    episode_length = int(base.lengths[ep])
    episode_data = base._load_slice(ep, 0, episode_length)
    pixels, raw_actions = episode_data[args.pixels_key], episode_data["action"]
    if not torch.is_tensor(pixels):
        pixels = torch.as_tensor(np.asarray(pixels))
    raw_actions = raw_actions if torch.is_tensor(raw_actions) else torch.as_tensor(np.asarray(raw_actions))
    if raw_mode:
        if pixels.ndim == 4 and pixels.shape[1] == 3:                  # loader hands back CHW
            frames = pixels.contiguous()
        elif pixels.ndim == 4 and pixels.shape[-1] == 3:               # HWC
            frames = pixels.permute(0, 3, 1, 2).contiguous()
        else:
            raise AssertionError(f"unrecognized pixel layout {tuple(pixels.shape)}")
        assert frames.shape[-1] == args.img_size and frames.shape[-2] == args.img_size, \
            f"raw mode stores unresized frames; source is {tuple(frames.shape)} not {args.img_size}"
        if frames.dtype != torch.uint8:
            assert frames.max() > 1.5, "expected 0..255 pixel range"
            frames = frames.to(torch.uint8)
        actions = ((raw_actions.float() - action_mean) / action_std).half()               # (L, adim) per-step
        return frames, actions
    frames = img_t({args.pixels_key: pixels})["pixels"].float()
    n_observations = episode_length // frameskip
    actions = raw_actions[:n_observations * frameskip].reshape(n_observations, frameskip, raw_actions.shape[1])
    actions = (actions - action_mean) / action_std
    actions = actions.reshape(n_observations, frameskip * raw_actions.shape[1]).half()
    n_kept = min(frames.shape[0], n_observations + 1)
    if args.u8:
        fr = frames[:n_kept]
        u8 = torch.clamp(torch.round((fr * _U8_STD + _U8_MEAN) * 255.0), 0, 255).to(torch.uint8)
        return u8, actions
    return frames[:n_kept].half(), actions

start_time = time.time()
image_size = args.img_size
row_bytes = 3 * image_size * image_size * (1 if (raw_mode or args.u8) else 2)
bin_path = f"{tmp}/{tag}.frames.bin"
frames_per_episode = []
n_valids = []
action_values, target_indices, maxh_values, episode_bases = [], [], [], []
frame_offset = 0
with open(bin_path, "wb") as frame_file:
    for ep in range(n_eps):
        frames, actions = episode_tensors(ep)
        n_frames = frames.shape[0]
        frames_per_episode.append(n_frames)
        frame_file.write(frames.numpy().tobytes())
        if raw_mode:
            # Keep only anchors with a complete action block and next frame.
            n_valid = max(0, n_frames - frameskip)
            n_valids.append(n_valid)
            last = n_frames - 1
            for t in range(n_valid):
                target_indices.append(frame_offset + t); maxh_values.append(last - t); episode_bases.append(frame_offset)
            for t in range(n_frames):
                action_values.append(actions[t] if t < actions.shape[0] else torch.zeros_like(actions[0]))
        else:
            n_valid = min(actions.shape[0], n_frames - 1)
            n_valids.append(n_valid)
            last = n_frames - 1
            for t in range(n_valid):
                target_indices.append(frame_offset + t); maxh_values.append(last - t); episode_bases.append(frame_offset)
                action_values.append(actions[t])
        frame_offset += n_frames
        if (ep + 1) % 500 == 0:
            print(f"[cache] {ep+1}/{n_eps} ({time.time()-start_time:.0f}s)", flush=True)
N = frame_offset
print(f"[cache] streamed N={N} frames in {time.time()-start_time:.0f}s", flush=True)

FRAME_DTYPE = np.uint8 if (raw_mode or args.u8) else np.float16
# --u8: memmap the final .npy directly under --out (HDFS) so the pod's local disk holds only
# the .bin; the fp16 fs3 reacher build died at 49 min with .bin+.npy both under /tmp.
frames_path = f"{args.out}/{tag}.frames.npy" if args.u8 else f"{tmp}/{tag}.frames.npy"
start_time = time.time()
chunk_size = 4096
raw = np.memmap(bin_path, dtype=FRAME_DTYPE, mode="r", shape=(N, 3, image_size, image_size))
if args.u8:
    # --u8 writes the .npy straight onto HDFS (fuse): no write-mmap there (Errno 95), so
    # emit a standard .npy header and stream the local .bin through sequential write().
    with open(frames_path, "wb") as fh:
        np.lib.format.write_array_header_1_0(
            fh, {"descr": np.lib.format.dtype_to_descr(np.dtype(FRAME_DTYPE)),
                 "fortran_order": False, "shape": (N, 3, image_size, image_size)})
        for i in range(0, N, chunk_size):
            fh.write(np.ascontiguousarray(raw[i:i + chunk_size]).tobytes())
else:
    frames_memmap = np.lib.format.open_memmap(frames_path, mode="w+", dtype=FRAME_DTYPE,
                                   shape=(N, 3, image_size, image_size))
    for i in range(0, N, chunk_size):
        frames_memmap[i:i + chunk_size] = raw[i:i + chunk_size]
    frames_memmap.flush(); del frames_memmap
del raw
os.remove(bin_path)
print(f"[cache] finalized npy {frames_path} in {time.time()-start_time:.0f}s", flush=True)
A_flat = torch.stack(action_values).numpy() if action_values else np.zeros((0, 1), np.float16)
if args.terminal_state == "success":
    verify_success_order()
terminal_obs, n_episodes_without_success = terminal_obs_per_episode(n_eps, frames_per_episode)
frames_to_terminal = np.concatenate(
    [terminal_obs[ep] - np.arange(n_valid, dtype=np.int64) for ep, n_valid in enumerate(n_valids)]
) if target_indices else np.zeros((0,), np.int64)
maxh = np.asarray(maxh_values, np.int64)
assert (frames_to_terminal <= maxh).all(), "terminal past the episode's last frame"
if args.terminal_state == "last":
    assert (frames_to_terminal == maxh).all(), "'last' must reproduce maxh exactly"
np.savez(f"{tmp}/{tag}.aux.npz",
         A_flat=A_flat, t_gidx=np.asarray(target_indices, np.int64),
         maxh=np.asarray(maxh_values, np.int64), ep_base=np.asarray(episode_bases, np.int64),
         frames_to_terminal=frames_to_terminal,
         act_mean=np.asarray(act_mean), act_std=np.asarray(act_std))
json.dump({"source_h5": args.h5, "img_size": image_size, "frameskip": frameskip,
           "anchor_rate": args.anchor_rate,
           "n_frames": int(N), "n_samples": len(target_indices), "n_episodes": n_eps,
           "terminal_state": args.terminal_state,
           "success_key": (args.success_key if args.terminal_state == "success" else None),
           "n_eps_no_success": n_episodes_without_success,
           "built": time.strftime("%F %T")},
          open(f"{tmp}/{tag}.meta.json", "w"), indent=1)
print(f"[cache] built: N={N} samples={len(target_indices)}", flush=True)

# Verify a sample of episodes byte-for-byte.
frames_memmap = np.load(frames_path, mmap_mode="r")
rng = np.random.default_rng(0)
episode_offsets = np.cumsum([0] + frames_per_episode[:-1])
for ep in rng.choice(n_eps, size=min(args.verify_eps, n_eps), replace=False):
    frames, _ = episode_tensors(int(ep))
    got = frames_memmap[episode_offsets[ep]:episode_offsets[ep] + frames_per_episode[ep]]
    assert np.array_equal(got, frames.numpy()), f"VERIFY FAIL ep{ep}"
print(f"[cache] VERIFY_OK ({args.verify_eps} episodes byte-identical)", flush=True)
del frames_memmap

for f in os.listdir(tmp):
    src = f"{tmp}/{f}"; dst = f"{args.out}/{f}"
    if os.path.exists(dst):
        os.remove(dst) if os.path.isfile(dst) else shutil.rmtree(dst)
    start_time = time.time()
    shutil.copy2(src, dst) if os.path.isfile(src) else shutil.copytree(src, dst)
    print(f"[cache] copied {f} -> {args.out} ({time.time()-start_time:.0f}s)", flush=True)
shutil.rmtree(tmp)
print("[cache] DONE", flush=True)