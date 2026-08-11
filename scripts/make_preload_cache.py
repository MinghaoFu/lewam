"""Materialize train_lewam_gc's preload as a reusable cache on a2f.

Streams episodes (RAM = one episode, no torch.cat peak) into an np.memmap holding the
EXACT tensor preload_frames+flatten_for_training would build (same img_t, same fp16,
same obs-granularity keep rule, same index columns), then verifies K random episodes
byte-for-byte against the reference per-episode transform.

Outputs under --out (default $A2F/preload_cache):
  <stem>_fs<FS>_i<IMG>.frames.npy   [N,3,IMG,IMG] float16 (np format, mmap-able)
  <stem>_fs<FS>_i<IMG>.aux.npz      A_flat/t_gidx/maxh/ep_base/frames_to_terminal
                                    + act_mean/act_std
  <stem>_fs<FS>_i<IMG>.meta.json    provenance + counts + terminal_state rule
Write to local /tmp first, then cp to --out (never in-place writes on fuse).

frames_to_terminal[row] = obs-steps from that anchor to the episode's TERMINAL frame
(--terminal_state: 'last' = final frame, == maxh; 'success' = first firing of --success_key).
--patch_terminal appends it to an existing cache's aux.npz without touching frames.npy.
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
p.add_argument("--terminal_state", choices=["last", "success"], default="last",
               help="what the episode's terminal state is: 'last' = its final frame (success-only "
                    "demos trimmed at completion, and all goal-reach datasets); 'success' = the "
                    "first frame where --success_key fires (demos that run past completion, e.g. "
                    "cube). Stored per anchor as aux frames_to_terminal = terminal - t; the "
                    "trainer's goal sampler clamps to it.")
p.add_argument("--success_key", default="success",
               help="per-row bool column read for --terminal_state success")
p.add_argument("--patch_terminal", action="store_true",
               help="don't rebuild: read an EXISTING cache's aux.npz under --out (flat or per-stem "
                    "layout), append frames_to_terminal per --terminal_state, and rewrite aux.npz "
                    "+ meta.json in place (write local, then atomic replace). frames.npy is never "
                    "touched. 'success' reads only the success column of --h5.")
p.add_argument("--anchor_rate", choices=["obs", "raw"], default="obs",
               help="obs (default): store every frameskip-th frame; anchors live on that grid and "
                    "action blocks are grid-aligned (today's cache, unchanged). raw: store ALL "
                    "frames as uint8 and per-step actions, so the trainer can anchor at any raw "
                    "frame with frameskip-strided windows (5x the positions, all block phasings). "
                    "raw aux arrays are in RAW-frame units; tag gains a _raw suffix.")
args = p.parse_args()
RAW = args.anchor_rate == "raw"

stem = os.path.basename(args.h5).replace(".h5", "")
tag = f"{stem}_fs{args.frameskip}_i{args.img_size}" + ("_raw" if RAW else "")
tmp = f"/tmp/preload_cache/{tag}"
os.makedirs(tmp, exist_ok=True)
os.makedirs(args.out, exist_ok=True)


def terminal_obs_per_episode(n_eps, keeps_per_ep):
    """Per-episode terminal frame at obs granularity, per --terminal_state.

    'last' -> keeps-1 for every episode. 'success' -> the first KEPT frame where the success
    column fires, i.e. first obs-rate index i with success[offset + FS*i] true -- cache frame i
    IS raw frame FS*i, so the stored goal frame really shows a completed state (a flag firing
    between kept frames does not count; success can flicker around the first contact). Episodes
    that never fire at a kept frame fall back to their last frame (counted + reported). Reads
    the h5 columns directly (ep_offset/ep_len are per-episode or per-row; both parse); episode
    order is cross-checked against the swm segmentation by the caller.
    """
    if args.terminal_state == "last":
        return np.asarray([k - 1 for k in keeps_per_ep], np.int64), 0
    import h5py
    with h5py.File(args.h5, "r") as f:
        success = np.asarray(f[args.success_key][:]).reshape(-1).astype(bool)
        ep_offset = np.asarray(f["ep_offset"][:]).reshape(-1)
        ep_len = np.asarray(f["ep_len"][:]).reshape(-1)
    starts = np.unique(ep_offset)
    assert starts.size == n_eps, f"h5 has {starts.size} episodes, cache/loader has {n_eps}"
    lens = np.asarray([int(ep_len[np.searchsorted(ep_offset, s)]) for s in starts], np.int64)
    stride = 1 if RAW else args.frameskip
    term_obs = np.empty(n_eps, np.int64)
    n_no_success = 0
    for i, (s, L) in enumerate(zip(starts, lens)):
        k = int(keeps_per_ep[i])
        n_keep = L if RAW else L // args.frameskip
        # the build keeps min(pixel-frames, n_keep+1) frames; anything outside [n_keep, n_keep+1]
        # means the h5's episode layout doesn't match the cache's
        assert n_keep <= k <= n_keep + 1, \
            f"ep{i}: cache keeps {k} frames but h5 ep_len {L} implies {n_keep}..{n_keep + 1}"
        hits = np.flatnonzero(success[s:s + L:stride][:k])
        if hits.size == 0:
            n_no_success += 1
            term_obs[i] = k - 1
        else:
            term_obs[i] = int(hits[0])
    return term_obs, n_no_success


def verify_success_order(n_verify=5):
    """Episode-order guard for --terminal_state success: the raw h5 rows [ep_offset, +ep_len) and
    swm's episode segmentation must agree per episode, or every terminal lands in the wrong
    episode (undetectable by the length check when episodes share a length). Compares the success
    column episode-by-episode through BOTH readers for n_verify random episodes."""
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
        L = int(sw.lengths[ep])
        via_swm = np.asarray(sw._load_slice(int(ep), 0, L)[args.success_key]).reshape(-1).astype(bool)
        # swm hands non-pixel keys back at OBS RATE here (verified on cube: slice == raw[::FS]);
        # subsample the raw segment the same way before comparing
        via_h5 = success[starts[ep]:starts[ep] + L:args.frameskip]
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
    # recover the per-episode layout from the aux rows: consecutive rows share ep_base;
    # keeps (frames kept per episode) = first row's t offset... derived instead from maxh:
    # row t of an episode has maxh = (keeps-1) - t, so the episode's first row gives keeps-1.
    bases, first_row = np.unique(ep_base, return_index=True)
    order = np.argsort(first_row)
    bases, first_row = bases[order], first_row[order]
    assert (np.diff(bases) > 0).all(), "aux ep_base not in ascending stream order"
    # any row t of an episode has maxh = (keeps-1) - t and t = t_gidx - ep_base, so keeps falls out
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
    term_obs, n_no_success = terminal_obs_per_episode(n_eps, keeps_per_ep)
    frames_to_terminal = term_obs[np.searchsorted(bases, ep_base)] - (t_gidx - ep_base)
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
                n_eps_no_success=n_no_success,
                terminal_patched=time.strftime("%F %T"))
    json.dump(meta, open(f"{tmp}/{tag}.meta.json", "w"), indent=1)
    n_term_anchor = int((frames_to_terminal >= 1).sum())
    print(f"[cache] patch {tag}: eps={n_eps} no_success={n_no_success} "
          f"anchors {n_term_anchor}/{len(t_gidx)} pre-terminal "
          f"median frames_to_terminal={int(np.median(frames_to_terminal))}", flush=True)
    for src, dst in ((f"{tmp}/{tag}.aux.npz", aux_path), (f"{tmp}/{tag}.meta.json", meta_path)):
        shutil.copy2(src, dst + ".new")
        os.replace(dst + ".new", dst)   # readers see old-or-new aux, never a half-written one
        print(f"[cache] patched {dst}", flush=True)
    shutil.rmtree(tmp)
    print("[cache] PATCH DONE", flush=True)
    sys.exit(0)

# EXACT same construction as train_lewam_gc main(). raw mode loads at frameskip=1 so the
# loader hands back EVERY pixel frame; args.frameskip stays the semantic stride for the tag
# and the trainer.
_ktl = ["pixels", "action"]
_load_fs = 1 if RAW else args.frameskip
base = swm.data.load_dataset(args.h5, transform=None, cache_dir=None, num_steps=4,
                             frameskip=_load_fs, keys_to_load=_ktl,
                             keys_to_cache=[k for k in _ktl if k != "pixels"],
                             format="hdf5")
assert int(base.frameskip) == _load_fs

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
    """EXACT mirror of preload_frames' per-episode body (obs mode). raw mode keeps every
    frame as uint8 CHW (the trainer normalizes at batch time) and every per-step action row."""
    L = int(base.lengths[ep])
    sl = base._load_slice(ep, 0, L)
    pix, raw_act = sl["pixels"], sl["action"]
    if not torch.is_tensor(pix):
        pix = torch.as_tensor(np.asarray(pix))
    raw_act = raw_act if torch.is_tensor(raw_act) else torch.as_tensor(np.asarray(raw_act))
    if RAW:
        if pix.ndim == 4 and pix.shape[1] == 3:                  # loader hands back CHW
            pp = pix.contiguous()
        elif pix.ndim == 4 and pix.shape[-1] == 3:               # HWC
            pp = pix.permute(0, 3, 1, 2).contiguous()
        else:
            raise AssertionError(f"unrecognized pixel layout {tuple(pix.shape)}")
        assert pp.shape[-1] == args.img_size and pp.shape[-2] == args.img_size, \
            f"raw mode stores unresized frames; source is {tuple(pp.shape)} not {args.img_size}"
        if pp.dtype != torch.uint8:
            assert pp.max() > 1.5, "expected 0..255 pixel range"
            pp = pp.to(torch.uint8)
        a = ((raw_act.float() - am) / astd).half()               # (L, adim) per-step
        return pp, a
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
n_valids = []
A_l, tg_l, mh_l, eb_l = [], [], [], []
off = 0
with open(bin_path, "wb") as fb:
    for ep in range(n_eps):
        pp, a = episode_tensors(ep)
        k = pp.shape[0]
        keeps.append(k)
        fb.write(pp.numpy().tobytes())
        if RAW:
            # anchors need a full action block + next frame: t + FS <= k-1.
            # A_l is FRAME-aligned (row i = per-step action at frame i), not anchor-aligned.
            n_valid = max(0, k - FS)
            n_valids.append(n_valid)
            last = k - 1
            for t in range(n_valid):
                tg_l.append(off + t); mh_l.append(last - t); eb_l.append(off)
            for t in range(k):
                A_l.append(a[t] if t < a.shape[0] else torch.zeros_like(a[0]))
        else:
            n_valid = min(a.shape[0], k - 1)
            n_valids.append(n_valid)
            last = k - 1
            for t in range(n_valid):
                tg_l.append(off + t); mh_l.append(last - t); eb_l.append(off)
                A_l.append(a[t])
        off += k
        if (ep + 1) % 500 == 0:
            print(f"[cache] {ep+1}/{n_eps} ({time.time()-t0:.0f}s)", flush=True)
N = off
print(f"[cache] streamed N={N} frames in {time.time()-t0:.0f}s", flush=True)

FRAME_DTYPE = np.uint8 if RAW else np.float16
frames_path = f"{tmp}/{tag}.frames.npy"
fr = np.lib.format.open_memmap(frames_path, mode="w+", dtype=FRAME_DTYPE,
                               shape=(N, 3, IMG, IMG))
t1 = time.time()
CH = 4096
raw = np.memmap(bin_path, dtype=FRAME_DTYPE, mode="r", shape=(N, 3, IMG, IMG))
for i in range(0, N, CH):
    fr[i:i + CH] = raw[i:i + CH]
fr.flush(); del fr; del raw
os.remove(bin_path)
print(f"[cache] finalized npy in {time.time()-t1:.0f}s", flush=True)
A_flat = torch.stack(A_l).numpy() if A_l else np.zeros((0, 1), np.float16)
if args.terminal_state == "success":
    verify_success_order()
term_obs, n_no_success = terminal_obs_per_episode(n_eps, keeps)
frames_to_terminal = np.concatenate(
    [term_obs[ep] - np.arange(nv, dtype=np.int64) for ep, nv in enumerate(n_valids)]
) if tg_l else np.zeros((0,), np.int64)
mh_arr = np.asarray(mh_l, np.int64)
assert (frames_to_terminal <= mh_arr).all(), "terminal past the episode's last frame"
if args.terminal_state == "last":
    assert (frames_to_terminal == mh_arr).all(), "'last' must reproduce maxh exactly"
np.savez(f"{tmp}/{tag}.aux.npz",
         A_flat=A_flat, t_gidx=np.asarray(tg_l, np.int64),
         maxh=np.asarray(mh_l, np.int64), ep_base=np.asarray(eb_l, np.int64),
         frames_to_terminal=frames_to_terminal,
         act_mean=np.asarray(act_mean), act_std=np.asarray(act_std))
json.dump({"source_h5": args.h5, "img_size": IMG, "frameskip": FS,
           "anchor_rate": args.anchor_rate,
           "n_frames": int(N), "n_samples": len(tg_l), "n_episodes": n_eps,
           "terminal_state": args.terminal_state,
           "success_key": (args.success_key if args.terminal_state == "success" else None),
           "n_eps_no_success": n_no_success,
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
