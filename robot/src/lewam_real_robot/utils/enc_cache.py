"""Precomputed frozen-encoder embeddings, so training never runs the backbone.

`cache_encodings:` (configs/base.yaml) precomputes each stream's encoder output for every
window training can request, under `augmentation_draws` augmentations, and training looks
embeddings up instead of encoding.

Entries are whole windows keyed by (episode, anchor), since the backbone attends across
the window it is given. Per stream, `data` is uint8 `[N, K, tokens, dim]` quantized per
token against fp16 `scale`/`zero`, with a sorted key array for `searchsorted` lookup.
Augmentation is seeded per (episode, anchor, field, draw), so streams reading one field
share the same crop.

Training batches carry `enc_cache/<field>` int64 `(episode, anchor, draw)` keys in place
of the pixel field (see datasets/lerobot.py).
"""

import hashlib
import json
import math
import zlib
from pathlib import Path

import numpy as np
import torch

_PACK = 1 << 32  # keys pack (episode ordinal, anchor) into one int64


def resolve_stride(subsample_cfg, dataset_fps):
    """`subsample_streams:` -> the anchor stride (decision points per episode are thinned
    to every stride-th frame). Streams' offsets are fixed by their `raw_index`, so the
    anchor grid is the only thing left to subsample; with window-level caching no
    divisibility between offsets and stride is required. Each pattern gives `every: N`
    or `fps: F` (F must divide the dataset fps); several patterns take the LCM.
    """
    if not subsample_cfg:
        return 1
    stride = 1
    for pattern, spec in dict(subsample_cfg).items():
        spec = dict(spec)
        if ("every" in spec) == ("fps" in spec):
            raise ValueError(
                f"subsample_streams['{pattern}'] needs exactly one of `every` or `fps`"
            )
        if "every" in spec:
            step = int(spec["every"])
        else:
            fps = float(spec["fps"])
            if dataset_fps % fps:
                raise ValueError(
                    f"subsample_streams['{pattern}']: fps {fps} does not divide the "
                    f"dataset's {dataset_fps} fps; use `every:` for uneven strides"
                )
            step = int(dataset_fps // fps)
        if step < 1:
            raise ValueError(f"subsample_streams['{pattern}']: stride must be >= 1")
        stride = stride * step // math.gcd(stride, step)
    return stride


_SEED_SALT = "lewam-enc-cache-v2"


def draw_seed(episode, anchor, field, draw):
    """The augmentation seed for one (window, draw). Keyed by field rather than stream,
    so every stream reading the field (context and future of one camera) shares the
    draw's parameters -- the coherence the live augmentation wrapper gives a sample.
    The salt is a constant, not the cache fingerprint, so the DATASET can recompute the
    exact same parameters without knowing the cache's identity: that is how a pixel
    window served next to cached latents (a decoder target) gets the same crop the
    cached draw was built with (see augment_like_draw).
    """
    return zlib.crc32(f"{_SEED_SALT}|{episode}|{anchor}|{field}|{draw}".encode())


def augment_like_draw(augmenter, field, window, episode, anchor, draw):
    """Augment a pixel window with draw `draw`'s exact parameters -- the counterpart of
    the build's _augment_window, exposed for dataset-side use: a field can then serve
    raw pixels (e.g. a pixel-decoder's target) geometrically coherent with the cached
    latents conditioning on it. Parameter draws come from the CPU generator on both
    sides, so crop/rotation/jitter match exactly; only gaussian noise (device generator)
    differs from a GPU-built cache, which at its 0.01 std is irrelevant.
    """
    augmenter.rng.reseed(draw_seed(episode, anchor, field, draw))
    return augmenter({field: window})[field]


def cache_fingerprint(streams_info, encoder_specs, augment_cfg, stride, draws, dataset_id):
    """Everything that changes the cached bytes, hashed into the cache's identity. A
    changed setup therefore resolves to a different directory and rebuilds, instead of
    silently loading embeddings computed under different settings.
    """
    from omegaconf import OmegaConf

    def plain(x):
        return OmegaConf.to_container(x, resolve=True) if OmegaConf.is_config(x) else x

    # only the encoders the cached streams run: anything else an arm declares touches
    # no cached byte and would split one cache into two
    specs = plain(encoder_specs) or {}
    used = {info["encoder"] for info in streams_info.values()}
    payload = {
        "streams": {
            name: {"field": info["field"], "offsets": list(info["offsets"]),
                   "encoder": info["encoder"]}
            for name, info in sorted(streams_info.items())
        },
        "encoders": {name: spec for name, spec in specs.items() if name in used},
        "augment": plain(augment_cfg),
        "stride": stride,
        "draws": draws,
        "dataset": plain(dataset_id),
        "quant": "uint8_per_token_v2",   # v2: constant-salt draw seeds (augment_like_draw)
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


# cgroup limit files, v2 then v1. /proc/meminfo always describes the HOST, which
# inside a pod is off by 10x. Absent files, the literal "max" and the v1 no-limit
# sentinel (~2^63) all mean no cgroup bound and fall through to meminfo.
_CGROUP_MEM = (
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
     "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
)
_CGROUP_NO_LIMIT = 1 << 60


def _read_bytes(path):
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value >= _CGROUP_NO_LIMIT else value


def _cgroup_available_bytes():
    """What the cgroup still allows this process to allocate, or None without a limit."""
    for limit_path, usage_path in _CGROUP_MEM:
        limit = _read_bytes(limit_path)
        if limit is None:
            continue
        usage = _read_bytes(usage_path) or 0
        return max(0, limit - usage)
    return None


def _available_ram_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
                cgroup = _cgroup_available_bytes()
                return host if cgroup is None else min(host, cgroup)
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


class StreamStore:
    """One stream's cached windows: sorted packed keys + quantized embeddings."""

    ARRAYS = ("keys", "data", "scale", "zero")

    def __init__(self, keys, data, scale, zero):
        self.keys = torch.as_tensor(np.asarray(keys))         # int64 [N] sorted
        self.data = data                                      # uint8 [N, K, tokens, dim]
        self.scale = scale                                    # fp16  [N, K, tokens, 1]
        self.zero = zero                                      # fp16  [N, K, tokens, 1]

    def lookup(self, key_tensor, device):
        """int64 [B, 3] (episode, anchor, draw) -> fp32 [B, tokens, dim] on `device`.
        A key the build never produced is a hard error: it means the dataset emitted a
        window the enumeration missed, and a silent zero would train on garbage.
        """
        cols = key_tensor.to("cpu", torch.int64)
        packed = cols[:, 0] * _PACK + cols[:, 1]
        idx = torch.searchsorted(self.keys, packed)
        idx = idx.clamp_max(len(self.keys) - 1)
        if not torch.equal(self.keys[idx], packed):
            missing = packed[self.keys[idx] != packed][0].item()
            raise KeyError(
                f"encoding cache has no window (episode {missing // _PACK}, "
                f"anchor {missing % _PACK}); the cache was built for a different "
                f"anchor grid or episode split -- delete it and rebuild"
            )
        draw = cols[:, 2]
        rows = idx.numpy()
        k = draw.numpy()
        q = torch.as_tensor(np.ascontiguousarray(self.data[rows, k])).to(device)
        scale = torch.as_tensor(np.ascontiguousarray(self.scale[rows, k])).to(device)
        zero = torch.as_tensor(np.ascontiguousarray(self.zero[rows, k])).to(device)
        return q.float() * scale.float() + zero.float()


def quantize_windows(z):
    """fp32 [n, tokens, dim] (any device) -> per-token uint8 + fp16 scale/zero numpy
    arrays, the storage format. Quantizing before leaving the device cuts the transfer
    ~4x versus shipping fp32 embeddings back.
    """
    lo = z.amin(dim=-1, keepdim=True)
    hi = z.amax(dim=-1, keepdim=True)
    scale = (hi - lo).clamp_min(1e-8) / 255.0
    q = ((z - lo) / scale).round().clamp(0, 255).to(torch.uint8)
    return q.cpu().numpy(), scale.half().cpu().numpy(), lo.half().cpu().numpy()


class EncodingCache:
    """The per-stream stores plus identity metadata; see the module docstring."""

    def __init__(self, stores, fingerprint, meta):
        self.stores = stores
        self.fingerprint = fingerprint
        self.meta = meta

    @property
    def draws(self):
        return int(self.meta["draws"])

    def fields(self):
        return {info["field"] for info in self.meta["streams"].values()}

    def lookup(self, stream, key_tensor, device):
        return self.stores[stream].lookup(key_tensor, device)


    @staticmethod
    def cache_dir(disk_path, fingerprint):
        """The fingerprint alone names the directory, so every run whose cached bytes
        would be identical resolves to the same one and the first to build it serves the
        rest. Naming it after the config as well would defeat that: the arms of a sweep
        differ in their trunk and losses, which change nothing the cache holds.
        """
        return Path(disk_path) / f"enc-{fingerprint}"

    def save(self, directory):
        """Write manifest + per-stream arrays. Streams built straight into memmaps under
        `directory` are already on disk; flushing them is all that is left.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, store in self.stores.items():
            safe = name.replace("/", "_")
            for attr in StreamStore.ARRAYS:
                arr = getattr(store, attr)
                arr = arr.numpy() if torch.is_tensor(arr) else arr
                target = directory / f"{safe}.{attr}.npy"
                if isinstance(arr, np.memmap) and Path(arr.filename) == target:
                    arr.flush()
                else:
                    np.save(target, np.ascontiguousarray(arr))
        (directory / "manifest.json").write_text(
            json.dumps({"fingerprint": self.fingerprint, **self.meta}, indent=2)
        )

    @classmethod
    def load(cls, directory, fingerprint, store_on="auto"):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError(
                f"cache at {directory} has fingerprint {manifest['fingerprint']}, "
                f"expected {fingerprint}: the config changed since it was built. "
                f"Delete the directory to rebuild."
            )
        total = sum(
            math.prod(s["data_shape"]) for s in manifest["streams"].values()
        )
        in_ram = cls._store_in_ram(store_on, total)
        stores = {}
        for name, spec in manifest["streams"].items():
            safe = name.replace("/", "_")
            arrays = {}
            for attr in StreamStore.ARRAYS:
                mode = None if (in_ram or attr == "keys") else "r"
                arrays[attr] = np.load(directory / f"{safe}.{attr}.npy", mmap_mode=mode)
                if in_ram and isinstance(arrays[attr], np.memmap):
                    arrays[attr] = np.asarray(arrays[attr])
            stores[name] = StreamStore(**arrays)
        meta = {k: v for k, v in manifest.items() if k != "fingerprint"}
        return cls(stores, fingerprint, meta)

    @staticmethod
    def _store_in_ram(store_on, total_bytes, headroom=0.5):
        """memory | disk | auto -> whether arrays live in RAM. `auto` takes RAM only when
        the cache fits in half of what is currently available, so the run cannot OOM the
        host just by loading its own cache.
        """
        if store_on == "memory":
            return True
        if store_on == "disk":
            return False
        if store_on != "auto":
            raise ValueError(f"store_on must be memory|disk|auto, got '{store_on}'")
        return total_bytes <= _available_ram_bytes() * headroom


def calibrate_encode_batch(encoder, window_frames, device, amp_dtype=None,
                           target_fraction=0.9, log=print):
    """Largest window batch the frozen encoder can forward at once, probed with the same
    affine-fit search training's `batch_size: auto` uses -- but forward-only under
    no_grad (precompute has no gradients) and unbounded by `effective_batch` (that is an
    optimization concept; precompute has no optimizer). Units are windows of this
    stream's shape, not training samples.

    The probe mirrors the real encode call (autocast forward + the fp32 upcast +
    quantize_windows) so its peak matches what `_encode_windows` actually allocates --
    a bare forward alone undercounts the upcast/quantize buffers and overestimates the
    batch that fits. `_encode_windows` still retries on OOM if this probe is wrong.
    """
    from .auto_batch import find_max_batch_size

    sample = window_frames[:1].to(device)
    use_amp = amp_dtype is not None and device.type == "cuda"

    def step_fn(batch_size):
        chunk = sample.expand(batch_size, *sample.shape[1:])
        with torch.no_grad():
            with torch.autocast(device.type, amp_dtype, enabled=use_amp):
                z = encoder(chunk).float()
            quantize_windows(z)

    return find_max_batch_size(step_fn, target_fraction=target_fraction,
                               device=device, log=log)


def assemble_shards(directory, world):
    """Concatenate `shard{r}/` builds into the final cache under `directory`.

    Shards are contiguous slices of the key-sorted key space (build_encoding_cache
    splits episodes into contiguous, count-balanced ranges), so rank-order concatenation
    IS the sorted layout -- no merge. Copies stream via memmaps, so nothing is ever
    whole in RAM, then removes the shard directories.
    """
    import shutil

    directory = Path(directory)
    shards = [directory / f"shard{r}" for r in range(world)]
    manifest = json.loads((shards[0] / "manifest.json").read_text())
    for name in manifest["streams"]:
        safe = name.replace("/", "_")
        for attr in StreamStore.ARRAYS:
            parts = [np.load(s / f"{safe}.{attr}.npy", mmap_mode="r") for s in shards]
            total = sum(len(p) for p in parts)
            out = np.lib.format.open_memmap(
                directory / f"{safe}.{attr}.npy", mode="w+",
                dtype=parts[0].dtype, shape=(total, *parts[0].shape[1:]),
            )
            row = 0
            for part in parts:
                out[row:row + len(part)] = part
                row += len(part)
            out.flush()
            if attr == "data":
                manifest["streams"][name]["data_shape"] = [total, *parts[0].shape[1:]]
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for shard in shards:
        shutil.rmtree(shard)


class _FrameJobs(torch.utils.data.Dataset):
    """(episode, frame) decode jobs for the build's DataLoader workers."""

    def __init__(self, source, jobs):
        self.source, self.jobs = source, jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        ep, t = self.jobs[i]
        return ep, t, self.source.read_frame(ep, t)


def build_encoding_cache(source, algo, streams_info, augment_cfg, *, draws, store_on,
                         directory, fingerprint, device, encode_batch="auto",
                         amp_dtype=None, decode_workers=0, rank=0, world=1, log=print):
    """Precompute every stream's windows into an EncodingCache.

    `amp_dtype` is the autocast dtype the encoder forward runs under (the same one
    training mixes precision with, e.g. `torch.float16`/`torch.bfloat16`); `None` runs
    full precision. Passed in rather than hardcoded, since bf16 has no tensor-core
    support on pre-Ampere GPUs (V100 and older) -- fp16 is the accelerated dtype there.

    `source` is the (unwrapped) training LeRobotSource: it provides the anchor space
    (`cache_anchor_space`) and single-frame decoding (`read_frame`), both already on the
    training split and anchor stride. Ranks split the episodes into contiguous ranges
    balanced by anchor count; packed keys sort by (episode, anchor), so each rank's rows
    are a contiguous slice of the final arrays and multi-rank assembly is plain
    rank-order concatenation (training/base.py owns the barrier and reload).
    """
    import fnmatch

    from .augment import Augmenter

    # only patterns touching a cached field: the precompute batches carry one field
    # each, and the Augmenter's matches-no-field guard would reject the rest
    cached_fields = {info["field"] for info in streams_info.values()}
    augment_cfg = {
        p: ops for p, ops in dict((augment_cfg or {}).get("streams") or {}).items()
        if any(fnmatch.fnmatchcase(f, p) for f in cached_fields)
    }

    anchor_space = source.cache_anchor_space()
    total = sum(len(a) for _, a in anchor_space)
    lo, hi = (rank * total) // world, ((rank + 1) * total) // world
    shard, seen = [], 0
    for ep, anchors in anchor_space:
        take = [a for i, a in enumerate(anchors, seen) if lo <= i < hi]
        if take:
            shard.append((ep, take))
        seen += len(anchors)
    n_rows = hi - lo

    augmenter = Augmenter(augment_cfg, seed=0)
    stores, arrays = {}, {}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    encoders = {}
    for name, info in streams_info.items():
        encoders[name] = algo.encoders[info["encoder"]].to(device).eval()

    # video decode is the build's bottleneck, so it runs in DataLoader workers while
    # this process augments and encodes. Items come back in submission order, so an
    # episode is complete exactly when its frame count has been consumed.
    needed_by_ep = {
        ep: sorted({
            max(0, a + off)
            for a in anchors
            for info in streams_info.values()
            for off in info["offsets"]
        })
        for ep, anchors in shard
    }
    jobs = [(ep, t) for ep, _ in shard for t in needed_by_ep[ep]]
    workers = min(decode_workers, len(jobs))
    loader = torch.utils.data.DataLoader(
        _FrameJobs(source, jobs), batch_size=None, num_workers=workers,
        **({"prefetch_factor": 4} if workers else {}),
    )
    items = iter(loader)

    batch_sizes = {}
    row = 0
    for ep, anchors in shard:
        frames = {}
        for _ in needed_by_ep[ep]:
            got_ep, t, item = next(items)
            assert int(got_ep) == ep
            frames[int(t)] = {
                f: v.to(device, non_blocking=True) for f, v in item.items()
            }

        for name, info in streams_info.items():
            field, offsets = info["field"], info["offsets"]
            windows, seeds = [], []
            for a in anchors:
                stack = torch.stack(
                    [frames[max(0, a + off)][field] for off in offsets]
                )
                for k in range(draws):
                    windows.append(stack)
                    seeds.append(draw_seed(ep, a, field, k))
            _encode_windows(
                name, windows, seeds, augmenter, field, encoders[name], device,
                arrays, batch_sizes, encode_batch, n_rows, draws, row,
                directory, store_on, amp_dtype, log,
            )
        for name in streams_info:
            arrays[name]["keys"][row:row + len(anchors)] = [
                ep * _PACK + a for a in anchors
            ]
        row += len(anchors)
        log(f"cache build rank {rank}: episode {ep} done "
            f"({row}/{n_rows} windows)")

    meta = {
        "draws": draws,
        "streams": {
            name: {
                "field": info["field"],
                "data_shape": list(arrays[name]["data"].shape),
            }
            for name, info in streams_info.items()
        },
    }
    for name in streams_info:
        stores[name] = StreamStore(**arrays[name])
    return EncodingCache(stores, fingerprint, meta)


def _encode_windows(name, windows, seeds, augmenter, field, encoder, device, arrays,
                    batch_sizes, encode_batch, n_rows, draws, row0, directory,
                    store_on, amp_dtype, log):
    """Augment + encode one episode's windows for one stream and write their rows.

    Windows arrive on `device` and augmentation runs there too (its tensor ops are ~free
    next to the encoder -- on CPU they were the build's second bottleneck after decode).
    Each encoded chunk is quantized on-device and written straight into its rows, so
    neither the fp32 embeddings nor the whole episode ever sit in memory at once.

    `amp_dtype` is the autocast dtype (`None` runs full precision); see
    `build_encoding_cache`.
    """
    safe = name.replace("/", "_")
    frames = torch.stack([
        _augment_window(augmenter, field, w, s) for w, s in zip(windows, seeds)
    ])
    use_amp = amp_dtype is not None and device.type == "cuda"
    if name not in batch_sizes:
        if encode_batch == "auto" and device.type == "cuda":
            batch_sizes[name] = calibrate_encode_batch(encoder, frames, device,
                                                        amp_dtype=amp_dtype, log=log)
        else:
            batch_sizes[name] = int(encode_batch) if encode_batch != "auto" else 8
        log(f"cache build: stream '{name}' encode batch {batch_sizes[name]} windows")
    with torch.no_grad():
        for i in range(0, len(frames), batch_sizes[name]):
            # no OOM fallback here on purpose: calibrate_encode_batch's probe mirrors
            # this exact call (autocast forward + upcast + quantize), so a miss here
            # means the calibration itself is wrong and should be fixed, not papered
            # over with a silent, ever-shrinking batch size
            chunk = frames[i:i + batch_sizes[name]]
            with torch.autocast(device.type, amp_dtype, enabled=use_amp):
                z = encoder(chunk).float()
            q, scale, zero = quantize_windows(z)
            if name not in arrays:
                tokens, dim = q.shape[-2:]
                make = (np.zeros if EncodingCache._store_in_ram(store_on,
                        n_rows * draws * tokens * dim) else
                        lambda shape, dtype: np.lib.format.open_memmap(
                            directory / f"{safe}.data.npy", mode="w+",
                            dtype=dtype, shape=shape))
                arrays[name] = {
                    "keys": np.zeros(n_rows, dtype=np.int64),
                    "data": make((n_rows, draws, tokens, dim), dtype=np.uint8),
                    "scale": np.zeros((n_rows, draws, tokens, 1), dtype=np.float16),
                    "zero": np.zeros((n_rows, draws, tokens, 1), dtype=np.float16),
                }
            idxs = np.arange(i, i + len(q))
            rows, ks = row0 + idxs // draws, idxs % draws
            arrays[name]["data"][rows, ks] = q
            arrays[name]["scale"][rows, ks] = scale
            arrays[name]["zero"][rows, ks] = zero


def _augment_window(augmenter, field, window, seed):
    """One draw of one window, parameters fixed by `seed` -- the same seed reproduces
    the same crop/jitter, which is how context and future windows of one sample stay
    coherent (see draw_seed). Runs on whatever device `window` lives on.
    """
    augmenter.rng.reseed(seed)
    out = augmenter({field: window})[field]
    if out.dtype != torch.uint8:
        out = (out * 255).round().clamp(0, 255).to(torch.uint8)
    return out
