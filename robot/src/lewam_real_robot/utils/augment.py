import fnmatch
import inspect
import math
import os
from typing import Any

import torch
import torch.nn.functional as F


def _flat(x):
    """(..., C, H, W) -> (N, C, H, W), so a whole time window goes through one conv/grid op."""
    return x.reshape(-1, *x.shape[-3:])


def _luma(x):
    """Perceptual grayscale, keeping the channel axis. Non-RGB streams fall back to a
    plain channel mean, since the luminance weights only mean something for RGB.
    """
    if x.shape[-3] != 3:
        return x.mean(dim=-3, keepdim=True)
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device)
    return (x * weights.view(-1, 1, 1)).sum(dim=-3, keepdim=True)


def _rgb_to_hsv(x):
    r, g, b = x.unbind(dim=-3)
    maxc = x.max(dim=-3).values
    minc = x.min(dim=-3).values
    span = maxc - minc
    achromatic = span == 0
    ones = torch.ones_like(maxc)
    saturation = span / torch.where(maxc == 0, ones, maxc)
    divisor = torch.where(achromatic, ones, span)
    rc, gc, bc = ((maxc - c) / divisor for c in (r, g, b))
    hue = torch.where(
        maxc == r, bc - gc,
        torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    hue = torch.where(achromatic, torch.zeros_like(hue), hue) / 6.0 % 1.0
    return hue, saturation, maxc


def _hsv_to_rgb(hue, saturation, value):
    sector = (hue % 1.0) * 6.0
    index = torch.floor(sector)
    frac = sector - index
    p = value * (1 - saturation)
    q = value * (1 - saturation * frac)
    t = value * (1 - saturation * (1 - frac))
    table = torch.stack([
        torch.stack([value, q, p, p, t, value]),
        torch.stack([t, value, value, q, p, p]),
        torch.stack([p, p, t, value, value, q]),
    ])
    index = index.to(torch.long) % 6
    picked = table.gather(1, index.expand(3, 1, *index.shape))
    return picked.squeeze(1).movedim(0, -3)


class _Rng:
    """Draws every augmentation parameter. `seed=None` uses the global torch RNG, which a
    DataLoader already reseeds per worker per epoch; a seed instead gives dedicated
    generators, reseeded once per worker process (see AugmentedSource).
    """

    def __init__(self, seed=None):
        self.seed = seed
        self._generators = {}

    def reseed(self, seed):
        self.seed = seed
        self._generators = {}

    def _generator(self, device):
        if self.seed is None:
            return None
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[key] = generator
        return self._generators[key]

    def uniform(self, low, high):
        if low == high:
            return float(low)
        return float(torch.empty(()).uniform_(low, high, generator=self._generator("cpu")))

    def randint(self, high):
        if high <= 1:
            return 0
        return int(torch.randint(high, (), generator=self._generator("cpu")))

    def chance(self, p):
        return p >= 1.0 or self.uniform(0.0, 1.0) < p

    def noise_like(self, tensor):
        return torch.empty_like(tensor).normal_(generator=self._generator(tensor.device))


def _pair(value, default_high=1.0):
    """A magnitude or an explicit [low, high] range -> (low, high)."""
    if isinstance(value, (int, float)):
        return (float(value), float(default_high))
    low, high = value
    return (float(low), float(high))


def random_crop(x, rng, scale=0.9, p=1.0):
    """Crop a random `scale`-fraction window and resize it back to the original size, so
    the field's shape is unchanged. `scale` is a side fraction, not an area fraction, and
    applies to both axes, so aspect is preserved.
    """
    if not rng.chance(p):
        return x
    height, width = x.shape[-2:]
    fraction = rng.uniform(*_pair(scale))
    crop_h = max(1, min(height, round(height * fraction)))
    crop_w = max(1, min(width, round(width * fraction)))
    top = rng.randint(height - crop_h + 1)
    left = rng.randint(width - crop_w + 1)
    cropped = x[..., top:top + crop_h, left:left + crop_w]
    if (crop_h, crop_w) == (height, width):
        return cropped
    resized = F.interpolate(
        _flat(cropped), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.reshape(x.shape)


def rotation(x, rng, degrees=0.0, p=1.0, padding_mode="border"):
    """Rotate about the image center by an angle drawn from +/- `degrees`. `padding_mode`
    fills the corners the rotation opens up ("border", "reflection", or "zeros").
    """
    if degrees == 0.0 or not rng.chance(p):
        return x
    angle = math.radians(rng.uniform(-degrees, degrees))
    height, width = x.shape[-2:]
    cos, sin = math.cos(angle), math.sin(angle)
    # affine_grid works in per-axis normalized coordinates, so a non-square image
    # needs the aspect ratio folded in or the rotation comes out sheared
    matrix = torch.tensor(
        [[cos, -sin * height / width, 0.0], [sin * width / height, cos, 0.0]],
        dtype=x.dtype, device=x.device,
    )
    flat = _flat(x)
    grid = F.affine_grid(matrix.expand(flat.shape[0], 2, 3), flat.shape, align_corners=False)
    rotated = F.grid_sample(
        flat, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False
    )
    return rotated.reshape(x.shape)


def color_jitter(x, rng, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, gamma=0.0,
                 p=1.0):
    """Photometric jitter. `brightness`/`contrast`/`saturation` are magnitudes: the factor is
    drawn from 1 +/- m. `hue` shifts along the hue circle by +/- h turns (0.5 is the full
    circle), and is a no-op on single-channel streams.

    `gamma` raises the image to a power drawn from exp(+/- gamma): a tone curve, moving the
    mid-tones while leaving black at black and white at white. It must run before
    `contrast`, whose mean subtraction would otherwise hand a negative base to a
    fractional power.
    """
    if not rng.chance(p):
        return x
    if brightness:
        x = x * rng.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
    if gamma:
        x = x.clamp_min(0.0) ** math.exp(rng.uniform(-gamma, gamma))
    if contrast:
        mean = _luma(x).mean(dim=(-2, -1), keepdim=True)
        x = (x - mean) * rng.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast) + mean
    if saturation:
        gray = _luma(x)
        x = (x - gray) * rng.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation) + gray
    if hue and x.shape[-3] == 3:
        h, s, v = _rgb_to_hsv(x.clamp(0.0, 1.0))
        x = _hsv_to_rgb(h + rng.uniform(-hue, hue), s, v)
    return x


def gaussian_noise(x, rng, std=0.0, p=1.0):
    """Additive white noise. On image fields `std` is in [0, 1] units (see Augmenter); on
    state/action fields it is in whatever units reach the augmenter -- normalized units,
    since augmentation runs outside NormalizedSource.
    """
    if not std or not rng.chance(p):
        return x
    return x + rng.noise_like(x) * std


# insertion order is apply order: geometry, then photometry, then noise, so noise is
# never smoothed away by an interpolation running after it
OPS = {
    "random_crop": random_crop,
    "rotation": rotation,
    "color_jitter": color_jitter,
    "gaussian_noise": gaussian_noise,
}
IMAGE_OPS = ("random_crop", "rotation", "color_jitter")


def _plain(value) -> Any:
    """OmegaConf containers -> plain dict/list, so ops take ordinary kwargs."""
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if not isinstance(value, str) and hasattr(value, "__iter__"):
        return [_plain(v) for v in value]
    return value


def live_augment_streams(streams, live_fields):
    """The `streams` patterns that touch a field in `live_fields`, for a batch that no
    longer carries the rest (encoding-cache keys stand in for cached pixels, whose draws
    are baked into the cache). A pattern matching none of the live fields is dropped
    here on purpose; one matching no field at all is still the config error
    `Augmenter._check_patterns` raises."""
    live = set(live_fields)
    return {
        pattern: ops for pattern, ops in dict(streams or {}).items()
        if any(fnmatch.fnmatchcase(field, pattern) for field in live)
    }


class Augmenter:
    """Config-driven, tensor-only training augmentation over the standard batch dict.

    `streams` maps a batch-field name (or an fnmatch glob over field names, e.g.
    `observation.images.*`) to the ops applied to it. Several patterns may match one field;
    their op tables merge, later patterns winning per op.

        streams:
          "observation.images.*":
            random_crop: {scale: [0.9, 1.0]}
            color_jitter: {brightness: 0.2, hue: 0.02}
          observation.state: {gaussian_noise: {std: 0.01}}

    Each field's parameters are drawn once per item and shared across its whole time
    window; different fields draw independently.

    Image ops need a `(..., C, H, W)` field. Those fields are worked on as floats in [0, 1]
    and cast back to the input dtype (uint8 streams are scaled by 255 both ways), so a
    magnitude in a config means the same thing whichever dtype a backend serves.
    """

    def __init__(self, streams, seed=None):
        self.streams = {}
        for pattern, ops in _plain(streams).items():
            if ops is None:
                continue            # a run config nulls a pattern its model reads no field for
            self.streams[pattern] = {
                name: self._check_op(pattern, name, params or {}) for name, params in ops.items()
            }
        self.rng = _Rng(seed)
        self._per_field = {}
        self._checked_patterns = False

    @staticmethod
    def _check_op(pattern, name, params):
        if name not in OPS:
            raise ValueError(
                f"unknown augmentation '{name}' for streams pattern '{pattern}'. "
                f"available: {sorted(OPS)}"
            )
        allowed = set(inspect.signature(OPS[name]).parameters) - {"x", "rng"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {sorted(unknown)} for augmentation '{name}'. "
                f"available: {sorted(allowed)}"
            )
        return dict(params)

    def _ops_for(self, field):
        if field not in self._per_field:
            merged = {}
            for pattern, ops in self.streams.items():
                if fnmatch.fnmatchcase(field, pattern):
                    merged.update(ops)
            self._per_field[field] = {name: merged[name] for name in OPS if name in merged}
        return self._per_field[field]

    def _check_patterns(self, batch):
        """Fail fast on a pattern that matches no field: a typo'd stream name would
        otherwise silently disable the augmentation it was meant to configure.
        """
        if self._checked_patterns:
            return
        for pattern in self.streams:
            if not any(fnmatch.fnmatchcase(field, pattern) for field in batch):
                raise ValueError(
                    f"augment streams pattern '{pattern}' matches no batch field. "
                    f"available: {sorted(batch)}"
                )
        self._checked_patterns = True

    def __call__(self, batch):
        self._check_patterns(batch)
        out = dict(batch)
        for field, tensor in batch.items():
            ops = self._ops_for(field)
            if ops and torch.is_tensor(tensor):
                out[field] = self._apply(field, tensor, ops)
        return out

    def apply_batch(self, batch):
        """`__call__` over a collated batch: every tensor field with ops is augmented one
        sample at a time along its leading batch axis, so each sample draws its own
        parameters exactly as the per-item wrapper would, and the whole thing runs on
        whatever device the batch lives on (`experiment.augment_on: device` moves the
        ~200 ms-per-sample CPU cost of the image ops onto the GPU, where it is ~ms).
        Non-tensor entries (task strings) pass through untouched."""
        self._check_patterns(batch)
        fields = [f for f, t in batch.items() if torch.is_tensor(t) and self._ops_for(f)]
        if not fields:
            return dict(batch)
        # samples outer, fields inner: the same draw order as `__call__` on successive
        # items, so a seed reproduces the worker path sample for sample
        n = batch[fields[0]].shape[0]
        per_sample = [
            {f: self._apply(f, batch[f][i], self._ops_for(f)) for f in fields}
            for i in range(n)
        ]
        out = dict(batch)
        for f in fields:
            out[f] = torch.stack([s[f] for s in per_sample])
        return out

    def _apply(self, field, tensor, ops):
        as_image = any(name in IMAGE_OPS for name in ops) or tensor.dtype == torch.uint8
        if not as_image:
            return self._run(tensor, ops)

        if tensor.ndim < 3 or tensor.shape[-3] not in (1, 3, 4):
            raise ValueError(
                f"field '{field}' has shape {tuple(tensor.shape)}, which is not "
                f"(..., C, H, W); image augmentations {list(IMAGE_OPS)} need a visual stream"
            )
        dtype = tensor.dtype
        x = tensor.float() / 255.0 if dtype == torch.uint8 else tensor.float()
        x = self._run(x, ops).clamp(0.0, 1.0)
        if dtype == torch.uint8:
            x = (x * 255.0).round()
        return x.to(dtype)

    def _run(self, x, ops):
        for name, params in ops.items():
            x = OPS[name](x, self.rng, **params)
        return x


class AugmentedSource:
    """Wraps any source (or a NormalizedSource around one), augmenting each item on
    `__getitem__`. Sits outside normalization: image streams arrive raw, while state and
    action noise is specified in normalized units.

    Attribute access falls through to the wrapped source, so `provided_modalities`,
    `stats`, and `method` stay reachable through the wrapper.
    """

    def __init__(self, dataset, cfg):
        self.dataset = dataset
        cfg = _plain(cfg)
        self.seed = cfg.get("seed")
        self.augmenter = Augmenter(cfg.get("streams") or {}, seed=self.seed)
        self._seeded_pid = None

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self._seed_process()
        return self.augmenter(self.dataset[idx])

    def _seed_process(self):
        # DataLoader workers fork with an identical generator state; offset the seed per
        # worker or every worker replays the same stream of augmentations
        pid = os.getpid()
        if self.seed is None or self._seeded_pid == pid:
            return
        info = torch.utils.data.get_worker_info()
        self.augmenter.rng.reseed(self.seed + (info.id if info is not None else 0))
        self._seeded_pid = pid
