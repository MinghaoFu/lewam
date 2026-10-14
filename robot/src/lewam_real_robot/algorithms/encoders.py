"""Observation encoders feeding the predictor, frozen or jointly trained.

An encoder maps one raw batch field to what its spec entry's `via` demands:
a (B, N, dim) token tensor for token streams (N = len(index) * grid area),
a (B, S, dim) tensor for a shared cross source, or a list of `depth` tensors
for a per-layer cross source.

Two spec keys compose encoders without new code:

  frozen  no grads, permanent eval mode. Pretrained types (vjepa2)
          default frozen; heads default trainable.
  input   names another `encoders:` entry whose output this one consumes. The
          stream's raw batch field feeds the chain's root (the encoder with no
          `input`); `resolve_chains` orders the chain and rejects cycles, so a
          chain always bottoms out at real data.

`raw_steps_per_index` declares how many raw time steps one index step of the
encoded stream consumes (2 frames per VJEPA2 tubelet), so the algorithm can
slice each spec entry's window out of a shared raw field before encoding.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention
from .layers import SwiGLULinear, make_mlp

__all__ = [
    "Encoder",
    "VJEPA2Encoder",
    "AttentivePoolEncoder",
    "MLPEncoder",
    "ChunkMLPEncoder",
    "build_encoder",
    "resolve_chains",
]


def build_encoder(spec):
    """One `encoders:` section entry -> module, keyed by its `type` field."""
    kind = spec["type"]
    if kind not in _ENCODER_TYPES:
        raise ValueError(f"unknown encoder type '{kind}'. available: {sorted(_ENCODER_TYPES)}")
    encoder = _ENCODER_TYPES[kind](spec)
    if encoder.frozen:
        encoder.requires_grad_(False)
        encoder.eval()
    return encoder


def resolve_chains(encoder_cfgs):
    """`encoders:` section -> {name: [root, ..., name]} execution order per encoder.

    Follows each entry's `input` link back to a root that consumes the raw batch
    field. Unknown names and cycles are build-time errors: a cyclic chain would
    have no raw input at all.
    """
    chains = {}

    def resolve(name, visiting):
        if name in chains:
            return chains[name]
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise ValueError(f"encoder chain feeds back into itself ({cycle}): no raw input")
        parent = encoder_cfgs[name].get("input")
        if parent is None:
            chain = [name]
        elif parent not in encoder_cfgs:
            raise ValueError(f"encoder '{name}' takes input from unknown encoder '{parent}'")
        else:
            chain = resolve(parent, [*visiting, name]) + [name]
        chains[name] = chain
        return chain

    for name in encoder_cfgs:
        resolve(name, [])
    return chains


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Encoder(nn.Module):
    """Shared contract: forward(raw) returns the tensor(s) the trunk expects for
    every spec entry that names this encoder. `frozen` (spec key, per-type
    default) is applied by build_encoder and pins the module in eval mode.

    `supports_encoding_cache` marks the pixel backbones whose frozen output the
    encoding cache (utils/enc_cache.py) may precompute. It is a property of the
    TYPE, not of `frozen`: a probe run freezes every encoder (freeze_world_model),
    which must not turn a cheap non-visual codec (chunk_mlp on actions) into a
    "cacheable" stream -- the cache stores visual windows, and the dataset rejects
    a non-visual cache field.
    """

    raw_steps_per_index = 1
    default_frozen = False
    supports_encoding_cache = False

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.frozen = bool(spec.get("frozen", self.default_frozen))

    def train(self, mode=True):
        # the owning algorithm flips train/eval globally; a frozen encoder must keep
        # eval-mode dropout/BN statistics regardless
        return super().train(mode and not self.frozen)

    def forward(self, raw):
        raise NotImplementedError


def vjepa_module(name):
    """Import module `name` (e.g. "app.vjepa_2_1.models.vision_transformer") from the
    vendored external/vjepa2 subset.

    The vjepa2 repo is not a package: its files import each other as top-level
    `src.*` / `app.*`, assuming the repo root is on sys.path. Append (not prepend)
    the vendored root so those names resolve without shadowing real packages.

    Installing it instead is not an option: its setup.py declares no packages, so
    discovery flattens `src/` to top-level `datasets`/`models`/`utils` (colliding with
    real ones) and drops `app/` altogether -- the half this actually imports.
    """
    import importlib

    root = Path(__file__).resolve().parents[3] / "external" / "vjepa2"
    if not (root / "app").is_dir():
        raise ImportError(f"vendored vjepa2 code not found at {root}")
    if str(root) not in sys.path:
        sys.path.append(str(root))
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as err:
        # the vendored code is source on sys.path, not an installed distribution, so its
        # own requirements never get resolved
        raise ImportError(
            f"the vendored vjepa2 code needs '{err.name}', which is not installed. "
            f"try: uv pip install 'lewam-real-robot[vjepa]'"
        ) from err


def _vjepa_vision_transformer():
    return vjepa_module("app.vjepa_2_1.models.vision_transformer")


class VJEPA2Encoder(Encoder):
    """Pretrained V-JEPA 2.1 video backbone over (B, T, C, H, W) uint8 frames.

    Frames are resized to `crop_size` (int or [H, W]), ImageNet-normalized and grouped into
    2-frame tubelets; output is (B, steps * H/16 * W/16, dim) patch latents.
    `frames_per_step` sets how a tubelet is filled:

      1 (default)  each frame is duplicated into its own tubelet (one step = one frame)
      2            consecutive frame pairs form tubelets (2 raw rows per index step)

    Spec: `arch` (vit_tiny / vit_small / vit_base / vit_large / vit_huge / vit_giant /
    vit_gigantic), `checkpoint` (local path, URL, `pretrained` for the official release, or
    null for random weights), `crop_size`, `frames_per_step`, `backbone` (extra factory
    kwargs), `frozen` (default true). Downloads are cached under `$TORCH_HOME/hub/checkpoints`.
    """

    default_frozen = True
    supports_encoding_cache = True
    TUBELET_SIZE = 2

    PRETRAINED_URLS = {
        "vit_base": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        "vit_large": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        "vit_giant": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
        "vit_gigantic": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
    }

    def __init__(self, spec):
        super().__init__(spec)
        # an int is a square crop; [H, W] keeps a wide input at its own aspect, and
        # the stream's `grid` becomes [H/16, W/16]
        crop = spec.get("crop_size", 256)
        self.crop_size = ((int(crop), int(crop)) if isinstance(crop, (int, float))
                          else tuple(int(v) for v in crop))
        self.raw_steps_per_index = int(spec.get("frames_per_step", 1))
        assert self.raw_steps_per_index in (1, self.TUBELET_SIZE), (
            f"frames_per_step must be 1 or {self.TUBELET_SIZE}"
        )
        # `per_frame: true` encodes every index step as its own SINGLE-FRAME clip in the
        # batch dimension (the backbone's `img_temporal_dim_size: 1` image path), so
        # encoder attention never crosses time; every step sits at temporal RoPE position 0
        self.per_frame = bool(spec.get("per_frame", False))
        assert not (self.per_frame and self.raw_steps_per_index != 1), (
            "per_frame feeds one raw frame per index step; set frames_per_step: 1 "
            "and un-tubeleted raw_index strings in the arm config"
        )

        vt = _vjepa_vision_transformer()
        arch = spec.get("arch", "vit_large")
        if not arch.startswith("vit_") or not hasattr(vt, arch):
            available = sorted(n for n in dir(vt) if n.startswith("vit_"))
            raise ValueError(f"unknown vjepa2 arch '{arch}'. available: {available}")
        kwargs = {
            "img_size": self.crop_size,
            "num_frames": 16,
            "tubelet_size": self.TUBELET_SIZE,
            "use_sdpa": True,
            "use_silu": False,
            "wide_silu": True,
            "uniform_power": True,
            "use_rope": True,
            "img_temporal_dim_size": 1,
            "interpolate_rope": True,
            **(spec.get("backbone") or {}),
        }
        self.backbone = getattr(vt, arch)(**kwargs)
        if spec.get("checkpoint"):
            self._load_checkpoint(self._resolve_checkpoint(spec["checkpoint"], arch))

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(3, 1, 1))

    def _resolve_checkpoint(self, ref, arch):
        """`pretrained` -> official URL for the arch; URLs -> cached local download."""
        ref = str(ref)
        if ref == "pretrained":
            if arch not in self.PRETRAINED_URLS:
                raise ValueError(
                    f"no published V-JEPA 2.1 weights for '{arch}'; "
                    f"available: {sorted(self.PRETRAINED_URLS)}"
                )
            ref = self.PRETRAINED_URLS[arch]
        if ref.startswith(("http://", "https://")):
            dest = Path(torch.hub.get_dir()) / "checkpoints" / ref.rsplit("/", 1)[-1]
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                # download_url_to_file writes a temp file and renames, so concurrent ranks
                # at worst download twice, never read a torn file
                torch.hub.download_url_to_file(ref, str(dest), progress=True)
            ref = str(dest)
        return ref

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # a V-JEPA 2.1 training checkpoint nests the weights (EMA preferred)
        state = next((ckpt[k] for k in ("ema_encoder", "target_encoder", "encoder")
                      if isinstance(ckpt, dict) and k in ckpt), ckpt)
        state = {
            k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()
        }
        self.backbone.load_state_dict(state, strict=True)

    @property
    def gradient_checkpointing(self):
        return self.backbone.use_activation_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self.backbone.use_activation_checkpointing = bool(value)

    def forward(self, frames):
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)
        if self.raw_steps_per_index == 1 and not self.per_frame:
            frames = frames.repeat_interleave(self.TUBELET_SIZE, dim=1)
        B, T = frames.shape[:2]
        if not self.per_frame:
            assert T % self.TUBELET_SIZE == 0, f"window of {T} frames does not fill tubelets"

        x = frames.float()
        if frames.dtype == torch.uint8:
            x = x / 255.0
        x = x.flatten(0, 1)                                       # (B*T, C, H, W)
        if x.shape[-2:] != self.crop_size:
            x = F.interpolate(x, size=self.crop_size, mode="bilinear", antialias=True)
        x = (x - self.mean) / self.std
        dtype = next(self.backbone.parameters()).dtype
        if self.per_frame:
            # each step its own T=1 clip; output stays time-major (B, T*P, D), the
            # same layout the clip path produces
            out = self.backbone(x.unsqueeze(2).to(dtype))         # (B*T, C, 1, H, W)
            return out.unflatten(0, (B, T)).flatten(1, 2)
        x = x.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)         # (B, C, T, H, W)
        return self.backbone(x.to(dtype))


class AttentivePoolEncoder(Encoder):
    """Learned-query attention pooling over another encoder's patch tokens:
    (B, steps * tokens_per_step, in_dim) -> (B, steps * num_queries, out_dim).

    Chained after a patch-token encoder (`input: <name>`). Each step is pooled independently
    by the shared queries, so the output keeps one group of `num_queries` tokens per step.
    The consuming stream declares `dim: out_dim` and `grid: [1, num_queries]`.

    Spec: `in_dim`, `out_dim`, `tokens_per_step` (the source's grid area, e.g. 36 for a 6x6
    grid), `num_queries` (default 1), `num_heads` (default 8), `mlp_ratio` (default 4.0),
    `frozen` (default false).
    """

    def __init__(self, spec):
        super().__init__(spec)
        in_dim, out_dim = int(spec["in_dim"]), int(spec["out_dim"])
        self.tokens_per_step = int(spec["tokens_per_step"])
        self.num_queries = int(spec.get("num_queries", 1))
        num_heads = int(spec.get("num_heads", 8))
        assert out_dim % num_heads == 0, (
            f"attentive pool out_dim {out_dim} must divide into {num_heads} heads"
        )

        self.query = nn.Parameter(torch.randn(1, self.num_queries, out_dim) * 0.02)
        self.norm_src = nn.LayerNorm(in_dim, eps=1e-6)
        self.attn = CrossAttention(out_dim, num_heads, [in_dim])
        # the query enters attention unnormalized: it is a free parameter, so a norm
        # in front of it only reparametrizes what the parameter already learns
        self.norm_q = nn.LayerNorm(out_dim, eps=1e-6)
        self.mlp = make_mlp(out_dim, int(out_dim * float(spec.get("mlp_ratio", 4.0))), out_dim)

    def forward(self, x):
        b, n, _ = x.shape
        span = self.tokens_per_step
        if n % span:
            raise ValueError(
                f"attentive pool got {n} tokens, not a multiple of tokens_per_step="
                f"{span}; tokens_per_step must be the source encoder's grid area"
            )
        src = self.norm_src(x.reshape(b * (n // span), span, -1))
        q = self.query.expand(src.shape[0], -1, -1)
        q = q + self.attn(q, [src])
        q = q + self.mlp(self.norm_q(q))
        return q.reshape(b, -1, q.shape[-1])


class MLPEncoder(Encoder):
    """SwiGLU MLP over the last dim: norm(w(x)) * silu(w_gate(x)) -> Linear,
    i.e. the trunk's make_mlp with an optional norm on the value path. Encodes a
    raw low-dimensional field (e.g. the stacked action block conditioning the
    trunk via AdaLN) or, chained with `input:`, adapts another encoder's output.

    Spec: `in_dim`, `out_dim`, `hidden_dim` (default 4 * out_dim), `norm`
    ("batch", "layer", or null -- default null). BatchNorm sees the flattened
    leading dims as the batch.
    """

    def __init__(self, spec):
        super().__init__(spec)
        in_dim, out_dim = int(spec["in_dim"]), int(spec["out_dim"])
        hidden = int(spec.get("hidden_dim", 4 * out_dim))
        norm = spec.get("norm")
        norms = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}
        assert norm is None or norm in norms, f"unknown MLPEncoder norm '{norm}'"
        self.mlp = nn.Sequential(
            SwiGLULinear(in_dim, hidden, norm=norms[norm](hidden) if norm else None),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        x = x.float()
        lead, d = x.shape[:-1], x.shape[-1]
        return self.mlp(x.reshape(-1, d)).reshape(*lead, -1)


class ChunkMLPEncoder(Encoder):
    """Chunk-level MLP codec: (B, k * in_steps, in_dim) -> (B, k * out_steps, out_dim),
    each in_steps-row chunk flattened and mapped as one vector, chunks independently.

    With `in_steps: k, out_steps: 1` it compresses each k-row window (e.g. a state
    history) into one token.

    Spec: `in_dim`, `out_dim`, `in_steps` (rows consumed per chunk, default 1; becomes
    `raw_steps_per_index`), `out_steps` (rows emitted per chunk, default 1), `hidden_dim`
    (default 4 * out_dim), `depth` (hidden SwiGLU layers, default 1), `norm` ("batch",
    "layer", or null), `frozen` (default false).
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.in_dim, self.out_dim = int(spec["in_dim"]), int(spec["out_dim"])
        self.in_steps = int(spec.get("in_steps", 1))
        self.out_steps = int(spec.get("out_steps", 1))
        self.raw_steps_per_index = self.in_steps
        hidden = int(spec.get("hidden_dim", 4 * self.out_dim))
        depth = int(spec.get("depth", 1))
        norm = spec.get("norm")
        norms = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}
        assert norm is None or norm in norms, f"unknown ChunkMLPEncoder norm '{norm}'"
        layers, width = [], self.in_steps * self.in_dim
        for _ in range(depth):
            layers.append(SwiGLULinear(width, hidden, norm=norms[norm](hidden) if norm else None))
            width = hidden
        layers.append(nn.Linear(width, self.out_steps * self.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        b, n, d = x.shape
        if n % self.in_steps:
            raise ValueError(
                f"chunk_mlp got {n} rows, not a multiple of in_steps={self.in_steps}"
            )
        k = n // self.in_steps
        z = self.net(x.reshape(b * k, self.in_steps * d))
        return z.reshape(b, k * self.out_steps, self.out_dim)


_ENCODER_TYPES = {
    "vjepa2": VJEPA2Encoder,
    "attentive_pool": AttentivePoolEncoder,
    "mlp": MLPEncoder,
    "chunk_mlp": ChunkMLPEncoder,
}
