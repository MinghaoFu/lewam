"""N-dimensional rotary position embeddings, built on rotary-embedding-torch.

RopeND assembles per-token rotation frequencies from named position axes; the rotation
math lives in the library. Compute freqs once per forward pass and apply them to Q/K in
every block with `apply_rotary_emb(freqs, x)`.

An algorithm declares its axes; there is no default set.

    model:
      rope:
        time: {share: 0.75, period: auto}
        seq: {share: 0.25, min_period: 4.0, max_period: 512.0}

`share` is a portion of the head dim, normalized and rounded to whole rotation pairs so
the parts cover it exactly. Periods are in that axis's own units -- seconds for `time`,
token counts for `seq`, grid cells for `height`/`width` -- and either end may be a number
or `auto` to be fitted by `fit_ladder`; `period: auto` fits both.

One ladder per axis, never per stream: `q_m . k_n` reduces to a function of (m - n) only
when both tokens rotate on the same frequencies.
"""

import math

import torch
import torch.nn as nn
from rotary_embedding_torch import apply_rotary_emb

__all__ = [
    "RopeND", "apply_rotary_emb", "make_pos_ids", "resolve_axes", "axes_from_positions",
]

POS_AXES = ("time", "height", "width")


def make_pos_ids(times, grid=(1, 1)):
    """Position ids for one stream's tokens: axis name -> (len(times) * H * W,).

    times: seconds relative to decision time t=0. grid: (H, W) patch grid per timestep;
    (1, 1) attaches no spatial extent (h=w=0). Tokens are ordered time-major.

    No `seq` axis: it indexes the whole concatenated sequence, so only the caller that
    concatenates the streams can assign it.
    """
    h, w = grid
    # the grid aranges must live where `times` lives: the scratch ViT passes a CUDA
    # tensor (trunk callers pass CPU lists) and meshgrid refuses mixed devices
    t = torch.as_tensor(times, dtype=torch.float32)
    ids = torch.meshgrid(
        t,
        torch.arange(h, dtype=torch.float32, device=t.device),
        torch.arange(w, dtype=torch.float32, device=t.device),
        indexing="ij",
    )
    return dict(zip(POS_AXES, (i.flatten() for i in ids)))


def fit_ladder(ids):
    """Fit a period range to ids: min_period = 2 * smallest gap, max_period = 2 * span.

    Every dial then sweeps at most half a turn over the scale it resolves, the most it can
    sweep and stay unambiguous since cos is injective on [0, pi]. The fastest dial lands on
    a multiple of pi at every position, so sin is zero there and that one pair carries a
    parity bit rather than a rotation.

    Returns None for ids that never vary.
    """
    if ids is None:
        return None
    # round before unique: two streams whose times coincide can land a float ULP apart,
    # which would fit min_period to the rounding error and alias everything
    vals = torch.unique(ids.detach().float().round(decimals=6))
    if vals.numel() < 2:
        return None
    # no clamp: span >= gap for any two distinct ids, so max >= min by construction
    return {
        "min_period": 2 * vals.diff().min().item(),
        "max_period": 2 * (vals[-1] - vals[0]).item(),
    }


SPEC_EXAMPLE = "{time: {share: 0.75, period: auto}, seq: {share: 0.25, period: auto}}"


def _resolve_periods(name, entry, fitted):
    """An axis's (min_period, max_period), each either a number or fitted via `auto`."""
    if "period" in entry:
        assert entry["period"] == "auto", (
            f"rope axis '{name}': `period` only takes 'auto'; for explicit values set "
            "min_period and max_period"
        )
        clash = {"min_period", "max_period"} & set(entry)
        assert not clash, (
            f"rope axis '{name}' sets both `period: auto` and {sorted(clash)} -- pick one"
        )
        wanted = {"min_period": "auto", "max_period": "auto"}
    else:
        missing = {"min_period", "max_period"} - set(entry)
        assert not missing, (
            f"rope axis '{name}' must set {sorted(missing)}, or `period: auto` to fit both "
            "to the positions this arm builds. Either end may be 'auto' on its own."
        )
        wanted = {k: entry[k] for k in ("min_period", "max_period")}

    out = {}
    for key, value in wanted.items():
        if value != "auto":
            out[key] = float(value)
            continue
        assert fitted is not None, (
            f"rope axis '{name}' asks for an automatic {key}, but its positions never vary "
            "so there is nothing to fit -- set it explicitly, or drop the axis"
        )
        out[key] = fitted[key]
    return out


def axes_from_positions(pos, spec):
    """Declared axes + the positions they will carry -> {axis: {share, min_period, max_period}}.

    `spec` is an algorithm's `model.rope` and is required: an axis rotates only if it was
    asked for, and its ladder is fitted only where it says `auto`.
    """
    assert spec, (
        f"model.rope must declare its axes, each one's share of the head dim, and its "
        f"period range, e.g. {SPEC_EXAMPLE} -- there is no default axis set"
    )
    axes = {}
    for name, entry in spec.items():
        # a leaf drops an inherited axis by setting it to null (OmegaConf merge cannot
        # delete a key) -- e.g. 1-query pooling leaves `width` with nothing to rotate
        if entry is None:
            continue
        assert isinstance(entry, dict) or hasattr(entry, "keys"), (
            f"rope axis '{name}' must be a mapping with `share` and a period range, "
            f"e.g. {SPEC_EXAMPLE}"
        )
        entry = dict(entry)
        assert "share" in entry, (
            f"rope axis '{name}' must say what share of the head dim it takes"
        )
        axes[name] = {"share": entry["share"], **_resolve_periods(name, entry, fit_ladder(pos.get(name)))}
    return axes


def split_dims(shares, head_dim):
    """{axis: share} -> {axis: even dim count}, covering head_dim exactly.

    Allocation is in rotation pairs (RoPE turns two dims at a time), by largest remainder.
    """
    names = list(shares)
    assert names, "rope needs at least one axis"
    assert head_dim % 2 == 0, f"head_dim {head_dim} must be even"
    pairs = head_dim // 2
    assert pairs >= len(names), (
        f"head_dim {head_dim} cannot give a rotation pair to each of {len(names)} rope axes"
    )
    assert all(s > 0 for s in shares.values()), f"rope shares must be positive, got {shares}"

    total = sum(shares.values())
    raw = {n: pairs * shares[n] / total for n in names}
    got = {n: int(math.floor(raw[n])) for n in names}
    short = pairs - sum(got.values())
    for n in sorted(names, key=lambda k: raw[k] - got[k], reverse=True)[:short]:
        got[n] += 1
    # a small enough share rounds to nothing; every declared axis still gets a pair
    while min(got.values()) == 0:
        got[max(names, key=lambda k: got[k])] -= 1
        got[min(names, key=lambda k: got[k])] += 1
    return {n: got[n] * 2 for n in names}


def resolve_axes(axes, head_dim):
    """{axis: {share, min_period, max_period}} -> {axis: (dims, min_period, max_period)}."""
    dims = split_dims({n: float(a["share"]) for n, a in axes.items()}, head_dim)
    resolved = {}
    for name, a in axes.items():
        lo, hi = float(a["min_period"]), float(a["max_period"])
        assert 0 < lo <= hi, f"axis '{name}': need 0 < min_period <= max_period, got {lo}, {hi}"
        resolved[name] = (dims[name], lo, hi)
    return resolved


class RopeND(nn.Module):
    """Rotation frequencies for tokens positioned on any number of named axes.

    Position ids are continuous floats, so `time` in seconds rather than frame index keeps
    one config working at 30 fps and 2.5 fps. An axis a stream does not populate is passed
    as zeros, leaving that band an identity rotation while still costing its head dims.
    """

    def __init__(self, head_dim, axes):
        super().__init__()
        self.head_dim = head_dim
        self.axes = resolve_axes(axes, head_dim)
        for name, (dims, lo, hi) in self.axes.items():
            n = dims // 2
            fraction = torch.linspace(0.0, 1.0, n) if n > 1 else torch.zeros(1)
            period = lo * (hi / lo) ** fraction
            self.register_buffer(f"freqs_{name}", 2 * math.pi / period, persistent=False)

    def extra_repr(self):
        return ", ".join(f"{n}: {d}d [{lo:g}, {hi:g}]" for n, (d, lo, hi) in self.axes.items())

    def forward(self, ids):
        """ids: {axis: (N,) float}. Axes absent from `ids` rotate by nothing.

        Returns (N, head_dim) rotation freqs, laid out in the order the axes were declared.
        """
        ref = next(iter(ids.values()))
        parts = []
        for name, (dims, _, _) in self.axes.items():
            pos = ids.get(name)
            if pos is None:
                parts.append(ref.new_zeros(ref.shape[0], dims))
                continue
            angles = pos[:, None].float() * getattr(self, f"freqs_{name}")[None]
            # the library pairs adjacent dims (`rotate_half` reshapes '... (d r)', r=2),
            # so each frequency has to land on both dims of its pair
            parts.append(angles.repeat_interleave(2, dim=-1))
        return torch.cat(parts, dim=-1).to(ref.dtype)
