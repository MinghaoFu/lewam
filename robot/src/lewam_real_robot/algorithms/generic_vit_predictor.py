"""Generic ViT predictor: one transformer over configurable token streams.

The `conditioning` and `predict` specs describe the model's interface; stream names are
arbitrary keys. Each predicted stream declares how it is produced:

    flow    tokens enter as noise x_t and leave as a velocity; target = x1 - x0.
    direct  adds no tokens: reads a head off its `input` stream's tokens and regresses the
            encoded value at its own `index`, position i predicting target i.

Both types can be mixed in one `predict:` block.

    conditioning:
      video:  { dim: 192, index: "-2..0", grid: [1, 1] }   # via: tokens (default)
      state:  { dim: 6, via: cross }                       # cross-attention source
      action: { dim: 2, via: cond, index: "-2..0" }        # per-frame AdaLN conditioning
    predict:
      video:  { type: direct, input: video, dim: 192, index: "-1..1" }
      action: { type: flow, dim: 2, index: "0..47", fps: 30, block_size: 2 }

Token and flow streams share one time axis in declaration order, conditioning first. Token
counts, RoPE positions and the block-causal mask derive from each stream's
index/fps/grid/block_size. Attention is unmasked unless `model.causal: true`, which a
`direct` stream requires.
"""

import torch
import torch.nn as nn

from ..utils.spec import parse_index
from .block import DiTBlock, MoTBlock
from .layers import (
    FinalLayer, InputLayer, MLPFinalLayer, MLPInputLayer, TimestepEmbedder, make_mlp,
)
from .masking import (assert_no_attention_path, block_causal_flex_mask,
                      block_causal_mask, block_intervals, build_stream_visibility)
from .predictive_model import PredictiveModel
from .rope import RopeND, axes_from_positions, make_pos_ids

__all__ = ["GenericViTPredictor", "GenericViTTrunk"]


def _resolve_attn_backend(backend):
    """"auto" -> flex on Ampere+ GPUs, sdpa otherwise."""
    if backend != "auto":
        return backend
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return "flex"
    return "sdpa"


def _stream_geometry(spec):
    """index/fps/grid/block_size of one token stream."""
    indices = torch.tensor(parse_index(spec["index"]), dtype=torch.long)
    fps = float(spec.get("fps", 1.0))
    grid = tuple(spec.get("grid", (1, 1)))
    block_size = int(spec.get("block_size", 1))
    return indices, indices.to(torch.float64) / fps, grid, fps, block_size


def _stream_times(spec):
    """The time (seconds) of each index step of a stream, ignoring its spatial grid."""
    indices = torch.tensor(parse_index(spec["index"]), dtype=torch.long)
    return indices.to(torch.float64) / float(spec.get("fps", 1.0))


def is_direct(spec):
    return spec.get("type") == "direct"


def is_mse(spec):
    return spec.get("type") == "mse"


def _token_io(name, spec, key, default_mlp_dim):
    """`(kind, mlp_dim)` for a predicted stream's `token_in`/`token_out` block.

    `mlp` is DreamZero's action encoder/decoder, an MLP whose inner width `mlp_dim`
    names, with the timestep concatenated at the input; it is the default for every
    predicted stream (flow or direct) so noise/action/state/vision tokens all leave the
    trunk the same way. `adaln` (InputLayer/FinalLayer, the modulated projections) is
    still available by declaring it explicitly. Declaring nothing keeps `mlp` at
    `default_mlp_dim` (the trunk's `model_dim`).
    """
    block = spec.get(key)
    if block is None:
        return "mlp", default_mlp_dim
    block = dict(block)
    kind, mlp_dim = block.pop("type", "mlp"), block.pop("mlp_dim", None)
    if block:
        raise ValueError(f"'{name}' {key} declares unknown keys {sorted(block)}")
    if kind == "adaln":
        if mlp_dim is not None:
            raise ValueError(f"'{name}' {key} is adaln, which has no mlp_dim to set")
        return kind, None
    if kind == "mlp":
        if mlp_dim is None:
            raise ValueError(f"'{name}' {key} is mlp and must declare its `mlp_dim`")
        return kind, int(mlp_dim)
    raise ValueError(f"'{name}' {key} declares unknown type '{kind}'; use 'mlp' or 'adaln'")


def sample_flow_time(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001):
    """Training timesteps for the flow streams: pi0/pi0.5/SmolVLA's Beta schedule, mirrored
    onto this repo's time axis. Returns (bsize,) float32 in (1 - scale - offset, 1 - offset).

    They draw `Beta(alpha, beta) * scale + offset`, which at (1.5, 1.0) has density ~sqrt(t)
    and so puts most mass at t=1 -- their noise end, where the velocity field is hardest and
    worth the extra samples. Here t=0 is noise and t=1 is data (see `_forward_train`), so the
    draw is flipped to skew the same way. `1 - Beta(alpha, beta)` rather than the identical
    `Beta(beta, alpha)` keeps the parameters reading as theirs do.

    Beta goes through _sample_dirichlet, which MPS does not implement, so the draw is taken
    on CPU and moved -- meaning it consumes the CPU RNG stream, as lerobot's pi0 does.
    """
    dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta))
    t = dist.sample((bsize,)) * scale + offset
    return (1.0 - t).to(device=device, dtype=torch.float32)


def build_expert_ids(order, clean, noisy, experts):
    """(expert names, (N_streams,) long) from `model.experts`: a mapping expert name ->
    list of token streams (names, or the `all` / `context` / `noisy` groups `attends:`
    accepts). Every token stream must land in exactly one expert -- a stream left out or
    named twice is an error, not a default -- and a single expert is refused: that is the
    shared trunk, spelled by omitting the key."""
    idx = {n: i for i, n in enumerate(order)}
    groups = {"all": list(order), "context": list(clean), "noisy": list(noisy)}
    names = [str(n) for n in experts]
    if len(names) < 2:
        raise ValueError(
            f"model.experts declares {names}: one expert is the shared trunk -- drop the "
            "key for that, or name at least two experts"
        )
    owner = {}
    for name in names:
        listed = experts[name]
        if not listed:
            raise ValueError(f"model.experts['{name}'] lists no streams")
        for entry in listed:
            for stream in groups.get(str(entry), [str(entry)]):
                if stream not in idx:
                    raise ValueError(
                        f"model.experts['{name}'] names '{stream}', which is not a token "
                        f"stream (token streams: {order}; groups: all, context, noisy)"
                    )
                if stream in owner and owner[stream] != name:
                    raise ValueError(
                        f"model.experts assigns '{stream}' to both '{owner[stream]}' and "
                        f"'{name}'; a token is processed by exactly one expert"
                    )
                owner[stream] = name
    missing = [n for n in order if n not in owner]
    if missing:
        raise ValueError(
            f"model.experts leaves token streams {missing} unassigned; every stream must "
            f"name its expert (experts: {names})"
        )
    ids = torch.tensor([names.index(owner[n]) for n in order], dtype=torch.long)
    return names, ids


class GenericViTTrunk(nn.Module):
    def __init__(
        self,
        conditioning,
        predict,
        model_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        mlp_dim=None,
        dropout=0.0,
        attn_backend="auto",
        dim_head=None,
        causal=False,
        clean_attends_noisy=True,
        attention=None,
        rope=None,
        experts=None,
    ):
        super().__init__()
        self.depth = depth
        self.attn_backend = _resolve_attn_backend(attn_backend)
        self.gradient_checkpointing = False

        clean_streams, cross_sources, cond_sources = [], [], []
        for name, spec in conditioning.items():
            via = spec.get("via", "tokens")
            assert via in ("tokens", "cross", "cond", None), f"unknown via '{via}' for '{name}'"
            assert not (spec.get("token_in") or spec.get("token_out")), (
                f"conditioning stream '{name}' declares token_in/token_out, which configure "
                "a PREDICTED stream's projections; a conditioning stream enters through a "
                "plain linear and is never read by a head"
            )
            if via is None:
                # `via: null`: encoded for loss terms, never a trunk input
                continue
            {"cross": cross_sources, "cond": cond_sources}.get(via, clean_streams).append(
                (name, spec)
            )
        assert predict, "predict is empty"
        noisy_streams = [(n, s) for n, s in predict.items()
                         if not is_direct(s) and not is_mse(s)]
        # `type: mse`: the stream contributes LEARNED QUERY tokens (one shared embedding
        # per stream, positions told apart by rope) and its head regresses the encoded
        # target directly -- no noise, no integration
        query_streams = [(n, s) for n, s in predict.items() if is_mse(s)]
        direct_streams = [(n, s) for n, s in predict.items() if is_direct(s)]

        self.stream_order, self.stream_slices, self.stream_dims = [], {}, {}
        pos_parts, starts, ends = [], [], []
        offset = 0
        for name, spec in clean_streams + noisy_streams + query_streams:
            indices, times, grid, fps, block_size = _stream_geometry(spec)
            spatial = grid[0] * grid[1]
            n = len(indices) * spatial
            self.stream_order.append(name)
            self.stream_slices[name] = slice(offset, offset + n)
            self.stream_dims[name] = int(spec["dim"])
            offset += n
            pos_parts.append(make_pos_ids(times.float(), grid))
            s, e = block_intervals(indices.repeat_interleave(spatial), fps, block_size)
            starts.append(s)
            ends.append(e)
        self.num_tokens = offset

        # `seq` needs every stream's token count, so it can only be assigned here, and it
        # follows declaration order: reordering `conditioning:` moves every token position
        pos = {axis: torch.cat([p[axis] for p in pos_parts]) for axis in pos_parts[0]}
        pos["seq"] = torch.arange(self.num_tokens, dtype=torch.float32)
        self.rope = RopeND(dim_head or model_dim // num_heads, axes_from_positions(pos, rope))
        dropped = [a for a in ("height", "width") if pos[a].any() and a not in self.rope.axes]
        assert not dropped, (
            f"a stream declares a grid, but model.rope leaves {dropped} out of "
            f"{list(self.rope.axes)}: its patch coordinates would be dropped silently"
        )
        # every axis registered, not just rope-active ones: `_cond_map` reads `pos_time`
        # whether or not time rotates
        for axis, ids in pos.items():
            self.register_buffer(f"pos_{axis}", ids, persistent=False)
        self.pos_axes = tuple(a for a in pos if a in self.rope.axes)
        # tokens on a direct path: their timestep slot is zeroed so `t` cannot leak in
        t_free = torch.zeros(self.num_tokens, dtype=torch.bool)
        assert causal or not direct_streams, (
            f"direct streams {[n for n, _ in direct_streams]} need `model.causal: true`. "
            "A direct head reads its input stream's own tokens and targets the window one "
            "step later, so unmasked the token at index i attends the clean encoded "
            "observation at i + 1 -- the value it is being trained to predict"
        )
        for name, _ in query_streams:
            t_free[self.stream_slices[name]] = True
        for name, spec in direct_streams:
            assert spec.get("token_in") is None, (
                f"direct stream '{name}' declares token_in, but it adds no tokens of its "
                f"own -- it reads '{spec['input']}'s. Only token_out applies here"
            )
            src = spec["input"]
            assert src in self.stream_slices, (
                f"direct stream '{name}' names input '{src}', which is not a token stream"
            )
            n_input = self.stream_slices[src].stop - self.stream_slices[src].start
            n_target = len(parse_index(spec["index"])) * (
                lambda g: g[0] * g[1]
            )(tuple(spec.get("grid", (1, 1))))
            assert n_input == n_target, (
                f"direct stream '{name}': input '{src}' has {n_input} tokens but its target "
                f"window has {n_target}; the windows are paired positionally so they must match"
            )
            t_free[self.stream_slices[src]] = True
        self.register_buffer("t_free", t_free, persistent=False)

        # per-stream visibility: each token stream may declare `attends:` (stream names
        # or the groups all/context/noisy; a stream always attends itself), replacing
        # full visibility for its queries. The legacy `clean_attends_noisy: false` is
        # compiled into the same table: conditioning tokens are pure K/V -- no head
        # reads them -- so they never attend noise-seeded flow tokens even when their
        # time blocks overlap; a direct head's input stream is exempt, exactly as the
        # old role rule had it.
        clean_names = [n for n, _ in clean_streams]
        noisy_names = [n for n, _ in noisy_streams] + [n for n, _ in query_streams]
        for name, spec in cross_sources + cond_sources:
            assert spec.get("attends") is None, (
                f"'{name}' declares attends, but a via: {spec.get('via')} stream is not "
                "a token stream -- visibility only applies to self-attention tokens"
            )
        for name, spec in direct_streams:
            assert spec.get("attends") is None, (
                f"direct stream '{name}' declares attends, but it has no tokens of its "
                f"own -- put the attends on its input stream '{spec.get('input')}'"
            )
        attends = {n: (list(spec["attends"]) if spec.get("attends") is not None else None)
                   for n, spec in clean_streams + noisy_streams + query_streams}
        direct_inputs = {spec["input"] for _, spec in direct_streams}
        legacy_blocked = () if clean_attends_noisy else tuple(
            n for n in clean_names if n not in direct_inputs)
        vis = build_stream_visibility(self.stream_order, clean_names, noisy_names,
                                      attends, legacy_blocked)
        assert_no_attention_path(self.stream_order, vis,
                                 (attention or {}).get("no_path"))
        self._vis = vis
        self._clean_names = list(clean_names)
        self._subset_cache = {}
        stream_ids = torch.empty(self.num_tokens, dtype=torch.long)
        for i, name in enumerate(self.stream_order):
            stream_ids[self.stream_slices[name]] = i
        self.register_buffer("all_stream_ids", stream_ids, persistent=False)
        # per-stream experts (`model.experts`): the trunk owns the stream -> expert table
        # and hands every layout's routes to the blocks; absent = one shared DiT trunk
        self.expert_names = []
        self._route_cache = {}
        if experts is not None:
            self.expert_names, stream_experts = build_expert_ids(
                self.stream_order, clean_names, noisy_names, experts)
            self.register_buffer("all_expert_ids", stream_experts[stream_ids], persistent=False)
        start, end = torch.cat(starts), torch.cat(ends)
        # plain attributes, NOT buffers: the boundaries are float64, which .to(mps)
        # cannot convert. They are only read on CPU to build subset masks.
        self._time_start = start.clone()
        self._time_end = end.clone()
        self._time_causal = bool(causal)
        if not causal:
            # every token's block spans all time, so `kv_start < q_end` holds everywhere
            # and `block_size` stops meaning anything. A `direct` stream needs causal:
            # true -- its input window overlaps its own target window, so without the
            # time rule the token at index i attends the clean encoded observation at
            # i + 1, which is exactly what it is being trained to predict
            start = torch.full_like(start, float("-inf"))
            end = torch.full_like(end, float("inf"))
        # with no rule left there is nothing to express, so skip the mask entirely
        # rather than build an all-True (N, N) tensor and hand it to SDPA
        self._unmasked = not causal and bool(vis.all())
        if not self._unmasked:
            full_vis = None if vis.all() else vis
            # the float64 block boundaries are buffers only on the flex path: MPS cannot
            # hold float64
            if self.attn_backend == "sdpa":
                self.register_buffer(
                    "dense_mask",
                    block_causal_mask(start, end, stream_ids=stream_ids, vis=full_vis),
                    persistent=False,
                )
            else:
                self.register_buffer("block_start", start, persistent=False)
                self.register_buffer("block_end", end, persistent=False)
                self.register_buffer("stream_ids", stream_ids, persistent=False)
                if full_vis is None:
                    self.vis_table = None
                else:
                    self.register_buffer("vis_table", full_vis, persistent=False)
        self._flex_mask = None

        # AdaLN cond slots: [timestep?] + one per `via: cond` stream
        self.has_flow = bool(noisy_streams)
        self.cond_order = [name for name, _ in cond_sources]
        self.t_embedder = TimestepEmbedder(model_dim) if self.has_flow else None
        self.cond_proj = nn.ModuleDict({
            name: nn.Linear(int(spec["dim"]), model_dim) for name, spec in cond_sources
        })
        self.per_token_cond = bool(cond_sources) or bool(direct_streams)
        cond_dim = model_dim * (int(self.has_flow) + len(cond_sources))

        # matched on time, so one per-frame action conditions every token of its frame
        for name, spec in cond_sources:
            self.register_buffer(
                f"cond_map_{name}", self._cond_map(_stream_times(spec)), persistent=False
            )

        # clean/conditioning streams enter through the same MLP lift, minus the
        # timestep: they are never denoised, so `t` has nothing to tell them
        self.clean_proj = nn.ModuleDict({
            name: make_mlp(int(spec["dim"]), model_dim, model_dim) for name, spec in clean_streams
        })
        self.noisy_proj = nn.ModuleDict()
        # the streams whose token_in reads the bare timestep slot instead of the full cond
        self.time_concat = set()
        for name, spec in noisy_streams:
            kind, mlp_dim = _token_io(name, spec, "token_in", model_dim)
            if kind == "mlp":
                self.noisy_proj[name] = MLPInputLayer(int(spec["dim"]), model_dim, mlp_dim)
                self.time_concat.add(name)
            else:
                self.noisy_proj[name] = InputLayer(
                    int(spec["dim"]), model_dim, cond_dim=cond_dim
                )
        self.query_embed = nn.ParameterDict()
        for name, _ in query_streams:
            sl = self.stream_slices[name]
            emb = torch.empty(1, sl.stop - sl.start, model_dim)
            nn.init.trunc_normal_(emb, std=0.02)
            self.query_embed[name] = nn.Parameter(emb)

        self.heads = nn.ModuleDict()
        for name, spec in noisy_streams + query_streams + direct_streams:
            kind, mlp_dim = _token_io(name, spec, "token_out", model_dim)
            self.heads[name] = (
                MLPFinalLayer(model_dim, int(spec["dim"]), mlp_dim) if kind == "mlp"
                else FinalLayer(model_dim, int(spec["dim"]), cond_dim=cond_dim)
            )
        self.predict_names = [name for name, _ in
                              noisy_streams + query_streams + direct_streams]
        self.flow_names = [name for name, _ in noisy_streams]
        self.query_names = [name for name, _ in query_streams]
        self.direct_input = {name: spec["input"] for name, spec in direct_streams}

        self.source_order = [name for name, _ in cross_sources]
        source_dims = [int(spec["dim"]) for _, spec in cross_sources]
        block_kwargs = dict(source_dims=source_dims or None, mlp_ratio=mlp_ratio,
                            mlp_dim=mlp_dim, dropout=dropout, cond_dim=cond_dim,
                            dim_head=dim_head)
        if self.expert_names:
            self.blocks = nn.ModuleList([
                MoTBlock(model_dim, num_heads, len(self.expert_names), **block_kwargs)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlock(model_dim, num_heads, **block_kwargs) for _ in range(depth)
            ])

    def _cond_map(self, cond_times):
        """Row of a cond stream feeding each token, by matching the token's time; -1 = none."""
        idx = torch.full((self.num_tokens,), -1, dtype=torch.long)
        for row, time in enumerate(cond_times):
            idx[torch.isclose(self.pos_time.double(), time.double())] = row
        return idx

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()

    def required_streams(self, streams):
        """The `via: tokens` conditioning streams a subset integration over `streams`
        needs: the attends-closure of the requested predict streams, restricted to the
        clean token streams. Everything outside it can be omitted from `tokens` -- the
        separability guard proves the retained outputs are unchanged."""
        idx = {n: i for i, n in enumerate(self.stream_order)}
        want = [n for n in streams if n in idx] + [
            self.direct_input[n] for n in streams if n in self.direct_input]
        seen = set(want)
        frontier = list(want)
        while frontier:
            q = frontier.pop()
            for k in self.stream_order:
                if k not in seen and self._vis[idx[q], idx[k]]:
                    seen.add(k)
                    frontier.append(k)
        return [n for n in self.stream_order if n in seen and n in self._clean_names]

    def _assert_separable(self, active, missing):
        idx = {n: i for i, n in enumerate(self.stream_order)}
        bad = [(q, k) for q in active for k in missing if self._vis[idx[q], idx[k]]]
        if bad:
            raise ValueError(
                f"cannot integrate {active} without {sorted({k for _, k in bad})}: "
                f"attends edges {bad} make the retained outputs depend on the omitted "
                f"tokens. Include them, or cut the edges in the arm's `attends:` lists."
            )

    def _subset_layout(self, active):
        """(token index, per-stream slices in the subset layout, attention mask) for an
        active-stream subset, cached per subset."""
        device = self.all_stream_ids.device
        key = (tuple(active), device)
        hit = self._subset_cache.get(key)
        if hit is not None:
            return hit
        self._assert_separable(active, [n for n in self.stream_order if n not in active])
        pieces, slices, off = [], {}, 0
        for name in active:
            sl = self.stream_slices[name]
            n = sl.stop - sl.start
            pieces.append(torch.arange(sl.start, sl.stop))
            slices[name] = slice(off, off + n)
            off += n
        idx_cpu = torch.cat(pieces)
        idx = idx_cpu.to(device)
        # mask assembly on CPU (float64 boundaries), result moved as bool
        start, end = self._time_start[idx_cpu], self._time_end[idx_cpu]
        if not self._time_causal:
            start = torch.full_like(start, float("-inf"))
            end = torch.full_like(end, float("inf"))
        vis = None if bool(self._vis.all()) else self._vis
        ids = self.all_stream_ids[idx].cpu()
        mask = None
        if self._time_causal or vis is not None:
            if self.attn_backend == "sdpa":
                mask = block_causal_mask(start, end, stream_ids=ids, vis=vis).to(device)
            else:
                mask = block_causal_flex_mask(start.to(device), end.to(device),
                                              stream_ids=ids.to(device),
                                              vis=None if vis is None else vis.to(device))
        self._subset_cache[key] = (idx, slices, mask)
        return idx, slices, mask

    def _routes(self, idx=None):
        """One (n_e,) LongTensor of token positions per expert for the full layout (idx
        None) or a subset layout (`idx` = full-layout token index of each subset token),
        or () for the shared trunk. Cached per layout and device."""
        if not self.expert_names:
            return ()
        device = self.all_expert_ids.device
        key = (None if idx is None else tuple(idx.tolist()), device)
        hit = self._route_cache.get(key)
        if hit is None:
            ids = self.all_expert_ids if idx is None else self.all_expert_ids[idx]
            hit = tuple((ids == e).nonzero().squeeze(-1) for e in range(len(self.expert_names)))
            self._route_cache[key] = hit
        return hit

    def _mask(self):
        if self._unmasked:
            return None
        if self.attn_backend == "sdpa":
            return self.dense_mask
        if self._flex_mask is None or self._flex_mask.kv_num_blocks.device != self.block_start.device:
            self._flex_mask = block_causal_flex_mask(
                self.block_start, self.block_end,
                stream_ids=self.stream_ids, vis=self.vis_table,
            )
        return self._flex_mask

    def _build_cond(self, t, cond, batch, device, dtype):
        """The fixed cond slots concatenated -> (B, cond_dim) or (B, num_tokens, cond_dim),
        and the timestep slot on its own, which an MLPInputLayer concatenates instead of
        reading the whole cond (None when no stream is flow)."""
        slots, temb = [], None
        if self.has_flow:
            if t is None:
                t = torch.zeros(batch, device=device, dtype=dtype)
            temb = self.t_embedder(t)
            if self.per_token_cond:
                temb = temb.unsqueeze(1).expand(-1, self.num_tokens, -1)
                temb = temb * (~self.t_free).view(1, -1, 1)
            slots.append(temb)
        for name in self.cond_order:
            v = self.cond_proj[name](cond[name])
            m = getattr(self, f"cond_map_{name}")
            gathered = v[:, m.clamp(min=0)]
            slots.append(gathered * (m >= 0).view(1, -1, 1))
        return torch.cat(slots, dim=-1), temb

    @staticmethod
    def _slice_cond(cond, sl):
        return cond[:, sl] if cond.dim() == 3 else cond

    def _layer_source(self, value, layer):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            assert len(value) == self.depth, "per-layer source must provide one tensor per block"
            return value[layer]
        return value

    def forward(self, tokens, t=None, sources=None, source_masks=None, cond=None,
                use_kv_cache=False):
        """
        tokens:       dict with one (B, N_name, dim_name) tensor per token stream; clean streams
                      carry encoded observations, flow streams x_t
        t:            (B,) flow-matching timestep in [0, 1]; unused when no stream is flow
        sources:      dict name -> (B, S, dim) tensor or list of `depth` tensors
        source_masks: dict name -> (B, S) bool, True = ignore
        cond:         dict name -> (B, rows, dim) per-frame AdaLN conditioning
        Returns dict name -> velocity (flow streams) / next-value prediction (direct streams).
        """
        sources, source_masks, cond = sources or {}, source_masks or {}, cond or {}
        ref = next(iter(tokens.values()))
        c, temb = self._build_cond(t, cond, ref.shape[0], ref.device, ref.dtype)

        active = [n for n in self.stream_order if n in tokens]
        subset = len(active) < len(self.stream_order)
        # cond/temb stay full-size: per-stream slots index the full layout either way
        parts = []
        for name in active:
            sl = self.stream_slices[name]
            if name in self.query_embed:
                parts.append(self.query_embed[name].expand(ref.shape[0], -1, -1))
                continue
            assert tokens[name].shape[1] == sl.stop - sl.start, (
                f"stream '{name}' got {tokens[name].shape[1]} tokens, layout expects "
                f"{sl.stop - sl.start}: the batch window is missing rows (a clean stream "
                f"reading future rows at inference? integrate with streams= to omit it)"
            )
            if name in self.clean_proj:
                parts.append(self.clean_proj[name](tokens[name]))
            else:
                slot = temb if name in self.time_concat else c
                parts.append(self.noisy_proj[name](
                    tokens[name], self._slice_cond(slot, self.stream_slices[name])
                ))
        x = torch.cat(parts, dim=1)

        if subset:
            idx, out_slices, attn_mask = self._subset_layout(active)
            freqs = self.rope({a: getattr(self, f"pos_{a}")[idx] for a in self.pos_axes})
            routes = self._routes(idx)
        else:
            out_slices = self.stream_slices
            freqs = self.rope({a: getattr(self, f"pos_{a}") for a in self.pos_axes})
            attn_mask = self._mask()
            routes = self._routes()
        masks = [source_masks.get(name) for name in self.source_order] or None
        # MoTBlock takes the routes as a trailing positional; DiTBlock takes none
        extra = (routes,) if self.expert_names else ()

        for i, block in enumerate(self.blocks):
            layer_sources = [self._layer_source(sources.get(name), i) for name in self.source_order]
            if not any(s is not None for s in layer_sources):
                layer_sources = None

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, layer_sources, masks,
                    freqs, attn_mask, use_kv_cache, *extra, use_reentrant=False,
                )
            else:
                x = block(x, c, layer_sources, masks, freqs, attn_mask, use_kv_cache, *extra)

        out = {}
        for name in self.predict_names:
            src = self.direct_input.get(name, name)
            if src not in out_slices:
                continue
            sl = out_slices[src]
            full_sl = self.stream_slices[src]
            out[name] = self.heads[name](x[:, sl], self._slice_cond(c, full_sl))
        return out

    @torch.no_grad()
    def rollout(self, tokens, sources=None, source_masks=None, cond=None, num_steps=10,
                cfg_scale=1.0, cfg_drop=(), streams=None):
        """One inference pass: Euler-integrate the flow streams from noise, or -- when
        nothing is flow -- a single deterministic forward.
        Returns dict name -> prediction. `streams` restricts integration to those
        predict streams; `tokens` then carries only the clean streams they attend
        (`required_streams`), and the separability guard refuses subsets whose outputs
        would differ from the full pass.
        """
        assert tokens, "rollout needs at least one clean stream"
        if not self.flow_names:
            tokens = {**tokens, **{n: self.query_embed[n] for n in self.query_names
                                   if streams is None or n in streams}}
            return self(tokens, None, sources, source_masks, cond=cond)
        return self.ode_solve(tokens, sources, source_masks, cond=cond, num_steps=num_steps,
                              cfg_scale=cfg_scale, cfg_drop=cfg_drop, streams=streams)

    @torch.no_grad()
    def ode_solve(self, tokens, sources=None, source_masks=None, cond=None, num_steps=10,
                  cfg_scale=1.0, cfg_drop=(), streams=None):
        """Euler integration under no_grad: the inference path. See `integrate`."""
        return self.integrate(tokens, sources, source_masks, cond=cond, num_steps=num_steps,
                              cfg_scale=cfg_scale, cfg_drop=cfg_drop, streams=streams)

    def integrate(self, tokens, sources=None, source_masks=None, cond=None, num_steps=10,
                  cfg_scale=1.0, cfg_drop=(), use_kv_cache=True, streams=None):
        """Euler integration from noise; returns dict name -> prediction per predicted stream.

        tokens holds the clean streams only; noise for flow streams is drawn internally.
        cfg_scale > 1.0 amplifies the velocity against a second pass with the sources named in
        cfg_drop removed; the conditional pass caches cross-attention K/V across steps, the drop
        pass runs uncached. Direct streams (if any) are read off the final step.

        This is the differentiable form -- `ode_solve` is the same thing under no_grad.
        An `integration` loss calls it WITH grad to supervise the endpoint the sampler
        actually produces, at N times the cost of one forward: the backward then runs the
        full Euler chain, so peak activation memory grows with `num_steps`.
        `use_kv_cache=False` recomputes the
        cross-attention K/V every step rather than reusing step 0's under autograd.
        """
        assert tokens, "integrate needs at least one clean stream"
        ref = next(iter(tokens.values()))
        B, device, dtype = ref.shape[0], ref.device, ref.dtype
        sources = sources or {}

        flow_active = self.flow_names
        query_active = self.query_names
        if streams is not None:
            unknown = [n for n in streams if n not in self.predict_names]
            assert not unknown, f"streams={unknown} are not predict streams ({self.predict_names})"
            flow_active = [n for n in self.flow_names if n in streams]
            query_active = [n for n in self.query_names if n in streams]
        # value unused (forward reads the embedding); presence marks the stream active
        tokens = {**tokens, **{n: self.query_embed[n] for n in query_active}}
        x = {}
        for name in flow_active:
            n = self.stream_slices[name].stop - self.stream_slices[name].start
            x[name] = torch.randn(B, n, self.stream_dims[name], device=device, dtype=dtype)

        use_cfg = cfg_scale != 1.0 and any(sources.get(name) is not None for name in cfg_drop)
        uncond_sources = {k: v for k, v in sources.items() if k not in cfg_drop}
        self.clear_cache()

        dt = 1.0 / num_steps
        out = {}
        for step in range(num_steps):
            t = torch.full((B,), step * dt, device=device, dtype=dtype)
            out = self({**tokens, **x}, t, sources, source_masks, cond=cond,
                       use_kv_cache=use_kv_cache)
            v = out
            if use_cfg:
                u = self({**tokens, **x}, t, uncond_sources, source_masks, cond=cond,
                         use_kv_cache=False)
                v = {name: u[name] + cfg_scale * (v[name] - u[name]) for name in v}
            x = {name: x[name] + v[name] * dt for name in x}

        return {**{n: out[n] for n in out if n not in x
                   and (streams is None or n in streams)}, **x}


class GenericViTPredictor(PredictiveModel):
    """GenericViTTrunk plus the encoders that feed it.

    loss(batch) builds a per-call context (encoder latents, predictions, targets) via the base
    and sums the configured `losses:` terms. The pred/target rule is generic: a flow stream is
    seeded with noise and targets x1 - x0, a direct stream reads its input stream's tokens and targets its own encoded window.
    predict(obs) integrates the flow streams from the same start, or runs a single
    deterministic forward when nothing is flow. No path here is stop-gradded: a jointly
    trained encoder is shaped by the flow loss through both x0 and x1, and collapse is held
    off by `sigreg`.

    An `integration` loss term additionally runs that same solve WITH gradients during
    training (`_forward_integrate`), so the sampler's endpoint -- not just the velocity at a
    random t -- is supervised, at the cost of `num_steps` extra trunk passes per step.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_flow_steps = int(cfg.get("num_flow_steps", 1))
        self.cfg_scale = float(cfg.get("cfg_scale", 1.0))
        self.cfg_drop = tuple(cfg.get("cfg_drop", ()) or ())
        inf = cfg.get("inference_streams")
        self.inference_streams = [str(n) for n in inf] if inf else None
        if self.inference_streams:
            unknown = [n for n in self.inference_streams if n not in self.predict_spec]
            assert not unknown, (
                f"inference_streams {unknown} are not predict streams "
                f"({sorted(self.predict_spec)})"
            )

    def _build_predictor(self, cfg):
        return GenericViTTrunk(cfg.conditioning, cfg.predict, **cfg.model)

    def _forward_train(self, clean, sources, cond, targets):
        flow = {n: targets[n] for n in self.predictor.flow_names}
        t, x_t, target = None, {}, {}
        if flow:
            ref = next(iter(flow.values()))
            t = sample_flow_time(ref.shape[0], ref.device)
            tb = t.view(-1, 1, 1)
            x0 = {n: torch.randn_like(v) for n, v in flow.items()}
            x_t = {n: (1 - tb) * x0[n] + tb * flow[n] for n in flow}
            target.update({n: flow[n] - x0[n] for n in flow})
        target.update({n: targets[n] for n in self.predictor.direct_input})
        target.update({n: targets[n] for n in self.predictor.query_names})

        queries = {n: self.predictor.query_embed[n] for n in self.predictor.query_names}
        pred = self.predictor({**clean, **x_t, **queries}, t, sources, cond=cond)
        return pred, target

    def _forward_integrate(self, clean, sources, cond, num_steps):
        # deliberately unguided: the term supervises the unguided endpoint, and a guided
        # solve would double its already num_steps-fold cost
        out = self.predictor.integrate(
            clean, sources, cond=cond, num_steps=num_steps, use_kv_cache=False,
        )
        return {name: out[name] for name in self.predictor.flow_names}

    def predict(self, obs, streams=None):
        """Integrate the flow streams. `streams` (default: cfg `inference_streams`,
        else all) restricts integration to those predict streams; only the clean token
        streams they attend are encoded, so the batch needs no rows for the rest --
        e.g. a policy-only pass on a selfwam arm needs no future action chunk."""
        self.eval()
        streams = list(streams) if streams is not None else self.inference_streams
        with torch.no_grad():
            token_subset = None
            if streams is not None:
                token_subset = set(self.predictor.required_streams(streams))
            clean, sources, cond, _ = self._condition(obs, token_streams=token_subset)
            return self.predictor.rollout(
                clean, sources, cond=cond, num_steps=self.num_flow_steps,
                cfg_scale=self.cfg_scale, cfg_drop=self.cfg_drop, streams=streams,
            )

    def summary(self):
        return {
            **super().summary(),
            "num_tokens": self.predictor.num_tokens,
            "num_flow_steps": self.num_flow_steps,
            "experts": list(self.predictor.expert_names) or None,
        }
