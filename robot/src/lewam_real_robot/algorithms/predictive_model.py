"""Shared base for the predictor algorithms.

Builds the encoders, reads the modality spec (`conditioning`/`predict`), slices each entry's
window out of its raw batch field, routes conditioning streams by `via`, and runs the
`losses:` terms. A subclass supplies the predictor, `_forward_train` and the eval rollouts.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.spec import (
    LAST, action_entry, parse_index, raw_index, spec_fields, split_index,
)
from .base import BaseAlgorithm
from .encoders import build_encoder, resolve_chains
from .losses import build_loss_term


def stitch_views(views, mode="horizontal"):
    """Several (B, T, C, H, W) camera windows -> one wider (or taller) window.

    Views are resized to the first one's frame size before joining. Flat (B, T, D) streams
    join on the feature axis instead: two 7-dim state columns become one 14-dim vector per
    step, and only `horizontal` means anything there.
    """
    if mode not in ("horizontal", "vertical"):
        raise ValueError(f"unknown stitch '{mode}'; use 'horizontal' or 'vertical'")
    if views[0].ndim == 3:
        steps = {tuple(view.shape[:2]) for view in views}
        if len(steps) > 1:
            raise ValueError(
                f"flat streams stitch on the feature axis, so they must share a window: "
                f"got {sorted(steps)}"
            )
        if mode != "horizontal":
            raise ValueError(
                f"stitch '{mode}' needs a spatial axis; a flat stream only has features, "
                f"so it joins horizontally or not at all"
            )
        return torch.cat(views, dim=-1)
    dim = -1 if mode == "horizontal" else -2
    size = views[0].shape[-2:]
    out = []
    for view in views:
        if view.shape[-2:] != size:
            resized = F.interpolate(
                view.flatten(0, 1).float(), size=size, mode="bilinear", align_corners=False
            )
            if view.dtype == torch.uint8:
                resized = resized.round().clamp(0, 255)
            view = resized.to(view.dtype).unflatten(0, view.shape[:2])
        out.append(view)
    return torch.cat(out, dim=dim)


class PredictiveModel(BaseAlgorithm):
    def __init__(self, cfg):
        super().__init__()
        self.conditioning = cfg.conditioning
        self.predict_spec = cfg.predict

        self.encoder_specs = {
            name: dict(spec) for name, spec in (cfg.get("encoders") or {}).items()
        }
        self.encoders = nn.ModuleDict({
            name: build_encoder(spec) for name, spec in (cfg.get("encoders") or {}).items()
        })
        self._encoder_chains = resolve_chains(cfg.get("encoders") or {})

        # entries sharing a batch field are served one tensor covering the UNION of their
        # windows, past first; each slices its own window back out before encoding
        entries = {**cfg.conditioning, **cfg.predict}
        field_union, field_has_last = {}, {}
        for name, spec in entries.items():
            enc = spec.get("encoder")
            assert enc is None or enc in self.encoders, f"'{name}' names unknown encoder '{enc}'"
            if "index" in spec:
                # raw rows, not index steps: a tubelet step spans several rows
                rel, has_last = split_index(raw_index(spec))
                for field in spec_fields(name, spec):
                    field_union.setdefault(field, set()).update(rel)
                    field_has_last[field] = field_has_last.get(field, False) or has_last
        # "last" is appended after the relative offsets, so it is always the final row
        field_union = {
            field: sorted(steps) + ([LAST] if field_has_last.get(field) else [])
            for field, steps in field_union.items()
        }
        self._last_fields = {f for f, hl in field_has_last.items() if hl}

        # An index step spans multiple rows in TIME (`raw_index`, e.g. 2 per VJEPA2 tubelet)
        # and in SPACE (grid area), and the two compose. "last" rows are addressed from the
        # END so the same selectors work at rollout, where the field lacks the future rows.
        self._selectors = {}
        self._raw_per_step = {}
        for name, spec in entries.items():
            if "index" not in spec:
                self._selectors[name] = None
                continue
            enc = spec.get("encoder")
            if enc is not None:
                raw_per_step = 1
                for stage in self._encoder_chains[enc]:
                    raw_per_step *= self.encoders[stage].raw_steps_per_index
                span = 1
            else:
                raw_per_step = 1
                grid = spec.get("grid", (1, 1))
                span = grid[0] * grid[1]
            self._raw_per_step[name] = raw_per_step
            steps = len(parse_index(spec["index"]))
            raw = parse_index(raw_index(spec))
            if len(raw) != steps * raw_per_step:
                needed = steps * raw_per_step
                raise ValueError(
                    f"stream '{name}' consumes {raw_per_step} raw rows per index step, so "
                    f"its {steps} index steps need a `raw_index` naming the {needed} rows "
                    f"they are built from; got {len(raw)}"
                )
            fields = spec_fields(name, spec)
            unions = {f: field_union[f] for f in fields}
            if len({tuple(u) for u in unions.values()}) > 1:
                raise ValueError(
                    f"stream '{name}' stitches {fields}, but they carry different row "
                    f"layouts ({unions}); every stitched view must be read with one window"
                )
            union = field_union[fields[0]]
            n_last = raw.count(LAST)
            seen_last = 0
            entry_rows = []
            for i in raw:
                if i == LAST:
                    seen_last += 1
                    entry_rows.append(seen_last - n_last - 1)
                else:
                    entry_rows.append(union.index(i))
            rows = [e * span + j for e in entry_rows for j in range(span)]
            contiguous = rows == list(range(rows[0], rows[0] + len(rows)))
            stop = rows[0] + len(rows)
            self._selectors[name] = slice(rows[0], stop or None) if contiguous else rows

        # eval rollout geometry: obs_len is the SPAN back to the earliest conditioning
        # offset, and a window may be sparse, so drivers sample `field_offsets` out of a
        # dense obs_len history.
        cond_union = {}
        for name, spec in dict(cfg.conditioning).items():
            if "index" in spec:
                rel, _ = split_index(raw_index(spec))          # "last" is not history
                for field in spec_fields(name, spec):
                    cond_union.setdefault(field, set()).update(rel)
        self.field_offsets = {f: sorted(steps) for f, steps in cond_union.items() if steps}
        self.obs_len = max(
            (1 - min(steps) for steps in self.field_offsets.values()), default=1
        )
        # both stay None for a model that predicts no actions, e.g. a world model
        action = action_entry(cfg.predict)
        self.action_stream, self.action_field = action or (None, None)
        if self.action_stream is not None:
            spec = cfg.predict[self.action_stream]
            self.chunk_len = len(parse_index(spec["index"]))
            self.action_dim = int(spec["dim"])
            self.execute_len = int(cfg.get("execute_len", self.chunk_len))
            assert 1 <= self.execute_len <= self.chunk_len

        default_streams = [
            name for name, spec in cfg.predict.items()
            if spec.get("type", "flow") in ("flow", "prediction")
        ]
        loss_cfg = cfg.get("losses") or [
            {"type": "flow", "stream": name, "weight": 1.0} for name in default_streams
        ]
        stream_dims = {
            name: int(spec["dim"])
            for name, spec in {**dict(cfg.conditioning), **dict(cfg.predict)}.items()
            if "dim" in spec
        }
        stream_frames = {
            name: len(parse_index(spec["index"]))
            for name, spec in {**dict(cfg.conditioning), **dict(cfg.predict)}.items()
            if "index" in spec
        }
        # seconds, so streams at different rates land on identical floats for one moment
        stream_times = {
            name: tuple(
                LAST if isinstance(i, str) else i / float(spec.get("fps", 1.0))
                for i in parse_index(spec["index"])
            )
            for name, spec in {**dict(cfg.conditioning), **dict(cfg.predict)}.items()
            if "index" in spec
        }
        self.loss_terms = nn.ModuleList(
            [build_loss_term(term, default_streams, stream_dims, stream_frames, stream_times)
             for term in loss_cfg]
        )
        # `weight: 0.0` declares a MONITORING term: evals report it, training never
        # pays for it -- loss() skips these in train mode and their artifacts (an
        # observed `integration` costs num_steps extra trunk passes) are not built
        scored = [float(term.get("weight", 1.0)) != 0.0 for term in loss_cfg]
        if loss_cfg and not any(scored):
            raise ValueError(
                "every `losses:` term has weight 0.0, so training has no objective. At "
                "least one term must be scored; weight 0.0 means eval-only."
            )

        # every integration term reads one shared solve, so num_steps is one setting
        self._integration = None
        int_terms = [term for term in loss_cfg if term.get("type") == "integration"]
        if int_terms:
            unknown = {t["stream"] for t in int_terms} - set(cfg.predict)
            if unknown:
                raise ValueError(f"integration names {sorted(unknown)}, not predict streams")
            not_flow = [t["stream"] for t in int_terms
                        if cfg.predict[t["stream"]].get("type", "flow") != "flow"]
            if not_flow:
                raise ValueError(
                    f"integration names {sorted(not_flow)}, which are not flow streams: "
                    f"there is no ODE to integrate"
                )
            solve = {int(t.get("num_steps") or cfg.get("num_flow_steps", 1))
                     for t in int_terms}
            if len(solve) > 1:
                raise ValueError(
                    f"integration terms disagree on the solve ({sorted(solve)} steps); every "
                    f"term reads the same endpoint, so num_steps must match"
                )
            num_steps = solve.pop()
            assert num_steps >= 1, "integration num_steps must be >= 1"
            self._integration = {
                "num_steps": num_steps,
                "streams": [t["stream"] for t in int_terms],
                # false when every integration term is observed-only: the solve then runs
                # in evals alone, and a training step costs exactly what it did before
                "scored": any(float(t.get("weight", 1.0)) != 0.0 for t in int_terms),
            }

        self._stream_axes = {}
        for name, spec in entries.items():
            if "index" not in spec:
                continue
            grid = tuple(spec.get("grid", (1, 1)))
            self._stream_axes[name] = (
                len(parse_index(spec["index"])), int(grid[0]) * int(grid[1])
            )

        self.predictor = self._build_predictor(cfg)

    def _build_predictor(self, cfg):
        """Return the predictor nn.Module (built from cfg.model + the spec)."""
        raise NotImplementedError

    def _forward_train(self, clean, sources, cond, targets):
        """Turn the encoded context into a (pred, target) pair, both dicts keyed by predict
        stream, for the `losses:` terms. clean/sources/cond are the encoded conditioning
        streams routed by `via` (tokens / cross / cond); targets are the encoded predict
        streams.
        """
        raise NotImplementedError

    @property
    def backbone(self):
        """The predictor the loss layer runs against."""
        return self.predictor

    def run_encoder(self, name, x):
        """Run x through encoder `name`, first through the upstream chain its `input`
        links declare. A frozen stage runs under no_grad unless a trainable stage
        upstream needs gradients through it.
        """
        return self._run_stages(self._encoder_chains[name], x)

    def _run_stages(self, stages, x):
        for stage in stages:
            encoder = self.encoders[stage]
            if encoder.frozen and not (torch.is_tensor(x) and x.requires_grad):
                with torch.no_grad():
                    x = encoder(x)
            else:
                x = encoder(x)
        return x

    def cacheable_streams(self):
        """The spec entries whose frozen-root encoder output can be precomputed, as
        `{stream: {field, offsets, encoder}}`. A stream qualifies when its encoder chain
        starts frozen and it reads one un-stitched visual field with no `"last"` row.
        """
        entries = {**dict(self.conditioning), **dict(self.predict_spec)}
        out = {}
        for name, spec in entries.items():
            if "index" not in spec:
                continue
            fields = spec_fields(name, spec)
            enc = spec.get("encoder")
            rel, has_last = split_index(raw_index(spec))
            root = self.encoders[self._encoder_chains[enc][0]] if enc is not None else None
            # frozen alone is not enough: a probe run freezes everything, and the cache
            # holds visual windows only (supports_encoding_cache is per encoder type)
            if (root is not None and len(fields) == 1 and not has_last
                    and root.frozen and root.supports_encoding_cache):
                out[name] = {
                    "field": fields[0],
                    "offsets": parse_index(raw_index(spec)),
                    "encoder": self._encoder_chains[enc][0],
                }
        return out

    def raw_pixel_fields(self):
        """Fields some entry reads WITHOUT a cacheable frozen-root chain -- these must
        keep arriving as pixels even when other streams on the field are served from
        the encoding cache (the trainer builds the dataset in keys+pixels mode for the
        intersection).
        """
        cacheable = set(self.cacheable_streams())
        needed = set()
        entries = {**dict(self.conditioning), **dict(self.predict_spec)}
        for name, spec in entries.items():
            if "index" in spec and name not in cacheable:
                needed.update(spec_fields(name, spec))
        return needed

    def attach_encoding_cache(self, cache):
        """Serve the cached streams from `cache` whenever a batch carries their
        `enc_cache/<field>` keys. Batches with pixel fields (validation, rollouts) keep
        the live path -- the dispatch is the batch's schema, decided per batch.
        """
        missing = set(self.cacheable_streams()) - set(cache.stores)
        if missing:
            raise ValueError(f"encoding cache lacks streams {sorted(missing)}")
        self._enc_cache = cache

    def _raw(self, name, spec, batch):
        """A stream's own raw window, sliced out of the shared field and stitched, but not
        yet encoded -- what its encoder consumes, and what a pixel probe compares against.
        """
        sel = self._selectors[name]
        views = [batch[field] for field in spec_fields(name, spec)]
        if sel is not None:
            views = [view[:, sel] for view in views]
        return views[0] if len(views) == 1 else stitch_views(views, spec.get("stitch", "horizontal"))

    def _encode(self, name, spec, batch):
        fields = spec_fields(name, spec)
        key_field = f"enc_cache/{fields[0]}"
        cache = getattr(self, "_enc_cache", None)
        if key_field in batch:
            # dispatch is per stream: a field can carry keys AND pixels, when another
            # entry reads it raw
            if cache is not None and name in cache.stores:
                device = next(self.parameters()).device
                x = cache.lookup(name, batch[key_field], device)
                return self._run_stages(self._encoder_chains[spec["encoder"]][1:], x)
            if fields[0] not in batch:
                raise ValueError(
                    f"batch carries '{key_field}' but stream '{name}' has neither a "
                    f"cache entry nor the field's pixels: attach a cache serving it "
                    f"(attach_encoding_cache) or feed pixel batches"
                )
        raw = self._raw(name, spec, batch)
        enc = spec.get("encoder")
        return self.run_encoder(enc, raw) if enc else raw

    def _condition(self, batch, token_streams=None):
        """Encode every conditioning stream once, routing it by `via`.

        `via: null` encodes the stream but feeds it to nothing in the predictor; it is
        read from ctx["latent"] by loss terms and by stream-mode `source:` blocks.
        `token_streams` (None = all) restricts which `via: tokens` streams are encoded
        AT ALL -- a subset integration's omitted clean streams then need no batch rows.
        cross/cond/null streams are always encoded.
        """
        clean, sources, cond, hidden = {}, {}, {}, {}
        for name, spec in self.conditioning.items():
            via = spec.get("via", "tokens")
            assert via in ("tokens", "cross", "cond", None), f"unknown via '{via}' for '{name}'"
            if via == "tokens" and token_streams is not None and name not in token_streams:
                continue
            enc = self._encode(name, spec, batch)
            {"cross": sources, "cond": cond, None: hidden}.get(via, clean)[name] = enc
        return clean, sources, cond, hidden

    def _forward_integrate(self, clean, sources, cond, num_steps):
        """The differentiable ODE endpoint per flow stream, under the same conditioning as
        `_forward_train`.
        """
        raise NotImplementedError

    def _build_context(self, batch):
        """Assemble the artifacts the loss terms read: per-stream encoder `latent`, predictor
        `pred`, the regression `target`, and the `integrated` ODE endpoint when an integration
        term asks for it.
        Conditioning latents feed the predictor; predicted latents are the targets.
        """
        clean, sources, cond, hidden = self._condition(batch)
        latent = {**clean, **sources, **cond, **hidden}
        targets = {name: self._encode(name, spec, batch) for name, spec in self.predict_spec.items()}
        latent.update(targets)
        pred, target = self._forward_train(clean, sources, cond, targets)
        ctx = {"latent": latent, "pred": pred, "target": target}
        # observed (`weight: 0.0`) terms run in eval only, so in train mode build only what
        # the scored terms read -- see the `scored` note where the terms are built
        if self._integration is not None and (
            self._integration["scored"] or not self.training
        ):
            ctx["integrated"] = self._forward_integrate(
                clean, sources, cond, self._integration["num_steps"],
            )
            # only `units: raw` terms read this; building it is a couple of tiny tensors
            scales = {n: self._unit_scale(n) for n in self._integration["streams"]}
            ctx["unit_scale"] = {n: v for n, v in scales.items() if v is not None}
        return ctx

    def _unit_scale(self, name):
        """Per-dim factor turning a NORMALIZED error on this stream into the field's own
        units, or None when the stream has no such field (a codec latent) or the model
        carries no stats.

        Normalization is `(x - q_lo) / span * 2 - 1` (percentile) or `(x - mean) / std`, so
        an error scales by `span / 2` or `std` respectively -- the inverse of the slope, and
        the reason a wider dataset silently lowers every normalized metric.
        """
        stats = getattr(self, "norm_stats", None)
        if not stats:
            return None
        spec = self.predict_spec[name]
        field = spec.get("from", name)
        entry = stats.get(field)
        if not entry:
            return None
        method = getattr(self, "norm_method", "mean_std")
        if method == "percentile":
            lo = torch.as_tensor(entry["q_lo"], dtype=torch.float32)
            hi = torch.as_tensor(entry["q_hi"], dtype=torch.float32)
            return (hi - lo).clamp_min(1e-8) / 2
        return torch.as_tensor(entry["std"], dtype=torch.float32).clamp_min(1e-8)

    def _variance_logs(self, latent):
        """Share of variance carried along the time and grid axes, per stream FAMILY.

        Streams reading the same source field with matching grid span and latent width --
        a camera's context and its predicted future -- are concatenated along time and
        measured as ONE trajectory, so the estimate covers the full context+future token
        set rather than each half individually (grouping is automatic, from the specs'
        `from:` fields; no configuration). A family labels itself by its members' common
        prefix (scene_context + scene_future -> `var/scene/...`); a stream with no
        partner keeps its own name, exactly as before.

        Each value is the variance along that axis over the family's total: 1.0 is what
        independent tokens give, 0 means constant along the axis, i.e. collapsed.
        """
        streams = {**dict(self.conditioning), **dict(self.predict_spec)}
        groups = {}
        for name, (steps, span) in self._stream_axes.items():
            z = latent.get(name)
            if z is None or z.ndim != 3 or z.shape[1] != steps * span:
                continue
            z = z.detach().float().unflatten(1, (steps, span))
            spec = streams.get(name)
            field = spec.get("from", name) if spec is not None else name
            groups.setdefault((field, span, z.shape[-1]), []).append((name, z))
        logs = {}
        for (field, span, _), members in groups.items():
            names = [n for n, _ in members]
            z = torch.cat([m for _, m in members], dim=1)
            label = (names[0] if len(names) == 1
                     else os.path.commonprefix(names).rstrip("_") or field)
            total = z.reshape(-1, z.shape[-1]).var(dim=0).mean().clamp_min(1e-12)
            if z.shape[1] > 1:
                logs[f"var/{label}/temporal"] = z.var(dim=1).mean() / total
            if span > 1:
                logs[f"var/{label}/spatial"] = z.var(dim=2).mean() / total
        return logs

    def loss(self, batch):
        ctx = self._build_context(batch)
        logs, total = {}, 0.0
        for term in self.loss_terms:
            # a `weight: 0.0` term is observed, not optimized: computing it during training
            # would cost real time and add a `loss/<label>` series that no gradient backs
            if self.training and term.weight == 0.0:
                continue
            val = term(ctx)
            total = total + term.weight * val
            logs[f"loss/{term.label}"] = val.detach()
        return {"loss": total, **logs, **self._variance_logs(ctx["latent"])}

    def summary(self):
        return {**super().summary(), "obs_len": self.obs_len}
