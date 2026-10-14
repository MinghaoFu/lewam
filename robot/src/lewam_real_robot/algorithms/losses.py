"""Configurable training losses for the predictor algorithms.

A loss term is an nn.Module built from one `losses:` config entry. Each term reads a
per-call context dict (see `PredictiveModel._build_context`):

    ctx["latent"]      encoder output per encoder-backed stream
    ctx["pred"]        predictor output per predict stream
    ctx["target"]      regression target per predict stream
    ctx["integrated"]  ODE endpoint per flow stream, only when an `integration` term needs it

A term with `weight: 0.0` is observed only: evals report it as `loss/<label>` and training
skips it.

`PredictionLoss` (types `flow` and `prediction`) is the MSE of `pred` against `target`.
`sigreg` regularizes `latent` toward an isotropic Gaussian.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SIGReg", "PredictionLoss", "SIGRegLoss", "IntegrationLoss",
           "build_loss_term"]


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (LeWorldModel): pushes embeddings toward an
    isotropic Gaussian via the Epps-Pulley squared ECF distance over random 1D projections.
    Reference: https://github.com/lucas-maes/le-wm/blob/main/module.py

    forward(embeddings: (B, N, D)) -> scalar. Tokens are flattened to (B*N, D) and scored as
    one bag of samples. The statistic is scaled by the sample count B*N, so an iid isotropic
    Gaussian scores ~1.05 regardless of B, N or D; correlated tokens score higher.
    """

    def __init__(self, knots=17, num_proj=1024, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=dtype)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=dtype)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings):
        x = embeddings.reshape(-1, embeddings.size(-1)).to(self.dtype)
        A = torch.randn(x.size(-1), self.num_proj, device=x.device, dtype=self.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (x @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        return (err @ self.weights).mean() * x.size(0)


class PredictionLoss(nn.Module):
    """MSE between the predictor's output and its regression target, summed over the term's
    streams: velocity vs `x1 - x0` for flow streams, next latent vs next latent for direct ones.
    """

    def __init__(self, streams, weight, label="prediction"):
        super().__init__()
        self.streams = list(streams)
        self.weight = weight
        # a single-stream term logs under its stream name; a term spanning several
        # logs under the loss type's label
        self.label = self.streams[0] if len(self.streams) == 1 else label

    def forward(self, ctx):
        return sum(F.mse_loss(ctx["pred"][n], ctx["target"][n]) for n in self.streams)


class IntegrationLoss(nn.Module):
    """Inference-consistency over one flow stream: score the endpoint the SAMPLER produces,
    not the velocity at a random t.

    The flow term supervises v(x_t, t) at independently drawn t, which is only an unbiased
    surrogate in the infinite-step limit; a 4-step Euler rollout accumulates the per-step
    error, and nothing in training ever saw the states its own trajectory visits. This term
    closes that loop: `ctx["integrated"]` is the endpoint of a differentiable N-step solve
    started from noise as at eval, and the error is taken against the same x1 the flow term regresses toward.

    Nothing here is stop-gradded: the term's error propagates into the predictor through the
    whole solve, and into a jointly trained encoder through the target it aims at. `num_steps` configures the solve itself and is read by the
    algorithm, not here (see `PredictiveModel._integration`).
    """

    def __init__(self, stream, weight, norm="mse", units="normalized"):
        super().__init__()
        assert norm in ("l1", "mse"), f"unknown integration norm '{norm}'"
        assert units in ("normalized", "raw"), (
            f"unknown integration units '{units}'; use 'normalized' or 'raw'"
        )
        self.streams = [stream]
        self.stream = stream
        self.weight = weight
        self.norm = norm
        self.units = units
        self.label = f"int_{stream}" + ("_raw" if units == "raw" else "")

    def forward(self, ctx):
        pred, target = ctx["integrated"][self.stream], ctx["latent"][self.stream]
        if self.units == "raw":
            # pred and target are in normalized units; rescaling by the normalizer's per-dim span is what
            # reaches the robot's units
            scale = (ctx.get("unit_scale") or {}).get(self.stream)
            if scale is None:
                raise ValueError(
                    f"integration term on '{self.stream}' asks for `units: raw`, but the "
                    f"model carries no norm_stats for the field it reads -- there is no "
                    f"scale to convert by. Train with a `normalize:` dataset, or drop to "
                    f"`units: normalized`"
                )
            scale = scale.to(pred.dtype).to(pred.device)
            pred, target = pred * scale, target * scale
        return F.l1_loss(pred, target) if self.norm == "l1" else F.mse_loss(pred, target)


class SIGRegLoss(nn.Module):
    """SIGReg over the encoder latents of the term's streams.

    `statistic` picks how (B, T, S, D) latents are grouped into sample bags:

        pooled (default)  one bag of all B*T*S tokens.
        per_timestep      one bag per frame position, averaged over T.
        per_spatial       one bag per grid/query position, averaged over S.

    `concat` joins the term's streams before the statistic runs; without it each stream is
    scored separately and the scores are summed:

        null (default)  independent terms, summed.
        temporal        streams joined along T (e.g. context and future frames of one camera).
        spatial         streams joined along S (e.g. two cameras at the same timesteps).
        batch           streams joined along B.

    Joined streams must match on the other two axes.

    Each term has one bias-free linear head shared by its streams, trained only by this term
    and unused at prediction time. `proj_dim` sets its width (default: the stream dim, with an
    identity init).
    """

    CONCAT_DIMS = {"batch": 0, "temporal": 1, "spatial": 2}

    def __init__(self, streams, weight, knots=17, num_proj=1024, proj_dim=None,
                 stream_dims=None, statistic="pooled", stream_frames=None, concat=None,
                 stream_times=None):
        super().__init__()
        assert statistic in ("pooled", "per_timestep", "per_spatial"), (
            f"unknown statistic '{statistic}'"
        )
        assert concat is None or concat in self.CONCAT_DIMS, (
            f"unknown concat '{concat}'; use {sorted(self.CONCAT_DIMS)} or omit it"
        )
        self.streams = list(streams)
        self.weight = weight
        # statistic and concat qualify the label, so several terms over one set of
        # streams do not overwrite each other's logs
        self.label = "sigreg_" + "_".join(self.streams)
        if statistic != "pooled":
            self.label += f"_{statistic}"
        if concat is not None:
            self.label += f"_{concat}"
        self.sigreg = SIGReg(knots=knots, num_proj=num_proj)
        self.statistic = statistic
        self.concat = concat
        self.stream_frames = dict(stream_frames or {})
        self.stream_times = {
            n: t for n, t in (stream_times or {}).items() if n in self.streams
        }
        if statistic in ("per_timestep", "per_spatial") or concat is not None:
            missing = [n for n in self.streams if not self.stream_frames.get(n)]
            assert not missing, (
                f"{concat or statistic} sigreg needs the frame count of {missing} "
                f"(streams without an `index` have no time axis)"
            )
        dims = {n: (stream_dims or {}).get(n) for n in self.streams}
        if any(d is None for d in dims.values()):
            raise ValueError(
                f"sigreg needs the latent dim of every stream it reads to size its head, "
                f"got {dims}; a stream must declare `dim`"
            )
        if len(set(dims.values())) != 1:
            raise ValueError(
                f"a sigreg term shares one head across its streams, which requires equal "
                f"stream dims, got {dims}; split into separate terms"
            )
        in_dim = dims[self.streams[0]]
        out_dim = in_dim if proj_dim is None else int(proj_dim)
        self.head = nn.Linear(in_dim, out_dim, bias=False)
        if out_dim == in_dim:
            # a square head starts as the identity, so omitting proj_dim costs nothing at
            # init and the term measures the raw latents until the head moves
            nn.init.eye_(self.head.weight)

    def _score(self, grid):
        """(B, T, S, D) -> the statistic, averaged over however many bags it splits into."""
        if self.statistic == "pooled":
            bags = [grid.flatten(1, 2)]
        elif self.statistic == "per_timestep":
            bags = [grid[:, i] for i in range(grid.shape[1])]
        else:
            bags = [grid[:, :, j] for j in range(grid.shape[2])]
        return sum(self.sigreg(bag) for bag in bags) / len(bags)

    def _grid(self, ctx, name):
        z = ctx["latent"][name]
        if self.head is not None:
            z = self.head(z)
        # no concat with the pooled statistic never splits the token axis, so it does
        # not need a real frame count
        t = self.stream_frames.get(name) or 1
        return z.unflatten(1, (t, -1))

    def _join_temporal(self, grids):
        """Frame slices of every stream, each distinct time kept once.

        Windows overlap whenever a context and a future stream share frames (le-wm's
        shift-by-one target does, at 2 of 4 frames), and those frames are one moment of
        one timestream, not two samples of it. Without times to match on, nothing is
        dropped. The first stream listed wins a tie, which matters only for an encoder
        whose latent depends on the surrounding window.
        """
        if not self.stream_times:
            return torch.cat(grids, dim=1)
        seen, keep = set(), []
        for name, grid in zip(self.streams, grids):
            for i, t in enumerate(self.stream_times[name]):
                if t in seen:
                    continue
                seen.add(t)
                keep.append(grid[:, i : i + 1])
        return torch.cat(keep, dim=1)

    def forward(self, ctx):
        grids = [self._grid(ctx, n) for n in self.streams]
        if self.concat is None:
            return sum(self._score(g) for g in grids)

        dim = self.CONCAT_DIMS[self.concat]
        kept = [i for i in range(4) if i != dim]
        shapes = {n: tuple(g.shape) for n, g in zip(self.streams, grids)}
        if len({tuple(s[i] for i in kept) for s in shapes.values()}) != 1:
            axes = "(B, T, S, D)"
            raise ValueError(
                f"concat '{self.concat}' joins streams along {axes} axis {dim}, so their "
                f"other axes must match; got {shapes}"
            )
        if self.concat == "temporal":
            return self._score(self._join_temporal(grids))
        return self._score(torch.cat(grids, dim=dim))


def build_loss_term(term, default_streams, stream_dims=None, stream_frames=None,
                    stream_times=None):
    """One `losses:` entry -> loss module. `default_streams` is the default set a prediction
    term covers when it names no `stream`/`streams` (all predict streams marked
    `type: flow`/`prediction`). `stream_dims` maps stream name -> latent dim, which every
    sigreg term needs to size its head; `stream_frames` maps stream name -> frame count
    (its `index` length), needed when a sigreg term sets a non-pooled `statistic` or any
    `concat`; `stream_times` maps stream name -> per-frame times, letting `concat:
    temporal` tell which frames of two streams are the same moment.
    """
    kind = term["type"]
    weight = float(term.get("weight", 1.0))
    if kind in ("flow", "prediction"):
        streams = term.get("streams")
        if streams is None:
            streams = [term["stream"]] if term.get("stream") is not None else list(default_streams)
        return PredictionLoss(streams, weight, label=kind)
    if kind == "integration":
        return IntegrationLoss(term["stream"], weight, norm=term.get("norm", "mse"),
                               units=term.get("units", "normalized"))
    if kind == "sigreg":
        return SIGRegLoss(
            list(term["streams"]), weight,
            knots=int(term.get("knots", 17)), num_proj=int(term.get("num_proj", 1024)),
            proj_dim=term.get("proj_dim"), stream_dims=stream_dims,
            statistic=term.get("statistic", "pooled"), stream_frames=stream_frames,
            concat=term.get("concat"), stream_times=stream_times,
        )
    raise ValueError(f"unknown loss type '{kind}'")
