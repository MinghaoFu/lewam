"""Chunked action drivers for real-robot rollouts: dataset-keyed observation frames in,
one raw action vector per control tick out.

`ChunkDriver` runs the model in-process, `RemoteDriver` on scripts/serve.py; the eval
config's `server:` knob picks between them.

Must stay free of any lerobot import: the inference server imports ChunkDriver without
the hardware stack installed.
"""

import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from ...datasets.lerobot import resolve_slice
from ...utils.normalize import Normalizer
from ...utils.smoothing import smooth_actions
from ...utils.spec import action_entry, raw_index, spec_fields, split_index
from ...utils.wire import connect, encode_jpeg, recv_msg, send_msg


def _filter_kwargs(cfg):
    """eval `action_filter:` -> smooth_actions kwargs, or None when disabled."""
    opts = dict(cfg.get("action_filter") or {})
    return opts if opts.pop("enabled", False) else None


SAMPLING_KNOBS = {"num_flow_steps": int, "cfg_scale": float}


def _apply_sampling(model, cfg):
    """Set the eval config's sampling knobs on the model; `None` keeps the trained value.
    Returns the effective values.

    The trained values must be stashed on first use, not read back each time: the server
    reuses one cached model, so an omitted override would inherit the previous client's.
    """
    trained = getattr(model, "_trained_sampling", None)
    if trained is None:
        trained = {k: getattr(model, k) for k in SAMPLING_KNOBS if hasattr(model, k)}
        model._trained_sampling = trained
    effective = {}
    for knob, cast in SAMPLING_KNOBS.items():
        if knob not in trained:
            continue
        override = cfg.get(knob)
        value = trained[knob] if override is None else cast(override)
        if knob == "num_flow_steps" and value < 1:
            raise ValueError(f"eval num_flow_steps={value} must be at least 1")
        setattr(model, knob, value)
        effective[knob] = value
    return effective


def make_driver(model, cfg, obs_features):
    """ChunkDriver for a policy that predicts an action chunk."""
    if model.action_stream is None:
        raise ValueError(f"{type(model).__name__} predicts no action stream")
    return ChunkDriver(model, cfg, obs_features)


def _warm_history(hold, acts, action_len, frame):
    """Fill a still-empty executed-action history with this episode's hold action.

    `hold` is a raw action, or a callable deriving one from the episode's first frame.
    """
    if hold is None or not action_len or acts:
        return
    row = hold(frame) if callable(hold) else hold
    if row is not None:
        acts.extend([np.asarray(row, dtype=np.float32)] * action_len)


class BaseDriver:
    """Chunk bookkeeping shared by both implementations: keep a dense history of the
    observation columns the model reads, replan when the executed chunk runs out, and
    hand each replan only the frames at the model's conditioning offsets.

    The offsets may be sparse (e.g. `[-11, -5, 0]` out of a 12-frame history): the dense
    buffer is what the control loop fills, the offset window is what inference sees.

    No rig reports the actions already executed -- an observation frame does not carry the
    action that produced it -- so this driver is their only source of truth, remembering
    what it commanded and replaying that as a second window.
    """

    def _configure(self, columns, offsets, execute_len, action_offsets=()):
        self.columns = list(columns)
        self.offsets = list(offsets)
        self.obs_len = 1 - self.offsets[0]
        self.execute_len = int(execute_len)
        self.action_offsets = list(action_offsets)
        self.action_len = -self.action_offsets[0] if self.action_offsets else 0
        self.reset()

    def reset(self, hold_action=None):
        """`hold_action` is the raw action that leaves the robot where it stands, or a
        callable deriving one from the episode's first frame. It warms the executed-action
        history to match training, which clamps that window onto the episode's first row.
        """
        self._hist = deque(maxlen=self.obs_len)
        self._acts = deque(maxlen=self.action_len)
        self._buffer = deque()
        self._latency = None
        self._hold = hold_action


    def _record_latency(self, **spans) -> None:
        """Seconds spent on each part of the replan that just ran, held for the caller
        to collect."""
        self._latency = spans

    def take_latency(self) -> dict | None:
        """The last replan's timings, once, or None if none has happened since."""
        latency, self._latency = self._latency, None
        return latency

    def step(self, frame):
        """One dataset-keyed observation frame in, one raw action vector out."""
        self._hist.append({col: frame[col] for col in self.columns})
        _warm_history(self._hold, self._acts, self.action_len, frame)
        if not self._buffer:
            hist = list(self._hist)
            hist = [hist[0]] * (self.obs_len - len(hist)) + hist
            window = [hist[self.obs_len - 1 + o] for o in self.offsets]
            self._buffer.extend(self.predict_window(window, self._executed()))
        action = self._buffer.popleft()
        if self.action_len:
            # recorded after the pop, so a replan at tick t sees actions from ticks < t
            self._acts.append(np.asarray(action, dtype=np.float32))
        return action

    def _executed(self):
        """One already-commanded action per `action_offsets` entry, `None` where the
        episode is not yet old enough to have one.
        """
        acts = list(self._acts)
        return [acts[len(acts) + o] if len(acts) + o >= 0 else None
                for o in self.action_offsets]

    def predict_window(self, window, actions=()):
        """Frames at `self.offsets` plus the actions at `self.action_offsets` -> the next
        `execute_len` raw action vectors.
        """
        raise NotImplementedError

    def close(self):
        pass


class ChunkDriver(BaseDriver):
    """In-process inference: map columns onto the model's conditioning fields, normalize
    observations, run the model, and unnormalize actions with the stats restored from its
    checkpoint.

    Also the server half of the remote path: scripts/serve.py builds one of these from the
    observation schema the client sent and replays received windows through it.
    """

    def __init__(self, model, cfg, obs_features):
        self.model = model
        self.device = next(model.parameters()).device
        stats = getattr(model, "norm_stats", None)
        method = getattr(model, "norm_method", "mean_std")
        self.normalizer = Normalizer(stats, method=method).to(self.device) if stats else None
        size = cfg.get("image_size")
        self.image_size = tuple(int(v) for v in size) if size else None
        self.action_filter = _filter_kwargs(cfg)
        self.sampling = _apply_sampling(model, cfg)

        self.action_stream = model.action_stream
        self.action_field = model.action_field
        self._check_model(model)

        fields = {f for name, spec in model.conditioning.items() for f in spec_fields(name, spec)}
        if getattr(model, "_last_fields", None):
            raise NotImplementedError("goal ('last') conditioning is not supported on-robot yet")

        self.action_context, self.action_context_field = (
            action_entry(model.conditioning) or (None, None)
        )
        action_offsets = ()
        if self.action_context is not None:
            spec = model.conditioning[self.action_context]
            action_offsets = self._history_offsets(spec)
            # an A2A source reading the action field widens the window the
            # executed-action history must fill, so union in its declared offsets
            declared_action = (dict(getattr(model, "field_offsets", None) or {})
                               .get(self.action_context_field) or [])
            action_offsets = sorted(set(action_offsets) | {o for o in declared_action if o < 0})
            # a codec stream's `dim` is its latent width; the rows handed to the model are
            # raw actions, which the stream's encoder turns into that latent
            enc = spec.get("encoder")
            enc_spec = (getattr(model, "encoder_specs", None) or {}).get(enc) or {}
            self.action_context_dim = int(enc_spec.get("in_dim") or spec["dim"])
            predicted_dim = int(getattr(model, "action_dim", 0)
                                or model.predict_spec[self.action_stream]["dim"])
            if self.action_context_dim != predicted_dim:
                raise ValueError(
                    f"stream '{self.action_context}' reads {self.action_context_dim}-dim "
                    f"actions but '{self.action_stream}' predicts {predicted_dim}-dim ones; "
                    f"the executed chunk is what feeds the history, so they must match."
                )
            fields.discard(self.action_context_field)

        self.task = cfg.get("task") or None
        columns = dict(cfg.get("columns") or {})
        self.field_to_col = {field: columns.get(field, field) for field in sorted(fields)}
        missing = [c for c in self.field_to_col.values() if c not in obs_features]
        if missing:
            raise ValueError(
                f"model fields read columns {missing} which this robot does not produce; "
                f"available: {sorted(obs_features)}. Remap with eval `columns:`."
            )
        self._visual_fields = {
            f for f, c in self.field_to_col.items()
            if obs_features[c]["dtype"] in ("video", "image")
        }
        self.visual = {self.field_to_col[f] for f in self._visual_fields}
        self.slices = {
            field: np.asarray(
                resolve_slice(patterns, obs_features[self.field_to_col[field]]["names"], field),
                dtype=np.int64,
            )
            for field, patterns in dict(cfg.get("slices") or {}).items()
        }

        dense = list(range(1 - int(getattr(model, "obs_len", 1)), 1))
        declared = dict(getattr(model, "field_offsets", None) or {})
        self.field_offsets = {f: list(declared.get(f) or dense) for f in self.field_to_col}
        union = sorted({o for offs in self.field_offsets.values() for o in offs})
        override = cfg.get("execute_len")
        execute_len = (
            getattr(model, "execute_len", model.chunk_len) if override is None else int(override)
        )
        if not 1 <= execute_len <= model.chunk_len:
            raise ValueError(
                f"eval execute_len={execute_len} is outside the model's chunk of "
                f"{model.chunk_len} steps."
            )
        self._configure(self.field_to_col.values(), union, execute_len, action_offsets)

    def _executed_window(self, actions):
        """Already-commanded actions -> the (1, T, dim) conditioning block, normalized.

        Slots older than the episode are front-clamped onto the oldest action there is,
        matching the dataset reader. Clamping must happen after normalization, so that
        with nothing executed yet the rows stay at normalized zero, the dataset's centre.
        """
        rows = torch.zeros(
            1, len(self.action_offsets), self.action_context_dim, device=self.device
        )
        present = [i for i, action in enumerate(actions) if action is not None]
        if present:
            raw = torch.as_tensor(
                np.stack([np.asarray(actions[i], dtype=np.float32) for i in present]),
                device=self.device,
            ).unsqueeze(0)
            if self.normalizer is not None:
                raw = self.normalizer.normalize(self.action_context_field, raw)
            rows[0, present] = raw[0]
            # `_executed` drops the OLDEST offsets first, so the gap is a leading run
            rows[0, :present[0]] = raw[0, 0]
        return rows

    def predict_window(self, window, actions=()):
        return self.predict_windows([window], [actions])[0]

    def _check_model(self, model):
        if self.action_stream is None:
            raise ValueError(
                "chunked eval executes a predicted action chunk, but no stream of this "
                "model's `predict:` carries actions. Mark one with `role: action`."
            )

    def _history_offsets(self, spec):
        """The executed-action offsets the action-conditioning stream reads."""
        action_offsets, has_last = split_index(raw_index(spec))
        if has_last or (action_offsets and action_offsets[-1] >= 0):
            raise ValueError(
                f"stream '{self.action_context}' conditions on actions at "
                f"'{spec['index']}', but a rollout only knows actions it has already "
                f"executed: every offset must be negative."
            )
        return action_offsets

    def _prepare(self, windows, actions_per_env=None):
        """`B` conditioning windows (+ executed actions) -> the model's `obs` dict: visual
        fields as (B, T, C, H, W) uint8 at `image_size`, the rest normalized floats."""
        batch = len(windows)
        actions_per_env = list(actions_per_env or [()] * batch)
        rows = [dict(zip(self.offsets, window, strict=True)) for window in windows]
        obs = {}
        for field, col in self.field_to_col.items():
            sel = self.slices.get(field)
            stack = [
                np.stack([row[o][col] if sel is None else row[o][col][sel]
                          for o in self.field_offsets[field]])
                for row in rows
            ]
            raw = torch.as_tensor(np.stack(stack))   # (B, T, ...), on the CPU
            if field in self._visual_fields:
                raw = raw.permute(0, 1, 4, 2, 3)                # (B, T, C, H, W) uint8
                if self.image_size and raw.shape[-2:] != self.image_size:
                    # MPS rejects an area pool whose input is not divisible by its output, so
                    # that backend resizes on the CPU like the dataset does
                    if self.device.type != "mps":
                        raw = raw.to(self.device)
                    steps = raw.shape[1]
                    flat = F.interpolate(raw.flatten(0, 1).float(), size=self.image_size,
                                         mode="area")
                    raw = (flat.round().clamp(0, 255).to(torch.uint8)
                           .view(batch, steps, *flat.shape[1:]))
                raw = raw.to(self.device)
            else:
                raw = raw.to(self.device).float()
                if self.normalizer is not None:
                    raw = self.normalizer.normalize(field, raw)
            obs[field] = raw
        if self.action_context is not None:
            obs[self.action_context_field] = torch.cat(
                [self._executed_window(actions) for actions in actions_per_env], dim=0
            )
        if self.task is not None:
            obs["task"] = [self.task] * batch
        return obs

    def predict_windows(self, windows, actions_per_env=None):
        """`B` conditioning windows -> `B` chunks, in ONE forward pass."""
        started = time.perf_counter()
        obs = self._prepare(windows, actions_per_env)
        chunk = self.model.predict(obs)
        if isinstance(chunk, dict):
            chunk = chunk[self.action_stream]
        if self.action_filter is not None:
            # must run on the whole chunk, before the cut to execute_len: the filter's
            # window reaches past the executed prefix, so cutting first changes its edges
            chunk = smooth_actions(chunk, **self.action_filter)
        if self.normalizer is not None:
            chunk = self.normalizer.unnormalize(self.action_field, chunk)
        out = chunk[:, : self.execute_len].cpu().numpy()
        self._record_latency(compute=time.perf_counter() - started)
        return list(out)


class RemoteDriver(BaseDriver):
    """Inference on scripts/serve.py over one blocking TCP connection.

    The server owns the model and every field/column decision: this side names the run to
    serve, describes the columns its robot produces, and is told back which of them to
    buffer and at what offsets.

    Blocking by design: the arm holds its last commanded pose for the round trip rather
    than executing actions predicted from a staler observation.
    """

    @staticmethod
    def _model_overrides(cfg):
        """Config overrides forwarded to the server's model build: the eval config's
        explicit `model_overrides` list plus any `algorithm.*` override on this
        process's own hydra command line -- so `+algorithm.inference_streams=[action]`
        typed on the CLIENT reaches the server, keeping the server config-free."""
        out = [str(o) for o in (cfg.get("model_overrides") or [])]
        try:
            from hydra.core.hydra_config import HydraConfig
            if HydraConfig.initialized():
                out += [str(o) for o in HydraConfig.get().overrides.task
                        if str(o).lstrip("+").startswith("algorithm.")]
        except Exception:
            pass
        return out

    def __init__(self, cfg, obs_features, run):
        self.address = str(cfg.get("server"))
        self.quality = int(cfg.get("jpeg_quality", 95))
        if not run:
            raise ValueError(
                "remote eval needs the wandb run to serve: pass load=<run_id> "
                "(or set eval.run=<entity>/<project>/<run_id> directly)."
            )
        self.sock = connect(self.address, timeout=float(cfg.get("server_timeout", 120)))
        ack = self._request({
            "type": "init",
            "run": run,
            "obs_features": obs_features,
            "image_size": list(cfg.image_size) if cfg.get("image_size") else None,
            "columns": dict(cfg.get("columns") or {}),
            "slices": {k: list(v) if not isinstance(v, str) else v
                       for k, v in dict(cfg.get("slices") or {}).items()},
            "action_filter": dict(cfg.get("action_filter") or {}),
            "execute_len": cfg.get("execute_len"),
            "task": cfg.get("task"),
            "model_overrides": self._model_overrides(cfg),
            **{knob: cfg.get(knob) for knob in SAMPLING_KNOBS},
        })
        self.checkpoint_step = ack.get("step")
        self.sampling = ack.get("sampling") or {}
        self.visual = {
            c for c in ack["columns"] if obs_features[c]["dtype"] in ("video", "image")
        }
        self._configure(ack["columns"], ack["offsets"], ack["execute_len"],
                        ack.get("action_offsets") or ())

    def _request(self, msg):
        send_msg(self.sock, msg)
        reply = recv_msg(self.sock)
        if "error" in reply:
            raise RuntimeError(f"inference server {self.address}: {reply['error']}")
        return reply

    def predict_window(self, window, actions=()):
        started = time.perf_counter()
        frames = [
            {col: encode_jpeg(v, self.quality) if col in self.visual else v
             for col, v in frame.items()}
            for frame in window
        ]
        encoded = time.perf_counter()
        reply = self._request(
            {"type": "predict", "frames": frames, "actions": list(actions)}
        )
        # both durations are measured on their own machine's clock, so subtracting
        # them needs no clock agreement between the two
        round_trip = time.perf_counter() - encoded
        compute = float(reply.get("compute_s") or 0.0)
        self._record_latency(
            encode=encoded - started,
            compute=compute,
            network=max(0.0, round_trip - compute),
        )
        return reply["actions"]

    def close(self):
        self.sock.close()
