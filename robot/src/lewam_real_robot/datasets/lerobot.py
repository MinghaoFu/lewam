import fnmatch
import os

import numpy as np
import torch

from ..utils.spec import raw_index, spec_fields, split_index
from .base import BaseSource


def split_episodes(episodes, fraction, split, seed=0):
    """Partition episode indices into a training set and a held-out validation set.

    `split` is "train" or "val"; `fraction` is the share of episodes held out. The draw is
    a seeded shuffle rather than a tail slice, because episodes are recorded in session
    order: the last N of a session share lighting, wear and object placements that the
    earlier ones never see, which flatters or wrecks a held-out score for reasons that have
    nothing to do with the policy. The same seed on both sides is what keeps the two halves
    disjoint.

    A non-zero fraction always holds out at least one episode, and never the last one left.
    """
    episodes = list(episodes)
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    fraction = float(fraction or 0.0)
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"validation_split must be in [0, 1), got {fraction}")
    if not episodes:
        return []
    held = int(round(len(episodes) * fraction))
    if fraction > 0.0:
        held = max(1, held)
    held = min(held, len(episodes) - 1)
    if split == "val" and held == 0:
        raise ValueError(
            f"validation_split={fraction} holds out no episodes of {len(episodes)}: "
            f"raise it, or add episodes, to run a held-out eval"
        )
    order = np.random.default_rng(seed).permutation(len(episodes))
    picked = order[:held] if split == "val" else order[held:]
    return sorted(int(episodes[i]) for i in picked)


def episodes_for_tasks(meta, tasks):
    """Instruction strings -> the indices of the episodes demonstrating them.

    A task is named by its instruction text rather than by an index, because the index a
    dataset assigns a task is an artifact of how the dataset was packed and does not line
    up with the index the eval environment uses for the same task. The text is the one
    identifier both sides agree on, so a single list of instructions can drive training
    and evaluation without either drifting.
    """
    wanted = [str(task) for task in tasks]
    known = list(meta.tasks.index) if meta.tasks is not None else []
    missing = [task for task in wanted if task not in known]
    if missing:
        raise ValueError(
            f"tasks {missing} are not in this dataset. its {len(known)} tasks: {known}"
        )
    wanted = set(wanted)
    picked = [
        index
        for index, episode_tasks in enumerate(meta.episodes["tasks"])
        if wanted & set(episode_tasks)
    ]
    if not picked:
        raise ValueError(f"tasks {sorted(wanted)} are named by no episode")
    return picked


def resolve_slice(patterns, names, field):
    """Feature-name globs -> dimension indices into a flat column. Shared with the
    lerobot eval backend, which slices live robot observations the same way.

    `patterns` is a list, or a comma-separated string, of fnmatch globs over the column's
    per-dimension names (`"*.pos"`, `"wrist_flex.torq"`, `"gripper.*"`). Indices come out in
    pattern order, dataset order within a pattern, deduplicated, so overlapping patterns are
    fine and the caller controls the layout of the sliced vector.
    """
    if isinstance(patterns, str):
        patterns = patterns.split(",")
    selected = []
    for pattern in patterns:
        pattern = str(pattern).strip()
        matched = [i for i, name in enumerate(names) if fnmatch.fnmatchcase(name, pattern)]
        if not matched:
            raise ValueError(
                f"slice pattern '{pattern}' for field '{field}' matches no dimension. "
                f"available: {list(names)}"
            )
        selected.extend(i for i in matched if i not in selected)
    return selected


class LeRobotSource(BaseSource):
    """Expert demonstrations from a LeRobot v3.0 dataset, windowed by the temporal index
    spec (`cfg.conditioning` / `cfg.predict`, see utils/spec.py). Each spec entry reads the
    batch field named by its `from` key (default: the entry name) as a window of step offsets
    around a sampled decision point t=0. A `"last"` index entry appends the episode's final
    frame (a goal frame). Loads from `cfg.root` if given, otherwise downloads `cfg.repo_id`.

    Config knobs:
      repo_id        HF dataset id (e.g. `lerobot/pusht`). Required.
      root           local dataset dir; overrides the HF cache.
      revision       dataset revision/tag (default: the codebase version).
      episodes       episode indices to keep; null uses all.
      tasks          instruction strings to keep the episodes of; null uses all.
      validation_split
                     fraction of episodes held out for offline validation.
      validation_split_seed
                     seed for that split; training and validation must agree on it.
      columns        `{from_key: lerobot_column}` remap; default identity.
      slices         `{from_key: patterns}` keeping some dimensions of a flat column by
                     feature name (e.g. `"*.pos"`, `"gripper.pos, *.vel"`).
      image_size     `[H, W]` to resize visual streams to; null keeps native size.
      return_uint8   uint8 [0,255] images (default); false gives float [0,1].
      drop_boundary  drop decision points whose window runs past the episode end (default
                     true); offsets before the start clamp onto the first frame.
    """

    def __init__(self, cfg, split=None, subsample=None, cache_fields=None, cache_draws=1,
                 keep_pixels=None):
        from lerobot.datasets import LeRobotDatasetMetadata

        self._repo_id = cfg.repo_id
        self._root = cfg.get("root")
        self._revision = cfg.get("revision")
        episodes = cfg.get("episodes")
        requested = [int(e) for e in episodes] if episodes is not None else None
        self._columns = dict(cfg.get("columns") or {})
        slices = dict(cfg.get("slices") or {})
        self._return_uint8 = cfg.get("return_uint8", True)
        self._video_backend = cfg.get("video_backend")  # None -> lerobot's default
        # re-encoded videos re-quantize frame pts by ~1e-4 s, right at lerobot's default
        # tolerance, so a run dies on pts jitter unless this is raised
        self._tolerance_s = cfg.get("tolerance_s")
        size = cfg.get("image_size")
        self._image_size = tuple(int(v) for v in size) if size else None
        drop_boundary = cfg.get("drop_boundary", True)

        meta = LeRobotDatasetMetadata(self._repo_id, root=self._root, revision=self._revision)
        features = meta.features
        fps = meta.fps
        self._meta_stats = getattr(meta, "stats", None) or {}

        from ..utils.enc_cache import resolve_stride
        self._anchor_stride = resolve_stride(subsample, fps)
        self._cache_fields = frozenset(cache_fields or ())
        self._cache_draws = int(cache_draws)
        # cached fields that must ALSO serve pixels. Their pixel windows take the SAME
        # augmentation draw as the emitted key, or pixels and latents disagree.
        self._keep_pixels = frozenset(keep_pixels or ())
        if not self._keep_pixels <= self._cache_fields:
            raise ValueError(
                f"keep_pixels {sorted(self._keep_pixels - self._cache_fields)} are not "
                f"cache fields"
            )
        aug_streams = dict((cfg.get("augment") or {}).get("streams") or {})
        self._draw_aug_cfg = {
            p: ops for p, ops in aug_streams.items()
            if any(fnmatch.fnmatchcase(f, p) for f in self._keep_pixels)
        } if self._keep_pixels else {}
        self._draw_augmenter = None

        tasks = cfg.get("tasks")
        if tasks:
            by_task = episodes_for_tasks(meta, tasks)
            requested = (
                sorted(set(by_task) & set(requested)) if requested is not None else by_task
            )
            if not requested:
                raise ValueError(
                    f"`tasks:` and `episodes:` select no episodes in common of "
                    f"'{self._repo_id}'"
                )

        # `split=None` is every episode, for a caller asking only what modalities exist
        if split is None:
            self._episodes = requested
        else:
            self._episodes = split_episodes(
                requested if requested is not None else range(len(meta.episodes["length"])),
                cfg.get("validation_split", 0.0),
                split,
                seed=int(cfg.get("validation_split_seed", 0)),
            )

        # "last" is tracked apart from the window: it is not a relative offset
        windows, has_last, field_to_col, visual = {}, {}, {}, set()
        entries = {**cfg.conditioning, **cfg.predict}
        for key, spec in entries.items():
            for field in spec_fields(key, spec):
                col = self._columns.get(field, field)
                if col in features and "index" in spec:
                    relative, last = split_index(raw_index(spec))
                    windows.setdefault(field, set()).update(relative)
                    has_last[field] = has_last.get(field, False) or last
                    field_to_col[field] = col
                    if features[col].get("dtype") in ("video", "image"):
                        visual.add(field)
        if not windows:
            raise ValueError(
                f"none of the requested modalities are columns of '{self._repo_id}'. "
                f"available: {sorted(features)}"
            )
        self._windows = {field: sorted(steps) for field, steps in windows.items()}
        self._has_last = {field for field, last in has_last.items() if last}
        self._field_to_col = field_to_col
        self._visual = visual
        self.provided_modalities = set(field_to_col) | set(field_to_col.values())

        self._slices = {}
        for field, patterns in slices.items():
            if field not in field_to_col:
                raise ValueError(
                    f"slice given for field '{field}', which no spec entry reads. "
                    f"sliceable fields: {sorted(field_to_col)}"
                )
            col = field_to_col[field]
            names = features[col].get("names")
            if field in visual or not isinstance(names, list) or len(features[col]["shape"]) != 1:
                raise ValueError(
                    f"field '{field}' reads column '{col}', which is not a flat vector with "
                    f"per-dimension names and so cannot be sliced by name"
                )
            indices = resolve_slice(patterns, names, field)
            self._slices[field] = torch.tensor(indices, dtype=torch.long)

        for field in self._cache_fields:
            if field not in field_to_col:
                raise ValueError(
                    f"cache field '{field}' is read by no spec entry; "
                    f"readable fields: {sorted(field_to_col)}"
                )
            if field not in visual:
                raise ValueError(f"cache field '{field}' is not a visual stream")
            if field in self._has_last:
                raise ValueError(
                    f"cache field '{field}' uses a 'last' (goal) row, which the "
                    f"anchor-keyed encoding cache cannot represent"
                )

        # one entry per lerobot COLUMN, covering the union its fields ask for; a column
        # whose every field is cached is dropped, so its videos never decode
        col_offsets = {}
        for field, steps in self._windows.items():
            col_offsets.setdefault(field_to_col[field], set()).update(steps)
        col_offsets = {col: sorted(steps) for col, steps in col_offsets.items()}
        uncached_cols = {
            col for field, col in field_to_col.items()
            if field not in self._cache_fields or field in self._keep_pixels
        }
        self._delta_timestamps = {
            col: [s / fps for s in steps] for col, steps in col_offsets.items()
            if steps and col in uncached_cols
        }
        self._rows = {
            field: torch.tensor(
                [col_offsets[field_to_col[field]].index(s) for s in steps], dtype=torch.long
            )
            for field, steps in self._windows.items()
        }

        # offsets before an episode clamp onto its first frame, so only the `ahead` side
        # can drop; clamping there would repeat the final action as a hold never commanded
        offsets = [s for steps in self._windows.values() for s in steps]
        ahead = max(0, max(offsets, default=0))
        lengths_all = np.asarray(meta.episodes["length"])
        selected = self._episodes if self._episodes is not None else range(len(lengths_all))
        # (global frame, episode-final global frame, episode ordinal, local frame); the
        # ordinal+local pair is the encoding-cache key space, stable because `selected` is sorted
        self._index = []
        self._ep_starts = []
        start = 0
        for ep in selected:
            length = int(lengths_all[ep])
            self._ep_starts.append(start)
            bound = length - ahead if drop_boundary else length
            last = start + length - 1
            ordinal = len(self._ep_starts) - 1
            self._index.extend(
                (start + t, last, ordinal, t) for t in range(0, bound, self._anchor_stride)
            )
            start += length

        self._ds = None
        self._goal_ds = None
        self._plain_ds = None
        self._ds_pid = None

    def stats_columns(self, keys):
        """`keys` as flat per-frame arrays over this split's episodes, sliced exactly like
        the served data. None when any key is not a plain parquet column.

        Per-frame, not per-window: a window overlaps its neighbours, so window statistics
        weight each frame by how many windows contain it. Reading the columns also avoids
        the video decode `__getitem__` cannot, since it decodes every key in
        `delta_timestamps` and the cameras are there whenever the algorithm uses them.
        """
        ds, _ = self._datasets()
        hf = getattr(ds.reader, "hf_dataset", None)
        if hf is None:
            return None
        out = {}
        for key in keys:
            col = self._field_to_col.get(key, self._columns.get(key, key))
            if col in self._visual or col not in hf.column_names:
                return None
            values = np.asarray(hf.with_format("numpy")[col])
            values = values.reshape(len(values), -1)
            dims = self._slices.get(key)
            if dims is not None:
                values = values[:, dims.numpy()]
            out[key] = values
        return out or None

    def _datasets(self):
        from lerobot.datasets import LeRobotDataset

        # re-open per process: video decoder handles are not fork-safe
        pid = os.getpid()
        if self._ds is None or self._ds_pid != pid:
            # lerobot's module-global torchcodec cache survives the fork, and its handles
            # share file offsets with the parent, so concurrent reads corrupt each other's
            # packets ("Invalid data found"). Drop the inherited entries.
            from lerobot.datasets import video_utils
            video_utils._default_decoder_cache.clear()
            def open_dataset(delta_timestamps):
                return LeRobotDataset(
                    self._repo_id,
                    root=self._root,
                    revision=self._revision,
                    episodes=self._episodes,
                    delta_timestamps=delta_timestamps,
                    return_uint8=self._return_uint8,
                    video_backend=self._video_backend,
                    **({"tolerance_s": self._tolerance_s}
                       if self._tolerance_s is not None else {}),
                    decode_unrequested_videos=False,
                )

            self._ds = open_dataset(self._delta_timestamps or None)
            self._prune_hf_columns(self._ds)
            # a second handle with no delta_timestamps decodes the "last" frame as ONE row
            self._goal_ds = open_dataset(None) if self._has_last else None
            self._ds_pid = pid
        return self._ds, self._goal_ds

    def _prune_hf_columns(self, ds):
        """Restrict the reader's hf_dataset to the columns train-time fetches read.

        On a repo whose cameras are inline parquet Image columns rather than video files,
        every lerobot row fetch decodes EVERY stored column of the touched rows, so an
        unread camera is JPEG-decoded once per row of every delta window.
        """
        reader = ds.reader
        hf = reader.hf_dataset
        if hf is None:
            return
        needed = set(self._delta_timestamps or ()) | {
            "episode_index", "frame_index", "index", "timestamp", "task_index",
        }
        keep = [c for c in hf.column_names if c in needed]
        if len(keep) == len(hf.column_names):
            return
        from lerobot.datasets.io_utils import hf_transform_to_torch

        pruned = hf.select_columns(keep)
        pruned.set_transform(hf_transform_to_torch)
        reader.hf_dataset = pruned

    def __getstate__(self):
        return {**self.__dict__, "_ds": None, "_goal_ds": None, "_ds_pid": None,
                "_plain_ds": None, "_draw_augmenter": None}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        frame, last_frame, ep_ord, local_t = self._index[idx]
        ds, goal_ds = self._datasets()
        item = ds[frame]
        goal = goal_ds[last_frame] if goal_ds is not None else {}
        batch = {}
        for field, col in self._field_to_col.items():
            draw = None
            if field in self._cache_fields:
                # one draw per field per sample, so two cameras augment independently
                draw = int(torch.randint(self._cache_draws, ()))
                batch[f"enc_cache/{field}"] = torch.tensor(
                    [ep_ord, local_t, draw], dtype=torch.long
                )
                if field not in self._keep_pixels:
                    continue
            rows = self._rows[field]
            value = item[col]
            # lerobot squeezes a single-timestamp video query to (C, H, W); without
            # this the row select below eats the channel axis
            if field in self._visual and value.dim() == 3:
                value = value.unsqueeze(0)
            parts = [value.index_select(0, rows)] if len(rows) else []
            if field in self._has_last:
                # after every relative offset: predictive_model.py addresses it from the end
                parts.append(goal[col].unsqueeze(0))
            tensor = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if field in self._visual and self._image_size:
                tensor = self._resize(tensor)
            if field in self._slices:
                tensor = tensor.index_select(-1, self._slices[field])
            if draw is not None and self._draw_aug_cfg:
                # the AugmentedSource wrapper skips cached fields, so this is the only
                # augmentation these pixels get: it must match the emitted draw exactly
                if self._draw_augmenter is None:
                    from ..utils.augment import Augmenter
                    self._draw_augmenter = Augmenter(self._draw_aug_cfg)
                from ..utils.enc_cache import augment_like_draw
                tensor = augment_like_draw(
                    self._draw_augmenter, field, tensor, ep_ord, local_t, draw
                )
            batch[field] = tensor
        # a plain string survives default_collate (it becomes a list per batch)
        if "task" in item:
            batch["task"] = item["task"]
        return batch

    def _resize(self, tensor):
        if tuple(tensor.shape[-2:]) == self._image_size:
            return tensor
        resized = torch.nn.functional.interpolate(
            tensor.float(), size=self._image_size, mode="area"
        )
        if tensor.dtype == torch.uint8:
            resized = resized.round().clamp(0, 255).to(torch.uint8)
        return resized

    def cache_anchor_space(self):
        """Every (episode ordinal, [anchors]) this split can sample, in key-sort order --
        exactly the decision points `__getitem__` draws from, so the cache the build
        produces covers precisely what training will request and nothing else.
        """
        by_ep = {}
        for _, _, ep_ord, local_t in self._index:
            by_ep.setdefault(ep_ord, []).append(local_t)
        return sorted(by_ep.items())

    def read_frame(self, ep_ord, local_t):
        """One frame of every visual field, resized like training frames -- the build
        driver's decode primitive. Uses a windowless dataset handle so each call decodes
        one frame per video, not a whole delta window. Safe inside DataLoader workers:
        each process builds its own handle (and drops inherited decoder-cache entries,
        whose file offsets are shared with the parent -- see _datasets).
        """
        from lerobot.datasets import LeRobotDataset

        if self._plain_ds is None or getattr(self, "_plain_pid", None) != os.getpid():
            from lerobot.datasets import video_utils
            video_utils._default_decoder_cache.clear()
            self._plain_pid = os.getpid()
            self._plain_ds = None
        if self._plain_ds is None:
            self._plain_ds = LeRobotDataset(
                self._repo_id, root=self._root, revision=self._revision,
                episodes=self._episodes, return_uint8=self._return_uint8,
                video_backend=self._video_backend,
                **({"tolerance_s": self._tolerance_s}
                   if self._tolerance_s is not None else {}),
                decode_unrequested_videos=False,
            )
        item = self._plain_ds[self._ep_starts[ep_ord] + int(local_t)]
        out = {}
        for field in self._visual:
            tensor = item[self._field_to_col[field]]
            if self._image_size:
                tensor = self._resize(tensor.unsqueeze(0)).squeeze(0)
            out[field] = tensor
        return out
