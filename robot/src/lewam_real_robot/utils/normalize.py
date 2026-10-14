import numpy as np
import torch


def compute_stats(dataset, keys, max_samples=None, seed=0, percentiles=None):
    """Mean/std/min/max per key, over the WHOLE dataset by default.

    Reduces over every dim except the last (assumed to be the feature dim), so it works the
    same whether a key's tensor carries a time dim (e.g. an action chunk) or not.

    A source offering `stats_columns(keys)` is read through that instead of item by item,
    exactly and without building windows; `max_samples` applies only to the item fallback,
    where indices are sampled randomly because a source's order can be grouped by episode.

    `percentiles` (a `[lo, hi]` pair) adds `q_lo`/`q_hi` per key.
    """
    columns = getattr(dataset, "stats_columns", None)
    columns = columns(keys) if columns is not None else None
    if columns:
        stats = {}
        for key, arr in columns.items():
            arr = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
            stats[key] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": np.sqrt(np.clip(arr.var(axis=0), 1e-8, None)).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }
            if percentiles is not None:
                lo, hi = percentiles
                stats[key]["q_lo"] = np.percentile(arr, lo, axis=0).tolist()
                stats[key]["q_hi"] = np.percentile(arr, hi, axis=0).tolist()
        return stats

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    indices = np.random.default_rng(seed).choice(len(dataset), size=n, replace=False)
    sums, sq_sums, mins, maxs, counts, values = {}, {}, {}, {}, {}, {}

    for i in indices:
        item = dataset[i]
        for key in keys:
            if key not in item:
                continue
            value = item[key]
            arr = np.asarray(value, dtype=np.float64).reshape(-1, value.shape[-1])
            if key not in sums:
                sums[key] = arr.sum(axis=0)
                sq_sums[key] = (arr**2).sum(axis=0)
                mins[key] = arr.min(axis=0)
                maxs[key] = arr.max(axis=0)
                counts[key] = arr.shape[0]
            else:
                sums[key] += arr.sum(axis=0)
                sq_sums[key] += (arr**2).sum(axis=0)
                mins[key] = np.minimum(mins[key], arr.min(axis=0))
                maxs[key] = np.maximum(maxs[key], arr.max(axis=0))
                counts[key] += arr.shape[0]
            if percentiles is not None:
                values.setdefault(key, []).append(arr)

    stats = {}
    for key in sums:
        mean = sums[key] / counts[key]
        var = sq_sums[key] / counts[key] - mean**2
        std = np.sqrt(np.clip(var, 1e-8, None))
        stats[key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": mins[key].tolist(),
            "max": maxs[key].tolist(),
        }
        if percentiles is not None:
            lo, hi = percentiles
            stacked = np.concatenate(values[key], axis=0)
            stats[key]["q_lo"] = np.percentile(stacked, lo, axis=0).tolist()
            stats[key]["q_hi"] = np.percentile(stacked, hi, axis=0).tolist()
    return stats


class Normalizer:
    """Applies mean-std, min-max, or percentile normalization to selected batch keys, and inverts
    predictions back to raw units at inference time. Operates purely on the standard
    batch dict, so the same class works regardless of which backend produced the data.
    `stats` is a plain JSON-able dict (see `compute_stats`), so it travels naturally as
    checkpoint metadata: attach it to the model as `model.norm_stats`, and it gets saved
    and reloaded with the weights (see utils/checkpoint.py), no separate stats file.
    """

    def __init__(self, stats, method="mean_std"):
        self.method = method
        self.stats = {
            key: {stat: torch.tensor(v, dtype=torch.float32) for stat, v in per_key.items()}
            for key, per_key in stats.items()
        }

    def to(self, device):
        self.stats = {
            key: {stat: v.to(device) for stat, v in per_key.items()}
            for key, per_key in self.stats.items()
        }
        return self

    def normalize(self, key, tensor):
        if key not in self.stats:
            return tensor
        s = self.stats[key]
        if self.method == "mean_std":
            return (tensor - s["mean"]) / s["std"]
        if self.method == "min_max":
            span = (s["max"] - s["min"]).clamp_min(1e-8)
            return (tensor - s["min"]) / span * 2 - 1
        if self.method == "percentile":
            # not clamped: the tails outside q_lo/q_hi are real data, and clamping them
            # also caps what `unnormalize` can emit
            span = (s["q_hi"] - s["q_lo"]).clamp_min(1e-8)
            return (tensor - s["q_lo"]) / span * 2 - 1
        raise ValueError(f"unknown normalization method '{self.method}'")

    def unnormalize(self, key, tensor):
        if key not in self.stats:
            return tensor
        s = self.stats[key]
        if self.method == "mean_std":
            return tensor * s["std"] + s["mean"]
        if self.method == "min_max":
            span = (s["max"] - s["min"]).clamp_min(1e-8)
            return (tensor + 1) / 2 * span + s["min"]
        if self.method == "percentile":
            span = (s["q_hi"] - s["q_lo"]).clamp_min(1e-8)
            return (tensor + 1) / 2 * span + s["q_lo"]
        raise ValueError(f"unknown normalization method '{self.method}'")

    def normalize_batch(self, batch):
        return {k: self.normalize(k, v) for k, v in batch.items()}


class NormalizedSource:
    """Wraps any BaseSource, normalizing the configured keys on `__getitem__`. Stats are
    recomputed from this run's data by default -- no cross-run caching, so a fresh run is
    always faithful to the data it's about to train on. Pass `stats` to reuse an existing
    set instead (e.g. a checkpoint's own `norm_stats` when resuming).
    """

    def __init__(self, dataset, keys, method="mean_std", max_samples=2000, stats=None,
                 percentiles=None):
        self.dataset = dataset
        self.provided_modalities = dataset.provided_modalities
        self.method = method
        pct = percentiles if method == "percentile" else None
        self.stats = stats if stats is not None else compute_stats(
            dataset, keys, max_samples=max_samples, percentiles=pct
        )
        self.normalizer = Normalizer(self.stats, method=method)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.normalizer.normalize_batch(self.dataset[idx])
