def live_fields(source):
    """The batch fields a source serves (lerobot: its spec fields; other backends: their
    provided modalities) -- what an augmentation pattern can match."""
    return set(getattr(source, "_field_to_col", None) or source.provided_modalities)


def build_dataset(cfg, norm_stats_override=None, augment=True, split=None,
                  subsample=None, cache_fields=None, cache_draws=1, keep_pixels=None):
    """Lazy backend factory. A backend module is imported only when selected, so a
    base install never imports a backend that is not present. If `cfg.normalize` is set,
    wraps the source so training sees normalized data (see utils/normalize.py); if
    `cfg.augment` is set, wraps that in turn so training sees augmented data
    (see utils/augment.py).

    `norm_stats_override`, if given, is used as-is instead of recomputing stats from this
    dataset -- pass a checkpoint's own `norm_stats` when resuming, so a run stays
    consistent with the stats it was originally trained with.

    `augment=False` builds the same source without the augmentation wrapper, for the
    callers that want the data as recorded (held-out eval, modality checks).

    `split="train"` / `"val"` keeps only that side of the episode partition
    `cfg.validation_split` defines; None keeps every episode.

    `subsample` is the root `subsample_streams:` config (anchor thinning) and
    `cache_fields`/`cache_draws` put the source in encoding-cache mode: those fields are
    served as `enc_cache/<field>` window keys instead of pixels, and their augmentation
    patterns are dropped from the wrapper here because the draws are baked into the
    cache (utils/enc_cache.py). Both are lerobot-only.
    """
    backend = cfg.backend
    if (subsample or cache_fields) and backend != "lerobot":
        raise ValueError(
            f"subsample_streams / cache_encodings need the lerobot backend, got '{backend}'"
        )
    # a backend with no episodes has nothing to hold out, so "train" is all of it;
    # "val" would silently hand back the training data
    if split == "val" and backend != "lerobot":
        raise ValueError(
            f"split='val' needs episodes to hold out, which the '{backend}' backend does "
            f"not have; only the lerobot backend supports a held-out split"
        )
    if backend == "lerobot":
        from .lerobot import LeRobotSource
        source = LeRobotSource(cfg, split=split, subsample=subsample,
                               cache_fields=cache_fields, cache_draws=cache_draws,
                               keep_pixels=keep_pixels)
    else:
        raise ValueError(
            f"backend '{backend}' not installed. "
            f"try: uv pip install 'lewam-real-robot[{backend}]'"
        )

    norm_cfg = cfg.get("normalize")
    if norm_cfg:
        from ..utils.normalize import NormalizedSource
        method = norm_cfg.get("method", "mean_std")
        # Never read off meta/stats.json: lerobot builds its quantiles by averaging
        # per-episode quantiles, which is much narrower than the global quantile.
        stats = norm_stats_override
        source = NormalizedSource(
            source,
            keys=list(norm_cfg["keys"]),
            method=method,
            percentiles=norm_cfg.get("percentiles"),
            stats=stats,
        )

    aug_cfg = cfg.get("augment")
    if augment and aug_cfg and aug_cfg.get("streams"):
        streams = dict(aug_cfg["streams"])
        if cache_fields:
            # a cached field is not in the batch and its draws are already baked in, so keep
            # only the patterns that still match a live field
            from ..utils.augment import live_augment_streams
            streams = live_augment_streams(streams, live_fields(source) - set(cache_fields))
        if streams:
            from ..utils.augment import AugmentedSource
            source = AugmentedSource(source, {**dict(aug_cfg), "streams": streams})
    return source
