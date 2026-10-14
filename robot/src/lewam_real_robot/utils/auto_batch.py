"""Pick the largest per-device batch size that fits in VRAM.

`find_max_batch_size` gathers cheap doubling probes until one crosses
`fit_threshold` of device memory (or goes over target/OOM), then fits peak
usage against batch size (usage is affine in batch: fixed weights/optimizer
memory plus per-sample activations) and probes the solve of
`usage == target_fraction` directly, re-fitting with each new measurement.
With fewer than `min_fit_points` measurements the fit is unreliable, so it
degrades to plain doubling + bisection. An OOM probe bounds the bracket and
the next probe is the bracket midpoint, which yields a measurable point.
The search stops when a probe lands within `tolerance` under the target, the
bracket closes, or a safe probe reaches `target_cap` (the per-rank effective
batch: anything larger would be clamped by the caller anyway).

The caller's `step_fn(batch_size)` must run a complete iteration -- forward,
backward, optimizer step and zero_grad -- so the peak includes activations,
gradients and optimizer state. The step mutates model and optimizer, so the
caller snapshots and restores their state around the search (see
TrainingMixin._calibrate_batch_size).
"""

from contextlib import contextmanager

import torch


@contextmanager
def peak_vram_fraction(device=None):
    """Yields a tracker whose `.fraction` is the block's peak allocated / total VRAM."""

    class _Tracker:
        fraction = 0.0

    tracker = _Tracker()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        yield tracker
    finally:
        peak = torch.cuda.max_memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        tracker.fraction = peak / total


def _solve_target(points, target_fraction):
    """Least-squares usage = a + m*batch over the measured probes, solved for the
    target. None when the fit is degenerate (no spread, or non-positive slope)."""
    n = len(points)
    sx = sum(b for b, _ in points)
    sy = sum(u for _, u in points)
    sxx = sum(b * b for b, _ in points)
    sxy = sum(b * u for b, u in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    m = (n * sxy - sx * sy) / denom
    if m <= 0:
        return None
    a = (sy - m * sx) / n
    return int((target_fraction - a) / m)


def find_max_batch_size(step_fn, target_fraction=0.85, device=None, log=print,
                        target_cap=None, fit_threshold=0.15, min_fit_points=4,
                        tolerance=0.02, max_probes=30):
    assert torch.cuda.is_available(), "auto batch size calibration needs CUDA"

    points = []
    lo, hi = 0, None

    def probe(batch_size):
        nonlocal lo, hi
        try:
            with peak_vram_fraction(device) as tracker:
                step_fn(batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"  batch_size={batch_size}: OOM")
            hi = batch_size if hi is None else min(hi, batch_size)
            return None
        used = tracker.fraction
        log(f"  batch_size={batch_size}: {used * 100:.1f}% VRAM peak")
        points.append((batch_size, used))
        if used >= target_fraction:
            hi = batch_size if hi is None else min(hi, batch_size)
        else:
            lo = max(lo, batch_size)
        return used

    def done():
        return (target_cap is not None and lo >= target_cap) or (
            hi is not None and hi - lo <= 1
        )

    def result():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return lo if target_cap is None else min(lo, target_cap)

    batch = 1
    while True:
        used = probe(batch)
        if used is None and batch == 1:
            raise RuntimeError(
                "OOM at batch_size=1: shrink the model or its inputs"
            ) from None
        if used is None or used >= min(fit_threshold, target_fraction) or done():
            break
        batch *= 2

    for _ in range(max_probes):
        if done():
            return result()
        cand = _solve_target(points, target_fraction) if len(points) >= min_fit_points else None
        if cand is None:
            cand = (lo + hi) // 2 if hi is not None else lo * 2
        if target_cap is not None:
            cand = min(cand, target_cap)
        if hi is not None:
            cand = min(cand, hi - 1)
        cand = max(cand, lo + 1)
        if any(b == cand for b, _ in points):
            cand = (lo + hi) // 2 if hi is not None else lo * 2
            cand = max(cand, lo + 1)
            if hi is not None:
                cand = min(cand, hi - 1)
            if any(b == cand for b, _ in points):
                return result()
        used = probe(cand)
        if used is None and hi is not None and hi - lo > 1:
            # an OOM measures nothing: take the bracket midpoint as a real data point
            probe(max(lo + 1, (lo + hi) // 2))
            continue
        if used is not None and target_fraction - tolerance <= used < target_fraction:
            return result()
    return result()
