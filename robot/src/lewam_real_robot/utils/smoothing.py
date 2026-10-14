"""Action-chunk post-processing."""

import torch
import torch.nn.functional as F

__all__ = ["smooth_actions"]


def smooth_actions(actions, window=21, poly_order=3):
    """Smooth an action chunk: upsample 2x, Savitzky-Golay filter, downsample.

    actions: (B, N, D) -> (B, N, D). window is clamped to an odd size within
    [5, 2N]; poly_order is clamped below the effective window.
    """
    from scipy.signal import savgol_filter

    device = actions.device
    dtype = actions.dtype
    B, N, _ = actions.shape

    upsampled = F.interpolate(
        actions.float().permute(0, 2, 1),
        size=N * 2,
        mode="linear",
        align_corners=True,
    ).permute(0, 2, 1)

    up_np = upsampled.cpu().numpy()
    window = min(window, N * 2)
    if window % 2 == 0:
        window -= 1
    window = max(window, 5)
    poly_order = min(poly_order, window - 1)

    for b in range(B):
        up_np[b] = savgol_filter(up_np[b], window, poly_order, axis=0)

    smoothed = torch.from_numpy(up_np).to(device=device)

    downsampled = F.interpolate(
        smoothed.permute(0, 2, 1),
        size=N,
        mode="linear",
        align_corners=True,
    ).permute(0, 2, 1)

    return downsampled.to(dtype=dtype)
