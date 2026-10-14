"""Time-based block-causal attention masks.

Each token carries a time interval, the causal attention block it belongs to, on the
time axis shared by all modalities (seconds, relative to decision time t=0). The whole
mask is then a single predicate:

    attend  iff  kv_block_start < q_block_end

Within a block, tokens attend each other fully. Across modalities running at different
rates (video tubelets vs. action steps), time alignment falls out of the comparison,
there is no index arithmetic tying streams together. New modalities just declare their
step indices; an always-visible conditioning token can pass start=-inf.

Intervals are built from the modality spec's integer step indices, and boundaries are
computed as (integer step count) / fps in float64. IEEE division is correctly rounded,
so boundaries that are equal in exact arithmetic are bit-identical across modalities
and the strict < comparison never misfires.
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask

__all__ = ["block_intervals", "block_causal_mask", "block_causal_flex_mask"]


def block_intervals(indices, fps, block_size=1):
    """Per-token causal block (start, end) times in seconds, each shape like `indices`.

    indices: integer step indices from the modality spec (negative = past, positive =
    future). Video tokens repeat their timestep's index once per spatial patch
    (`repeat_interleave`), matching the token order of make_pos_ids. fps: steps per
    second for this modality. block_size: steps per attention block, the experiment
    knob for causality granularity (1 = fully causal per step). Blocks are anchored
    at t=0, so alignment across modalities is deterministic.
    """
    idx = torch.as_tensor(indices, dtype=torch.long)
    block = torch.div(idx, block_size, rounding_mode="floor")
    start = (block * block_size).to(torch.float64) / fps
    end = ((block + 1) * block_size).to(torch.float64) / fps
    return start, end


def block_causal_mask(start, end, blocked_q=None, noisy_kv=None, stream_ids=None, vis=None):
    """Dense (N, N) bool mask for SDPA, True = attend. Rows are queries.

    blocked_q/noisy_kv (both (N,) bool or both None) add a role rule on top of the
    time predicate: a blocked_q query never attends a noisy_kv key, whatever their
    time blocks say. stream_ids ((N,) long) + vis ((S, S) bool, rows = query stream)
    add per-stream visibility the same way: a query attends a key only if
    vis[stream_ids[q], stream_ids[k]].
    """
    mask = start[None, :] < end[:, None]
    if blocked_q is not None:
        mask &= ~(blocked_q[:, None] & noisy_kv[None, :])
    if vis is not None:
        mask &= vis[stream_ids[:, None], stream_ids[None, :]]
    return mask


def block_causal_flex_mask(start, end, blocked_q=None, noisy_kv=None, stream_ids=None,
                           vis=None, flex_block_size=128):
    """The same predicate as a flex-attention BlockMask.

    flex_block_size is the kernel's sparsity granularity, it does not change mask
    semantics (partial blocks are re-checked against the predicate inside the kernel).
    """
    def mask_fn(b, h, q_idx, kv_idx):
        ok = start[kv_idx] < end[q_idx]
        if blocked_q is not None:
            ok = ok & ~(blocked_q[q_idx] & noisy_kv[kv_idx])
        if vis is not None:
            ok = ok & vis[stream_ids[q_idx], stream_ids[kv_idx]]
        return ok

    n = start.shape[0]
    return create_block_mask(
        mask_fn, B=None, H=None, Q_LEN=n, KV_LEN=n,
        device=start.device.type, BLOCK_SIZE=flex_block_size,
    )


def build_stream_visibility(order, clean, noisy, attends, legacy_blocked=()):
    """(S, S) bool visibility over token streams, rows = query stream.

    order: stream names in token order. clean/noisy: the names in each group, for
    resolving the `context` / `noisy` / `all` group aliases in an `attends:` list.
    attends: dict name -> list of stream names/groups, or None for full visibility.
    A stream always attends itself. legacy_blocked: clean streams barred from noisy
    keys (the old `clean_attends_noisy: false` rule), applied only where no explicit
    `attends:` overrides it.
    """
    idx = {n: i for i, n in enumerate(order)}
    groups = {"all": list(order), "context": list(clean), "noisy": list(noisy)}
    vis = torch.ones(len(order), len(order), dtype=torch.bool)
    for q in legacy_blocked:
        for k in noisy:
            vis[idx[q], idx[k]] = False
    for name, allowed in attends.items():
        if allowed is None:
            continue
        row = idx[name]
        vis[row] = False
        vis[row, row] = True
        for entry in allowed:
            targets = groups.get(entry, [entry])
            for t in targets:
                if t not in idx:
                    raise ValueError(
                        f"stream '{name}' attends '{t}', which is not a token stream "
                        f"(token streams: {order}; groups: all, context, noisy)"
                    )
                vis[row, idx[t]] = True
    return vis


def assert_no_attention_path(order, vis, no_path):
    """Raise if any multi-layer attention path lets `src`'s content reach `dst`'s queries.

    Content flows from key stream k into query stream q whenever vis[q, k]; over layers
    that composes transitively. For each (src, dst) pair, walk the reads-from graph
    backward from dst and fail with the offending chain if it reaches src.
    """
    idx = {n: i for i, n in enumerate(order)}
    for src, dst in no_path or ():
        for n in (src, dst):
            if n not in idx:
                raise ValueError(f"attention.no_path names unknown stream '{n}' ({order})")
        parent, frontier, seen = {dst: None}, [dst], {dst}
        while frontier:
            q = frontier.pop()
            for k in order:
                if k in seen or not vis[idx[q], idx[k]]:
                    continue
                parent[k] = q
                if k == src:
                    chain, node = [], src
                    while node is not None:
                        chain.append(node)
                        node = parent[node]
                    raise ValueError(
                        "attention.no_path violated: "
                        + " -> ".join(chain)
                        + " (each arrow = 'is read by'); cut one of these attends edges"
                    )
                seen.add(k)
                frontier.append(k)
