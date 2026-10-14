"""Self- and cross-attention for the DiT.

Both layers consume the precomputed RoPE angle matrix from rope.py and take the
attention mask explicitly: a dense bool mask (True = attend) dispatches to SDPA,
a flex-attention BlockMask dispatches to flex_attention.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .rope import apply_rotary_emb

__all__ = ["SelfAttention", "CrossAttention"]

# uncompiled flex_attention materializes the full B*H*N*N scores matrix; compile
# lowers it to a fused kernel. Some driver/triton setups fail to compile, so the
# first error falls back to eager permanently. LEWAM_NO_COMPILE_FLEX=1 skips it.
_flex = flex_attention if os.environ.get("LEWAM_NO_COMPILE_FLEX") else torch.compile(flex_attention)


def _run_flex(q, k, v, block_mask):
    """Compiled flex_attention with a one-shot, permanent fallback to eager on compile failure."""
    global _flex
    try:
        return _flex(q, k, v, block_mask=block_mask)
    except Exception:
        if _flex is flex_attention:
            raise
        _flex = flex_attention
        return _flex(q, k, v, block_mask=block_mask)


def split_heads(x, num_heads):
    B, N, D = x.shape
    return x.reshape(B, N, num_heads, D // num_heads).transpose(1, 2)


def merge_heads(x):
    B, H, N, Dh = x.shape
    return x.transpose(1, 2).reshape(B, N, H * Dh)


def attend(q, k, v, attn_mask=None):
    if isinstance(attn_mask, BlockMask):
        return _run_flex(q, k, v, attn_mask)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    """QKV self-attention with QK-norm and 3D RoPE. `dim_head` defaults to
    hidden_size // num_heads (inner width == residual width, unchanged); pass it to
    decouple the attention inner width (num_heads * dim_head) from the residual width,
    e.g. le-wm's 16 heads x 64 on a 192-wide stream. With RoPE, `freqs` must be sized to
    dim_head.
    """

    def __init__(self, hidden_size, num_heads, dim_head=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_head if dim_head is not None else hidden_size // num_heads
        inner = num_heads * head_dim
        self.qkv = nn.Linear(hidden_size, 3 * inner)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.out_proj = nn.Linear(inner, hidden_size)

    def forward(self, x, freqs=None, attn_mask=None):
        q, k, v = (split_heads(t, self.num_heads) for t in self.qkv(x).chunk(3, dim=-1))
        q = self.q_norm(q)
        k = self.k_norm(k)
        if freqs is not None:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        out = attend(q.to(v.dtype), k.to(v.dtype), v, attn_mask)
        return self.out_proj(merge_heads(out))


class CrossAttention(nn.Module):
    """Cross-attention over multiple conditioning sources with per-source K/V projections.

    source_dims gives each source's feature dim. A None source is skipped. Pass
    use_kv_cache=True to reuse K/V and the combined source mask across ODE steps; call
    clear_cache() before each new solve.

    No RoPE: sources sit off the token axis, so rotating Q against unrotated K would encode
    the query's absolute position rather than an offset. Source order reaches the model only
    through what its encoder baked into the features.
    """

    def __init__(self, hidden_size, num_heads, source_dims):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_projs = nn.ModuleList([nn.Linear(d, hidden_size) for d in source_dims])
        self.k_norms = nn.ModuleList([nn.LayerNorm(head_dim) for _ in source_dims])
        self.v_projs = nn.ModuleList([nn.Linear(d, hidden_size) for d in source_dims])
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.clear_cache()

    def clear_cache(self):
        self._cached_k = None
        self._cached_v = None
        self._cached_mask = None

    @property
    def has_cache(self):
        return self._cached_k is not None

    def _project_sources(self, sources, source_masks, batch, device):
        ks, vs = [], []
        combined_mask = None
        for i, src in enumerate(sources):
            if src is None:
                continue
            ks.append(self.k_norms[i](split_heads(self.k_projs[i](src), self.num_heads)))
            vs.append(split_heads(self.v_projs[i](src), self.num_heads))
            if source_masks is not None:
                m = source_masks[i]
                chunk = m if m is not None else torch.zeros(
                    batch, src.shape[1], dtype=torch.bool, device=device
                )
                combined_mask = chunk if combined_mask is None else torch.cat([combined_mask, chunk], dim=1)

        k = torch.cat(ks, dim=2)
        v = torch.cat(vs, dim=2)
        attn_mask = ~combined_mask.unsqueeze(1).unsqueeze(1) if combined_mask is not None else None
        return k, v, attn_mask

    def forward(self, x, sources, source_masks=None, use_kv_cache=False):
        """x: (B, N, hidden) queries. sources: list of (B, N_i, source_dims[i]) or None.
        source_masks: per-source (B, N_i) bool, True = ignore (padding), or None.
        """
        q = self.q_norm(split_heads(self.q_proj(x), self.num_heads))

        if use_kv_cache and self.has_cache:
            k, v, attn_mask = self._cached_k, self._cached_v, self._cached_mask
        else:
            k, v, attn_mask = self._project_sources(sources, source_masks, x.shape[0], x.device)
            if use_kv_cache:
                self._cached_k, self._cached_v, self._cached_mask = k, v, attn_mask

        out = attend(q.to(v.dtype), k.to(v.dtype), v, attn_mask)
        return self.out_proj(merge_heads(out))
