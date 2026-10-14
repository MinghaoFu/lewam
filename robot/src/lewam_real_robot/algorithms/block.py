"""DiT transformer block: self-attention -> optional cross-attention -> MLP; and its
Mixture-of-Transformers form, MoTBlock, with per-expert parameters under one attention."""

import torch.nn as nn

from .attention import CrossAttention, SelfAttention, attend, merge_heads
from .layers import bcast_cond, make_mlp, modulate
from .rope import apply_rotary_emb

__all__ = ["DiTBlock", "MoTBlock"]


class DiTBlock(nn.Module):
    """Self-attention and MLP branches are adaRMS-modulated and gated by cond; the
    modulation MLP is zero-initialized, so both branches are closed and the block is
    the identity at init. The cross-attention branch reads the conditioning sources
    through a plain pre-norm, is ungated, and sees no RoPE.

    Norms are RMSNorm, following pi0.5's action expert, which modulates shift/scale/gate
    off an RMSNorm exactly as this block does off DiT's LayerNorm.
    """

    def __init__(self, hidden_size, num_heads, source_dims=None, mlp_ratio=4.0, dropout=0.0,
                 cond_dim=None, dim_head=None, mlp_dim=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sa = SelfAttention(hidden_size, num_heads, dim_head=dim_head)

        self.ca = CrossAttention(hidden_size, num_heads, source_dims) if source_dims else None
        self.norm_ca = nn.RMSNorm(hidden_size, eps=1e-6) if source_dims else None

        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = make_mlp(hidden_size, int(mlp_dim or hidden_size * mlp_ratio), hidden_size)
        self.drop = nn.Dropout(dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim or hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def clear_cache(self):
        if self.ca is not None:
            self.ca.clear_cache()

    def forward(self, x, cond, sources=None, source_masks=None, freqs=None,
                attn_mask=None, use_kv_cache=False):
        """
        x:            (B, N, hidden_size)
        cond:         (B, cond_dim) shared across tokens, or (B, N, cond_dim) per token
        sources:      list of (B, N_i, source_dims[i]) or None per source
        source_masks: list of (B, N_i) bool (True = ignore) or None per source
        freqs:        (N, head_dim) RoPE angles for all tokens; self-attention only, as
                      cross-attention sources have no position to be relative to
        attn_mask:    dense bool (N, N) or flex BlockMask, True = attend
        use_kv_cache: reuse cross-attention K/V across ODE steps
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + bcast_cond(gate_msa) * self.drop(self.sa(h, freqs=freqs, attn_mask=attn_mask))

        if self.ca is not None and (sources is not None or (use_kv_cache and self.ca.has_cache)):
            x = x + self.drop(self.ca(
                self.norm_ca(x), sources, source_masks=source_masks,
                use_kv_cache=use_kv_cache,
            ))

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + bcast_cond(gate_mlp) * self.drop(self.mlp(h))
        return x


class MoTBlock(nn.Module):
    """Mixture-of-Transformers block (Liang et al. 2024): DiTBlock's structure with every
    parameterised sub-layer duplicated per expert and ONE self-attention over the union of
    tokens. Each token is routed to the expert its stream declares (`model.experts`): its
    QKV/output projections, QK-norms, MLP, cross-attention branch and adaRMS modulation are
    that expert's own, while the attention scores mix every token under the shared mask.
    The norms carry no parameters, so there is nothing of them to split.

    Streams therefore share nothing in the trunk but the attention pattern; an expert's
    weights receive gradient from another expert's loss only through the keys/values its
    tokens contribute where the mask lets the other expert read them. With a single
    expert this is DiTBlock exactly; the trunk builds DiTBlock in that case so existing
    checkpoints keep their parameter names.

    `routes` (one LongTensor of token indices per expert, in the layout of `x`) comes from
    the trunk, which owns the stream -> expert table and every subset layout.
    """

    def __init__(self, hidden_size, num_heads, num_experts, source_dims=None, mlp_ratio=4.0,
                 dropout=0.0, cond_dim=None, dim_head=None, mlp_dim=None):
        super().__init__()
        assert num_experts >= 2, (
            f"MoTBlock needs at least two experts, got {num_experts}; one expert is the "
            "shared trunk -- drop `model.experts` for that"
        )
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = dim_head if dim_head is not None else hidden_size // num_heads
        inner = num_heads * self.head_dim
        experts = range(num_experts)

        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.ModuleList(nn.Linear(hidden_size, 3 * inner) for _ in experts)
        self.q_norm = nn.ModuleList(nn.LayerNorm(self.head_dim) for _ in experts)
        self.k_norm = nn.ModuleList(nn.LayerNorm(self.head_dim) for _ in experts)
        self.out_proj = nn.ModuleList(nn.Linear(inner, hidden_size) for _ in experts)

        if source_dims:
            self.ca = nn.ModuleList(
                CrossAttention(hidden_size, num_heads, source_dims) for _ in experts)
            self.norm_ca = nn.ModuleList(nn.RMSNorm(hidden_size, eps=1e-6) for _ in experts)
        else:
            self.ca = self.norm_ca = None

        self.mlp = nn.ModuleList(
            make_mlp(hidden_size, int(mlp_dim or hidden_size * mlp_ratio), hidden_size)
            for _ in experts)
        self.drop = nn.Dropout(dropout)

        self.adaLN_modulation = nn.ModuleList()
        for _ in experts:
            ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim or hidden_size, 6 * hidden_size, bias=True))
            nn.init.zeros_(ada[-1].weight)
            nn.init.zeros_(ada[-1].bias)
            self.adaLN_modulation.append(ada)

    def clear_cache(self):
        if self.ca is not None:
            for ca in self.ca:
                ca.clear_cache()

    @staticmethod
    def _scatter(parts, num_tokens):
        """Reassemble per-expert outputs [(idx, (B, n_e, ...))] into one (B, N, ...) tensor.
        Every token belongs to exactly one expert, so the union of the routes covers N."""
        first = parts[0][1]
        out = first.new_empty((first.shape[0], num_tokens) + tuple(first.shape[2:]))
        for idx, y in parts:
            out.index_copy_(1, idx, y)
        return out

    def _route(self, modules, x, routes):
        """modules[e] applied to x's expert-e tokens, results scattered back to x's layout."""
        parts = [(idx, modules[e](x.index_select(1, idx)))
                 for e, idx in enumerate(routes) if idx.numel()]
        return self._scatter(parts, x.shape[1])

    def _attend(self, h, routes, freqs, attn_mask):
        B, N, _ = h.shape
        qs, ks, vs = [], [], []
        for e, idx in enumerate(routes):
            if not idx.numel():
                continue
            n = idx.numel()
            q, k, v = self.qkv[e](h.index_select(1, idx)).chunk(3, dim=-1)
            qs.append((idx, self.q_norm[e](q.reshape(B, n, self.num_heads, self.head_dim))))
            ks.append((idx, self.k_norm[e](k.reshape(B, n, self.num_heads, self.head_dim))))
            vs.append((idx, v.reshape(B, n, self.num_heads, self.head_dim)))
        # (B, N, H, Dh) -> (B, H, N, Dh), the layout SelfAttention attends in
        q, k, v = (self._scatter(p, N).transpose(1, 2) for p in (qs, ks, vs))
        if freqs is not None:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        out = merge_heads(attend(q.to(v.dtype), k.to(v.dtype), v, attn_mask))
        return self._route(self.out_proj, out, routes)

    def forward(self, x, cond, sources=None, source_masks=None, freqs=None,
                attn_mask=None, use_kv_cache=False, routes=None):
        """DiTBlock.forward plus `routes`: one (n_e,) LongTensor of token indices per expert,
        covering every token of `x` exactly once."""
        assert routes is not None and len(routes) == self.num_experts, (
            "MoTBlock needs one route per expert"
        )
        B, N, _ = x.shape
        c = cond if cond.dim() == 3 else cond[:, None].expand(B, N, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self._route(self.adaLN_modulation, c, routes).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.drop(self._attend(h, routes, freqs, attn_mask))

        if self.ca is not None and (sources is not None
                                    or (use_kv_cache and any(ca.has_cache for ca in self.ca))):
            parts = []
            for e, idx in enumerate(routes):
                if not idx.numel():
                    continue
                xe = x.index_select(1, idx)
                parts.append((idx, self.ca[e](self.norm_ca[e](xe), sources,
                                              source_masks=source_masks,
                                              use_kv_cache=use_kv_cache)))
            x = x + self.drop(self._scatter(parts, N))

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.drop(self._route(self.mlp, h, routes))
        return x
