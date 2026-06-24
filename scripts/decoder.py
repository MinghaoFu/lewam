"""Lightweight transformer decoder for LeWM visualization (per paper App. D).

CLS token (192-D) → cross-attention transformer with learnable patch queries
→ 16×16×3 pixel patches → 224×224 RGB image.

Diagnostic-only: not on the planning path.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.dropout = dropout

    def forward(self, q, kv):
        # q: (B, Nq, D), kv: (B, Nkv, D)
        q_, kv_ = self.norm_q(q), self.norm_kv(kv)
        Q = rearrange(self.to_q(q_), 'b n (h d) -> b h n d', h=self.heads)
        K = rearrange(self.to_k(kv_), 'b n (h d) -> b h n d', h=self.heads)
        V = rearrange(self.to_v(kv_), 'b n (h d) -> b h n d', h=self.heads)
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout if self.training else 0.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        h = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, dim),
        )

    def forward(self, x):
        return self.net(x)


class LeWMVisDecoder(nn.Module):
    """CLS (192-D) → 224×224 RGB. Patch size 16 → 14×14 = 196 queries."""

    def __init__(
        self,
        cls_dim: int = 192,
        hidden_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        mlp_ratio: float = 4.0,
        img_size: int = 224,
        patch_size: int = 16,
        out_channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches_side = img_size // patch_size
        self.n_patches = self.n_patches_side ** 2  # 196
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Project CLS → hidden_dim (becomes K/V — single token)
        self.cls_proj = nn.Sequential(
            nn.Linear(cls_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable patch queries
        self.query = nn.Parameter(torch.randn(1, self.n_patches, hidden_dim) * 0.02)

        # Stack of (cross-attn + MLP) blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                'attn': CrossAttention(hidden_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                'mlp': MLPBlock(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout),
            }))

        # Final norm + linear → pixel patches
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)

    def forward(self, cls_emb):
        """cls_emb: (B, cls_dim) or (B, T, cls_dim). Returns (B, 3, H, W) or (B, T, 3, H, W)."""
        if cls_emb.ndim == 3:
            B, T = cls_emb.shape[:2]
            flat = cls_emb.reshape(B * T, -1)
            out = self._decode(flat)
            return out.reshape(B, T, self.out_channels, self.n_patches_side * self.patch_size, self.n_patches_side * self.patch_size)
        else:
            return self._decode(cls_emb)

    def _decode(self, cls_emb):
        B = cls_emb.shape[0]
        # cls_emb: (B, cls_dim) → kv: (B, 1, hidden)
        kv = self.cls_proj(cls_emb).unsqueeze(1)
        # queries: (1, P, hidden) → (B, P, hidden)
        q = self.query.expand(B, -1, -1).clone()
        for blk in self.blocks:
            q = q + blk['attn'](q, kv)
            q = q + blk['mlp'](q)
        q = self.norm_out(q)
        patches = self.head(q)  # (B, P, 16*16*3)
        # Reshape to image
        ps, ch = self.patch_size, self.out_channels
        n = self.n_patches_side
        img = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=n, w=n, p1=ps, p2=ps, c=ch)
        return img  # (B, 3, 224, 224)
