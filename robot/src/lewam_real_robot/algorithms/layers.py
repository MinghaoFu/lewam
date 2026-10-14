"""adaRMS (adaptive RMSNorm) layers for the DiT.

A conditioning vector (the flow-matching timestep embedding) modulates RMSNorm
output with a learned shift and scale. InputLayer projects tokens into the trunk
width and modulates them; FinalLayer modulates and projects out to the target dim,
and is zero-initialized so the model predicts zero at init (adaLN-Zero).

The norm is RMSNorm rather than DiT's LayerNorm, following pi0.5's action expert:
same `x * (1 + scale) + shift` modulation, no mean subtraction. Both norms are
parameterless here, so a checkpoint trained before this change still loads without
complaint while computing a different function -- re-evaluate those from the commit
that trained them.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["modulate", "bcast_cond", "make_mlp", "TimestepEmbedder", "InputLayer", "FinalLayer",
           "MLPInputLayer", "MLPFinalLayer"]


def bcast_cond(v):
    """A modulation term is either (B, D), one vector shared by every token, or (B, N, D), one
    per token. Give the shared form a length-1 token axis so both broadcast against (B, N, D).
    """
    return v.unsqueeze(1) if v.dim() == 2 else v


def modulate(x, shift, scale):
    """x: (B, N, D); shift, scale: (B, D) shared across tokens, or (B, N, D) per token."""
    return x * (1 + bcast_cond(scale)) + bcast_cond(shift)


class SwiGLULinear(nn.Module):
    """`norm`, an optional module applied to the value path before gating, must
    accept the same shape x has at call time (mind BatchNorm1d's 2D input)."""

    def __init__(self, in_dim, out_dim, norm=None):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
        self.w_gate = nn.Linear(in_dim, out_dim)
        self.norm = norm if norm is not None else nn.Identity()

    def forward(self, x):
        return self.norm(self.w(x)) * F.silu(self.w_gate(x))


def make_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(SwiGLULinear(in_dim, hidden_dim), nn.Linear(hidden_dim, out_dim))


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of a scalar timestep in [0, 1], refined by a SwiGLU MLP.

    Periods are spaced geometrically over [min_period, max_period]; the defaults are
    pi0/pi0.5/SmolVLA's, picked so the features resolve a timestep in [0, 1] -- the
    fastest channel turns 250 times across the range, the slowest a quarter turn. DiT's
    own constants assume instead that t is an integer diffusion step in [0, 1000], which
    in this parameterization is (2*pi, 2*pi*1e4) -- every period longer than [0, 1], so
    most channels sit near-constant over the range a flow timestep actually spans.
    """

    def __init__(self, dim, min_period=4e-3, max_period=4.0):
        super().__init__()
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        self.mlp = make_mlp(dim, dim, dim)

    @staticmethod
    def timestep_embedding(t, dim, min_period=4e-3, max_period=4.0):
        """t: (B,) -> (B, dim) sinusoidal features."""
        half = dim // 2
        fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32, device=t.device)
        period = min_period * (max_period / min_period) ** fraction
        args = t[:, None].float() * (2 * math.pi / period)[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        emb = self.timestep_embedding(t, self.dim, self.min_period, self.max_period)
        return self.mlp(emb.to(t.dtype))


class InputLayer(nn.Module):
    """Project tokens to hidden_size, then modulate: (B, N, in_dim), (B, cond_dim) or
    (B, N, cond_dim) -> (B, N, hidden_size). `cond_dim` defaults to hidden_size (a plain
    timestep cond); a concatenated multi-source cond is wider than the residual stream.
    """

    def __init__(self, in_dim, hidden_size, cond_dim=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_size)
        self.norm = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim or hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, cond):
        x = self.linear(x)
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class FinalLayer(nn.Module):
    """Modulate, then project to out_dim: (B, N, hidden_size), (B, cond_dim) or
    (B, N, cond_dim) -> (B, N, out_dim).

    Modulation and projection weights start at zero, so the output is zero at init.
    """

    def __init__(self, hidden_size, out_dim, cond_dim=None):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim or hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class MLPInputLayer(nn.Module):
    """Project tokens to hidden_size and fold the timestep in by concatenation:
    (B, N, in_dim), (B, hidden_size) or (B, N, hidden_size) -> (B, N, hidden_size).

    DreamZero's action encoder, itself pi0's: a linear lift, the timestep embedding
    concatenated onto it, then an MLP back down to the trunk width. `t` therefore reaches
    the tokens twice -- here, and again as the adaLN cond every block applies. InputLayer
    is the alternative, where adaLN is the only path `t` has in.
    """

    def __init__(self, in_dim, hidden_size, mlp_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_size)
        self.mlp = make_mlp(2 * hidden_size, mlp_dim, hidden_size)

    def forward(self, x, temb):
        x = self.linear(x)
        return self.mlp(torch.cat([x, bcast_cond(temb).expand_as(x)], dim=-1))


class MLPFinalLayer(nn.Module):
    """Project out through a hidden bottleneck: (B, N, hidden_size) -> (B, N, out_dim).

    DreamZero's action decoder, with no norm and no modulation -- `t` has already entered
    through MLPInputLayer and the blocks. Its output projection is zero-initialized as
    FinalLayer's is, so the predicted velocity is exactly zero at init.
    """

    def __init__(self, hidden_size, out_dim, mlp_dim):
        super().__init__()
        self.mlp = make_mlp(hidden_size, mlp_dim, out_dim)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond=None):
        return self.mlp(x)
