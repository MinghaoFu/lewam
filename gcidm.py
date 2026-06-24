"""GC-IDM: goal-conditioned inverse dynamics on FROZEN LeWM latents (planning-free).

Faithful reproduction of `gcidm` (Nguyen/Xu/Huang, UMich, arXiv 2605.08732 --
"Latent Geometry Beyond Search: Amortizing Planning in World Models"). The GC-IDM
maps (current latent z_t, goal latent z_goal, remaining horizon h) -> next raw
action in ONE forward pass; NO test-time search (no CEM / no WM rollout). It is
trained by HINDSIGHT MSE on frozen LeWM latents.

Architecture (verbatim from the paper spec):
  - backbone input = concat [z_t || z_goal]   (2 x D, D=192 for vit-tiny)
  - 3 MLP layers, hidden 512, GELU + LayerNorm + dropout 0.1  (~1.5M params)
  - horizon conditioning = AdaLN-Zero (LOAD-BEARING; paper reports -42pp without):
      h_norm = min(steps_left, H_max)/H_max -> sinusoidal(64 freqs) -> 2-layer MLP
      -> per-MLP-layer (scale, shift), applied as
            x = LayerNorm(x) * (1 + scale) + shift
      with the (scale,shift) projection INITIALIZED TO ZERO  (AdaLN-Zero =>
      identity at init).
  - output = a single raw action vector (env action dim x frameskip block).

This module is additive: it is imported only by `train_gcidm.py` and by the
`mode=gcidm` branch of `gip.py`/`eval_gip.py`. Existing bc/guided/planning paths
are untouched.
"""

import math

import torch
from torch import nn


def _sinusoidal_embedding(h_norm: torch.Tensor, n_freqs: int = 64) -> torch.Tensor:
    """Sinusoidal embedding of a scalar in [0, 1].

    h_norm: (B,) float tensor in [0, 1]. Returns (B, 2*n_freqs).
    Mirrors the standard transformer/diffusion timestep embedding: log-spaced
    frequencies, concat(sin, cos). 64 freqs -> 128-d (paper says "64 freqs").
    """
    device = h_norm.device
    # log-spaced frequencies (1 .. 10000 style), matching DiffusionHead._temb scaling
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(n_freqs, device=device).float() / max(n_freqs - 1, 1)
    )
    ang = h_norm.float()[:, None] * freqs[None, :]  # (B, n_freqs)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)  # (B, 2*n_freqs)


class AdaLNBlock(nn.Module):
    """One GC-IDM MLP layer with AdaLN-Zero horizon conditioning.

    x_in (B, in_dim) -> LayerNorm (no affine) -> modulate by (scale, shift)
    derived from the horizon embedding -> Linear -> GELU -> Dropout -> x_out.

    The (scale, shift) come from a per-layer projection of the shared horizon
    embedding; that projection is initialized to ZERO so at init scale=shift=0
    and modulate() is the identity (AdaLN-Zero). The horizon signal therefore
    starts as a no-op and is learned in.
    """

    def __init__(self, in_dim, out_dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False, eps=1e-6)
        # per-layer (scale, shift) projection of the horizon embedding -> 2*in_dim
        self.cond_proj = nn.Linear(cond_dim, 2 * in_dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)  # AdaLN-Zero: identity at init
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond):
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)  # each (B, in_dim)
        x = self.norm(x) * (1 + scale) + shift
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class GCIDMHead(nn.Module):
    """Goal-conditioned inverse dynamics head on frozen LeWM latents.

    forward(z_t, z_goal, h_norm) -> raw action block (B, action_block_dim).
      z_t, z_goal : (B, emb_dim)   frozen LeWM latents (encode(...)['emb'][:,0])
      h_norm      : (B,)           remaining horizon, ALREADY normalized to [0,1]
                                   (= min(steps_left, H_max)/H_max). Caller owns
                                   the normalization so train and eval agree.

    3 AdaLN MLP layers (hidden 512) on concat[z_t || z_goal]; the final layer
    maps hidden -> action_dim (no activation on the output). Horizon is injected
    via AdaLN-Zero in every layer.
    """

    def __init__(self, emb_dim=192, action_dim=25, hidden_dim=512, n_freqs=64,
                 dropout=0.1, cond_dim=128):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.action_dim = int(action_dim)
        self.n_freqs = int(n_freqs)
        in_dim = 2 * self.emb_dim  # [z_t || z_goal]

        # horizon embedding: sinusoidal(64 freqs)=128-d -> 2-layer MLP -> cond_dim.
        # cond_dim kept compact (128) so the per-layer AdaLN (scale,shift)
        # projections stay small and the head lands near the paper's ~1.5M params.
        sin_dim = 2 * self.n_freqs  # 128
        self.horizon_mlp = nn.Sequential(
            nn.Linear(sin_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # 3-layer MLP backbone, hidden 512, AdaLN-Zero conditioned per layer.
        self.block1 = AdaLNBlock(in_dim, hidden_dim, cond_dim, dropout)
        self.block2 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.block3 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.out = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, z_t, z_goal, h_norm):
        x = torch.cat([z_t, z_goal], dim=-1)          # (B, 2D)
        cond = self.horizon_mlp(_sinusoidal_embedding(h_norm, self.n_freqs))  # (B, cond_dim)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)
        return self.out(x)                            # (B, action_dim)


if __name__ == "__main__":
    # Unit test: shapes + param count + AdaLN-Zero identity-at-init sanity.
    torch.manual_seed(0)
    head = GCIDMHead(emb_dim=192, action_dim=25, hidden_dim=512, n_freqs=64)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[gcidm] GCIDMHead params = {n_params:,} ({n_params/1e6:.3f}M)")
    B = 8
    z_t = torch.randn(B, 192)
    z_goal = torch.randn(B, 192)
    h = torch.rand(B)  # h_norm in [0,1]
    a = head(z_t, z_goal, h)
    print(f"[gcidm] forward: z_t{tuple(z_t.shape)} z_goal{tuple(z_goal.shape)} "
          f"h{tuple(h.shape)} -> action{tuple(a.shape)}")
    assert a.shape == (B, 25), a.shape

    # AdaLN-Zero: at init, varying horizon must NOT change the output (cond_proj=0
    # => scale=shift=0 => modulation is identity, horizon is a no-op at init).
    head.eval()
    with torch.no_grad():
        a0 = head(z_t, z_goal, torch.zeros(B))
        a1 = head(z_t, z_goal, torch.ones(B))
    max_diff = (a0 - a1).abs().max().item()
    print(f"[gcidm] AdaLN-Zero init check: max|a(h=0)-a(h=1)| = {max_diff:.2e} "
          f"(should be ~0 at init)")
    assert max_diff < 1e-5, "AdaLN-Zero not identity at init!"
    print("[gcidm] unit test PASSED")
