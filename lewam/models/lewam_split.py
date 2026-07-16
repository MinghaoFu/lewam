"""LeWAM-Split model: a ViT encoder (+ projector) with two separate ("split")
heads on top of its latent -- a goal-conditioned action head (GCHead) and a
goal-conditioned dynamics predictor (GoalCondDynamics). Both heads condition on
the remaining horizon via AdaLN-Zero (`AdaLNBlock`/`sinusoidal_embedding`, shared
with gcidm.py's head via `lewam.models.module`, not redefined here).

Used by `scripts/train_lewam_gc.py`:
  model = LeWAMSplit(embed_dim=192, action_dim=action_block_dim, hidden_dim=...)
  a_pred, z_pred = model(z_t, z_goal, h_norm, a_t)
Anti-collapse regularization on the encoder latent is SIGReg (also from `module`).
"""

import torch
from torch import nn

from lewam.models.module import AdaLNBlock, sinusoidal_embedding, ViTEncoder

class GCHead(nn.Module):
    """cat[state, z_goal] -> 3 AdaLN blocks -> action. `state` is one z_dim vector by
    default (the split's z_t); `state_dim` widens it so a caller can pass a concatenated
    state (e.g. the unified model's [z_t, c_t] skip). Defaulting state_dim=z_dim keeps the
    split's head byte-identical (in_dim = 2*z_dim)."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512,
                 n_freqs=64, cond_dim=128, dropout=0.1, state_dim=None):
        super().__init__()
        self.n_freqs = n_freqs
        state_dim = z_dim if state_dim is None else state_dim
        in_dim = state_dim + z_dim
        sin_dim = 2 * n_freqs
        self.horizon_mlp = nn.Sequential(
            nn.Linear(sin_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.block1 = AdaLNBlock(in_dim, hidden_dim, cond_dim, dropout)
        self.block2 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.block3 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, z_t, z_goal, h_norm):
        x = torch.cat([z_t, z_goal], dim=-1)
        cond = self.horizon_mlp(sinusoidal_embedding(h_norm, self.n_freqs))
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)
        return self.out(x)


class GoalCondDynamics(nn.Module):
    """cat[z_t, a_t, z_goal] -> z_{t+1}."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512):
        super().__init__()
        in_dim = 2 * z_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_t, a_t, z_goal):
        return self.net(torch.cat([z_t, a_t, z_goal], dim=-1))


class LeWAMSplit(nn.Module):
    """Shared ViT encoder + two independent goal-conditioned heads on its
    latent: `gc_head` (action) and `dynamics` (next-latent prediction)."""

    def __init__(self, encoder_size="tiny", embed_dim=192, action_dim=25, hidden_dim=512,
                 img_size=224, dropout=0.1, proj_hidden=None):
        super().__init__()
        self.encoder = ViTEncoder(size=encoder_size, output_type="cls",
                                  output_dim=embed_dim, img_size=img_size,
                                  proj_hidden=proj_hidden)
        self.gc_head = GCHead(z_dim=embed_dim, action_dim=action_dim,
                              hidden_dim=hidden_dim, dropout=dropout)
        self.dynamics = GoalCondDynamics(z_dim=embed_dim, action_dim=action_dim,
                                         hidden_dim=hidden_dim)

    def encode(self, pixels):
        """pixels: (N, 3, H, W) -> (N, embed_dim) cls latent."""
        return self.encoder(pixels)

    def forward(self, z_t, z_goal, h_norm, a_t):
        """z_t, z_goal: (B, embed_dim); h_norm, a_t: (B,)/(B, action_dim).
        Returns (a_pred, z_pred) from the action head and the dynamics head."""
        a_pred = self.gc_head(z_t, z_goal, h_norm)
        z_pred = self.dynamics(z_t, a_t, z_goal)
        return a_pred, z_pred
