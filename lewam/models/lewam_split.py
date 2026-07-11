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
from stable_pretraining.backbone.utils import vit_hf

from lewam.models.module import AdaLNBlock, MLP, sinusoidal_embedding


class GCHead(nn.Module):
    """cat[z_t, z_goal] -> 3 AdaLN blocks -> action."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512,
                 n_freqs=64, cond_dim=128, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 2 * z_dim
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


class ViTEncoder(nn.Module):
    """ViT encoder + BN-MLP projector -> [CLS] latent"""

    def __init__(self, size="tiny", output_dim=192, img_size=224):
        super().__init__()
        self.vit = vit_hf(size=size, patch_size=14, image_size=img_size,
                              pretrained=False, use_mask_token=False)
        self.projector = MLP(input_dim=self.vit.config.hidden_size, output_dim=output_dim,
                             hidden_dim=2048, norm_fn=nn.BatchNorm1d)

    def forward(self, pixels):
        """pixels: (N, 3, H, W) -> (N, embed_dim) cls latent."""
        out = self.vit(pixels, interpolate_pos_encoding=True)
        return self.projector(out.last_hidden_state[:, 0])


class LeWAMSplit(nn.Module):
    """Shared ViT encoder + two independent goal-conditioned heads on its
    latent: `gc_head` (action) and `dynamics` (next-latent prediction)."""

    def __init__(self, embed_dim=192, action_dim=25, hidden_dim=512,
                 img_size=224, dropout=0.1):
        super().__init__()
        self.encoder = ViTEncoder(output_dim=embed_dim, img_size=img_size)
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
