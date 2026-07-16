"""LeWAM-Unified: the working split (ViT encoder -> gc_head + dynamics) given a short
TEMPORAL STATE-window and moved CLOSER to a unified predictor, without re-introducing the
seq's failure mode (a deep predictor entangled with dynamics, or interleaved action tokens).
See docs/LEWAM_UNIFIED_HANDOFF.md Part 2.

  pixels_{t-k..t} -> encoder -> [z_{t-k}, ..., z_t]
  c_t = z_t + Aggr([z_{t-k..t}])                # shallow causal aggregator, ZERO-INIT residual
  a_pred = gc_head(state, z_goal, h_norm)       # state = c_t  (or [z_t, c_t] if action_skip)
  z_pred = dynamics(c_t, a_t, z_goal)           # off the SHARED c_t (or z_t if dyn_on_ct=False)

ONE flexible model spans a family of ablation arms, ALL present so every checkpoint strict-loads
under the single eval adapter (gip.LeWAMUnifiedPolicy). The architecture is fully determined by
the config the trainer writes, and the loader rebuilds from it. The arms (config flags):
  * window        W: temporal context length (1 = no context / split-like control).
  * agg_depth     causal-transformer depth of the aggregator (keep SHALLOW: 1-2).
  * agg_residual  c_t = z_t + Aggr (residual, zero-init -> boots as the split) vs c_t = Aggr.
  * action_skip   action head reads [z_t, c_t] (explicit raw-z_t skip) vs just c_t.
  * dyn_on_ct     dynamics reads the shared c_t (more unified) vs raw z_t (action-only agg).

`GCHead`/`GoalCondDynamics` are imported from lewam_split so the heads match the split's; the
only new component is the aggregator (and, for action_skip, a wider gc_head input).
"""

import torch
from torch import nn

from lewam.models.module import Block, ViTEncoder
from lewam.models.lewam_split import GCHead, GoalCondDynamics


class StateWindowAggregator(nn.Module):
    """Shallow causal transformer over a fixed-length window of state latents
    [z_{t-k}, ..., z_t] -> an output at the last position. A learned positional embedding is
    added over the (fixed) window positions; blocks are causal; only the LAST position (the
    current frame) is read out. With `zero_init` the output projection is zero-initialized so
    the residual model boots as the single-frame split (c_t == z_t at init)."""

    def __init__(self, z_dim=192, window=8, depth=2, heads=4, dim_head=48,
                 mlp_dim=None, dropout=0.0, zero_init=True):
        super().__init__()
        self.window = int(window)
        self.z_dim = int(z_dim)
        mlp_dim = mlp_dim or 4 * z_dim
        self.pos_emb = nn.Parameter(torch.zeros(1, self.window, z_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([
            Block(z_dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                  dropout=dropout, causal=True)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(z_dim)
        self.out_proj = nn.Linear(z_dim, z_dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)  # corr==0 at init -> residual c_t == z_t
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, window):
        """window: (B, W, D) state latents, oldest..current. Returns (B, D) at last position."""
        x = window + self.pos_emb[:, : window.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.out_proj(self.norm(x[:, -1]))


class LeWAMUnified(nn.Module):
    """Shared ViT encoder + a shallow state-window aggregator producing one context latent
    `c_t` feeding both the goal-conditioned action head and the goal-conditioned dynamics."""

    def __init__(self, encoder_size="tiny", embed_dim=192, action_dim=25, hidden_dim=512,
                 img_size=224, dropout=0.1, proj_hidden=None, window=8, agg_depth=2,
                 agg_heads=4, agg_residual=True, action_skip=False, dyn_on_ct=True):
        super().__init__()
        self.window = int(window)
        self.agg_residual = bool(agg_residual)
        self.action_skip = bool(action_skip)
        self.dyn_on_ct = bool(dyn_on_ct)
        self.encoder = ViTEncoder(size=encoder_size, output_type="cls",
                                  output_dim=embed_dim, img_size=img_size,
                                  proj_hidden=proj_hidden)
        self.aggregator = StateWindowAggregator(
            z_dim=embed_dim, window=window, depth=agg_depth, heads=agg_heads,
            dim_head=max(embed_dim // agg_heads, 1), zero_init=self.agg_residual)
        # action head reads [z_t, c_t] (2*embed_dim state) if action_skip else c_t (embed_dim)
        act_state_dim = 2 * embed_dim if self.action_skip else embed_dim
        self.gc_head = GCHead(z_dim=embed_dim, action_dim=action_dim,
                              hidden_dim=hidden_dim, dropout=dropout, state_dim=act_state_dim)
        self.dynamics = GoalCondDynamics(z_dim=embed_dim, action_dim=action_dim,
                                         hidden_dim=hidden_dim)

    def encode(self, pixels):
        """pixels: (N, 3, H, W) -> (N, embed_dim) cls latent."""
        return self.encoder(pixels)

    def aggregate(self, window):
        """window: (B, W, embed_dim) state latents (last = current) -> c_t: (B, embed_dim)."""
        out = self.aggregator(window)
        return window[:, -1] + out if self.agg_residual else out

    def _act_state(self, window, c_t):
        """State vector fed to gc_head: [z_t, c_t] (action_skip) or c_t."""
        return torch.cat([window[:, -1], c_t], dim=-1) if self.action_skip else c_t

    def gc_action(self, window, z_goal, h_norm):
        """Reactive action from a state window. Encapsulates aggregate + skip-assembly so the
        eval adapter is arm-agnostic. window: (B, W, embed_dim)."""
        c_t = self.aggregate(window)
        return self.gc_head(self._act_state(window, c_t), z_goal, h_norm)

    def forward(self, window, z_goal, h_norm, a_t):
        """window: (B, W, embed_dim) (last = current); z_goal: (B, embed_dim);
        h_norm: (B,); a_t: (B, action_dim). Returns (a_pred, z_pred)."""
        c_t = self.aggregate(window)
        a_pred = self.gc_head(self._act_state(window, c_t), z_goal, h_norm)
        dyn_in = c_t if self.dyn_on_ct else window[:, -1]
        z_pred = self.dynamics(dyn_in, a_t, z_goal)
        return a_pred, z_pred
