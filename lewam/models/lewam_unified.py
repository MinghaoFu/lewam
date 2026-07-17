"""LeWAM-Unified: the working split (ViT encoder -> gc_head + dynamics) given a short
TEMPORAL STATE-window and moved CLOSER to a unified predictor, without re-introducing the
seq's failure mode (a deep predictor entangled with dynamics, or interleaved action tokens).
See docs/LEWAM_UNIFIED_HANDOFF.md Part 2.

  pixels_{t-k..t} -> encoder -> [z_{t-k}, ..., z_t]
  c_t = z_t + g(z_t) * Aggr([z_{t-k..t}])       # shallow causal aggregator, ZERO-INIT residual
  a_pred = gc_head(c_t, z_goal, h_norm)          # off the shared c_t
  z_pred = dynamics(c_t, a_t, z_goal)            # off the SHARED c_t (or z_t if dyn_on_zt)

The action head reads c_t, which is `z_t + gate*Aggr(win)` (residual) or `Aggr(win)`
(no-residual) -- i.e. the residual flag IS the "include the raw z_t read" lever, and the
aggregator is always used. dynamics reads the same shared c_t (unification); `dyn_on_zt`
reverts it to the split's raw-z_t dynamics (a labeled control).

ONE flexible model spans a family of ablation arms, ALL present so every checkpoint
strict-loads under the single eval adapter (gip.LeWAMUnifiedPolicy); the architecture is fully
determined by the config the trainer writes, and the loader rebuilds from it. Arms (flags):
  * window        W: temporal context length (1 = no context / split-like control).
  * agg_depth     causal-transformer depth of the aggregator (keep SHALLOW: 1-2).
  * agg_residual  c_t = z_t + Aggr (residual, zero-init -> boots as the split) vs c_t = Aggr.
  * agg_gate      input-dependent sigmoid gate on the residual correction (still boots as split).
  * agg_action_cond  each window token z_tau conditions (AdaLN, ConditionalBlock) on the
                  embedded PREVIOUS action a_{tau-1} (learnable null-action at episode start);
                  actions enter as CONDITIONING, not as sequence tokens (no eval covariate
                  shift from action tokens, no doubled sequence length).

Both read-outs share c_t (the unification): the action head reads it, and the dynamics head
reads it too.

`GCHead`/`GoalCondDynamics` are imported from lewam_split so the heads match the split's; the
only new module is the aggregator (+ its optional action embedder / null-action / gate).
"""

import torch
from torch import nn

from lewam.models.module import MLP, Block, ConditionalBlock, ViTEncoder
from lewam.models.lewam_split import GCHead, GoalCondDynamics


class StateWindowAggregator(nn.Module):
    """Shallow causal transformer over a fixed-length window of state latents
    [z_{t-k}, ..., z_t] -> a raw correction at the last position. A learned positional embedding
    is added over the (fixed) window positions; blocks are causal; only the LAST position (the
    current frame) is read out. With `zero_init` the output projection is zero-initialized so a
    residual model boots as the single-frame split (correction == 0 -> c_t == z_t at init).

    If `action_cond`, each token z_tau is AdaLN-conditioned (ConditionalBlock) on the embedded
    PREVIOUS action a_{tau-1}; positions with no valid previous action (episode start / left-pad)
    use a learnable null-action embedding. Actions are CONDITIONING here, never sequence tokens."""

    def __init__(self, z_dim=192, window=8, depth=2, heads=4, dim_head=48,
                 mlp_dim=None, dropout=0.0, zero_init=True, action_cond=False, action_dim=25):
        super().__init__()
        self.window = int(window)
        self.z_dim = int(z_dim)
        self.action_cond = bool(action_cond)
        mlp_dim = mlp_dim or 4 * z_dim
        self.pos_emb = nn.Parameter(torch.zeros(1, self.window, z_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        block_cls = ConditionalBlock if self.action_cond else Block
        self.blocks = nn.ModuleList([
            block_cls(z_dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                      dropout=dropout, causal=True)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(z_dim)
        self.out_proj = nn.Linear(z_dim, z_dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)  # corr==0 at init -> residual c_t == z_t
            nn.init.zeros_(self.out_proj.bias)
        if self.action_cond:
            self.action_embed = MLP(action_dim, z_dim, z_dim, act_fn=nn.SiLU)
            self.null_action = nn.Parameter(torch.zeros(z_dim))

    def forward(self, window, a_prev=None, a_prev_mask=None):
        """window: (B, W, D) oldest..current. a_prev: (B, W, action_dim) previous-action blocks
        aligned to each token (only used if action_cond); a_prev_mask: (B, W) bool, False where
        there is no valid previous action (-> null-action). Returns (B, D) at the last position."""
        x = window + self.pos_emb[:, : window.shape[1]]
        if self.action_cond:
            cond = self.action_embed(a_prev)                              # (B, W, D)
            cond = torch.where(a_prev_mask.unsqueeze(-1), cond,
                               self.null_action.to(cond.dtype))           # null where invalid
            for blk in self.blocks:
                x = blk(x, cond)
        else:
            for blk in self.blocks:
                x = blk(x)
        return self.out_proj(self.norm(x[:, -1]))


class LeWAMUnified(nn.Module):
    """Shared ViT encoder + a shallow state-window aggregator producing one context latent
    `c_t` feeding both the goal-conditioned action head and the goal-conditioned dynamics."""

    def __init__(self, encoder_size="tiny", embed_dim=192, action_dim=25, hidden_dim=512,
                 img_size=224, dropout=0.1, proj_hidden=None, window=8, agg_depth=2,
                 agg_heads=4, agg_residual=True, agg_gate=False, agg_action_cond=False):
        super().__init__()
        self.window = int(window)
        self.agg_residual = bool(agg_residual)
        self.agg_gate = bool(agg_gate) and self.agg_residual  # gate only modulates the residual
        self.agg_action_cond = bool(agg_action_cond)
        self.encoder = ViTEncoder(size=encoder_size, output_type="cls",
                                  output_dim=embed_dim, img_size=img_size,
                                  proj_hidden=proj_hidden)
        self.aggregator = StateWindowAggregator(
            z_dim=embed_dim, window=window, depth=agg_depth, heads=agg_heads,
            dim_head=max(embed_dim // agg_heads, 1), zero_init=self.agg_residual,
            action_cond=self.agg_action_cond, action_dim=action_dim)
        if self.agg_gate:
            # input-dependent sigmoid gate on the correction; bias zero-init -> g == 0.5 at init.
            self.gate_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.gate_proj.bias)
        self.gc_head = GCHead(z_dim=embed_dim, action_dim=action_dim,
                              hidden_dim=hidden_dim, dropout=dropout)
        self.dynamics = GoalCondDynamics(z_dim=embed_dim, action_dim=action_dim,
                                         hidden_dim=hidden_dim)

    def encode(self, pixels):
        """pixels: (N, 3, H, W) -> (N, embed_dim) cls latent."""
        return self.encoder(pixels)

    def aggregate(self, window, a_prev=None, a_prev_mask=None):
        """window: (B, W, embed_dim) (last = current) -> c_t: (B, embed_dim)."""
        u = self.aggregator(window, a_prev, a_prev_mask)
        if not self.agg_residual:
            return u
        z_t = window[:, -1]
        if self.agg_gate:
            u = torch.sigmoid(self.gate_proj(z_t)) * u
        return z_t + u

    def gc_action(self, window, z_goal, h_norm, a_prev=None, a_prev_mask=None):
        """Reactive action from a state window. Encapsulates aggregate() so the eval adapter is
        arm-agnostic. window: (B, W, embed_dim)."""
        c_t = self.aggregate(window, a_prev, a_prev_mask)
        return self.gc_head(c_t, z_goal, h_norm)

    def forward(self, window, z_goal, h_norm, a_t, a_prev=None, a_prev_mask=None):
        """window: (B, W, embed_dim) (last = current); z_goal: (B, embed_dim); h_norm: (B,);
        a_t: (B, action_dim); a_prev/a_prev_mask: previous-action conditioning (agg_action_cond).
        Returns (a_pred, z_pred)."""
        c_t = self.aggregate(window, a_prev, a_prev_mask)
        a_pred = self.gc_head(c_t, z_goal, h_norm)
        z_pred = self.dynamics(c_t, a_t, z_goal)  # shared c_t feeds the dynamics too
        return a_pred, z_pred
