"""LeWAM-Unified: the working split (ViT encoder -> gc_head + dynamics) given TEMPORAL context
and moved CLOSER to a unified predictor, without re-introducing the seq's failure mode (a deep
predictor entangled with dynamics, or interleaved action tokens). See docs/LEWAM_UNIFIED_HANDOFF.

  pixels_0..t -> encoder -> [z_0, ..., z_t]
  [c_0..c_t] = [z_0..z_t] + g(z)*Aggr([z_0..z_t])    # causal aggregator, per-position, ZERO-INIT residual
  a_pred_tau = gc_head(c_tau, z_goal_tau, h_tau)      # off the shared c
  z_pred_tau = dynamics(c_tau, a_tau, z_goal_tau)     # off the SHARED c

The aggregator is a CAUSAL Transformer over the state-latent sequence: one pass emits a context
latent c_tau at EVERY position (position tau attends only to z_{<=tau}), so it trains
sequence-parallel. c_tau = z_tau + g*Aggr(...) (residual, zero-init -> boots as the split at every
position, preserving the endgame-precise direct read of z_tau). dynamics reads the same shared c
(the unification). Action history, if enabled, enters as AdaLN CONDITIONING on the previous action
a_{tau-1} (ConditionalBlock), never as sequence tokens -> no doubled length, no action-token eval
covariate shift; a_tau is never an aggregator input -> no leak.

Two entry points:
  * forward_seq(seq, z_goal, h, a_t, ...) -> (a_pred[B,L,.], z_pred[B,L,.]) : per-position, the
    sequence-parallel TRAINING path (supervise every position from one causal pass).
  * forward/gc_action(seq, z_goal, h, ...) : LAST-position readout, the reactive EVAL path (and the
    current single-decision-point trainer). gc_action is arm-agnostic (hides aggregate + action-cond).

ONE flexible model spans all ablation arms (all present so every checkpoint strict-loads under the
single eval adapter); the config the trainer writes fully determines the architecture. Flags:
  window (context length fed to the model), agg_depth, agg_heads, agg_residual, agg_gate,
  agg_action_cond, agg_max_len (positional-embedding capacity).

`GCHead`/`GoalCondDynamics` are imported from lewam_split so the heads match the split's; the only
new module is the aggregator (+ its optional action embedder / null-action / gate).
"""

import torch
from torch import nn

from lewam.models.module import MLP, Block, ConditionalBlock, ViTEncoder
from lewam.models.lewam_split import GCHead, GoalCondDynamics


class CausalStateAggregator(nn.Module):
    """Causal Transformer over a state-latent sequence [z_0..z_t] -> a per-position raw correction
    [B, L, D] (position tau depends only on z_{<=tau}). A learned absolute positional embedding
    (capacity `max_len`) is added; blocks are causal. With `zero_init` the output projection is
    zero-initialized so a residual model boots as the single-frame split (correction == 0 at init).

    If `action_cond`, each token z_tau is AdaLN-conditioned (ConditionalBlock) on the embedded
    PREVIOUS action a_{tau-1}; positions with no valid previous action (episode start / left-pad)
    use a learnable null-action embedding. Actions are CONDITIONING here, never sequence tokens."""

    def __init__(self, z_dim=192, max_len=128, depth=2, heads=4, dim_head=48,
                 mlp_dim=None, dropout=0.0, zero_init=True, action_cond=False, action_dim=25):
        super().__init__()
        self.z_dim = int(z_dim)
        self.max_len = int(max_len)
        self.action_cond = bool(action_cond)
        mlp_dim = mlp_dim or 4 * z_dim
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, z_dim))
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
            nn.init.zeros_(self.out_proj.weight)  # corr==0 at init -> residual c == z (per position)
            nn.init.zeros_(self.out_proj.bias)
        if self.action_cond:
            self.action_embed = MLP(action_dim, z_dim, z_dim, act_fn=nn.SiLU)
            self.null_action = nn.Parameter(torch.zeros(z_dim))

    def forward(self, seq, a_prev=None, a_prev_mask=None):
        """seq: (B, L, D) oldest..current. a_prev: (B, L, action_dim) previous-action per token
        (used iff action_cond); a_prev_mask: (B, L) bool, False -> null-action.
        Returns the raw correction at EVERY position: (B, L, D)."""
        L = seq.shape[1]
        assert L <= self.max_len, f"seq len {L} exceeds agg_max_len {self.max_len}"
        x = seq + self.pos_emb[:, :L]
        if self.action_cond:
            cond = self.action_embed(a_prev)                              # (B, L, D)
            cond = torch.where(a_prev_mask.unsqueeze(-1), cond,
                               self.null_action.to(cond.dtype))           # null where invalid
            for blk in self.blocks:
                x = blk(x, cond)
        else:
            for blk in self.blocks:
                x = blk(x)
        return self.out_proj(self.norm(x))


class LeWAMUnified(nn.Module):
    """Shared ViT encoder + a causal state aggregator producing a per-position context latent c
    feeding both the goal-conditioned action head and the goal-conditioned dynamics."""

    def __init__(self, encoder_size="tiny", embed_dim=192, action_dim=25, hidden_dim=512,
                 img_size=224, dropout=0.1, proj_hidden=None, window=8, agg_depth=2,
                 agg_heads=4, agg_residual=True, agg_gate=False, agg_action_cond=False,
                 agg_max_len=128):
        super().__init__()
        self.window = int(window)
        self.agg_residual = bool(agg_residual)
        self.agg_gate = bool(agg_gate) and self.agg_residual  # gate only modulates the residual
        self.agg_action_cond = bool(agg_action_cond)
        self.encoder = ViTEncoder(size=encoder_size, output_type="cls",
                                  output_dim=embed_dim, img_size=img_size,
                                  proj_hidden=proj_hidden)
        self.aggregator = CausalStateAggregator(
            z_dim=embed_dim, max_len=agg_max_len, depth=agg_depth, heads=agg_heads,
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

    def aggregate(self, seq, a_prev=None, a_prev_mask=None):
        """seq: (B, L, embed_dim) -> per-position context latent c: (B, L, embed_dim)."""
        u = self.aggregator(seq, a_prev, a_prev_mask)
        if not self.agg_residual:
            return u
        if self.agg_gate:
            u = torch.sigmoid(self.gate_proj(seq)) * u
        return seq + u

    def forward_seq(self, seq, z_goal, h_norm, a_t, a_prev=None, a_prev_mask=None):
        """Sequence-parallel training path: one causal pass -> c at every position, heads applied
        per-position. seq: (B,L,D); z_goal: (B,L,D); h_norm: (B,L); a_t: (B,L,action_dim);
        a_prev/a_prev_mask: (B,L,action_dim)/(B,L) if action_cond. Returns
        (a_pred (B,L,action_dim), z_pred (B,L,D))."""
        B, L, D = seq.shape
        c = self.aggregate(seq, a_prev, a_prev_mask)                      # (B,L,D)
        cf = c.reshape(B * L, D)                                          # flatten for the per-vector heads
        gf = z_goal.reshape(B * L, D)
        hf = h_norm.reshape(B * L)
        af = a_t.reshape(B * L, a_t.shape[-1])
        a_pred = self.gc_head(cf, gf, hf).reshape(B, L, -1)
        z_pred = self.dynamics(cf, af, gf).reshape(B, L, D)
        return a_pred, z_pred

    def gc_action(self, seq, z_goal, h_norm, a_prev=None, a_prev_mask=None):
        """Reactive LAST-position action from a sequence (the eval path; arm-agnostic).
        seq: (B,L,D); z_goal: (B,D); h_norm: (B,). Returns a_pred: (B, action_dim)."""
        c_last = self.aggregate(seq, a_prev, a_prev_mask)[:, -1]          # (B,D)
        return self.gc_head(c_last, z_goal, h_norm)

    def forward(self, seq, z_goal, h_norm, a_t, a_prev=None, a_prev_mask=None):
        """LAST-position readout (the single-decision-point path used by the current trainer/eval).
        seq: (B,L,D); z_goal: (B,D); h_norm: (B,); a_t: (B,action_dim). Returns (a_pred, z_pred)
        at the current (last) position."""
        c_last = self.aggregate(seq, a_prev, a_prev_mask)[:, -1]          # (B,D)
        a_pred = self.gc_head(c_last, z_goal, h_norm)
        z_pred = self.dynamics(c_last, a_t, z_goal)                       # shared c feeds dynamics too
        return a_pred, z_pred
