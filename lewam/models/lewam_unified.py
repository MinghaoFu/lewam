"""LeWAM-Unified: the working split (ViT encoder -> gc_head + dynamics) given TEMPORAL context
and moved CLOSER to a unified predictor, without re-introducing the seq's failure mode (a deep
predictor entangled with dynamics, or interleaved action tokens). See docs/LEWAM_UNIFIED_HANDOFF.

  pixels_t..t+W-1 (+ goal-tail frames) -> encoder -> [z_t, ..]
  [c_t..c_{t+W-1}] = z + g(z)*Aggr(z)                # causal aggregator, per-position, ZERO-INIT residual
  a_pred_tau = gc_head(c_tau, z_goal_tau, h_tau)      # goal PER POSITION: z_{tau+h}, h~U[1,H_max];
  z_pred_tau = dynamics(c_tau, a_tau, z_goal_tau)     #   NEVER an aggregator input. Off the SHARED c

The aggregator is a CAUSAL Transformer over the state-latent sequence (sinusoidal positions): one
pass emits a context latent c_tau at EVERY position (position tau attends only to z_{<=tau}), so it
trains sequence-parallel. c_tau = z_tau + g*Aggr(...) (residual, zero-init -> boots as the split at
every position, preserving the endgame-precise direct read of z_tau). dynamics reads the same
shared c (the unification). Action history, if enabled, enters as AdaLN CONDITIONING on the previous
action a_{tau-1} (ConditionalBlock), never as sequence tokens -> no doubled length, no action-token
eval covariate shift; a_tau is never an aggregator input -> no leak.

Two entry points:
  * forward_seq(seq, z_goal, h, a_t, ...) -> (a_pred[B,L,.], z_pred[B,L,.]) : per-position, the
    sequence-parallel TRAINING path (supervise every position from one causal pass).
  * gc_action(seq, z_goal, h, ...) : LAST-position readout, the reactive EVAL path (arm-agnostic;
    hides aggregate + action-cond).

ONE flexible model spans all ablation arms (all present so every checkpoint strict-loads under the
single eval adapter); the config the trainer writes fully determines the architecture. Flags:
  agg_depth, agg_heads, agg_residual, agg_gate, agg_action_cond.

`GCHead`/`GoalCondDynamics` are imported from lewam_split so the heads match the split's; the only
new module is the aggregator (+ its optional action embedder / null-action / gate).
"""

import math

import torch
from torch import nn

from lewam.models.module import MLP, Block, ConditionalBlock, ViTEncoder
from lewam.models.lewam_split import GCHead, GoalCondDynamics


def sinusoidal_position_encoding(length, dim, device, dtype):
    """Standard transformer sinusoidal positional encoding -> (length, dim). No parameters and no
    max-length cap (works for any sequence length; positions are absolute from the sequence start,
    which is a fresh start == the eval episode start)."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)   # (L,1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))                                  # (dim/2,)
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class CausalStateAggregator(nn.Module):
    """Causal Transformer over a state-latent sequence [z_0..z_t] -> a per-position raw correction
    [B, L, D] (position tau depends only on z_{<=tau}). Sinusoidal positional encoding; blocks are
    causal. With `zero_init` the output projection is zero-initialized so a residual model boots as
    the single-frame split (correction == 0 at init).

    If `action_cond`, each token z_tau is AdaLN-conditioned (ConditionalBlock) on the embedded
    PREVIOUS action a_{tau-1}; positions with no valid previous action (sequence start) use a
    learnable null-action embedding. Actions are CONDITIONING here, never sequence tokens."""

    def __init__(self, z_dim=192, depth=2, heads=4, dim_head=48,
                 mlp_dim=None, dropout=0.0, zero_init=True, action_cond=False, action_dim=25):
        super().__init__()
        self.z_dim = int(z_dim)
        self.action_cond = bool(action_cond)
        mlp_dim = mlp_dim or 4 * z_dim
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
        x = seq + sinusoidal_position_encoding(L, self.z_dim, seq.device, seq.dtype).unsqueeze(0)
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
                 img_size=224, dropout=0.1, proj_hidden=None, agg_depth=4,
                 agg_heads=4, agg_residual=False, agg_gate=False, agg_action_cond=False,
                 dyn_goal_cond=True, head_type="mse", n_mix=5, flow_H=1,
                 agg_dim_head=None, agg_mlp_dim=None, dyn_action_embed_dim=0):
        super().__init__()
        self.agg_residual = bool(agg_residual)
        self.agg_gate = bool(agg_gate) and self.agg_residual  # gate only modulates the residual
        self.agg_action_cond = bool(agg_action_cond)
        self.encoder = ViTEncoder(size=encoder_size, output_type="cls",
                                  output_dim=embed_dim, img_size=img_size,
                                  proj_hidden=proj_hidden)
        # aggregator width knobs (default = the tied MHA of the 9.3M baseline). agg_dim_head>0
        # DECOUPLES the attention inner dim (heads*dim_head) from embed_dim, and agg_mlp_dim widens
        # the FFN -- so the aggregator (our "predictor") can be scaled up (heads/head-dim/width) to
        # LeWM-predictor size WITHOUT touching the encoder latent width or the heads.
        _agg_dh = int(agg_dim_head) if agg_dim_head else max(embed_dim // agg_heads, 1)
        self.aggregator = CausalStateAggregator(
            z_dim=embed_dim, depth=agg_depth, heads=agg_heads,
            dim_head=_agg_dh, mlp_dim=(int(agg_mlp_dim) if agg_mlp_dim else None),
            zero_init=self.agg_residual,
            action_cond=self.agg_action_cond, action_dim=action_dim)
        if self.agg_gate:
            # input-dependent sigmoid gate on the correction; bias zero-init -> g == 0.5 at init.
            self.gate_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.gate_proj.bias)
        self.flow_H = int(flow_H)
        if head_type == "flow":
            # generative flow-matching action-chunk head (deterministic dynamics unchanged). Drop-in:
            # forward->cond, action_loss(cond,tgt), point/sample via few-step ODE. H=1 fits the current
            # single-block data path; H>1 (chunking) needs the trainer to feed (N,H,d) targets + mask.
            from lewam.models.flow_policy_head import FlowPolicyHead
            self.gc_head = FlowPolicyHead(z_dim=embed_dim, action_dim=action_dim,
                                          hidden_dim=hidden_dim, dropout=dropout, H=self.flow_H)
        else:
            self.gc_head = GCHead(z_dim=embed_dim, action_dim=action_dim,
                                  hidden_dim=hidden_dim, dropout=dropout,
                                  head_type=head_type, n_mix=n_mix)
        # dynamics goal-conditioning is independent of the gc_head's: the reactive policy stays
        # goal-conditioned; dyn_goal_cond=False makes ONLY the dynamics a pure forward model.
        self.dynamics = GoalCondDynamics(z_dim=embed_dim, action_dim=action_dim,
                                         hidden_dim=hidden_dim, goal_cond=bool(dyn_goal_cond),
                                         action_embed_dim=int(dyn_action_embed_dim))

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

    def forward_seq(self, seq, z_goal, h_norm, a_t, a_prev=None, a_prev_mask=None,
                    dyn_action_mix=0.0, dyn_action_detach=False):
        """Sequence-parallel training path: one causal pass -> c at every position, heads applied
        per-position. seq: (B,L,D); z_goal: (B,L,D); h_norm: (B,L); a_t: (B,L,action_dim);
        a_prev/a_prev_mask: (B,L,action_dim)/(B,L) if action_cond. Returns
        (a_pred (B,L,action_dim), z_pred (B,L,D)).

        dyn_action_mix (alpha) in [0,1] feeds the dynamics head a convex mix of the POLICY action
        and the ground-truth action: a_dyn = alpha*a_pred + (1-alpha)*a_t. alpha=0 (default) is the
        original split behaviour (dynamics on ground-truth actions). alpha>0 closes the train/rollout
        covariate gap -- at eval the dynamics is rolled on the policy's own actions, never on ground
        truth. With dyn_action_detach the policy action enters dynamics stop-gradient (dynamics adapts
        to the policy's action distribution, but the dyn loss never reshapes the BC policy); otherwise
        gradients flow policy->dynamics (the coupled 'unified' arm). The action is still supervised by
        the separate BC loss on a_pred, so this is NOT a latent-action model."""
        B, L, D = seq.shape
        c = self.aggregate(seq, a_prev, a_prev_mask)                      # (B,L,D)
        cf = c.reshape(B * L, D)                                          # flatten for the per-vector heads
        gf = z_goal.reshape(B * L, D)
        hf = h_norm.reshape(B * L)
        a_out_flat = self.gc_head(cf, gf, hf)                            # (B*L, out_dim) RAW head output
        a_out = a_out_flat.reshape(B, L, -1)                             # params (gmm) or action (mse)
        af_gt = a_t.reshape(B * L, a_t.shape[-1])
        if dyn_action_mix > 0.0:
            # dynamics is fed the actual POLICY ACTION (mixture mean for gmm, point for mse), never raw params
            a_pol = self.gc_head.point(a_out_flat)                       # (B*L, action_dim)
            a_src = a_pol.detach() if dyn_action_detach else a_pol
            af = dyn_action_mix * a_src.to(af_gt.dtype) + (1.0 - dyn_action_mix) * af_gt
        else:
            af = af_gt
        z_pred = self.dynamics(cf, af, gf).reshape(B, L, D)
        return a_out, z_pred                                             # a_out: consume via gc_head.action_loss

    def gc_action(self, seq, z_goal, h_norm, a_prev=None, a_prev_mask=None):
        """Reactive LAST-position action from a sequence (the eval path; arm-agnostic).
        seq: (B,L,D); z_goal: (B,D); h_norm: (B,). Returns a_pred: (B, action_dim)."""
        c_last = self.aggregate(seq, a_prev, a_prev_mask)[:, -1]          # (B,D)
        return self.gc_head.point(self.gc_head(c_last, z_goal, h_norm))   # point action (mixture mean for gmm)
