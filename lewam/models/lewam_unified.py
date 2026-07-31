"""LeWAM-Unified: ViT encoder -> a causal state aggregator -> a shared context latent c that feeds
both the goal-conditioned action head and the goal-conditioned dynamics. Over a context window:

  pixels_t..t+W-1 (+ per-position goal frames) -> encoder -> z_t..z_{t+W-1}
  c_tau = z_tau + g(z_tau) * Aggr(z_<=tau)          # causal aggregator, per-position residual (zero-init)
  a_pred_tau = gc_head(c_tau, z_goal_tau, h_tau)    # goal = z_{tau+h}, h ~ U[1, H_max], per position
  z_pred_tau = dynamics(c_tau, a_tau, z_goal_tau)   # dynamics reads the same shared c

The aggregator is a causal Transformer over the state-latent sequence (sinusoidal positions): one pass
emits c_tau at every position (tau attends only to z_<=tau), so training is sequence-parallel. The
residual is zero-init, so at init c_tau == z_tau and the model boots as the single-frame split. Goals
and actions are never aggregator inputs (no leak); optional action history enters as AdaLN conditioning
on a_{tau-1} (ConditionalBlock), not as extra tokens.

Two entry points:
  forward_seq(seq, z_goal, h, a_t, ...) -> (a_pred[B,L,.], z_pred[B,L,.]): sequence-parallel training,
    every position supervised from one causal pass.
  gc_action(seq, z_goal, h, ...): last-position reactive readout, the eval path (arm-agnostic).

One model spans all ablation arms so every checkpoint strict-loads under the single eval adapter; the
trainer's config fully determines the architecture. GCHead/GoalCondDynamics come from lewam_split; the
new module here is the aggregator.
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
    """Causal Transformer over a state-latent sequence z_0..z_t -> a per-position correction (B, L, D),
    position tau depending only on z_<=tau. Sinusoidal positions; causal blocks. With zero_init the
    output projection starts at zero, so a residual model boots as the single-frame split (correction 0
    at init).

    With action_cond, each token z_tau is AdaLN-conditioned (ConditionalBlock) on the embedded previous
    action a_{tau-1}; the sequence start (no valid previous action) uses a learnable null-action
    embedding. Actions are conditioning here, never sequence tokens."""

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
        """seq: (B, L, D) oldest..current. a_prev: (B, L, action_dim) previous action per token
        (used when action_cond); a_prev_mask: (B, L) bool, False -> null-action.
        Returns the correction at every position: (B, L, D)."""
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


class PrefixDynamics(nn.Module):
    """Fast-LeWM action-prefix dynamics (arXiv 2606.26217, adapted). Predicts, in parallel from a single
    anchor context, the future latent reached after each action prefix -- no autoregressive latent
    feedback, so multi-step predictions don't accumulate error. Drop-in alternative to GoalCondDynamics;
    the anchor is the aggregated context c_t (a generalization of the paper's single-frame z_t).

        ẑ_{t+k} = predictor(c_t, p_{t,k}),   p_{t,k} = prefix token over actions a_t..a_{t+k-1},  k=1..H

    Action-prefix encoder: a causal Transformer over [state_token(c_t), tok(a_t)..tok(a_{t+H-1})]. Under
    the causal mask the k-th action position attends only to the state token and a_t..a_{t+k-1}, so it
    summarizes exactly the length-k prefix (no leakage from future actions, eq 13); the 0-th (state)
    output is discarded. Predictor: concat[c_t, p_{t,k}] -> ẑ_{t+k}, the anchor re-injected alongside each
    prefix token. goal_cond adds z_goal to the predictor (ablation; default off = the forward model for
    planning).

    The k=1 output is a valid one-step transition (H-invariant by causality), so the same module serves
    autoregressive rollout (cem_dyn_mode='rollout') and parallel prefix scoring ('prefix')."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512, depth=2, heads=4,
                 dim_head=None, mlp_dim=None, dropout=0.0, goal_cond=False):
        super().__init__()
        self.z_dim = int(z_dim)
        self.goal_cond = bool(goal_cond)
        self.action_embed = MLP(action_dim, z_dim, z_dim, act_fn=nn.SiLU)   # action block -> token
        self.state_token = MLP(z_dim, z_dim, z_dim, act_fn=nn.SiLU)         # c_t -> 0-th (state) token
        dh = int(dim_head) if dim_head else max(z_dim // heads, 1)
        self.blocks = nn.ModuleList([
            Block(z_dim, heads=heads, dim_head=dh,
                  mlp_dim=(int(mlp_dim) if mlp_dim else 4 * z_dim), dropout=dropout, causal=True)
            for _ in range(int(depth))])
        self.norm = nn.LayerNorm(z_dim)
        self.predictor = MLP((3 * z_dim) if self.goal_cond else (2 * z_dim), hidden_dim, z_dim)

    def forward(self, c, action_seq, z_goal=None):
        """c: (N, D) anchor context per decision point; action_seq: (N, H, action_dim) the H action
        blocks a_t..a_{t+H-1}; z_goal: (N, D) or None. Returns (N, H, D) = [ẑ_{t+1}, .., ẑ_{t+H}]."""
        N, H, _ = action_seq.shape
        D = self.z_dim
        atok = self.action_embed(action_seq)                              # (N, H, D)
        stok = self.state_token(c).unsqueeze(1)                           # (N, 1, D)
        x = torch.cat([stok, atok], dim=1)                               # (N, H+1, D)
        x = x + sinusoidal_position_encoding(H + 1, D, x.device, x.dtype).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)                                                   # causal: position k attends to <=k
        x = self.norm(x)
        p = x[:, 1:]                                                      # (N, H, D) prefix tokens; 0-th discarded
        c_exp = c.unsqueeze(1).expand(N, H, D).to(p.dtype)                # anchor re-injected per horizon
        if self.goal_cond and z_goal is not None:
            pin = torch.cat([c_exp, p, z_goal.unsqueeze(1).expand(N, H, D).to(p.dtype)], dim=-1)
        else:
            pin = torch.cat([c_exp, p], dim=-1)
        return self.predictor(pin.reshape(N * H, -1)).reshape(N, H, D)


class LeWAMUnified(nn.Module):
    """Shared ViT encoder + a causal state aggregator producing a per-position context latent c
    feeding both the goal-conditioned action head and the goal-conditioned dynamics."""

    def __init__(self, encoder_size="tiny", embed_dim=192, action_dim=25, hidden_dim=512,
                 img_size=224, dropout=0.1, proj_hidden=None, agg_depth=4,
                 agg_heads=4, agg_residual=False, agg_gate=False, agg_action_cond=False,
                 dyn_goal_cond=True, head_type="mse", n_mix=5, flow_H=1,
                 agg_dim_head=None, agg_mlp_dim=None, dyn_action_embed_dim=0,
                 encoder_backbone="scratch", encoder_ckpt=None, use_idm=False,
                 use_prefix=False, prefix_H=5, prefix_depth=2, prefix_heads=4,
                 dyn_prefix_goal=False):
        super().__init__()
        self.agg_residual = bool(agg_residual)
        self.agg_gate = bool(agg_gate) and self.agg_residual  # gate only modulates the residual
        self.agg_action_cond = bool(agg_action_cond)
        self.encoder = ViTEncoder(size=encoder_size, output_type="cls",
                                  output_dim=embed_dim, img_size=img_size,
                                  proj_hidden=proj_hidden,
                                  backbone=encoder_backbone, backbone_ckpt=encoder_ckpt)
        # aggregator width knobs. agg_dim_head>0 decouples the attention inner dim (heads*dim_head) from
        # embed_dim, agg_mlp_dim widens the FFN -- so the aggregator can be scaled to LeWM-predictor size
        # without touching the encoder latent width.
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
            # flow-matching action-chunk head (dynamics unchanged). H=1 fits the single-block data path;
            # H>1 (chunking) needs the trainer to feed (N,H,d) targets + mask.
            from lewam.models.flow_policy_head import FlowPolicyHead
            self.gc_head = FlowPolicyHead(z_dim=embed_dim, action_dim=action_dim,
                                          hidden_dim=hidden_dim, dropout=dropout, H=self.flow_H)
        else:
            self.gc_head = GCHead(z_dim=embed_dim, action_dim=action_dim,
                                  hidden_dim=hidden_dim, dropout=dropout,
                                  head_type=head_type, n_mix=n_mix)
        # dynamics goal-conditioning is independent of the gc_head's: the policy stays goal-conditioned;
        # dyn_goal_cond=False makes only the dynamics a pure forward model. use_prefix swaps the
        # single-step GoalCondDynamics for the Fast-LeWM PrefixDynamics. Both expose a dynamics(...)
        # module; the trainer/eval pick the matching forward path.
        self.use_prefix = bool(use_prefix)
        self.prefix_H = int(prefix_H)
        if self.use_prefix:
            self.dynamics = PrefixDynamics(z_dim=embed_dim, action_dim=action_dim,
                                           hidden_dim=hidden_dim, depth=int(prefix_depth),
                                           heads=int(prefix_heads), goal_cond=bool(dyn_prefix_goal))
        else:
            self.dynamics = GoalCondDynamics(z_dim=embed_dim, action_dim=action_dim,
                                             hidden_dim=hidden_dim, goal_cond=bool(dyn_goal_cond),
                                             action_embed_dim=int(dyn_action_embed_dim))
        # inverse-dynamics head: recover a_t from (c_t, predicted z_{t+1}); c_t is pre-action, so the
        # action can only be recovered through the dynamics' prediction -> forces the dynamics to be
        # action-aware. Train-only aux loss (w_idm).
        self.idm_head = MLP(input_dim=2 * embed_dim, output_dim=action_dim,
                            hidden_dim=hidden_dim) if use_idm else None

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

        dyn_action_mix (alpha) in [0,1] feeds the dynamics a convex mix of the policy action and the
        ground-truth action: a_dyn = alpha*a_pred + (1-alpha)*a_t. alpha=0 (default) = dynamics on
        ground-truth actions; alpha>0 closes the train/rollout covariate gap (at eval the dynamics rolls
        on the policy's own actions). dyn_action_detach stop-gradients the policy action into dynamics
        (dynamics adapts to the policy's actions without the dyn loss reshaping the BC policy); otherwise
        gradients flow policy->dynamics. The action is still BC-supervised on a_pred, so this is not a
        latent-action model."""
        B, L, D = seq.shape
        c = self.aggregate(seq, a_prev, a_prev_mask)                      # (B,L,D)
        cf = c.reshape(B * L, D)                                          # flatten for the per-vector heads
        gf = z_goal.reshape(B * L, D)
        hf = h_norm.reshape(B * L)
        a_out_flat = self.gc_head(cf, gf, hf)                            # (B*L, out_dim) raw head output
        a_out = a_out_flat.reshape(B, L, -1)                             # params (gmm) or action (mse)
        af_gt = a_t.reshape(B * L, a_t.shape[-1])
        if dyn_action_mix > 0.0:
            # dynamics is fed the policy action (mixture mean for gmm, point for mse), not raw params
            a_pol = self.gc_head.point(a_out_flat)                       # (B*L, action_dim)
            a_src = a_pol.detach() if dyn_action_detach else a_pol
            af = dyn_action_mix * a_src.to(af_gt.dtype) + (1.0 - dyn_action_mix) * af_gt
        else:
            af = af_gt
        z_pred = self.dynamics(cf, af, gf).reshape(B, L, D)
        a_idm = None
        if self.idm_head is not None:                                    # inverse dynamics: (z_t, z_pred)->a_t
            # z_t (raw encoder latent), not c_t: same z-space as z_pred, and the action can only be
            # recovered via z_pred -> forces the dynamics to be action-aware.
            a_idm = self.idm_head(torch.cat([seq.reshape(B * L, D), z_pred.reshape(B * L, D)], dim=-1)
                                  ).reshape(B, L, -1)
        return a_out, z_pred, a_idm                                      # a_out: consume via gc_head.action_loss

    def forward_seq_prefix(self, state_seq, goal_seq, horizon_norm, action_prefix_seq,
                           a_prev=None, a_prev_mask=None):
        """Fast-LeWM prefix training path. One causal aggregation -> shared per-position context; the
        policy head reads it per position (as in forward_seq), and PrefixDynamics predicts all H prefix
        latents per anchor in parallel (no autoregression).
          state_seq:         (B, L, D)        z_t at each decision point (aggregator input)
          goal_seq:          (B, L, D)        per-position policy goal (gc_head; also the WM goal when
                                              dyn_prefix_goal)
          horizon_norm:      (B, L)           per-position horizon for gc_head
          action_prefix_seq: (B, L, H, adim)  per anchor, action blocks a_t..a_{t+H-1} (the WM prefix)
        Returns (action_out (B,L,.), pred_seq (B,L,H,D)); pred_seq[...,k-1] = ẑ_{t+k}. action_out is the
        raw gc_head output (consume via gc_head.action_loss on the anchor action a_t = prefix block 0)."""
        B, L, D = state_seq.shape
        H, action_dim = action_prefix_seq.shape[2], action_prefix_seq.shape[3]
        ctx_seq = self.aggregate(state_seq, a_prev, a_prev_mask)          # (B, L, D) shared context c_t
        ctx_flat = ctx_seq.reshape(B * L, D)
        action_out = self.gc_head(ctx_flat, goal_seq.reshape(B * L, D),
                                  horizon_norm.reshape(B * L)).reshape(B, L, -1)
        goal_flat = goal_seq.reshape(B * L, D) if self.dynamics.goal_cond else None
        pred_seq = self.dynamics(ctx_flat, action_prefix_seq.reshape(B * L, H, action_dim),
                                 goal_flat).reshape(B, L, H, D)
        return action_out, pred_seq

    def rollout_dyn(self, states, z_pred, next_tgt, z_goal, actions, n_pos, K,
                    a_prev=None, a_prev_mask=None):
        """V-JEPA2-style K-step latent rollout for the dynamics, trained alongside teacher forcing
        (L_dyn = L_tf + L_rollout). Feeds the model's own predicted latent back in for up to K-1 steps
        and supervises the deeper predictions against the real future latents, targeting compounding
        (exposure bias). Only the state is rolled; actions and goals stay ground truth, isolating
        state-compounding from action-awareness. Full BPTT -> depth-1 preds are pushed to be good
        rollout inputs too.

        states:   (B, L, D)  real z_p at each decision point (the aggregator input, p=0..L-1)
        z_pred:   (B, L, D)  depth-1 teacher-forced prediction ẑ_{p+1} (reused from forward_seq)
        next_tgt: (B, L, D)  real z_{p+1} target (z_window[:, 1:])
        z_goal:   (B, L, D)  per-position goal latent;  actions: (B, L, adim);  n_pos: (B,)
        K:        rollout length (K=2 == ONE rollout step; K<=1 -> no rollout, returns 0)

        Anchor t at depth s (s=1..K-1) predicts ẑ_{t+s+1} from the causal sequence
        [z_0..z_t, ẑ_{t+1}..ẑ_{t+s}] (real prefix + the s predicted states) with the REAL action
        a_{t+s} and goal g_{t+s}; target = real z_{t+s+1}. Anchors are masked to (t+s < n_pos).
        Returns a scalar per-element MSE over all valid (anchor, depth) terms (same scale as L_tf)."""
        B, L, D = states.shape
        dev = states.device
        if K <= 1 or L <= 1:
            return states.new_zeros(())
        idx = torch.arange(L, device=dev)
        # a_prev is per-token (B,L,.), identical across anchors -> broadcast onto the anchor batch
        ap_e = am_e = None
        if a_prev is not None:
            ap_e = a_prev.unsqueeze(1).expand(B, L, L, a_prev.shape[-1]).reshape(B * L, L, -1)
            am_e = a_prev_mask.unsqueeze(1).expand(B, L, L).reshape(B * L, L)
        frontier = [z_pred]                                # frontier[j-1][:, t] = ẑ_{t+j}
        tot_sq = states.new_zeros(())
        tot_n = states.new_zeros(())
        for s in range(1, K):
            # per-anchor sequence copy: seq[:, t] starts as the real states, then positions t+1..t+s
            # are overwritten with this anchor's own rollout predictions (causal read at t+s ignores
            # the untouched positions > t+s, so the real states left there are inert).
            seq = states.unsqueeze(1).expand(B, L, L, D).clone()      # (B, anchor, pos, D)
            for j in range(1, s + 1):
                keep = (idx + j) <= (L - 1)                           # in-range positions only
                ai = idx[keep]
                # frontier preds come out of the autocast dynamics (bf16); seq is fp32 (states.float()).
                # index_put requires matching dtypes -> cast the source to seq's dtype.
                seq[:, ai, ai + j, :] = frontier[j - 1][:, ai, :].to(seq.dtype)
            c = self.aggregate(seq.reshape(B * L, L, D), ap_e, am_e).reshape(B, L, L, D)
            rp = (idx + s).clamp(max=L - 1)                           # read position t+s (clamp: masked anyway)
            c_read = c[:, idx, rp, :]                                 # (B, L, D)  anchor t reads pos t+s
            z_next = self.dynamics(c_read.reshape(B * L, D),
                                   actions[:, rp, :].reshape(B * L, actions.shape[-1]),
                                   z_goal[:, rp, :].reshape(B * L, D)).reshape(B, L, D)
            frontier.append(z_next)                                  # feed back for depth s+1
            m = ((idx.unsqueeze(0) + s) < n_pos.unsqueeze(1)).float()  # (B, L) valid anchors
            tot_sq = tot_sq + ((z_next - next_tgt[:, rp, :]) ** 2 * m.unsqueeze(-1)).sum()
            tot_n = tot_n + m.sum()
        return tot_sq / (tot_n.clamp(min=1) * D)

    def gc_action(self, seq, z_goal, h_norm, a_prev=None, a_prev_mask=None):
        """Reactive LAST-position action from a sequence (the eval path; arm-agnostic).
        seq: (B,L,D); z_goal: (B,D); h_norm: (B,). Returns a_pred: (B, action_dim)."""
        c_last = self.aggregate(seq, a_prev, a_prev_mask)[:, -1]          # (B,D)
        return self.gc_head.point(self.gc_head(c_last, z_goal, h_norm))   # point action (mixture mean for gmm)
