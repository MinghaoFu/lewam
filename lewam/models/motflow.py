"""MoT-flow (owner design 2026-08-28): a Mixture-of-Transformers joint model. Two streams with
separate parameters (state stream: LN/QKV/out/FFN/AdaLN; action stream: the same), one global
self-attention over all tokens with a designed mask, so the policy and the dynamics exchange
information only where intended and never through a target.

Tokens
  state stream : z_hist (history latents, clean), z* (S state slots: the next-state prediction -- flow-noised tokens or MSE queries),
                 z_g (goal token, optional)
  action stream: a (clean a_1..a_{S*fs}: exactly the actions the state tokens depend on),
                 a* (A action slots: the next-action prediction -- flow-noised tokens or MSE queries)

Attention map (row attends to columns)
  z_hist -> z_hist
  z*_q   -> z_hist, z*_{<=q}, a_{<= q*fs}
  z_g    -> z_hist, z_g
  a_j    -> z_hist, a_{<=j}
  a*_j   -> z_hist, z_g, a*_{<=j}
a* never sees the clean actions or z* (no target leakage into the policy); z* never sees a* or
z_g (clean-action conditioning, goal-free dynamics); z_hist sees no target.

Views (num_views = N > 1): every camera is encoded by the same encoder and gets a learned view
embedding. The history holds N*H tokens (view-major: view 0's frames, then view 1's, ...), and the
policy reads all of them. The dynamics predicts one latent per view and step: N*S state slots,
slot (v, q) predicting view v at step q. Its map keeps the chain in q and reads every view:
  z*_{v,q} -> z_hist (all views), z*_{v',q'} for every v' and q' <= q, a_{<= q*fs}
With N = 1 the layout and the map are the single-view ones above.

Conditioning: tau_a via AdaLN on a* only, tau_s on z* only, h on a* only; clean tokens get the
fixed tau=1 embedding. The goal reaches the policy as the z_g token, not through the readout.

goal_cond=head (owner 2026-09-05; jointflow's arrangement): no z_g token and no h term anywhere in
the trunk -- the action stream is goal- and horizon-free -- and the action readout is GCHeadMSE
(per-token feature ++ goal latent through AdaLN-Zero MLP layers modulated by the horizon), so goal
and horizon condition ONLY the decoding head.

Losses: rectified flow on a*; on z* either rectified flow (`state_head=flow`) or MSE with a
learned query token (`state_head=mse`, clean-action-conditioned regression).

Inference (two phases through the same stack): Euler on a* (z*, a inert by the mask); then
a := the sampled first block(s) and Euler (or one pass) on z*. sample_inpaint = phase 2 alone.
Interface = jointflow's, so the trainer, eval adapters and probes run unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewam.models.module import VisionEncoder, GCHeadMSE, sinusoid


class MoTBlock(nn.Module):
    """Two-stream transformer block with one joint masked self-attention over the union of tokens.
    Token order: [history | state slots | goal | clean actions | action slots]. The first n_state tokens
    are the state stream, the rest the action stream; each stream has its own QKV / output / MLP.
    "Slots" are the tokens where a branch's prediction happens: the state slots hold the next-state
    prediction, the action slots the next-action prediction. Conditioning is classic AdaLN-Zero (shift,
    scale, gate for the attention and the MLP sub-layers) and reaches ONLY the slots of a branch that
    has it: a flow branch's tau, and the horizon on the action slots under goal reaching. Every other
    token -- history, goal, clean actions -- and every slot without conditioning is the plain pre-LN
    block: LayerNorm, no shift, no scale, gate 1."""

    def __init__(self, dim, n_heads, cond_state, cond_action, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim, self.n_heads, self.attn_dropout = dim, n_heads, dropout
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv_state, self.qkv_action = nn.Linear(dim, 3 * dim), nn.Linear(dim, 3 * dim)
        self.out_state, self.out_action = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.mlp_state, self.mlp_action = self._mlp(dim, mlp_ratio, dropout), self._mlp(dim, mlp_ratio, dropout)
        self.ada_state = self._adaln_zero(dim) if cond_state else None
        self.ada_action = self._adaln_zero(dim) if cond_action else None

    @staticmethod
    def _mlp(dim, mlp_ratio, dropout):
        return nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(mlp_ratio * dim, dim))

    @staticmethod
    def _adaln_zero(dim):
        ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(ada[-1].weight)
        nn.init.zeros_(ada[-1].bias)
        return ada

    @staticmethod
    def _pre_norm(norm, x, shift, scale):
        return norm(x) * (1 + scale) + shift

    def _modulation(self, x, layout, cond_state, cond_action):
        """The six (B, n, dim) AdaLN tensors -- shift, scale, gate for the attention, then for the mlp.
        Identity on every token (shift 0, scale 0, gate 1); the slots of a conditioned branch are filled
        from that branch's AdaLN, whose output is chunked in the same order."""
        _, state_slots, action_slots = layout
        shift_attn, scale_attn, gate_attn = torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)
        shift_mlp, scale_mlp, gate_mlp = torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)
        modulations = (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        for cond, ada, slots in ((cond_state, self.ada_state, state_slots), (cond_action, self.ada_action, action_slots)):
            if cond is None:
                continue
            for full, slot_values in zip(modulations, ada(cond)[:, None].chunk(6, dim=-1)):
                full[:, slots] = slot_values
        return modulations

    @staticmethod
    def _per_stream(x, n_state, f_state, f_action):
        return torch.cat([f_state(x[:, :n_state]), f_action(x[:, n_state:])], dim=1)

    def forward(self, x, mask, layout, cond_state, cond_action):
        """x (B, n, dim); mask (B, 1, n, n) additive (0 / -inf); layout = (n_state, state_slots, action_slots);
        cond_state / cond_action (B, dim) or None."""
        B, n, d = x.shape
        n_state = layout[0]
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            self._modulation(x, layout, cond_state, cond_action)
        h = self._pre_norm(self.norm_attn, x, shift_attn, scale_attn)
        qkv = self._per_stream(h, n_state, self.qkv_state, self.qkv_action)
        q, k, v = qkv.reshape(B, n, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.attn_dropout if self.training else 0.0)
        att = att.transpose(1, 2).reshape(B, n, d)
        x = x + gate_attn * self._per_stream(att, n_state, self.out_state, self.out_action)
        h = self._pre_norm(self.norm_mlp, x, shift_mlp, scale_mlp)
        return x + gate_mlp * self._per_stream(h, n_state, self.mlp_state, self.mlp_action)


class MoTFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_dim = cfg["z_dim"]
        self.fs = cfg["fs"]
        self.num_actions = cfg["num_actions_pred"]
        self.num_states = cfg["num_states_pred"]
        assert self.num_actions >= self.num_states * self.fs, "num_actions_pred must cover num_states_pred*fs"
        self.action_raw_dim = cfg["action_raw_dim"]
        self.history_len = cfg["policy_history_len"]
        self.n_flow_steps = cfg["n_flow_steps"]
        self.dim = cfg["d_model"]
        self.goal_conditioning = bool(cfg.get("goal_conditioning", False))
        # goal_cond: "token" = goal token in the trunk + horizon AdaLN on the action slots
        # (the design above); "head" = the trunk never sees goal or horizon, both condition only
        # the GCHeadMSE action readout (owner 2026-09-05). Old checkpoints lack the key -> token.
        self.goal_cond = str(cfg.get("goal_cond", "token"))
        assert self.goal_cond in ("token", "head"), self.goal_cond
        assert not (self.goal_cond == "head" and not self.goal_conditioning), \
            "goal_cond=head needs goal_conditioning"
        self.goal_token = self.goal_conditioning and self.goal_cond == "token"
        # sep_policy_state (owner design 2026-09-01): the action branch attends ONLY to a
        # projected view of the latent -- every history token gets a projected copy that feeds
        # the policy, and the goal latent is projected in place (the state stream never
        # consumes it). The policy's gradient reaches z only through this projection; the
        # dynamics owns z raw.
        self.sep_policy_state = bool(cfg.get("sep_policy_state", False))
        # SIGReg options live in the model so the projection checkpoints with it. Two
        # independent features (owner 2026-09-01): sigreg_pertime = SIGReg per latent group
        # (each history slot, each state target, the goal as policy context), losses averaged;
        # sigreg_proj_dim > 0 = SIGReg sees a learned z_dim -> proj_dim linear of each latent
        # (colleague recipe): z itself no longer has to be white -- the projection can whiten
        # any full-rank z -- while a full-rank output keeps the anti-collapse pressure.
        # Old checkpoints lack both keys and get the original pooled, unprojected loss.
        self.sigreg_pertime = bool(cfg.get("sigreg_pertime", False))
        self.sigreg_proj_dim = int(cfg.get("sigreg_proj_dim", 0))
        self.state_head = str(cfg.get("state_head", "flow"))
        assert self.state_head in ("flow", "mse")
        # action chunk objective, the mirror of state_head: flow = rectified flow on the action slots (sampled);
        # mse = regression from a learned query (action_query) read out in one pass at tau one (deterministic,
        # the conditional mean).
        self.action_head = str(cfg.get("action_head", "flow"))
        assert self.action_head in ("flow", "mse")
        self.rollout_only = True        # planning = autoregressive rollout over predict_state, nothing else
        self.state_residual = bool(cfg.get("state_residual", False))
        # state_prior: "gauss" (x_0 ~ N(0, I)) or "prev" (x_0 = z_t, the current latent: the flow
        # learns the displacement to z_{t+fs} along a straight path; owner 2026-08-29)
        self.state_prior = str(cfg.get("state_prior", "gauss"))
        assert self.state_prior in ("gauss", "prev"), self.state_prior
        self.state_prior_sigma = float(cfg.get("state_prior_sigma", 0.0))     # relative to rms(z_t)
        # state_param: "v" (the state token outputs the velocity) or "x" (JiT-style: it outputs a
        # prediction of the clean z_{t+fs}; the loss and the sampler go through
        # v_hat = (z_hat - x_tau) / max(1 - tau, state_x_eps))
        self.state_param = str(cfg.get("state_param", "v"))
        assert self.state_param in ("v", "x"), self.state_param
        self.state_x_eps = float(cfg.get("state_x_eps", 0.05))
        # state-flow tau draw: uniform^(1/alpha) (default) or logit-normal (JiT: mu -0.8, sigma 0.8)
        self.state_tau_logit = cfg.get("state_tau_logit", None)     # None or (mu, sigma)
        if self.state_tau_logit is not None:
            self.state_tau_logit = (float(self.state_tau_logit[0]), float(self.state_tau_logit[1]))
        assert not (self.state_prior == "prev" and self.state_residual), \
            "state_prior=prev already anchors the flow at z_t; state_residual is the other way to do that"
        self.tau_alpha = float(cfg.get("tau_alpha", 1.0))
        self.tau_alpha_state = float(cfg.get("tau_alpha_state", 0) or self.tau_alpha)
        self.n_clean_actions = self.num_states * self.fs
        self.num_views = int(cfg.get("num_views", 1))          # cameras; each adds H history and S state tokens
        self.n_state_tokens = self.num_states * self.num_views

        self.encoder = VisionEncoder(size=cfg["encoder_size"], output_type="cls",
                                     output_dim=self.z_dim, img_size=cfg["img_size"],
                                     backbone=cfg["encoder_backbone"],
                                     backbone_ckpt=cfg.get("encoder_ckpt"),
                                     proj_hidden=cfg["proj_hidden"])
        d = self.dim
        # state stream embeddings
        self.frame_in = nn.Linear(self.z_dim, d)
        self.frame_pos = nn.Parameter(torch.zeros(1, self.history_len, d))
        self.state_in = nn.Linear(self.z_dim, d)
        self.state_pos = nn.Parameter(torch.zeros(1, self.num_states, d))
        self.state_query = nn.Parameter(torch.randn(1, self.num_states, d) * 0.02)   # mse head
        if self.num_views > 1:            # single-view checkpoints keep their parameter set
            self.view_pos = nn.Parameter(torch.zeros(1, self.num_views, d))
        if self.goal_cond == "token":     # also the goal-free layout: existing checkpoints carry these
            self.goal_in = nn.Linear(self.z_dim, d)
            self.null_goal = nn.Parameter(torch.zeros(1, 1, d))
        if self.sep_policy_state:
            r = int(cfg.get("policy_proj_rank", 0) or 0)
            if r > 0:
                # bottlenecked policy view (owner 2026-09-03): LoRA-style rank-r factorization
                # P = U V, output stays z_dim so frame_in sharing is untouched; init U = V^T with
                # V row-orthonormal -> P starts as an orthogonal projector onto a random r-dim
                # subspace (an unbiased narrow view the policy then rotates)
                self.policy_proj = nn.Sequential(nn.Linear(self.z_dim, r, bias=False),
                                                 nn.Linear(r, self.z_dim))
                with torch.no_grad():
                    q, _ = torch.linalg.qr(torch.randn(self.z_dim, r))
                    self.policy_proj[0].weight.copy_(q.T)
                    self.policy_proj[1].weight.copy_(q)
                    self.policy_proj[1].bias.zero_()
            else:
                self.policy_proj = nn.Linear(self.z_dim, self.z_dim)
        if self.sigreg_proj_dim > 0:
            self.sigreg_proj = nn.Linear(self.z_dim, self.sigreg_proj_dim, bias=False)
            with torch.no_grad():
                if self.sigreg_proj_dim == self.z_dim:
                    self.sigreg_proj.weight.copy_(torch.eye(self.z_dim))   # plain SIGReg at step 0
                else:
                    nn.init.orthogonal_(self.sigreg_proj.weight)
        self.state_type = nn.Embedding(3, d)                 # 0 history, 1 next-state, 2 goal
        # action stream embeddings (shared projection for the clean actions and the action slots; the type embedding tells them apart)
        self.action_in = nn.Linear(self.action_raw_dim, d)
        self.action_pos = nn.Parameter(torch.zeros(1, self.num_actions, d))
        if self.action_head == "mse":       # mirror of state_query; registered only under the mse head so that
            self.action_query = nn.Parameter(torch.randn(1, self.num_actions, d) * 0.02)   # every flow ckpt keeps its layout
        self.action_type = nn.Embedding(2, d)                # 0 clean actions (context), 1 action slots
        # conditioning
        self.tau_in = nn.Linear(d, d)
        if self.goal_cond == "token":
            self.h_in = nn.Linear(d, d)
        self.blocks = nn.ModuleList(MoTBlock(d, cfg["n_heads"], cond_state=(self.state_head == "flow"),
                                             cond_action=(self.action_head == "flow" or self.goal_token),
                                             dropout=cfg["dropout"])
                                    for _ in range(cfg["depth"]))
        if self.goal_conditioning and self.goal_cond == "head":
            # jointflow's readout: per-token feature ++ goal latent -> 3 AdaLN-Zero MLP layers
            # (hidden 512, cond 128) modulated by the horizon embedding; learned null goal
            self.action_out = GCHeadMSE(d, self.z_dim, self.action_raw_dim, dropout=cfg["dropout"])
        else:
            self.action_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.action_raw_dim))
        self.state_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.z_dim))

        # ---- token layout and the attention map (fixed); history and state slots are view-major ----
        n_hist, n_state_slots = self.history_len * self.num_views, self.n_state_tokens
        n_clean_actions, n_action_slots = self.n_clean_actions, self.num_actions
        idx = {}
        offset = 0
        idx["hist"] = list(range(offset, offset + n_hist)); offset += n_hist
        idx["policy_hist"] = list(range(offset, offset + n_hist)) if self.sep_policy_state else []; offset += len(idx["policy_hist"])
        idx["state_slots"] = list(range(offset, offset + n_state_slots)); offset += n_state_slots
        idx["goal"] = list(range(offset, offset + 1)) if self.goal_token else []; offset += len(idx["goal"])
        idx["clean_actions"] = list(range(offset, offset + n_clean_actions)); offset += n_clean_actions
        idx["action_slots"] = list(range(offset, offset + n_action_slots)); offset += n_action_slots
        self.n_tokens = offset
        self.idx = idx
        # block layout: the state stream is the first n_state tokens; the slots (where each branch's
        # prediction happens) are the only tokens that can be conditioned
        state_rows, action_rows = idx["state_slots"], idx["action_slots"]
        self.layout = ((idx["clean_actions"] + idx["action_slots"])[0],
                       slice(state_rows[0], state_rows[-1] + 1) if state_rows else slice(0, 0),
                       slice(action_rows[0], action_rows[-1] + 1))
        attends = torch.zeros(self.n_tokens, self.n_tokens, dtype=torch.bool)      # attends[row, columns]
        hist, policy_hist, goal = idx["hist"], idx["policy_hist"], idx["goal"]
        state_slots, clean_actions, action_slots = idx["state_slots"], idx["clean_actions"], idx["action_slots"]
        policy_context = policy_hist if self.sep_policy_state else hist    # the history the action branch reads
        for row in hist:
            attends[row, hist] = True
        for row in policy_hist:
            attends[row, policy_hist] = True
        for slot_idx, row in enumerate(state_slots):
            step = slot_idx % self.num_states                    # slot (view, step), view-major
            attends[row, hist] = True
            attends[row, [state_slots[view * self.num_states + earlier]
                          for view in range(self.num_views) for earlier in range(step + 1)]] = True
            attends[row, clean_actions[:(step + 1) * self.fs]] = True
        for row in goal:
            attends[row, policy_context] = True
            attends[row, goal] = True
        for action_idx, row in enumerate(clean_actions):
            attends[row, policy_context] = True
            attends[row, clean_actions[:action_idx + 1]] = True
        for action_idx, row in enumerate(action_slots):
            attends[row, policy_context] = True
            attends[row, goal] = True
            attends[row, action_slots[:action_idx + 1]] = True
        self.register_buffer("attends", attends)

    # ---------------------------------------------------------------- helpers
    def encode(self, pixels):
        return self.encoder(pixels)

    def _mask(self, history_pad):
        """(B, 1, n, n) additive mask: the fixed map, with padded history keys removed."""
        B = history_pad.shape[0]
        attends = self.attends[None].expand(B, -1, -1).clone()
        if history_pad.any():
            key_pad = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=history_pad.device)
            key_pad[:, self.idx["hist"]] = history_pad.repeat(1, self.num_views)      # the same frames per view
            if self.idx["policy_hist"]:
                key_pad[:, self.idx["policy_hist"]] = history_pad.repeat(1, self.num_views)
            attends = attends & ~key_pad[:, None, :]
        mask = torch.zeros(B, self.n_tokens, self.n_tokens, device=history_pad.device)
        mask[~attends] = float("-inf")
        return mask[:, None]

    def _history_pos(self):
        """(1, views*H, d): the frame position of every history token plus its view embedding."""
        pos = self.frame_pos.repeat(1, self.num_views, 1)
        if self.num_views > 1:
            pos = pos + self.view_pos.repeat_interleave(self.history_len, dim=1)
        return pos

    def _state_slot_pos(self):
        """(1, views*S, d): the step position of every state slot plus its view embedding."""
        pos = self.state_pos.repeat(1, self.num_views, 1)
        if self.num_views > 1:
            pos = pos + self.view_pos.repeat_interleave(self.num_states, dim=1)
        return pos

    def _last_frames(self, z_history):
        """(B, views*S, z): each view's newest history latent repeated for that view's state slots
        (the prev prior and the residual anchor of the state flow)."""
        if z_history.dim() == 4:
            z_history = z_history.flatten(1, 2)
        B = z_history.shape[0]
        last = z_history.view(B, self.num_views, self.history_len, self.z_dim)[:, :, -1:]
        return last.expand(-1, -1, self.num_states, -1).reshape(B, self.n_state_tokens, self.z_dim)

    def _slot_conditioning(self, tau_a, tau_s, h_norm, B, device):
        """Per-branch conditioning vectors, (B, dim) or None. A tau exists only for a flow branch: tau_s on
        the state slots, tau_a on the action slots. The horizon conditions the action slots whenever the
        model is goal reaching (token variant; 0 when no horizon is given). Nothing else is conditioned."""
        d = self.dim
        cond_state = self.tau_in(sinusoid(tau_s, d)) if self.state_head == "flow" else None
        cond_action = self.tau_in(sinusoid(tau_a, d)) if self.action_head == "flow" else None
        if self.goal_token:
            horizon = h_norm if h_norm is not None else torch.zeros(B, device=device)
            h = self.h_in(sinusoid(horizon, d))
            cond_action = h if cond_action is None else cond_action + h
        return cond_state, cond_action

    def forward_tokens(self, z_history, history_pad, clean_actions, noisy_actions, noisy_state,
                       tau_a, tau_s, z_goal=None, h_norm=None, goal_keep=None):
        """One pass of the stack. z_history (B, views*H, z_dim) view-major, or (B, views, H, z_dim);
        clean_actions (B, n_clean_actions, adim) z-scored; noisy_actions (B, A, adim) (ignored under
        the mse action head: a learned query fills the slots); noisy_state (B, views*S, z_dim)
        (ignored under the mse state head, likewise). Returns (action_pred (B, A, adim),
        state_pred (B, views*S, z_dim)): each is the flow velocity under a flow head and the
        prediction itself under an mse head."""
        if z_history.dim() == 4:
            z_history = z_history.flatten(1, 2)
        B = z_history.shape[0]
        d = self.dim
        x = z_history.new_zeros(B, self.n_tokens, d)
        state_type_embed = self.state_type.weight
        history_pos = self._history_pos()
        x[:, self.idx["hist"]] = self.frame_in(z_history) + history_pos + state_type_embed[0]
        if self.sep_policy_state:
            x[:, self.idx["policy_hist"]] = self.frame_in(self.policy_proj(z_history)) \
                + history_pos + state_type_embed[0]
        slot_pos = self._state_slot_pos()
        if self.state_head == "mse":
            x[:, self.idx["state_slots"]] = self.state_query.repeat(1, self.num_views, 1).expand(B, -1, -1) \
                + slot_pos + state_type_embed[1]
        else:
            x[:, self.idx["state_slots"]] = self.state_in(noisy_state) + slot_pos + state_type_embed[1]
        if self.goal_conditioning and z_goal is not None and self.sep_policy_state:
            z_goal = self.policy_proj(z_goal)          # the policy's view of the goal (both modes)
        if self.goal_token:
            goal_embed = self.goal_in(z_goal)[:, None] if z_goal is not None else self.null_goal.expand(B, -1, -1)
            if goal_keep is not None:
                goal_embed = torch.where(goal_keep.view(B, 1, 1), goal_embed, self.null_goal.expand(B, -1, -1))
            x[:, self.idx["goal"]] = goal_embed + state_type_embed[2]
        action_type_embed = self.action_type.weight
        x[:, self.idx["clean_actions"]] = self.action_in(clean_actions) + self.action_pos[:, :self.n_clean_actions] + action_type_embed[0]
        if self.action_head == "mse":
            x[:, self.idx["action_slots"]] = self.action_query.expand(B, -1, -1) + self.action_pos + action_type_embed[1]
        else:
            x[:, self.idx["action_slots"]] = self.action_in(noisy_actions) + self.action_pos + action_type_embed[1]
        cond_state, cond_action = self._slot_conditioning(tau_a, tau_s, h_norm, B, z_history.device)
        mask = self._mask(history_pad)
        for block in self.blocks:
            x = block(x, mask, self.layout, cond_state, cond_action)
        action_slot_feats = x[:, self.idx["action_slots"]]
        if self.goal_conditioning and self.goal_cond == "head":
            action_pred = self.action_out(action_slot_feats, z_goal, h_norm, goal_keep)     # goal + horizon enter here only
        else:
            action_pred = self.action_out(action_slot_feats)
        return action_pred, self.state_out(x[:, self.idx["state_slots"]])

    @staticmethod
    def _masked_mse(pred, target, valid):
        per_token = ((pred - target) ** 2).mean(-1) * valid
        return (per_token.sum(-1) / valid.sum(-1).clamp(min=1)).mean()

    # ---------------------------------------------------------------- training
    def loss(self, z_history, history_pad, action_target, action_valid, state_target, state_valid,
             z_goal=None, h_norm=None, goal_keep=None):
        B = action_target.shape[0]
        dev = action_target.device
        clean_actions = action_target[:, :self.n_clean_actions]
        if self.action_head == "flow":                 # a tau and a noisy input exist only for a flow branch
            tau_a = torch.rand(B, device=dev) ** (1.0 / self.tau_alpha)
            noise_a = torch.randn_like(action_target)
            noisy_a = torch.lerp(noise_a, action_target, tau_a[:, None, None].to(action_target.dtype))
        else:
            tau_a = noise_a = noisy_a = None
        if self.state_head == "flow":
            tau_s = self._draw_tau_state(B, dev)
            if self.state_prior == "prev":
                noise_s = self._last_frames(z_history).to(state_target.dtype)                  # x_0 = z_t
                if self.state_prior_sigma > 0:
                    noise_s = noise_s + self._prior_scale(z_history) * torch.randn_like(state_target)
            else:
                noise_s = torch.randn_like(state_target)
            noisy_s = torch.lerp(noise_s, state_target, tau_s[:, None, None].to(state_target.dtype))
        else:
            tau_s = noise_s = noisy_s = None
        action_pred, state_pred = self.forward_tokens(z_history, history_pad, clean_actions, noisy_a, noisy_s,
                                         tau_a, tau_s, z_goal, h_norm, goal_keep)
        if self.action_head == "mse":                       # mirror of the mse state head: regress the chunk itself
            loss_action = self._masked_mse(action_pred, action_target, action_valid)
        else:
            loss_action = self._masked_mse(action_pred, action_target - noise_a, action_valid)
        if self.num_states == 0:
            return loss_action, loss_action.new_zeros(())
        if self.state_head == "mse":
            return loss_action, self._masked_mse(state_pred, state_target, state_valid)
        if self.state_param == "x":
            state_pred = self._x_to_v(state_pred, noisy_s, tau_s)
        return loss_action, self._masked_mse(state_pred, state_target - noise_s, state_valid)

    def _draw_tau_state(self, B, dev):
        if self.state_tau_logit is not None:
            mu, sd = self.state_tau_logit
            return torch.sigmoid(mu + sd * torch.randn(B, device=dev))
        return torch.rand(B, device=dev) ** (1.0 / self.tau_alpha_state)

    def _prior_scale(self, z_history):
        """sigma * rms(z_t) per state slot (detached), z_t being its view's newest history latent: the
        prev-prior noise scale in latent units."""
        rms = self._last_frames(z_history).detach().float().pow(2).mean(-1, keepdim=True).sqrt()
        return self.state_prior_sigma * rms.to(z_history.dtype)

    def _x_to_v(self, z_hat, x_tau, tau):
        """JiT-style reparameterization: velocity implied by a clean-state prediction."""
        denom = (1.0 - tau).clamp(min=self.state_x_eps)[:, None, None].to(z_hat.dtype)
        return (z_hat - x_tau) / denom

    # ---------------------------------------------------------------- inference
    def _state_phase(self, z_history, history_pad, clean_actions, noise_state, generator, z_goal, h_norm):
        B = z_history.shape[0]
        dev = z_history.device
        zero_action_slots = torch.zeros(B, self.num_actions, self.action_raw_dim, device=dev, dtype=clean_actions.dtype)
        one = torch.ones(B, device=dev)
        if self.state_head == "mse":
            _, state = self.forward_tokens(z_history, history_pad, clean_actions, zero_action_slots, None, one, one, z_goal, h_norm)
        else:
            if self.state_prior == "prev":
                state = self._last_frames(z_history)                                         # ODE starts at z_t
                if self.state_prior_sigma > 0:
                    eps = noise_state if noise_state is not None else \
                        torch.randn(B, self.n_state_tokens, self.z_dim, device=dev, generator=generator)
                    state = state + self._prior_scale(z_history) * eps
            else:
                state = noise_state if noise_state is not None else \
                    torch.randn(B, self.n_state_tokens, self.z_dim, device=dev, generator=generator)
            for i in range(self.n_flow_steps):
                tau = torch.full((B,), i / self.n_flow_steps, device=dev)
                _, v_s = self.forward_tokens(z_history, history_pad, clean_actions, zero_action_slots, state, one, tau, z_goal, h_norm)
                if self.state_param == "x":
                    v_s = self._x_to_v(v_s, state, tau)
                state = state + v_s / self.n_flow_steps
        if self.state_residual and self.num_states:
            state = state + self._last_frames(z_history)
        return state

    def _sample_impl(self, z_history, history_pad, generator=None, z_goal=None, h_norm=None,
                     noise_action=None, noise_state=None):
        B = z_history.shape[0]
        dev = z_history.device
        action = noise_action if noise_action is not None else \
            torch.randn(B, self.num_actions, self.action_raw_dim, device=dev, generator=generator)
        zero_clean_actions = torch.zeros(B, self.n_clean_actions, self.action_raw_dim, device=dev, dtype=action.dtype)
        zero_state_slots = torch.zeros(B, self.n_state_tokens, self.z_dim, device=dev, dtype=z_history.dtype)
        one = torch.ones(B, device=dev)
        if self.action_head == "mse":                            # mirror of the mse state head: one pass at tau one
            action, _ = self.forward_tokens(z_history, history_pad, zero_clean_actions, None, zero_state_slots, one, one, z_goal, h_norm)
        else:
            for i in range(self.n_flow_steps):                  # phase 1: the policy flow
                tau = torch.full((B,), i / self.n_flow_steps, device=dev)
                action_pred, _ = self.forward_tokens(z_history, history_pad, zero_clean_actions, action, zero_state_slots, tau, one, z_goal, h_norm)
                action = action + action_pred / self.n_flow_steps
        if self.num_states == 0:
            return action, action.new_zeros(B, 0, self.z_dim)
        state = self._state_phase(z_history, history_pad, action[:, :self.n_clean_actions], noise_state,
                                  generator, z_goal, h_norm)                       # phase 2
        return action, state

    @torch.no_grad()
    def sample(self, z_history, history_pad, generator=None, z_goal=None, h_norm=None):
        return self._sample_impl(z_history, history_pad, generator, z_goal, h_norm)

    @property
    def needs_state_noise(self):
        """True when the state phase of sampling starts from a noise tensor: a flow state head with
        a random prior, or a noisy previous-state prior. The mse state head never draws."""
        return self.num_states > 0 and self.state_head != "mse" and \
            (self.state_prior != "prev" or self.state_prior_sigma > 0)

    def draw_sample_noise(self, batch_size, device, generator=None):
        """The random tensors `sample` consumes, in order: the action noise, then the state noise if
        needs_state_noise (else None). A caller that imagines several steps can draw them all first
        and pass them to _sample_impl, keeping the generator stream identical to calling `sample`."""
        act_noise = torch.randn(batch_size, self.num_actions, self.action_raw_dim, device=device, generator=generator)
        state_noise = torch.randn(batch_size, self.n_state_tokens, self.z_dim, device=device, generator=generator) \
            if self.needs_state_noise else None
        return act_noise, state_noise

    def imagine_step(self, z_history, history_pad, actions, noise_state=None, z_goal=None, h_norm=None):
        """The MoT's one dynamics primitive, gradients enabled: the state token(s) one block ahead
        of the history from the CLEAN actions of that block (the first n_clean_actions of `actions`).
        Differentiable in `actions` (gradient planning) and in the history latents.
        Accepts a single block (fs actions) as well as the full n_clean_actions: a short input is zero-padded
        to n_clean_actions, which is inert for the first state token (under the causal map it attends only to
        its own block's fs clean actions) and that token is what the planners read."""
        clean_actions = actions[:, :self.n_clean_actions]
        if clean_actions.shape[1] < self.n_clean_actions:
            pad = clean_actions.new_zeros(clean_actions.shape[0], self.n_clean_actions - clean_actions.shape[1],
                                          clean_actions.shape[2])
            clean_actions = torch.cat([clean_actions, pad], dim=1)
        return self._state_phase(z_history, history_pad, clean_actions, noise_state, None, z_goal, h_norm)

    @torch.no_grad()
    def predict_state(self, z_history, history_pad, actions, noise_state=None, z_goal=None, h_norm=None):
        """imagine_step without gradients. Planning is the autoregressive rollout over this step
        (gip.JointFlowPlanPolicy._imagine); there is no one-block "inpaint" scorer for MoT."""
        return self.imagine_step(z_history, history_pad, actions, noise_state, z_goal, h_norm)

    @torch.no_grad()
    def sample_inpaint(self, z_history, history_pad, action_plan, noise_action=None,
                       noise_state=None, z_goal=None, h_norm=None):
        """Alias of predict_state under the joint model's name, for the shared probe adapters."""
        return self.predict_state(z_history, history_pad, action_plan, noise_state, z_goal, h_norm)

    # ---------------------------------------------------------------- lr groups
    def param_group_of(self, name):
        """encoder / dynamics (state stream + state embeddings/readout) / policy (the rest)."""
        if name.startswith("encoder."):
            return "encoder"
        if name.startswith(("state_in", "state_pos", "state_query", "state_out")):
            return "dynamics"
        if name.startswith("blocks."):
            parts = name.split(".")
            if len(parts) > 2 and parts[2].endswith("_state"):       # blocks.<i>.<module>_state.*
                return "dynamics"
        return "policy"


def build_model(cfg):
    defaults = dict(encoder_size="tiny", encoder_backbone="resnet18dp", encoder_ckpt=None,
                    img_size=224, z_dim=384, proj_hidden=768, d_model=384, n_heads=6, depth=8,
                    dropout=0.1, n_flow_steps=8, fs=5, num_actions_pred=10, num_states_pred=1,
                    policy_history_len=2, goal_conditioning=False, goal_cond="token", state_head="flow",
                    state_residual=False, tau_alpha=1.0, tau_alpha_state=0.0, num_views=1)
    return MoTFlow({**defaults, **cfg})
