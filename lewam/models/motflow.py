"""MoT-flow (owner design 2026-08-28): a Mixture-of-Transformers joint model. Two streams with
separate parameters (state stream: LN/QKV/out/FFN/AdaLN; action stream: the same), one global
self-attention over all tokens with a designed mask, so the policy and the dynamics exchange
information only where intended and never through a target.

Tokens
  state stream : z_hist (history latents, clean), z* (S noisy next-state tokens, or MSE queries),
                 z_g (goal token, optional)
  action stream: a (clean a_1..a_{S*fs}: exactly the actions the state tokens depend on),
                 a* (A noisy action tokens = the policy's flow)

Attention map (row attends to columns)
  z_hist -> z_hist
  z*_q   -> z_hist, z*_{<=q}, a_{<= q*fs}
  z_g    -> z_hist, z_g
  a_j    -> z_hist, a_{<=j}
  a*_j   -> z_hist, z_g, a*_{<=j}
a* never sees the clean actions or z* (no target leakage into the policy); z* never sees a* or
z_g (clean-action conditioning, goal-free dynamics); z_hist sees no target.

Conditioning: tau_a via AdaLN on a* only, tau_s on z* only, h on a* only; clean tokens get the
fixed tau=1 embedding. The goal reaches the policy as the z_g token, not through the readout.

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

from lewam.models.module import VisionEncoder, sinusoid


class MoTBlock(nn.Module):
    """Two-stream transformer block: per-stream LN-free AdaLN-Zero, QKV, output projection and
    MLP; a single masked self-attention over the union of tokens."""

    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim, self.n_heads = dim, n_heads
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(2)])
        self.out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                                nn.Dropout(dropout), nn.Linear(mlp_ratio * dim, dim))
                                  for _ in range(2)])
        self.ada = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim)) for _ in range(2)])
        for a in self.ada:
            nn.init.zeros_(a[-1].weight)
            nn.init.zeros_(a[-1].bias)
        self.attn_dropout = dropout

    @staticmethod
    def _select(stream, out0, out1):
        return torch.where(stream[None, :, None].bool(), out1, out0)

    def forward(self, x, stream, cond, mask):
        """x (B, n, dim); stream (n,) 0 = state stream, 1 = action stream; cond (B, n, dim);
        mask (B, 1, n, n) additive (0 / -inf)."""
        B, n, d = x.shape
        gates = self._select(stream, self.ada[0](cond), self.ada[1](cond)).chunk(6, dim=-1)
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = gates
        h = self.norm_attn(x) * (1 + sc_a) + sh_a
        qkv = self._select(stream, self.qkv[0](h), self.qkv[1](h))
        q, k, v = qkv.reshape(B, n, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.attn_dropout if self.training else 0.0)
        att = att.transpose(1, 2).reshape(B, n, d)
        x = x + g_a * self._select(stream, self.out[0](att), self.out[1](att))
        h = self.norm_mlp(x) * (1 + sc_m) + sh_m
        return x + g_m * self._select(stream, self.mlp[0](h), self.mlp[1](h))


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
        # policy_view (owner design 2026-09-01): the action branch attends ONLY to a projected
        # view of the latent -- every history token gets a projected copy that feeds the policy,
        # and the goal latent is projected in place (the state stream never consumes it). The
        # policy's gradient reaches z only through this projection; the dynamics owns z raw.
        self.policy_view = bool(cfg.get("policy_view", False))
        # sigreg_mode lives in the model so the projection checkpoints with it:
        #   pooled       -- trainer's original cat(z_t, state_target) single batch
        #   pertime      -- SIGReg per latent group (each history slot, each state target,
        #                   the goal as policy context), losses averaged
        #   pertime_proj -- pertime through a learned DxD linear (colleague recipe): z itself
        #                   no longer has to be white (the projection can whiten any full-rank
        #                   z); identity init = plain pertime at step 0; full-rank output keeps
        #                   the anti-collapse pressure
        self.sigreg_mode = str(cfg.get("sigreg_mode", "pooled"))
        assert self.sigreg_mode in ("pooled", "pertime", "pertime_proj"), self.sigreg_mode
        self.state_head = str(cfg.get("state_head", "flow"))
        assert self.state_head in ("flow", "mse")
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
        self.n_clean = self.num_states * self.fs

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
        self.goal_in = nn.Linear(self.z_dim, d)
        self.null_goal = nn.Parameter(torch.zeros(1, 1, d))
        if self.policy_view:
            self.policy_proj = nn.Linear(self.z_dim, self.z_dim)
        if self.sigreg_mode == "pertime_proj":
            self.sigreg_proj = nn.Linear(self.z_dim, self.z_dim, bias=False)
            with torch.no_grad():
                self.sigreg_proj.weight.copy_(torch.eye(self.z_dim))
        self.state_type = nn.Embedding(3, d)                 # 0 history, 1 next-state, 2 goal
        # action stream embeddings (shared projection for clean and noisy; type tells them apart)
        self.action_in = nn.Linear(self.action_raw_dim, d)
        self.action_pos = nn.Parameter(torch.zeros(1, self.num_actions, d))
        self.action_type = nn.Embedding(2, d)                # 0 clean, 1 noisy
        # conditioning
        self.tau_in = nn.Linear(d, d)
        self.h_in = nn.Linear(d, d)
        self.blocks = nn.ModuleList(MoTBlock(d, cfg["n_heads"], dropout=cfg["dropout"])
                                    for _ in range(cfg["depth"]))
        self.action_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.action_raw_dim))
        self.state_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.z_dim))

        # ---- token layout and the attention map (fixed) ----
        H, S, C, A = self.history_len, self.num_states, self.n_clean, self.num_actions
        idx = {}
        o = 0
        idx["hist"] = list(range(o, o + H)); o += H
        idx["phist"] = list(range(o, o + H)) if self.policy_view else []; o += len(idx["phist"])
        idx["state"] = list(range(o, o + S)); o += S
        idx["goal"] = list(range(o, o + 1)) if self.goal_conditioning else []; o += len(idx["goal"])
        idx["clean"] = list(range(o, o + C)); o += C
        idx["noisy"] = list(range(o, o + A)); o += A
        self.n_tokens = o
        self.idx = idx
        stream = torch.zeros(o, dtype=torch.long)
        stream[idx["clean"] + idx["noisy"]] = 1
        self.register_buffer("stream", stream)
        allow = torch.zeros(o, o, dtype=torch.bool)
        hist, st, gl, cl, ny = idx["hist"], idx["state"], idx["goal"], idx["clean"], idx["noisy"]
        ph = idx["phist"]
        pol_ctx = ph if self.policy_view else hist    # the history the action branch reads
        for r in hist:
            allow[r, hist] = True
        for r in ph:
            allow[r, ph] = True
        for q, r in enumerate(st):
            allow[r, hist] = True
            allow[r, st[:q + 1]] = True
            allow[r, cl[:(q + 1) * self.fs]] = True
        for r in gl:
            allow[r, pol_ctx] = True
            allow[r, gl] = True
        for j, r in enumerate(cl):
            allow[r, pol_ctx] = True
            allow[r, cl[:j + 1]] = True
        for j, r in enumerate(ny):
            allow[r, pol_ctx] = True
            allow[r, gl] = True
            allow[r, ny[:j + 1]] = True
        self.register_buffer("allow", allow)

    # ---------------------------------------------------------------- helpers
    def encode(self, pixels):
        return self.encoder(pixels)

    def _mask(self, history_pad):
        """(B, 1, n, n) additive mask: the fixed map, with padded history keys removed."""
        B = history_pad.shape[0]
        allow = self.allow[None].expand(B, -1, -1).clone()
        if history_pad.any():
            key_pad = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=history_pad.device)
            key_pad[:, self.idx["hist"]] = history_pad
            if self.idx["phist"]:
                key_pad[:, self.idx["phist"]] = history_pad
            allow = allow & ~key_pad[:, None, :]
        mask = torch.zeros(B, self.n_tokens, self.n_tokens, device=history_pad.device)
        mask[~allow] = float("-inf")
        return mask[:, None]

    def _cond(self, B, tau_a, tau_s, h_norm, device):
        d = self.dim
        one = torch.ones(B, device=device)
        c_clean = self.tau_in(sinusoid(one, d))
        cond = c_clean[:, None].expand(B, self.n_tokens, d).clone()
        cond[:, self.idx["state"]] = self.tau_in(sinusoid(tau_s, d))[:, None]
        c_noisy = self.tau_in(sinusoid(tau_a, d))
        if self.goal_conditioning:
            hn = h_norm if h_norm is not None else torch.zeros(B, device=device)
            c_noisy = c_noisy + self.h_in(sinusoid(hn, d))
        cond[:, self.idx["noisy"]] = c_noisy[:, None]
        return cond

    def forward_tokens(self, z_history, history_pad, clean_actions, noisy_actions, noisy_state,
                       tau_a, tau_s, z_goal=None, h_norm=None, goal_keep=None):
        """One pass of the stack. clean_actions (B, n_clean, adim) z-scored; noisy_actions
        (B, A, adim); noisy_state (B, S, z_dim) (ignored under the mse head). Returns
        (v_action (B, A, adim), state_out (B, S, z_dim))."""
        B = z_history.shape[0]
        d = self.dim
        x = z_history.new_zeros(B, self.n_tokens, d)
        st_type = self.state_type.weight
        x[:, self.idx["hist"]] = self.frame_in(z_history) + self.frame_pos + st_type[0]
        if self.policy_view:
            x[:, self.idx["phist"]] = self.frame_in(self.policy_proj(z_history)) \
                + self.frame_pos + st_type[0]
        if self.state_head == "mse":
            x[:, self.idx["state"]] = self.state_query.expand(B, -1, -1) + self.state_pos + st_type[1]
        else:
            x[:, self.idx["state"]] = self.state_in(noisy_state) + self.state_pos + st_type[1]
        if self.goal_conditioning:
            if z_goal is not None and self.policy_view:
                z_goal = self.policy_proj(z_goal)
            g = self.goal_in(z_goal)[:, None] if z_goal is not None else self.null_goal.expand(B, -1, -1)
            if goal_keep is not None:
                g = torch.where(goal_keep.view(B, 1, 1), g, self.null_goal.expand(B, -1, -1))
            x[:, self.idx["goal"]] = g + st_type[2]
        a_type = self.action_type.weight
        x[:, self.idx["clean"]] = self.action_in(clean_actions) + self.action_pos[:, :self.n_clean] + a_type[0]
        x[:, self.idx["noisy"]] = self.action_in(noisy_actions) + self.action_pos + a_type[1]
        cond = self._cond(B, tau_a, tau_s, h_norm, z_history.device)
        mask = self._mask(history_pad)
        for block in self.blocks:
            x = block(x, self.stream, cond, mask)
        return self.action_out(x[:, self.idx["noisy"]]), self.state_out(x[:, self.idx["state"]])

    @staticmethod
    def _masked_mse(pred, target, valid):
        per_token = ((pred - target) ** 2).mean(-1) * valid
        return (per_token.sum(-1) / valid.sum(-1).clamp(min=1)).mean()

    # ---------------------------------------------------------------- training
    def loss(self, z_history, history_pad, action_target, action_valid, state_target, state_valid,
             z_goal=None, h_norm=None, goal_keep=None):
        B = action_target.shape[0]
        dev = action_target.device
        tau_a = torch.rand(B, device=dev) ** (1.0 / self.tau_alpha)
        tau_s = self._draw_tau_state(B, dev)
        noise_a = torch.randn_like(action_target)
        noisy_a = torch.lerp(noise_a, action_target, tau_a[:, None, None].to(action_target.dtype))
        clean_a = action_target[:, :self.n_clean]
        if self.state_prior == "prev":
            noise_s = z_history[:, -1:].expand_as(state_target).to(state_target.dtype)     # x_0 = z_t
            if self.state_prior_sigma > 0:
                noise_s = noise_s + self._prior_scale(z_history) * torch.randn_like(state_target)
        else:
            noise_s = torch.randn_like(state_target)
        noisy_s = torch.lerp(noise_s, state_target, tau_s[:, None, None].to(state_target.dtype))
        v_a, out_s = self.forward_tokens(z_history, history_pad, clean_a, noisy_a, noisy_s,
                                         tau_a, tau_s, z_goal, h_norm, goal_keep)
        loss_action = self._masked_mse(v_a, action_target - noise_a, action_valid)
        if self.num_states == 0:
            return loss_action, loss_action.new_zeros(())
        if self.state_head == "mse":
            return loss_action, self._masked_mse(out_s, state_target, state_valid)
        if self.state_param == "x":
            out_s = self._x_to_v(out_s, noisy_s, tau_s)
        return loss_action, self._masked_mse(out_s, state_target - noise_s, state_valid)

    def _draw_tau_state(self, B, dev):
        if self.state_tau_logit is not None:
            mu, sd = self.state_tau_logit
            return torch.sigmoid(mu + sd * torch.randn(B, device=dev))
        return torch.rand(B, device=dev) ** (1.0 / self.tau_alpha_state)

    def _prior_scale(self, z_history):
        """sigma * rms(z_t) per sample (detached): the prev-prior noise scale in latent units."""
        rms = z_history[:, -1:].detach().float().pow(2).mean(-1, keepdim=True).sqrt()
        return self.state_prior_sigma * rms.to(z_history.dtype)

    def _x_to_v(self, z_hat, x_tau, tau):
        """JiT-style reparameterization: velocity implied by a clean-state prediction."""
        denom = (1.0 - tau).clamp(min=self.state_x_eps)[:, None, None].to(z_hat.dtype)
        return (z_hat - x_tau) / denom

    # ---------------------------------------------------------------- inference
    def _state_phase(self, z_history, history_pad, clean_a, noise_state, generator, z_goal, h_norm):
        B = z_history.shape[0]
        dev = z_history.device
        dummy_a = torch.zeros(B, self.num_actions, self.action_raw_dim, device=dev, dtype=clean_a.dtype)
        one = torch.ones(B, device=dev)
        if self.state_head == "mse":
            _, state = self.forward_tokens(z_history, history_pad, clean_a, dummy_a, None, one, one, z_goal, h_norm)
        else:
            if self.state_prior == "prev":
                state = z_history[:, -1:].expand(B, self.num_states, self.z_dim)            # ODE starts at z_t
                if self.state_prior_sigma > 0:
                    eps = noise_state if noise_state is not None else \
                        torch.randn(B, self.num_states, self.z_dim, device=dev, generator=generator)
                    state = state + self._prior_scale(z_history) * eps
            else:
                state = noise_state if noise_state is not None else \
                    torch.randn(B, self.num_states, self.z_dim, device=dev, generator=generator)
            for i in range(self.n_flow_steps):
                tau = torch.full((B,), i / self.n_flow_steps, device=dev)
                _, v_s = self.forward_tokens(z_history, history_pad, clean_a, dummy_a, state, one, tau, z_goal, h_norm)
                if self.state_param == "x":
                    v_s = self._x_to_v(v_s, state, tau)
                state = state + v_s / self.n_flow_steps
        if self.state_residual and self.num_states:
            state = state + z_history[:, -1:]
        return state

    def _sample_impl(self, z_history, history_pad, generator=None, z_goal=None, h_norm=None,
                     noise_action=None, noise_state=None):
        B = z_history.shape[0]
        dev = z_history.device
        action = noise_action if noise_action is not None else \
            torch.randn(B, self.num_actions, self.action_raw_dim, device=dev, generator=generator)
        dummy_c = torch.zeros(B, self.n_clean, self.action_raw_dim, device=dev, dtype=action.dtype)
        dummy_s = torch.zeros(B, self.num_states, self.z_dim, device=dev, dtype=z_history.dtype)
        one = torch.ones(B, device=dev)
        for i in range(self.n_flow_steps):                      # phase 1: the policy flow
            tau = torch.full((B,), i / self.n_flow_steps, device=dev)
            v_a, _ = self.forward_tokens(z_history, history_pad, dummy_c, action, dummy_s, tau, one, z_goal, h_norm)
            action = action + v_a / self.n_flow_steps
        if self.num_states == 0:
            return action, action.new_zeros(B, 0, self.z_dim)
        state = self._state_phase(z_history, history_pad, action[:, :self.n_clean], noise_state,
                                  generator, z_goal, h_norm)                       # phase 2
        return action, state

    @torch.no_grad()
    def sample(self, z_history, history_pad, generator=None, z_goal=None, h_norm=None):
        return self._sample_impl(z_history, history_pad, generator, z_goal, h_norm)

    def imagine_step(self, z_history, history_pad, actions, noise_state=None, z_goal=None, h_norm=None):
        """The MoT's one dynamics primitive, gradients enabled: the state token(s) one block ahead
        of the history from the CLEAN actions of that block (the first n_clean of `actions`).
        Differentiable in `actions` (gradient planning) and in the history latents."""
        return self._state_phase(z_history, history_pad, actions[:, :self.n_clean], noise_state,
                                 None, z_goal, h_norm)

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
            if len(parts) > 3 and parts[2] in ("qkv", "out", "mlp", "ada") and parts[3] == "0":
                return "dynamics"
        return "policy"


def build_model(cfg):
    defaults = dict(encoder_size="tiny", encoder_backbone="resnet18dp", encoder_ckpt=None,
                    img_size=224, z_dim=384, proj_hidden=768, d_model=384, n_heads=6, depth=8,
                    dropout=0.1, n_flow_steps=8, fs=5, num_actions_pred=10, num_states_pred=1,
                    policy_history_len=2, goal_conditioning=False, state_head="flow",
                    state_residual=False, tau_alpha=1.0, tau_alpha_state=0.0)
    return MoTFlow({**defaults, **cfg})
