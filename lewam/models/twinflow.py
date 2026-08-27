"""Twin flow: the jointflow action flow left exactly as it is, plus a SEPARATE state flow that
sees the clean ground-truth action chunk (owner design 2026-08-27).

Why a second model rather than a jointflow flag: in the joint flow the state token conditions
on action tokens at their own noise level, so the dynamics it learns is a tau-dependent mix of
marginal and conditional. Here the state trunk mirrors the action trunk (same CrossAttnBlock
stack, same token layout, cross-attention to the frame history) but its action tokens are the
clean chunk, never noised and carrying no tau, so it is a plain conditional dynamics model
p(z_{t+fs} | history, a). Scoring a candidate chunk for planning is then just running the state
flow on it, which sample_inpaint used to fake by clamping.

Conditioning is per-token: only the noisy state slot is modulated by tau_state; the clean
action tokens get a fixed tau=1 embedding (they carry no noise level of their own).

The two branches share the encoder and nothing else. `state_detach` cuts the state branch's
gradient into the encoder (its history memory and its target are stop-gradded), so the state
flow cannot outrun or collapse the encoder and the policy's representation is shaped by the
action loss alone; with it on, standardizing the state target (`state_target_norm`) is safe
because the 1/sd runaway needs a gradient path into the encoder.

Interface = jointflow's (loss / sample / sample_inpaint / _sample_impl / _inpaint_impl and
the attributes the eval adapters and probes read), so train_jointflow.py, gip.py and the
probe scripts run unchanged.
"""

import torch
import torch.nn as nn

from lewam.models.jointflow import JointFlow, joint_attn_mask, state_act_layout
from lewam.models.lewam_crossattn import CrossAttnBlock
from lewam.models.module import sinusoid


class StateFlow(nn.Module):
    """Rectified flow over the next-state latent token(s), conditioned on the clean action chunk
    (context tokens in the same [a.., z, a..] layout as jointflow) and the frame history (memory)."""

    def __init__(self, z_dim, action_raw_dim, num_actions, num_states, fs, history_len, dim,
                 n_heads, depth, dropout):
        super().__init__()
        self.z_dim, self.dim, self.num_states = z_dim, dim, num_states
        is_state, times = state_act_layout(num_actions, num_states, fs)
        self.register_buffer("is_state", is_state)
        self.register_buffer("action_slots", (~is_state).nonzero().squeeze(-1))
        self.register_buffer("state_slots", is_state.nonzero().squeeze(-1))
        # action tokens are context only: they never attend the noisy state token
        self.register_buffer("self_mask", joint_attn_mask(is_state, times, actions_attend_states=False))
        self.frame_in = nn.Linear(z_dim, dim)
        self.frame_pos = nn.Parameter(torch.zeros(1, history_len, dim))
        self.action_in = nn.Linear(action_raw_dim, dim)
        self.state_in = nn.Linear(z_dim, dim)
        self.slot_pos = nn.Parameter(torch.zeros(1, is_state.numel(), dim))
        self.modality_emb = nn.Embedding(2, dim)
        self.tau_in = nn.Linear(dim, dim)
        self.cond = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossAttnBlock(dim, n_heads, dropout=dropout) for _ in range(depth))
        self.state_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, z_dim))

    def _memory(self, z_history):
        return self.frame_in(z_history) + self.frame_pos

    def _cond(self, tau):
        """Per-token AdaLN conditioning: the noisy state slot(s) get tau; the clean action
        context slots get the fixed tau=1 ("clean") embedding, so their processing never
        depends on the state's noise level (per-modality, the jfv1 tau-bug lesson)."""
        cond_state = self.cond(self.tau_in(sinusoid(tau, self.dim)))                 # (B, dim)
        cond_clean = self.cond(self.tau_in(sinusoid(torch.ones_like(tau), self.dim)))
        cond = cond_clean.new_zeros(tau.shape[0], self.is_state.numel(), self.dim)
        cond[:, self.action_slots] = cond_clean.unsqueeze(1)
        cond[:, self.state_slots] = cond_state.unsqueeze(1)
        return cond

    def trunk(self, actions, noisy_state, memory, memory_pad, tau):
        B = actions.shape[0]
        a_proj = self.action_in(actions)
        x = a_proj.new_zeros(B, self.is_state.numel(), self.dim)
        x[:, self.action_slots] = a_proj
        x[:, self.state_slots] = self.state_in(noisy_state).to(a_proj.dtype)
        x = x + self.slot_pos + self.modality_emb(self.is_state.long())
        cond = self._cond(tau)
        for block in self.blocks:
            x = block(x, memory, cond, self.self_mask, memory_pad)
        return x

    def velocity(self, actions, noisy_state, memory, memory_pad, tau):
        x = self.trunk(actions, noisy_state, memory, memory_pad, tau)
        return self.state_out(x[:, self.state_slots])

    def loss(self, z_history, history_pad, actions, state_target, state_valid, tau_alpha=1.0):
        B = actions.shape[0]
        memory = self._memory(z_history)
        tau = torch.rand(B, device=actions.device) ** (1.0 / tau_alpha)
        noise = torch.randn_like(state_target)
        noisy = torch.lerp(noise, state_target, tau[:, None, None].to(state_target.dtype))
        v = self.velocity(actions, noisy, memory, history_pad, tau)
        per_token = ((v - (state_target - noise)) ** 2).mean(-1) * state_valid
        return (per_token.sum(-1) / state_valid.sum(-1).clamp(min=1)).mean()

    def sample(self, z_history, history_pad, actions, n_steps, noise_state=None, generator=None):
        """Euler ODE over the state token(s) under a CLEAN action chunk (grad-capable)."""
        B = actions.shape[0]
        memory = self._memory(z_history)
        state = noise_state if noise_state is not None else \
            torch.randn(B, self.num_states, self.z_dim, device=z_history.device, generator=generator)
        for i in range(n_steps):
            tau = torch.full((B,), i / n_steps, device=z_history.device)
            state = state + self.velocity(actions, state, memory, history_pad, tau) / n_steps
        return state


class TwinFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # the policy IS a jointflow with no state slot: same trunk, same GC readout, same sampler
        self.policy = JointFlow({**cfg, "num_states_pred": 0})
        self.z_dim = self.policy.z_dim
        self.fs = self.policy.fs
        self.num_actions = self.policy.num_actions
        self.num_states = int(cfg["num_states_pred"])
        self.action_raw_dim = self.policy.action_raw_dim
        self.history_len = self.policy.history_len
        self.n_flow_steps = self.policy.n_flow_steps
        self.goal_conditioning = self.policy.goal_conditioning
        self.state_detach = bool(cfg.get("state_detach", False))
        self.state_residual = bool(cfg.get("state_residual", False))
        self.tau_alpha_state = float(cfg.get("tau_alpha_state", 0) or cfg.get("tau_alpha", 1.0))
        self.state_target_norm = bool(cfg.get("state_target_norm", False))
        self.register_buffer("state_mu", torch.zeros(self.z_dim))
        self.register_buffer("state_sd", torch.ones(self.z_dim))
        self.state_stats_momentum = 0.99
        depth = int(cfg.get("state_depth", 0) or cfg["depth"])
        self.state_flow = StateFlow(self.z_dim, self.action_raw_dim, self.num_actions, self.num_states,
                                    self.fs, self.history_len, cfg["d_model"], cfg["n_heads"], depth,
                                    cfg["dropout"])

    # ---- encoder (shared; owned by the policy module) ----
    @property
    def encoder(self):
        return self.policy.encoder

    def encode(self, pixels):
        return self.policy.encode(pixels)

    # ---- target standardization (safe only under state_detach) ----
    def _norm_state(self, z):
        return (z - self.state_mu) / self.state_sd if self.state_target_norm else z

    def _denorm_state(self, z):
        return z * self.state_sd + self.state_mu if self.state_target_norm else z

    @torch.no_grad()
    def update_state_stats(self, z):
        z = z.detach().float().reshape(-1, self.z_dim)
        m, sd = z.mean(0), z.std(0).clamp_min(1e-3)
        mom = self.state_stats_momentum
        self.state_mu.mul_(mom).add_(m, alpha=1 - mom)
        self.state_sd.mul_(mom).add_(sd, alpha=1 - mom)

    # ---- training ----
    def loss(self, z_history, history_pad, action_target, action_valid, state_target, state_valid,
             z_goal=None, h_norm=None, goal_keep=None):
        loss_action, _ = self.policy.loss(z_history, history_pad, action_target, action_valid,
                                          state_target[:, :0], state_valid[:, :0],
                                          z_goal, h_norm, goal_keep)
        if self.num_states == 0:
            return loss_action, loss_action.new_zeros(())
        if self.state_detach:
            z_history, state_target = z_history.detach(), state_target.detach()
        if self.state_target_norm:
            if self.training:
                self.update_state_stats(state_target)
            state_target = self._norm_state(state_target)
        loss_state = self.state_flow.loss(z_history, history_pad, action_target, state_target,
                                          state_valid, self.tau_alpha_state)
        return loss_action, loss_state

    # ---- inference (jointflow signatures) ----
    def _finish_state(self, z_history, state):
        state = self._denorm_state(state)
        if self.state_residual and self.num_states:
            state = state + z_history[:, -1:]
        return state

    def _sample_impl(self, z_history, history_pad, generator=None, z_goal=None, h_norm=None,
                     noise_action=None, noise_state=None):
        action, _ = self.policy._sample_impl(z_history, history_pad, generator, z_goal, h_norm,
                                             noise_action, None)
        if self.num_states == 0:
            return action, action.new_zeros(action.shape[0], 0, self.z_dim)
        state = self.state_flow.sample(z_history, history_pad, action, self.n_flow_steps,
                                       noise_state, generator)
        return action, self._finish_state(z_history, state)

    @torch.no_grad()
    def sample(self, z_history, history_pad, generator=None, z_goal=None, h_norm=None):
        """Sample the action chunk from the policy flow, then the next-state latent from the
        state flow under that (clean) chunk."""
        return self._sample_impl(z_history, history_pad, generator, z_goal, h_norm)

    def _inpaint_impl(self, z_history, history_pad, action_plan, noise_action=None,
                      noise_state=None, z_goal=None, h_norm=None):
        """The state flow under a given clean chunk; noise_action is accepted for signature
        parity and unused (nothing is clamped -- the plan is the conditioning)."""
        state = self.state_flow.sample(z_history, history_pad, action_plan, self.n_flow_steps,
                                       noise_state, None)
        return self._finish_state(z_history, state)

    @torch.no_grad()
    def sample_inpaint(self, z_history, history_pad, action_plan, noise_action=None,
                       noise_state=None, z_goal=None, h_norm=None):
        return self._inpaint_impl(z_history, history_pad, action_plan, noise_action, noise_state,
                                  z_goal, h_norm)


def build_model(cfg):
    # jointflow's defaults + the twin's own keys
    defaults = dict(encoder_size="tiny", encoder_backbone="resnet18dp", encoder_ckpt=None,
                    img_size=224, z_dim=384, proj_hidden=768, d_model=384, n_heads=6, depth=8,
                    dropout=0.1, n_flow_steps=8, fs=5, num_actions_pred=10, num_states_pred=1,
                    policy_history_len=2, actions_attend_states=True, split_tau=False,
                    goal_conditioning=False, tau_cond="summed", state_residual=False,
                    tau_alpha=1.0, tau_alpha_state=0.0, state_target_norm=False,
                    state_detach=False, state_depth=0)
    return TwinFlow({**defaults, **cfg})
