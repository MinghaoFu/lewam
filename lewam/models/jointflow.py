"""Joint flow matching (docs/wip/LEWAM_FLOW_DESIGN.md): one rectified-flow objective over
per-timestep action tokens and block-boundary next-state latent tokens. The denoised sequence
interleaves frameskip actions with one state token per block boundary; causal temporal attention,
cross-attention into the frame-history memory. No separate dynamics head, no per-modality loss
weight: L = mean_MSE(v_action) + mean_MSE(v_state), each in its own native space."""

import torch
import torch.nn as nn

from lewam.models.lewam_crossattn import CrossAttnBlock, sinusoid
from lewam.models.module import VisionEncoder


def slot_layout(num_actions, num_states, frameskip):
    """Temporal slot order a[1..fs], z[fs], a[fs+1..2fs], z[2fs], ..., trailing actions.
    Returns (is_state bool (n_slots,), times float (n_slots,)); the state token for boundary q sits
    at time q*fs + 0.5 because a[q*fs] precedes the state it causes."""
    assert num_actions >= num_states * frameskip, "pad num_actions up to num_states*frameskip first"
    kinds, times = [], []
    for j in range(1, num_actions + 1):
        kinds.append(False)
        times.append(float(j))
        if j % frameskip == 0 and j // frameskip <= num_states:
            kinds.append(True)
            times.append(j + 0.5)
    return torch.tensor(kinds), torch.tensor(times)


def attn_mask(is_state, times, allow_imag):
    """Float mask (n_slots, n_slots), 0 = attend, -inf = blocked. A token attends tokens at earlier
    or equal time. allow_imag=False additionally blocks action tokens from attending (noisy) state
    tokens; state tokens always see the actions that cause them."""
    allowed = times[None, :] <= times[:, None]
    if not allow_imag:
        allowed &= ~(~is_state[:, None] & is_state[None, :])
    mask = torch.zeros(times.numel(), times.numel())
    mask[~allowed] = float("-inf")
    return mask


class JointFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_dim = cfg["z_dim"]
        self.fs = cfg["fs"]
        self.num_actions = cfg["num_actions_pred"]
        self.num_states = cfg["num_states_pred"]
        self.action_raw_dim = cfg["action_raw_dim"]
        self.history_len = cfg["policy_history_len"]
        self.allow_imag = cfg["allow_imag"]
        self.split_tau = cfg["split_tau"]
        self.n_flow_steps = cfg["n_flow_steps"]
        dim = cfg["d_model"]
        self.dim = dim

        self.encoder = VisionEncoder(size=cfg["encoder_size"], output_type="cls",
                                     output_dim=self.z_dim, img_size=cfg["img_size"],
                                     backbone=cfg["encoder_backbone"],
                                     backbone_ckpt=cfg.get("encoder_ckpt"))

        is_state, times = slot_layout(self.num_actions, self.num_states, self.fs)
        self.register_buffer("is_state", is_state)
        self.register_buffer("action_slots", (~is_state).nonzero().squeeze(-1))
        self.register_buffer("state_slots", is_state.nonzero().squeeze(-1))
        self.register_buffer("self_mask", attn_mask(is_state, times, self.allow_imag))

        self.frame_in = nn.Linear(self.z_dim, dim)
        self.frame_pos = nn.Parameter(torch.zeros(1, self.history_len, dim))
        self.action_in = nn.Linear(self.action_raw_dim, dim)
        self.state_in = nn.Linear(self.z_dim, dim)
        self.slot_pos = nn.Parameter(torch.zeros(1, is_state.numel(), dim))
        self.modality_emb = nn.Embedding(2, dim)
        # separate projections keep tau_action and tau_state distinguishable under --split_tau;
        # with the tied default both project the same sinusoid.
        self.tau_action_in = nn.Linear(dim, dim)
        self.tau_state_in = nn.Linear(dim, dim)
        self.cond = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossAttnBlock(dim, cfg["n_heads"], dropout=cfg["dropout"])
                                    for _ in range(cfg["depth"]))
        self.action_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.action_raw_dim))
        self.state_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.z_dim))

    def encode(self, pixels):
        return self.encoder(pixels)

    def _memory(self, z_ctx):
        return self.frame_in(z_ctx) + self.frame_pos

    def _cond(self, tau_action, tau_state):
        emb = self.tau_action_in(sinusoid(tau_action, self.dim)) \
            + self.tau_state_in(sinusoid(tau_state, self.dim))
        return self.cond(emb)

    def velocity(self, noisy_action, noisy_state, memory, memory_pad, tau_action, tau_state):
        B = noisy_action.shape[0]
        x = noisy_action.new_zeros(B, self.is_state.numel(), self.dim)
        x[:, self.action_slots] = self.action_in(noisy_action)
        if self.num_states:
            x[:, self.state_slots] = self.state_in(noisy_state)
        x = x + self.slot_pos + self.modality_emb(self.is_state.long())
        cond = self._cond(tau_action, tau_state)
        for block in self.blocks:
            x = block(x, memory, cond, self.self_mask, memory_pad)
        v_action = self.action_out(x[:, self.action_slots])
        v_state = self.state_out(x[:, self.state_slots])
        return v_action, v_state

    @staticmethod
    def _masked_mse(pred, target, valid):
        per_token = ((pred - target) ** 2).mean(-1) * valid
        return (per_token.sum(-1) / valid.sum(-1).clamp(min=1)).mean()

    def loss(self, z_ctx, ctx_pad, action_target, action_valid, state_target, state_valid):
        """Rectified flow, straight path: x_tau = (1-tau)*x0 + tau*x1, velocity target x1 - x0.
        One shared tau across both modalities unless split_tau draws them independently."""
        B = action_target.shape[0]
        memory = self._memory(z_ctx)
        tau_action = torch.rand(B, device=action_target.device)
        tau_state = torch.rand(B, device=action_target.device) if self.split_tau else tau_action

        noise_action = torch.randn_like(action_target)
        noisy_action = torch.lerp(noise_action, action_target, tau_action[:, None, None])
        noise_state = torch.randn_like(state_target)
        noisy_state = torch.lerp(noise_state, state_target, tau_state[:, None, None])

        v_action, v_state = self.velocity(noisy_action, noisy_state, memory, ctx_pad,
                                          tau_action, tau_state)
        loss_action = self._masked_mse(v_action, action_target - noise_action, action_valid)
        if self.num_states == 0:
            return loss_action, loss_action.new_zeros(())
        loss_state = self._masked_mse(v_state, state_target - noise_state, state_valid)
        return loss_action, loss_state

    @torch.no_grad()
    def sample(self, z_ctx, ctx_pad, generator=None):
        """Joint Euler ODE on the tied schedule. Returns the action chunk (z-scored raw space) and
        the imagined block-boundary latents."""
        B = z_ctx.shape[0]
        memory = self._memory(z_ctx)
        action = torch.randn(B, self.num_actions, self.action_raw_dim, device=z_ctx.device,
                             generator=generator)
        state = torch.randn(B, self.num_states, self.z_dim, device=z_ctx.device,
                            generator=generator)
        for i in range(self.n_flow_steps):
            tau = torch.full((B,), i / self.n_flow_steps, device=z_ctx.device)
            v_action, v_state = self.velocity(action, state, memory, ctx_pad, tau, tau)
            action = action + v_action / self.n_flow_steps
            state = state + v_state / self.n_flow_steps
        return action, state


def build_model(cfg):
    defaults = dict(encoder_size="tiny", encoder_backbone="resnet18dp", encoder_ckpt=None,
                    img_size=224, z_dim=512, d_model=256, n_heads=4, depth=6, dropout=0.1,
                    n_flow_steps=8, fs=5, num_actions_pred=5, num_states_pred=1,
                    policy_history_len=3, allow_imag=True, split_tau=False)
    return JointFlow({**defaults, **cfg})
