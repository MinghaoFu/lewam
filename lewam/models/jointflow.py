"""Joint flow matching over action and next-state latent tokens"""

import torch
import torch.nn as nn

from lewam.models.lewam_crossattn import CrossAttnBlock
from lewam.models.module import GCHeadMSE, VisionEncoder, sinusoid


def state_act_layout(num_actions, num_states, frameskip):
    """
    Orders a sequence of states and actions a[1..fs], z[fs], a[fs+1..2fs], z[2fs],...

    Args:
        num_actions (int): number of action tokens to predict
        num_states (int): number of state tokens to predict
        frameskip (int): number of action tokens between state tokens i.e. 
            how many actions are executed before the next state is observed
    Returns:
        is_state (torch.Tensor): boolean tensor of size (n_tokens,) indicating state vs action
        times (torch.Tensor): float tensor of size (n_tokens,) indicating the timestep
    """
    assert num_actions >= num_states * frameskip, "pad num_actions up to num_states*frameskip first"
    is_state, times = [], []
    for j in range(1, num_actions + 1):
        is_state.append(False)
        times.append(float(j))
        if j % frameskip == 0 and j // frameskip <= num_states:
            is_state.append(True)
            times.append(j + 0.5)
    return torch.tensor(is_state), torch.tensor(times)


def joint_attn_mask(is_state, times, actions_attend_states=True):
    """
    Returns attention mask (n_tokens, n_tokens), 0 = attend, -inf = blocked. 
    Temporal ordering: tokens attned only to tokens at earlier or equal time

    Args:
        is_state (torch.Tensor): boolean tensor of shape (n_tokens) indicating 
            which tokens are states
        times (torch.Tensor): float tensor of shape (n_tokens) of (env/trajectory) 
            timestpes for each token
        actions_attend_states (bool): if False, blocks action tokens from attending
            to (noisy) state tokens
    """
    attends = times[None, :] <= times[:, None]
    if not actions_attend_states:
        # action cannot attend to imagined state 
        attends &= ~(~is_state[:, None] & is_state[None, :])

    mask = torch.zeros(times.numel(), times.numel())
    mask[~attends] = float("-inf")
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
        self.actions_attend_states = cfg["actions_attend_states"]
        self.split_tau = cfg["split_tau"]   #RE: prefer indep_schedule or similar
        # state_residual: the state flow denoises delta = z[t+q*fs] - z[t]; sample paths
        # add z_t (the newest history anchor) back so callers always receive absolute z.
        self.state_residual = bool(cfg.get("state_residual", False))
        # tau_alpha: training tau ~ Beta(alpha, 1) via U^(1/alpha) -- alpha>1 upweights the
        # clean end (tau->1), where the curvature probe put the field's hard region (the
        # GR00T/LDA-1B heads bias the same way); alpha=1 recovers uniform.
        self.tau_alpha = float(cfg.get("tau_alpha", 1.0))
        # per-branch bias (owner 2026-08-27): oversampling near-clean STATES without touching
        # the action schedule is only possible with split_tau -- tau_alpha_state applies to
        # the state draw when split, and defaults to tau_alpha.
        self.tau_alpha_state = float(cfg.get("tau_alpha_state", 0) or self.tau_alpha)
        # state_target_norm (owner 2026-08-27, "noreg but z-score"): the state flow runs in
        # per-dim standardized latent coordinates. Running mean/std (EMA over training
        # batches, detached) live in buffers so eval inherits them; the ONLINE target keeps
        # its gradient -- only the scale changes. Motivation: unit-Gaussian source vs a
        # ~0.03-scale latent target makes the flow loss a denoising score and the samples a
        # noise-residual of fixed size (see docs/RECIPES.md cost-to-goal probe).
        self.state_target_norm = bool(cfg.get("state_target_norm", False))
        # state_mse (owner H1, 2026-08-27): the state slot is a plain regression readout that
        # shares the trunk's attention with the action flow tokens -- a learned query token
        # (no noise, no lerp, no tau of its own) whose output IS the next-latent prediction,
        # trained with MSE. It attends to the action tokens at their current noise level, so
        # it learns action-conditioned dynamics near tau=1 and the marginal near tau=0; the
        # samplers read it out with one pass at tau=1 over the finished action chunk.
        self.state_mse = bool(cfg.get("state_mse", False))
        self.register_buffer("state_mu", torch.zeros(self.z_dim))
        self.register_buffer("state_sd", torch.ones(self.z_dim))
        self.state_stats_momentum = 0.99
        self.tau_cond = cfg["tau_cond"]
        self.n_flow_steps = cfg["n_flow_steps"]
        self.dim = cfg["d_model"]   #RE: prefer embed_dim or similar; dim is too vague

        self.encoder = VisionEncoder(size=cfg["encoder_size"], output_type="cls",
                                     output_dim=self.z_dim, img_size=cfg["img_size"],
                                     backbone=cfg["encoder_backbone"],
                                     backbone_ckpt=cfg.get("encoder_ckpt"),
                                     proj_hidden=cfg["proj_hidden"])

        is_state, times = state_act_layout(self.num_actions, self.num_states, self.fs)
        self.register_buffer("is_state", is_state)
        self.register_buffer("action_slots", (~is_state).nonzero().squeeze(-1))
        self.register_buffer("state_slots", is_state.nonzero().squeeze(-1))
        self.register_buffer("self_mask", joint_attn_mask(is_state, times, self.actions_attend_states))

        self.frame_in = nn.Linear(self.z_dim, self.dim)
        self.frame_pos = nn.Parameter(torch.zeros(1, self.history_len, self.dim))
        self.action_in = nn.Linear(self.action_raw_dim, self.dim)
        self.state_in = nn.Linear(self.z_dim, self.dim)
        if self.state_mse:
            self.state_query = nn.Parameter(torch.randn(1, max(self.num_states, 1), self.dim) * 0.02)
        self.slot_pos = nn.Parameter(torch.zeros(1, is_state.numel(), self.dim))    #RE: idk, I think self.pos_emb is fine enough
        self.modality_emb = nn.Embedding(2, self.dim)

        self.tau_action_in = nn.Linear(self.dim, self.dim)
        self.tau_state_in = nn.Linear(self.dim, self.dim)
        self.cond = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))  #RE: pretty sure modules.MLP supports this

        self.blocks = nn.ModuleList(CrossAttnBlock(self.dim, cfg["n_heads"], dropout=cfg["dropout"])
                                    for _ in range(cfg["depth"]))
        # goal + horizon condition ONLY the action readout: the trunk and the state stream
        # never see the goal, so the imagined states stay goal-free dynamics
        self.goal_conditioning = bool(cfg.get("goal_conditioning", False))
        if self.goal_conditioning:
            self.action_out = GCHeadMSE(self.dim, self.z_dim, self.action_raw_dim,
                                        dropout=cfg["dropout"])
        else:
            self.action_out = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.action_raw_dim))
        self.state_out = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.z_dim))

    def encode(self, pixels):
        return self.encoder(pixels)

    def _memory(self, z_history):
        return self.frame_in(z_history) + self.frame_pos

    def _norm_state(self, z):
        return (z - self.state_mu) / self.state_sd if self.state_target_norm else z

    def _denorm_state(self, z):
        return z * self.state_sd + self.state_mu if self.state_target_norm else z

    @torch.no_grad()
    def update_state_stats(self, z):
        """EMA of per-dim mean/std of the online state targets (fp32, detached)."""
        z = z.detach().float().reshape(-1, self.z_dim)
        m, sd = z.mean(0), z.std(0).clamp_min(1e-3)
        mom = self.state_stats_momentum
        self.state_mu.mul_(mom).add_(m, alpha=1 - mom)
        self.state_sd.mul_(mom).add_(sd, alpha=1 - mom)

    def _cond(self, tau_action, tau_state):
        emb_action = self.tau_action_in(sinusoid(tau_action, self.dim))
        emb_state = self.tau_state_in(sinusoid(tau_state, self.dim))
        if self.tau_cond == "summed":
            return self.cond(emb_action + emb_state)
        cond_action = self.cond(emb_action)
        cond_state = self.cond(emb_state)
        cond = cond_action.new_zeros(cond_action.shape[0], self.is_state.numel(), self.dim)
        cond[:, self.action_slots] = cond_action.unsqueeze(1)
        cond[:, self.state_slots] = cond_state.unsqueeze(1)
        return cond

    def velocity(self, noisy_action, noisy_state, memory, memory_pad, tau_action, tau_state,
                 z_goal=None, h_norm=None, goal_keep=None):
        """Rectified flow forward pass"""
        B = noisy_action.shape[0]
        # buffer dtype follows the projections: under bf16 autocast the Linear outputs are
        # bf16 while noisy_action stays fp32, and index-put requires matching dtypes
        a_proj = self.action_in(noisy_action)
        x = a_proj.new_zeros(B, self.is_state.numel(), self.dim)
        x[:, self.action_slots] = a_proj
        if self.num_states:
            x[:, self.state_slots] = (self.state_query.expand(B, -1, -1).to(a_proj.dtype)
                                      if self.state_mse else self.state_in(noisy_state))

        x = x + self.slot_pos + self.modality_emb(self.is_state.long())
        cond = self._cond(tau_action, tau_state)
        for block in self.blocks:
            x = block(x, memory, cond, self.self_mask, memory_pad)

        if self.goal_conditioning:
            v_action = self.action_out(x[:, self.action_slots], z_goal, h_norm, goal_keep)
        else:
            v_action = self.action_out(x[:, self.action_slots])
        v_state = self.state_out(x[:, self.state_slots])
        return v_action, v_state

    @staticmethod
    def _masked_mse(pred, target, valid):
        per_token = ((pred - target) ** 2).mean(-1) * valid
        return (per_token.sum(-1) / valid.sum(-1).clamp(min=1)).mean()

    def loss(self, z_history, history_pad, action_target, action_valid, state_target, state_valid,
             z_goal=None, h_norm=None, goal_keep=None):
        """
        Rectified flow loss
            pred: velocity((1-tau)*x0 + tau*x1, tau, cond)
            target: x1 - x0
        """
        B = action_target.shape[0]
        memory = self._memory(z_history)
        if self.state_target_norm and self.num_states:
            if self.training:
                self.update_state_stats(state_target)
            state_target = self._norm_state(state_target)
        tau_action = torch.rand(B, device=action_target.device) ** (1.0 / self.tau_alpha)
        tau_state = (torch.rand(B, device=action_target.device) ** (1.0 / self.tau_alpha_state)
                     if (self.split_tau and not self.state_mse) else tau_action)

        noise_action = torch.randn_like(action_target)
        noisy_action = torch.lerp(noise_action, action_target,
                                  tau_action[:, None, None].to(action_target.dtype))
        noise_state = torch.randn_like(state_target)
        noisy_state = torch.lerp(noise_state, state_target,
                                 tau_state[:, None, None].to(state_target.dtype))

        v_action, v_state = self.velocity(noisy_action, noisy_state, memory, history_pad,
                                          tau_action, tau_state, z_goal, h_norm, goal_keep)
        loss_action = self._masked_mse(v_action, action_target - noise_action, action_valid)
        if self.num_states == 0:
            return loss_action, loss_action.new_zeros(())
        if self.state_mse:
            # v_state is the prediction itself (regression readout), not a velocity
            return loss_action, self._masked_mse(v_state, state_target, state_valid)
        loss_state = self._masked_mse(v_state, state_target - noise_state, state_valid)
        return loss_action, loss_state

    def _sample_impl(self, z_history, history_pad, generator=None, z_goal=None, h_norm=None,
                     noise_action=None, noise_state=None):
        """Grad-capable joint Euler ODE (steer-MPC differentiates it w.r.t. z_goal).
        Explicit noise tensors override the generator draws, making iterates of a
        test-time optimization comparable under one fixed noise realization."""
        B = z_history.shape[0]
        memory = self._memory(z_history)
        action = noise_action if noise_action is not None else \
            torch.randn(B, self.num_actions, self.action_raw_dim, device=z_history.device,
                        generator=generator)
        state = noise_state if noise_state is not None else \
            torch.randn(B, self.num_states, self.z_dim, device=z_history.device,
                        generator=generator)
        for i in range(self.n_flow_steps):
            tau = torch.full((B,), i / self.n_flow_steps, device=z_history.device)
            v_action, v_state = self.velocity(action, state, memory, history_pad, tau, tau,
                                              z_goal, h_norm)
            action = action + v_action / self.n_flow_steps
            if not self.state_mse:
                state = state + v_state / self.n_flow_steps
        if self.state_mse and self.num_states:
            tau1 = torch.ones(B, device=z_history.device)
            _, state = self.velocity(action, state, memory, history_pad, tau1, tau1, z_goal, h_norm)
        state = self._denorm_state(state)
        if self.state_residual and self.num_states:
            state = state + z_history[:, -1:]
        return action, state

    @torch.no_grad()
    def sample(self, z_history, history_pad, generator=None, z_goal=None, h_norm=None):
        """Joint Euler ODE on tied schedule.
        Returns the action chunks (z-scored raw space) and next-state latents."""
        return self._sample_impl(z_history, history_pad, generator, z_goal, h_norm)

    def _inpaint_impl(self, z_history, history_pad, action_plan, noise_action=None,
                      noise_state=None, z_goal=None, h_norm=None):
        """Grad-capable body of sample_inpaint (steer-MPC scores tilted-goal plans through
        it under the TRUE goal, with gradient flowing back through action_plan)."""
        B = z_history.shape[0]
        memory = self._memory(z_history)
        if noise_action is None:
            noise_action = torch.randn_like(action_plan)
        if noise_state is None:
            noise_state = torch.randn(B, self.num_states, self.z_dim, device=z_history.device)
        state = noise_state
        if self.state_mse:
            tau1 = torch.ones(B, device=z_history.device)
            _, state = self.velocity(action_plan, state, memory, history_pad, tau1, tau1,
                                     z_goal, h_norm)
        else:
            for i in range(self.n_flow_steps):
                tau = torch.full((B,), i / self.n_flow_steps, device=z_history.device)
                action = torch.lerp(noise_action, action_plan, tau[:, None, None].to(action_plan.dtype))
                _, v_state = self.velocity(action, state, memory, history_pad, tau, tau,
                                           z_goal, h_norm)
                state = state + v_state / self.n_flow_steps
        state = self._denorm_state(state)
        if self.state_residual and self.num_states:
            state = state + z_history[:, -1:]
        return state

    @torch.no_grad()
    def sample_inpaint(self, z_history, history_pad, action_plan, noise_action=None,
                       noise_state=None, z_goal=None, h_norm=None):
        """Euler ODE over the state slots only; action slots are clamped to the plan's
        flow path a_tau = (1-tau)*noise + tau*plan at every step. Passing the same
        noise tensors across calls makes candidate plans comparable within a replan.
        Returns the next-state latents imagined under the plan."""
        return self._inpaint_impl(z_history, history_pad, action_plan, noise_action,
                                  noise_state, z_goal, h_norm)


def build_model(cfg):
    defaults = dict(encoder_size="tiny", encoder_backbone="resnet18dp", encoder_ckpt=None,
                    img_size=224, z_dim=384, proj_hidden=768, d_model=384, n_heads=6, depth=8,
                    dropout=0.1, n_flow_steps=8, fs=5, num_actions_pred=5, num_states_pred=1,
                    policy_history_len=2, actions_attend_states=True, split_tau=False,
                    goal_conditioning=False, tau_cond="summed", state_residual=False,
                    tau_alpha=1.0, tau_alpha_state=0.0, state_target_norm=False,
                    state_mse=False)
    return JointFlow({**defaults, **cfg})
