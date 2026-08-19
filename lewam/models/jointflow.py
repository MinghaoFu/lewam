"""Joint flow matching over action and next-state latent tokens"""

import torch
import torch.nn as nn

from lewam.models.lewam_crossattn import CrossAttnBlock, sinusoid   #RE: stop rewriting sinusoid, put reusable module in one place
from lewam.models.module import VisionEncoder


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
        self.slot_pos = nn.Parameter(torch.zeros(1, is_state.numel(), self.dim))    #RE: idk, I think self.pos_emb is fine enough
        self.modality_emb = nn.Embedding(2, self.dim)
        self.tau_action_in = nn.Linear(self.dim, self.dim)
        self.tau_state_in = nn.Linear(self.dim, self.dim)
        self.cond = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))
        self.blocks = nn.ModuleList(CrossAttnBlock(self.dim, cfg["n_heads"], dropout=cfg["dropout"])
                                    for _ in range(cfg["depth"]))
        self.action_out = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.action_raw_dim))
        self.state_out = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.z_dim))

    def encode(self, pixels):
        return self.encoder(pixels)

    def _memory(self, z_history):
        return self.frame_in(z_history) + self.frame_pos

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

    def velocity(self, noisy_action, noisy_state, memory, memory_pad, tau_action, tau_state):
        """Rectified flow forward pass"""
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

    def loss(self, z_history, history_pad, action_target, action_valid, state_target, state_valid):
        """
        Rectified flow loss
            pred: velocity((1-tau)*x0 + tau*x1, tau, cond)
            target: x1 - x0
        """
        B = action_target.shape[0]
        memory = self._memory(z_history)
        tau_action = torch.rand(B, device=action_target.device)
        tau_state = torch.rand(B, device=action_target.device) if self.split_tau else tau_action

        noise_action = torch.randn_like(action_target)
        noisy_action = torch.lerp(noise_action, action_target, tau_action[:, None, None])
        noise_state = torch.randn_like(state_target)
        noisy_state = torch.lerp(noise_state, state_target, tau_state[:, None, None])

        v_action, v_state = self.velocity(noisy_action, noisy_state, memory, history_pad,
                                          tau_action, tau_state)
        loss_action = self._masked_mse(v_action, action_target - noise_action, action_valid)
        if self.num_states == 0:
            return loss_action, loss_action.new_zeros(())
        loss_state = self._masked_mse(v_state, state_target - noise_state, state_valid)
        return loss_action, loss_state

    @torch.no_grad()
    def sample(self, z_history, history_pad, generator=None):
        """Joint Euler ODE on tied schedule. 
        Returns the action chunks (z-scored raw space) and next-state latents."""
        B = z_history.shape[0]
        memory = self._memory(z_history)
        action = torch.randn(B, self.num_actions, self.action_raw_dim, device=z_history.device,
                             generator=generator)
        state = torch.randn(B, self.num_states, self.z_dim, device=z_history.device,
                            generator=generator)
        for i in range(self.n_flow_steps):
            tau = torch.full((B,), i / self.n_flow_steps, device=z_history.device)
            v_action, v_state = self.velocity(action, state, memory, history_pad, tau, tau)
            action = action + v_action / self.n_flow_steps
            state = state + v_state / self.n_flow_steps
        return action, state


def build_model(cfg):
    defaults = dict(encoder_size="tiny", encoder_backbone="resnet18dp", encoder_ckpt=None,
                    img_size=224, z_dim=384, proj_hidden=768, d_model=384, n_heads=6, depth=8,
                    dropout=0.1, n_flow_steps=8, fs=5, num_actions_pred=5, num_states_pred=1,
                    policy_history_len=2, actions_attend_states=True, split_tau=False,
                    tau_cond="summed")   # round-1 checkpoints: configs without the key
    return JointFlow({**defaults, **cfg})
