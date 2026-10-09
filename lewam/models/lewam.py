import torch
import torch.nn as nn
import torch.nn.functional as F

from lewam.models.encoders import build_encoder
from lewam.models.module import sinusoid


class MoTBlock(nn.Module):
    """
    MoT block over state, action modalities
    Token order: [history | state slots | goal | clean actions | action slots]
    """

    def __init__(
        self, dim: int, n_heads: int, cond_state: bool, cond_action: bool, 
        mlp_ratio: int = 4, dropout: float = 0.0
    ):
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
    def _mlp(dim: int, mlp_ratio: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim)
        )

    @staticmethod
    def _adaln_zero(dim: int) -> nn.Sequential:
        ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(ada[-1].weight)
        nn.init.zeros_(ada[-1].bias)
        return ada

    @staticmethod
    def _pre_norm(norm: nn.LayerNorm, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return norm(x) * (1 + scale) + shift

    def _modulate(
        self, x: torch.Tensor, layout: tuple[int, slice, slice], 
        cond_state: torch.Tensor | None, cond_action: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        """Return AdaLN tensors: shift, scale, gate for the attention, mlp.

        Args:
            x (torch.Tensor): embedding sequence (B, N, D)
            layout (tuple[int, slice, slice]): contains index of first action token, 
                slice of state slots, slice of action slots
            cond_state (torch.Tensor | None): (B, D) state slot conditioning
            cond_action (torch.Tensor | None): (B, D) action slot conditioning

        Returns:
            modulations (tuple[torch.Tensor, ...]): shift, scale, gate for attention, mlp, each (B, N, D)
        """
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
    def _per_stream(x: torch.Tensor, n_state: int, f_state: nn.Module, f_action: nn.Module) -> torch.Tensor:
        return torch.cat([f_state(x[:, :n_state]), f_action(x[:, n_state:])], dim=1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, layout: tuple[int, slice, slice], 
        cond_state: torch.Tensor | None, cond_action: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, N, D) token embeddings
            mask (torch.Tensor): (B, 1, N, N) additive attention mask
            layout (tuple[int, slice, slice]): contains index of first action token, 
                slice of state slots, slice of action slots
            cond_state (torch.Tensor | None): (B, D) state slot conditioning
            cond_action (torch.Tensor | None): (B, D) action slot conditioning

        Returns:
            x (torch.Tensor): (B, N, D) updated token embeddings
        """
        B, N, D = x.shape
        n_state = layout[0]
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            self._modulate(x, layout, cond_state, cond_action)
        h = self._pre_norm(self.norm_attn, x, shift_attn, scale_attn)
        qkv = self._per_stream(h, n_state, self.qkv_state, self.qkv_action)
        q, k, v = qkv.reshape(B, N, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.attn_dropout if self.training else 0.0)
        att = att.transpose(1, 2).reshape(B, N, D)
        x = x + gate_attn * self._per_stream(att, n_state, self.out_state, self.out_action)
        h = self._pre_norm(self.norm_mlp, x, shift_mlp, scale_mlp)
        return x + gate_mlp * self._per_stream(h, n_state, self.mlp_state, self.mlp_action)


class LeWAM(nn.Module):
    def __init__(self, cfg: dict) -> None:
        """
        Config:
            img_size (int): the size of the input image in pixels (assumed square)
            action_raw_dim (int): the action dim of the embodiment (e.g., dx,dy, 7 dof, etc)
            history_len (int): number of states used as context for next state/action prediction
            num_views (int): number of camera views of scene/env (e.g., scene1, wrist1, wrist2)
            num_actions_pred (int): number of actions / action chunk length of the policy
            num_states_pred (int): number of states predicted by the dynamics branch
            goal_type (str): "none" (goal-blind), "terminal" (goal token only) or "sampled" (goal token and horizon)
            goal_view_indices (list[int]): which of the camera views (0,...,num_views - 1) provide goal conditioning
                (1 or more)

            state_head (str): type of state output: "flow" (flow-matching) or "mse" (regression)
            action_head (str): type of action output: "flow" (flow-matching) or "mse" (regression)
            flow_alpha_act (float): if flow-matching action, controls sampling bias of flow timestep 
                (0 < alpha < 1: higher noise, 1: uniform, > 1: cleaner samples)
            flow_alpha_state (float): if flow-matching state, controls sampling bias of flow timestep 
                (0 < alpha < 1: higher noise, 1: uniform, > 1: cleaner samples)
            n_flow_steps (int): number of flow matching ODE integration steps

            encoder_backbone (str): the architecture of the encoder (vit, resnet18, resnet18dp, dinov3)
            encoder_size (str): encoder size: vit (tiny, small, base, large), dinov3 (small, base, large)
            encoder_ckpt (str): encoder_backbone=dinov3: weights to load (None downloads timm's)
            proj_hidden (int): the hidden layer size of the encoder's projection MLP

            sigreg_proj_dim (int): if > 0, the latent state is projected to this dim before SIGReg

            z_dim (int): the dim of the latent state (encoder output size)
            d_model (int): the embedding dim of the MoT
            n_heads (int): number of MoT attention heads
            depth (int): number of MoT blocks
            dropout (float): MoT block dropout
        """
        super().__init__()
        self.fs = cfg["fs"]
        self.num_actions_pred = cfg["num_actions_pred"]
        self.num_states_pred = cfg["num_states_pred"]
        self.n_clean_actions = self.num_states_pred * self.fs
        self.action_raw_dim = cfg["action_raw_dim"]
        self.history_len = cfg["history_len"]
        self.num_views = int(cfg.get("num_views", 1))
        self.goal_type = cfg["goal_type"]
        assert self.goal_type in ("none", "terminal", "sampled"), self.goal_type
        self.goal_cond = self.goal_type != "none"
        self.use_horizon = self.goal_type == "sampled"
        assert self.num_actions_pred >= self.num_states_pred * self.fs, "num_actions_pred must cover num_states_pred*fs"

        self.action_head = str(cfg.get("action_head", "flow"))
        self.flow_alpha_act = float(cfg.get("flow_alpha_act", 1.0))
        assert self.action_head in ("flow", "mse")
        self.state_head = str(cfg.get("state_head", "flow"))
        self.flow_alpha_state = float(cfg.get("flow_alpha_state", self.flow_alpha_act))
        assert self.state_head in ("flow", "mse")
        self.n_flow_steps = cfg["n_flow_steps"]

        self.sigreg_grouped = bool(cfg.get("sigreg_grouped", False))
        self.sigreg_proj_dim = int(cfg.get("sigreg_proj_dim", 0))

        self.n_state_tokens = self.num_states_pred * self.num_views
        self.goal_view_indices = [int(view) for view in cfg.get("goal_view_indices", [0])]
        assert all(0 <= view < self.num_views for view in self.goal_view_indices) and \
            len(set(self.goal_view_indices)) == len(self.goal_view_indices), self.goal_view_indices

        # Architecture
        self.z_dim = cfg["z_dim"]
        self.d_model = cfg["d_model"]
        d = self.d_model

        # encoder
        self.encoder = build_encoder(
            cfg["encoder_backbone"], self.z_dim, cfg["proj_hidden"], size=cfg["encoder_size"],
            img_size=cfg["img_size"], checkpoint=cfg.get("encoder_ckpt")
        )

        # state embeddings
        self.hist_proj = nn.Linear(self.z_dim, d)
        self.hist_pos_embed = nn.Parameter(torch.zeros(1, self.history_len, d))

        self.next_state_proj = nn.Linear(self.z_dim, d)
        self.next_state_pos = nn.Parameter(torch.zeros(1, self.num_states_pred, d))
        if self.state_head == "mse":
            self.next_state_query = nn.Parameter(torch.randn(1, self.num_states_pred, d) * 0.02) 

        if self.num_views > 1:
            self.view_pos = nn.Parameter(torch.zeros(1, self.num_views, d))
        if self.goal_cond:
            self.goal_proj = nn.Linear(self.z_dim, d)

        self.state_type = nn.Embedding(3, d)     # 0 history, 1 next-state, 2 goal

        # action embeddings
        self.action_proj = nn.Linear(self.action_raw_dim, d)
        self.action_pos_embed = nn.Parameter(torch.zeros(1, self.num_actions_pred, d))
        if self.action_head == "mse":
            self.action_query = nn.Parameter(torch.randn(1, self.num_actions_pred, d) * 0.02)
        self.action_type = nn.Embedding(2, d)    # 0 clean, 1 noisy/query

        # sigreg
        if self.sigreg_proj_dim > 0:
            self.sigreg_proj = nn.Linear(self.z_dim, self.sigreg_proj_dim, bias=False)
            with torch.no_grad():
                if self.sigreg_proj_dim == self.z_dim:
                    self.sigreg_proj.weight.copy_(torch.eye(self.z_dim))
                else:
                    nn.init.orthogonal_(self.sigreg_proj.weight)

        # conditioning
        self.tau_proj = nn.Linear(d, d)
        if self.use_horizon:
            self.h_proj = nn.Linear(d, d)

        # MoT
        self.blocks = nn.ModuleList(
            MoTBlock(
                d, cfg["n_heads"], cond_state=(self.state_head == "flow"),
                cond_action=(self.action_head == "flow" or self.use_horizon),
                dropout=cfg["dropout"]
            ) for _ in range(cfg["depth"])
        )
        self.action_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.action_raw_dim))
        self.state_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.z_dim))

        # Token layout
        n_hist, n_state_slots = self.history_len * self.num_views, self.n_state_tokens
        n_clean_actions, n_action_slots = self.n_clean_actions, self.num_actions_pred
        idx, offset = {}, 0

        idx["hist"] = list(range(offset, offset + n_hist))
        offset += n_hist
        idx["state_slots"] = list(range(offset, offset + n_state_slots))
        offset += n_state_slots
        idx["goal"] = list(range(offset, offset + len(self.goal_view_indices))) \
            if self.goal_cond else []
        offset += len(idx["goal"])
        idx["clean_actions"] = list(range(offset, offset + n_clean_actions))
        offset += n_clean_actions
        idx["action_slots"] = list(range(offset, offset + n_action_slots))
        offset += n_action_slots

        self.n_tokens = offset
        self.token_indices = idx

        # "slots" are noisy/query tokens which are decoded into predicted state and action (velocities)
        state_rows, action_rows = idx["state_slots"], idx["action_slots"]
        self.layout = (
            (idx["clean_actions"] + idx["action_slots"])[0],
            slice(state_rows[0], state_rows[-1] + 1) if state_rows else slice(0, 0),
            slice(action_rows[0], action_rows[-1] + 1)
        )

        # Attention map
        attends = torch.zeros(self.n_tokens, self.n_tokens, dtype=torch.bool)
        hist, goal = idx["hist"], idx["goal"]
        state_slots, clean_actions, action_slots = idx["state_slots"], idx["clean_actions"], idx["action_slots"]

        for row in hist:                        # history attends to itself
            attends[row, hist] = True
        # slots are ordered s.t. all `num_states_pred` timesteps for a view are contiguous 
        # i.e. slot_idx = (view_idx * num_states_pred) + step
        for i, row in enumerate(state_slots):   # state slots attend to history, all slots for all views up to the curr timestep,
            step = i % self.num_states_pred     #     all actions up to the curr timestep
            attends[row, hist] = True
            attends[row, [state_slots[view_idx * self.num_states_pred + t]
                          for view_idx in range(self.num_views) for t in range(step + 1)]] = True
            attends[row, clean_actions[:(step + 1) * self.fs]] = True
        for row in goal:                        # goal attends history and itself
            attends[row, hist] = True
            attends[row, goal] = True
        for i, row in enumerate(clean_actions): # clean actions attend history, and itself causally
            attends[row, hist] = True
            attends[row, clean_actions[:i + 1]] = True
        for i, row in enumerate(action_slots):  # action slots attend history, goal, and itself causally
            attends[row, hist] = True
            attends[row, goal] = True
            attends[row, action_slots[:i + 1]] = True
        self.register_buffer("attends", attends)

    # -- Helpers --
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixels)

    def _build_attn_mask(self, history_pad: torch.Tensor, goal_keep: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            history_pad (torch.Tensor): (B, history_len) boolean tensor indicating which history
                tokens are padded (True) and should not be attended to
            goal_keep (torch.Tensor | None): (B,) boolean tensor, False where the goal tokens should not be attended to

        Returns:
           mask (torch.Tensor): (B, 1, N, N) additive attention mask
        """
        B = history_pad.shape[0]
        attends = self.attends[None].expand(B, -1, -1).clone()
        drop_goal = goal_keep is not None and not bool(goal_keep.all())
        if history_pad.any() or drop_goal:
            key_pad = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=history_pad.device)
            key_pad[:, self.token_indices["hist"]] = history_pad.repeat(1, self.num_views)
            if drop_goal:
                key_pad[:, self.token_indices["goal"]] = ~goal_keep[:, None]
            attends = attends & ~key_pad[:, None, :]

        mask = torch.zeros(B, self.n_tokens, self.n_tokens, device=history_pad.device)
        mask[~attends] = float("-inf")
        return mask[:, None]

    def _history_pos_embed(self) -> torch.Tensor:
        """Returns history position embeddings (1, views*history_len, D)"""
        pos = self.hist_pos_embed.repeat(1, self.num_views, 1)
        if self.num_views > 1:
            pos = pos + self.view_pos.repeat_interleave(self.history_len, dim=1)
        return pos

    def _state_slot_pos_embed(self) -> torch.Tensor:
        """Returns the state slot position embeddings (1, views*num_states_pred, D)"""
        pos = self.next_state_pos.repeat(1, self.num_views, 1)
        if self.num_views > 1:
            pos = pos + self.view_pos.repeat_interleave(self.num_states_pred, dim=1)
        return pos

    def _compute_slot_conditioning(
        self, tau_a: torch.Tensor | None, tau_s: torch.Tensor | None, h_norm: torch.Tensor | None,
        goal_keep: torch.Tensor, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute the conditioning for state, action slots for AdaLN

        Args:
            tau_a (torch.Tensor | None): (B,) action flow time
            tau_s (torch.Tensor | None): (B,) state flow time
            h_norm (torch.Tensor | None): (B,) normalized goal horizon; None excludes it
            goal_keep (torch.Tensor): (B,) bool, False excludes the horizon of a dropped goal
            batch_size (int): batch size
            device (torch.device): device for the excluded horizon

        Returns:
            cond_state (torch.Tensor | None): (B, D) state slot conditioning
            cond_action (torch.Tensor | None): (B, D) action slot conditioning
        """
        assert self.use_horizon or h_norm is None, f"goal_type {self.goal_type} has no horizon input"
        cond_state = self.tau_proj(sinusoid(tau_s, self.d_model)) if self.state_head == "flow" else None
        cond_action = self.tau_proj(sinusoid(tau_a, self.d_model)) if self.action_head == "flow" else None
        if self.use_horizon:
            if h_norm is None:
                h = torch.zeros(batch_size, self.d_model, device=device)
            else:
                h = self.h_proj(sinusoid(h_norm, self.d_model)) * goal_keep[:, None]
            cond_action = h if cond_action is None else cond_action + h
        return cond_state, cond_action

    def forward_mot(
        self, z_history: torch.Tensor, history_pad: torch.Tensor, clean_actions: torch.Tensor,
        noisy_actions: torch.Tensor | None, noisy_state: torch.Tensor | None,
        tau_a: torch.Tensor | None, tau_s: torch.Tensor | None, 
        z_goal: torch.Tensor | None = None, h_norm: torch.Tensor | None = None, goal_keep: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_history (torch.Tensor): (B, views*history_len, z_dim) or (B, views, history_len, z_dim) history latents
            history_pad (torch.Tensor): (B, history_len) True where a history frame is padding
            clean_actions (torch.Tensor): (B, n_clean_actions, action_raw_dim) z-scored clean actions
            noisy_actions (torch.Tensor | None): (B, num_actions_pred, action_raw_dim) action slot inputs if flow-matching
            noisy_state (torch.Tensor | None): (B, views*num_states_pred, z_dim) state slot inputs inputs if flow-matching
            tau_a (torch.Tensor | None): (B,) action flow time
            tau_s (torch.Tensor | None): (B,) state flow time
            z_goal (torch.Tensor | None): (B, z_dim) or (B, goal views, z_dim) goal latents; None excludes the goal and horizon
            h_norm (torch.Tensor | None): (B,) normalized goal horizon
            goal_keep (torch.Tensor | None): (B,) bool, False excludes that sample's goal and horizon (goal dropout)

        Returns:
            action_pred (torch.Tensor): (B, num_actions_pred, action_raw_dim)
            state_pred (torch.Tensor): (B, views*num_states_pred, z_dim)
        """
        if z_history.dim() == 4:
            z_history = z_history.flatten(1, 2)
        B, d = z_history.shape[0], self.d_model
        tokens  = z_history.new_zeros(B, self.n_tokens, d)

        # embed history
        state_type_embed = self.state_type.weight
        hist_pos_embed = self._history_pos_embed()
        tokens[:, self.token_indices["hist"]] = self.hist_proj(z_history) + hist_pos_embed + state_type_embed[0]

        # embed state slots
        next_state_pos_embed = self._state_slot_pos_embed()
        if self.state_head == "mse":
            tokens[:, self.token_indices["state_slots"]] = self.next_state_query.repeat(1, self.num_views, 1).expand(B, -1, -1) \
                + next_state_pos_embed + state_type_embed[1]
        else:
            tokens[:, self.token_indices["state_slots"]] = self.next_state_proj(noisy_state) + next_state_pos_embed + state_type_embed[1]

        # embed goal; a dropped or missing goal is masked out of attention
        if goal_keep is None:
            goal_keep = torch.full((B,), z_goal is not None, dtype=torch.bool, device=z_history.device)
        if self.goal_cond and z_goal is not None:
            n_goal = len(self.goal_view_indices)
            goal_embed = self.goal_proj(z_goal[:, None] if z_goal.dim() == 2 else z_goal)
            assert goal_embed.shape[1] == n_goal, (goal_embed.shape, n_goal)
            if self.num_views > 1:
                goal_embed = goal_embed + self.view_pos[:, self.goal_view_indices]
            tokens[:, self.token_indices["goal"]] = goal_embed + state_type_embed[2]

        # embed action
        action_type_embed = self.action_type.weight
        tokens[:, self.token_indices["clean_actions"]] = self.action_proj(clean_actions) + self.action_pos_embed[:, :self.n_clean_actions] + action_type_embed[0]
        if self.action_head == "mse":
            tokens[:, self.token_indices["action_slots"]] = self.action_query.expand(B, -1, -1) + self.action_pos_embed + action_type_embed[1]
        else:
            tokens[:, self.token_indices["action_slots"]] = self.action_proj(noisy_actions) + self.action_pos_embed + action_type_embed[1]

        # MoT blocks (w/ conditioning)
        cond_state, cond_action = self._compute_slot_conditioning(tau_a, tau_s, h_norm, goal_keep, B, z_history.device)
        attn_mask = self._build_attn_mask(history_pad, goal_keep if self.goal_cond else None)
        for block in self.blocks:
            tokens = block(tokens, attn_mask, self.layout, cond_state, cond_action)
        action_slot_feats = tokens[:, self.token_indices["action_slots"]]
        action_pred, next_state_pred = self.action_out(action_slot_feats), self.state_out(tokens[:, self.token_indices["state_slots"]])

        return action_pred, next_state_pred

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        per_token = ((pred - target) ** 2).mean(-1) * valid
        return (per_token.sum(-1) / valid.sum(-1).clamp(min=1)).mean()

    # -- Training --
    def loss(
        self, z_history: torch.Tensor, history_pad: torch.Tensor, action_target: torch.Tensor, action_valid: torch.Tensor,
        state_target: torch.Tensor, state_valid: torch.Tensor, z_goal: torch.Tensor | None = None, 
        h_norm: torch.Tensor | None = None, goal_keep: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute state and action loss
        
        Args:
            z_history (torch.Tensor): (B, views*history_len, z_dim) or (B, views, history_len, z_dim) history latents
            history_pad (torch.Tensor): (B, history_len) True where a history frame is padding
            action_target (torch.Tensor): (B, num_actions_pred, action_raw_dim) z-scored action chunk
            action_valid (torch.Tensor): (B, num_actions_pred) 1 where the action lies inside the episode
            state_target (torch.Tensor): (B, views*num_states_pred, z_dim) target latents
            state_valid (torch.Tensor): (B, views*num_states_pred) 1 where the state lies inside the episode
            z_goal (torch.Tensor | None): (B, z_dim) or (B, goal views, z_dim) goal latents
            h_norm (torch.Tensor | None): (B,) normalized goal horizon
            goal_keep (torch.Tensor | None): (B,) bool, False excludes that sample's goal and horizon (goal dropout)

        Returns:
            loss_action (torch.Tensor): scalar action loss
            loss_state (torch.Tensor): scalar state loss, 0 when num_states_pred == 0
        """
        B = z_history.shape[0]
        device = z_history.device

        # sample flow timesteps
        if self.action_head == "flow":
            tau_a = self._sample_flow_time(B, self.flow_alpha_act, device)
            action_noise = torch.randn_like(action_target)
            noisy_actions = torch.lerp(action_noise, action_target, tau_a[:, None, None].to(action_target.dtype))
        else:
            tau_a = action_noise = noisy_actions = None
        if self.state_head == "flow":
            tau_s = self._sample_flow_time(B, self.flow_alpha_state, device)
            state_noise = torch.randn_like(state_target)
            noisy_states = torch.lerp(state_noise, state_target, tau_s[:, None, None].to(state_target.dtype))
        else:
            tau_s = state_noise = noisy_states = None

        # MoT forward
        clean_actions = action_target[:, :self.n_clean_actions]
        action_pred, state_pred = self.forward_mot(z_history, history_pad, clean_actions, noisy_actions, noisy_states,
                                                      tau_a, tau_s, z_goal, h_norm, goal_keep)
        
        # compute losses
        if self.action_head == "mse": 
            loss_action = self._masked_mse(action_pred, action_target, action_valid)
        else:
            loss_action = self._masked_mse(action_pred, action_target - action_noise, action_valid)
        if self.num_states_pred == 0:
            return loss_action, loss_action.new_zeros(())
        
        if self.state_head == "mse":
            return loss_action, self._masked_mse(state_pred, state_target, state_valid)
        return loss_action, self._masked_mse(state_pred, state_target - state_noise, state_valid)

    @staticmethod
    def _sample_flow_time(B: int, alpha: float, device: torch.device) -> torch.Tensor:
        return torch.rand(B, device=device) ** (1.0 / alpha)

    # -- Inference --
    def sample_action_noise(self, batch_size: int, device: torch.device,
                            generator: torch.Generator | None = None) -> torch.Tensor:
        """Initial action-flow noise, (batch_size, num_actions_pred, action_raw_dim)."""
        return torch.randn(batch_size, self.num_actions_pred, self.action_raw_dim, device=device, generator=generator)

    def sample_state_noise(self, batch_size: int, device: torch.device,
                           generator: torch.Generator | None = None) -> torch.Tensor:
        """Initial state-flow noise, (batch_size, views*num_states_pred, z_dim)."""
        return torch.randn(batch_size, self.n_state_tokens, self.z_dim, device=device, generator=generator)

    def predict_actions(
        self, z_history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor | None = None,
        h_norm: torch.Tensor | None = None, action_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """
        Predict the next action chunk via the policy

        Args:
            z_history (torch.Tensor): (B, views*history_len, z_dim) or (B, views, history_len, z_dim) history latents
            history_pad (torch.Tensor): (B, history_len) True where a history frame is padding
            z_goal (torch.Tensor | None): (B, z_dim) or (B, goal views, z_dim) goal latents
            h_norm (torch.Tensor | None): (B,) normalized goal horizon
            action_noise (torch.Tensor | None): (B, num_actions_pred, action_raw_dim) initial flow noise, drawn if None
            generator (torch.Generator | None): RNG for the drawn noise

        Returns:
            actions (torch.Tensor): (B, num_actions_pred, action_raw_dim) z-scored action chunk
        """
        B = z_history.shape[0]
        device = z_history.device

        # "turn off" dynamics prediction
        zero_clean_actions = torch.zeros(B, self.n_clean_actions, self.action_raw_dim, device=device, dtype=z_history.dtype)
        zero_state_slots = torch.zeros(B, self.n_state_tokens, self.z_dim, device=device, dtype=z_history.dtype)
        one = torch.ones(B, device=device)

        if self.action_head == "mse":
            actions, _ = self.forward_mot(z_history, history_pad, zero_clean_actions, None, zero_state_slots, None, one, z_goal, h_norm)
        else:
            # ODE integration
            actions = action_noise if action_noise is not None else self.sample_action_noise(B, device, generator)
            for i in range(self.n_flow_steps):
                tau = torch.full((B,), i / self.n_flow_steps, device=device)
                v_action, _ = self.forward_mot(z_history, history_pad, zero_clean_actions, actions, zero_state_slots, tau, one, z_goal, h_norm)
                actions = actions + v_action / self.n_flow_steps

        return actions

    def predict_states(
        self, z_history: torch.Tensor, history_pad: torch.Tensor, actions: torch.Tensor,
        state_noise: torch.Tensor | None = None, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """
        Predict the next latent state(s) via forward dynamics

        Args:
            z_history (torch.Tensor): (B, views*history_len, z_dim) or (B, views, history_len, z_dim) history latents
            history_pad (torch.Tensor): (B, history_len) True where a history frame is padding
            actions (torch.Tensor): (B, T, action_raw_dim) z-scored actions; the first n_clean_actions are used, zero-padded if T is shorter
            state_noise (torch.Tensor | None): (B, views*num_states_pred, z_dim) initial flow noise, drawn if None
            generator (torch.Generator | None): RNG for the drawn noise

        Returns:
            states (torch.Tensor): (B, views*num_states_pred, z_dim) predicted latent states
        """
        B = z_history.shape[0]
        device = z_history.device

        clean_actions = actions[:, :self.n_clean_actions]
        if clean_actions.shape[1] < self.n_clean_actions:
            pad = clean_actions.new_zeros(B, self.n_clean_actions - clean_actions.shape[1], clean_actions.shape[2])
            clean_actions = torch.cat([clean_actions, pad], dim=1)

        # "turn off" policy prediction
        zero_action_slots = torch.zeros(B, self.num_actions_pred, self.action_raw_dim, device=device, dtype=clean_actions.dtype)
        one = torch.ones(B, device=device)

        if self.state_head == "mse":
            _, states = self.forward_mot(z_history, history_pad, clean_actions, zero_action_slots, None, one, None)
        else:
            # ODE integration
            states = state_noise if state_noise is not None else self.sample_state_noise(B, device, generator)
            for i in range(self.n_flow_steps):
                tau = torch.full((B,), i / self.n_flow_steps, device=device)
                _, v_state = self.forward_mot(z_history, history_pad, clean_actions, zero_action_slots, states, one, tau)
                states = states + v_state / self.n_flow_steps

        return states

    def param_group_of(self, name: str) -> str:
        """Returns the parameter group of a named parameter for the optimizer"""
        if name.startswith("encoder."):
            return "encoder"
        if name.startswith(("next_state_proj", "next_state_pos", "next_state_query", "state_out")):
            return "dynamics"
        if name.startswith("blocks."):
            parts = name.split(".")
            if len(parts) > 2 and parts[2].endswith("_state"):       # blocks.<i>.<module>_state.*
                return "dynamics"
        return "policy"


def build_model(cfg: dict) -> LeWAM:
    defaults = dict(
        encoder_backbone="resnet18dp", encoder_ckpt=None,
        img_size=224, z_dim=192, proj_hidden=768, d_model=192, n_heads=4, depth=4,
        dropout=0.1, n_flow_steps=8, fs=5, num_actions_pred=10, num_states_pred=1,
        history_len=2, goal_type="none", state_head="mse",
        flow_alpha_act=1.0, num_views=1
    )
    return LeWAM({**defaults, **cfg})
