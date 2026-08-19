"""Cross-attention LeWAM. Both heads read a shared per-frame memory by cross-attention: the policy
denoises raw-action tokens conditioned on frame + goal tokens; the dynamics regresses the next latent
from a frame's history and its own action block. See docs/wip/CROSSATTN_DESIGN.md."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewam.models.module import VisionEncoder


def sinusoid(x, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
    a = x[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class CrossAttnBlock(nn.Module):
    """Causal self-attention over the action tokens, cross-attention into memory, MLP; each sublayer
    AdaLN-Zero modulated by the scalar conditioning (flow-time + horizon)."""

    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(mlp_ratio * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, memory, cond, self_mask, memory_pad):
        # cond: (B, dim) = one modulation for all tokens, or (B, n_tokens, dim) = per-token
        # modulation (jointflow's per-modality flow-time conditioning).
        gates = self.ada(cond).chunk(9, dim=-1)
        sh_s, sc_s, g_s, sh_c, sc_c, g_c, sh_m, sc_m, g_m = \
            (t.unsqueeze(1) if t.dim() == 2 else t for t in gates)
        h = self.norm_self(x) * (1 + sc_s) + sh_s
        x = x + g_s * self.self_attn(h, h, h, attn_mask=self_mask, need_weights=False)[0]
        h = self.norm_cross(x) * (1 + sc_c) + sh_c
        x = x + g_c * self.cross_attn(h, memory, memory, key_padding_mask=memory_pad,
                                      need_weights=False)[0]
        h = self.norm_mlp(x) * (1 + sc_m) + sh_m
        return x + g_m * self.mlp(h)


class PolicyHead(nn.Module):
    """Flow-matching over N raw-action tokens cross-attending to [frame tokens (+ goal)]."""

    def __init__(self, latent_dim, action_dim, n_tokens, history_len, dim, n_heads, depth,
                 n_flow_steps, dropout=0.0):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_flow_steps = n_flow_steps
        self.frame_in = nn.Linear(latent_dim, dim)
        self.frame_pos = nn.Parameter(torch.zeros(1, history_len, dim))   # most recent frame at index -1
        self.goal_in = nn.Linear(latent_dim, dim)
        self.type_emb = nn.Embedding(2, dim)                       # 0 frame, 1 goal
        self.action_in = nn.Linear(action_dim, dim)
        self.action_pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.cond = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossAttnBlock(dim, n_heads, dropout=dropout)
                                    for _ in range(depth))
        self.action_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, action_dim))
        self.dim = dim
        causal = torch.triu(torch.full((n_tokens, n_tokens), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal)

    def _memory(self, frames, frame_pad, goal, use_goal):
        m = self.frame_in(frames) + self.frame_pos + self.type_emb.weight[0]   # (M, H, dim)
        pad = frame_pad
        if use_goal:
            g = (self.goal_in(goal) + self.type_emb.weight[1]).unsqueeze(1)
            m = torch.cat([m, g], dim=1)
            pad = torch.cat([pad, pad.new_zeros(pad.shape[0], 1)], dim=1)
        return m, pad

    def _cond(self, tau, horizon):
        return self.cond(sinusoid(tau, self.dim) + sinusoid(horizon, self.dim))

    def velocity(self, noisy, memory, memory_pad, tau, horizon):
        x = self.action_in(noisy) + self.action_pos
        cond = self._cond(tau, horizon)
        for block in self.blocks:
            x = block(x, memory, cond, self.causal_mask, memory_pad)
        return self.action_out(x)

    def loss(self, frames, frame_pad, goal, use_goal, target, target_valid, horizon, item_valid):
        memory, memory_pad = self._memory(frames, frame_pad, goal, use_goal)
        tau = torch.rand(target.shape[0], device=target.device)
        noise = torch.randn_like(target)
        noisy = torch.lerp(noise, target, tau[:, None, None])
        pred = self.velocity(noisy, memory, memory_pad, tau, horizon)
        per_token = ((pred - (target - noise)) ** 2).mean(-1) * target_valid
        per_item = per_token.sum(-1) / target_valid.sum(-1).clamp(min=1)
        return (per_item * item_valid).sum() / item_valid.sum().clamp(min=1)

    @torch.no_grad()
    def act(self, frames, frame_pad, goal, use_goal, horizon):
        memory, memory_pad = self._memory(frames, frame_pad, goal, use_goal)
        a = torch.randn(frames.shape[0], self.n_tokens, self.action_out[-1].out_features,
                        device=frames.device)
        for i in range(self.n_flow_steps):
            tau = torch.full((a.shape[0],), i / self.n_flow_steps, device=a.device)
            a = a + self.velocity(a, memory, memory_pad, tau, horizon) / self.n_flow_steps
        return a


class DPHead(nn.Module):
    """DP's exact transformer denoiser + DDPM, conditioned on our encoder's frame latents. An
    isolation test: if this underperforms real DP, the gap is our encoder/training, not the head."""

    def __init__(self, latent_dim, action_dim, n_tokens, history_len, dim, n_heads, depth,
                 n_steps=100, dropout=0.1):
        super().__init__()
        from lewam.models.dp_transformer import TransformerForDiffusion
        self.net = TransformerForDiffusion(
            input_dim=action_dim, output_dim=action_dim, horizon=n_tokens, n_obs_steps=history_len,
            cond_dim=latent_dim, n_layer=depth, n_head=n_heads, n_emb=dim,
            causal_attn=True, time_as_cond=True, obs_as_cond=True, n_cond_layers=0,
            p_drop_attn=dropout, p_drop_emb=0.0)
        self.n_tokens = n_tokens
        self.n_flow_steps = n_steps                      # named to reuse build_policy's step override
        s = 0.008
        u = torch.linspace(0, n_steps, n_steps + 1) / n_steps
        acp = torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
        acp = acp / acp[0]
        betas = (1 - acp[1:] / acp[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("acp", torch.cumprod(alphas, 0))

    def _cond(self, frames, frame_pad):
        # DP feeds a fixed obs window; fill left-padded slots with the earliest real frame (its convention)
        B, hl, D = frames.shape
        idx = torch.arange(hl, device=frames.device).expand(B, hl)
        first_real = frame_pad.long().sum(1, keepdim=True)
        src = torch.maximum(idx, first_real).unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(frames, 1, src)

    def loss(self, frames, frame_pad, goal, use_goal, target, target_valid, horizon, item_valid):
        cond = self._cond(frames, frame_pad)
        t = torch.randint(0, len(self.betas), (target.shape[0],), device=target.device)
        noise = torch.randn_like(target)
        acp = self.acp[t][:, None, None]
        x_t = acp.sqrt() * target + (1 - acp).sqrt() * noise
        pred = self.net(x_t, t, cond)
        per_token = ((pred - noise) ** 2).mean(-1) * target_valid
        per_item = per_token.sum(-1) / target_valid.sum(-1).clamp(min=1)
        return (per_item * item_valid).sum() / item_valid.sum().clamp(min=1)

    @torch.no_grad()
    def act(self, frames, frame_pad, goal, use_goal, horizon):
        cond = self._cond(frames, frame_pad)
        x = torch.randn(frames.shape[0], self.n_tokens, self.net.head.out_features, device=frames.device)
        for t in reversed(range(len(self.betas))):
            eps = self.net(x, torch.full((x.shape[0],), t, device=x.device, dtype=torch.long), cond)
            acp, a, beta = self.acp[t], self.alphas[t], self.betas[t]
            mean = (x - beta / (1 - acp).sqrt() * eps) / a.sqrt()
            x = mean + (beta.sqrt() * torch.randn_like(x) if t > 0 else 0.0)
        return x


class DynamicsHead(nn.Module):
    """One state query per frameskip transition. Cross-attends to that transition's frame history and
    its own clean action block; regresses the next latent. The target is NOT detached here -- the
    encoder learns from being the prediction target (unified's default; anti-collapse is SIGReg on
    the latent). A caller wanting a stop-grad target detaches it before passing."""

    def __init__(self, latent_dim, action_dim, history_len, dim, n_heads, depth, dropout=0.0):
        super().__init__()
        self.query_in = nn.Linear(latent_dim, dim)
        self.frame_in = nn.Linear(latent_dim, dim)
        self.frame_pos = nn.Parameter(torch.zeros(1, history_len, dim))
        self.goal_in = nn.Linear(latent_dim, dim)
        self.action_in = nn.Linear(action_dim, dim)
        self.blocks = nn.ModuleList(_DynBlock(dim, n_heads, dropout) for _ in range(depth))
        self.state_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim))

    def predict(self, start, frames, frame_pad, action, goal, use_goal):
        q = self.query_in(start).unsqueeze(1)                     # (M, 1, dim)
        kv = torch.cat([self.frame_in(frames) + self.frame_pos, self.action_in(action)], dim=1)
        pad = torch.cat([frame_pad, frame_pad.new_zeros(action.shape[0], action.shape[1])], dim=1)
        if use_goal:
            kv = torch.cat([kv, self.goal_in(goal).unsqueeze(1)], dim=1)
            pad = torch.cat([pad, pad.new_zeros(pad.shape[0], 1)], dim=1)
        for block in self.blocks:
            q = block(q, kv, pad)
        return self.state_out(q.squeeze(1))

    def loss(self, start, frames, frame_pad, action, target, goal, use_goal, item_valid):
        pred = self.predict(start, frames, frame_pad, action, goal, use_goal)
        per_item = ((pred - target) ** 2).mean(-1)
        return (per_item * item_valid).sum() / item_valid.sum().clamp(min=1)


class _DynBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, q, kv, kv_pad):
        q = q + self.cross_attn(self.norm_cross(q), kv, kv, key_padding_mask=kv_pad,
                                need_weights=False)[0]
        return q + self.mlp(self.norm_mlp(q))


class LeWAMCrossAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_dim = cfg["z_dim"]
        self.fs = cfg["fs"]
        self.num_chunks = cfg["num_chunks"]
        self.n_tokens = self.fs * self.num_chunks
        self.action_raw_dim = cfg["action_raw_dim"]
        self.policy_history_len = cfg["policy_history_len"]
        self.dyn_history_len = cfg["dyn_history_len"]
        self.goal_conditioning = cfg["goal_conditioning"]
        self.dyn_goal_cond = cfg["dyn_goal_cond"]
        self.encoder = VisionEncoder(size=cfg["encoder_size"], output_type="cls",
                                     output_dim=self.z_dim, img_size=cfg["img_size"],
                                     backbone=cfg["encoder_backbone"],
                                     backbone_ckpt=cfg.get("encoder_ckpt"))
        head = cfg.get("head", "flow")
        if head == "dp":
            self.policy_head = DPHead(self.z_dim, self.action_raw_dim, self.n_tokens,
                                      self.policy_history_len, cfg["d_model"], cfg["n_heads"],
                                      cfg["policy_depth"], cfg["n_flow_steps"], cfg["dropout"])
        else:
            self.policy_head = PolicyHead(self.z_dim, self.action_raw_dim, self.n_tokens,
                                          self.policy_history_len, cfg["d_model"], cfg["n_heads"],
                                          cfg["policy_depth"], cfg["n_flow_steps"], cfg["dropout"])
        self.dynamics_head = DynamicsHead(self.z_dim, self.action_raw_dim, self.dyn_history_len,
                                          cfg["d_model"], cfg["n_heads"], cfg["dyn_depth"],
                                          cfg["dropout"])

    def encode(self, pixels):
        return self.encoder(pixels)

    def _frame_history(self, z_states, history_len):
        B, P, D = z_states.shape
        padded = torch.cat([z_states.new_zeros(B, history_len - 1, D), z_states], dim=1)
        history = padded.unfold(1, history_len, 1).permute(0, 1, 3, 2)          # (B, P, history_len, D)
        idx = torch.arange(P)[:, None] - (history_len - 1) + torch.arange(history_len)[None, :]
        pad = (idx < 0).to(z_states.device)                                     # (P, history_len)
        return history.reshape(B * P, history_len, D), pad[None].expand(B, P, -1).reshape(B * P, -1)

    def _raw_chunk(self, actions, valid):
        B, P, adim = actions.shape
        nc = self.num_chunks
        blocks = torch.cat([actions, actions.new_zeros(B, nc - 1, adim)], dim=1)
        blocks = blocks.unfold(1, nc, 1).permute(0, 1, 3, 2)                    # (B, P, nc, adim)
        raw = blocks.reshape(B, P, self.n_tokens, self.action_raw_dim).float()
        block_valid = torch.cat([valid, valid.new_zeros(B, nc - 1)], dim=1).unfold(1, nc, 1)
        raw_valid = block_valid.unsqueeze(-1).expand(B, P, nc, self.fs).reshape(B, P, self.n_tokens)
        return raw.reshape(B * P, self.n_tokens, self.action_raw_dim), raw_valid.reshape(B * P, self.n_tokens).float()

    def policy_loss(self, z_window, z_goal, actions, horizon, valid):
        B, P = valid.shape
        frames, frame_pad = self._frame_history(z_window[:, :P], self.policy_history_len)
        target, target_valid = self._raw_chunk(actions, valid)
        return self.policy_head.loss(
            frames, frame_pad, z_goal.reshape(B * P, self.z_dim), self.goal_conditioning,
            target, target_valid, horizon.reshape(B * P).float(), valid.reshape(B * P).float())

    def dyn_loss(self, z_window, z_goal, actions, valid):
        B, P = valid.shape
        z_states = z_window[:, :P]
        frames, frame_pad = self._frame_history(z_states, self.dyn_history_len)
        block = actions.reshape(B * P, self.fs, self.action_raw_dim)
        return self.dynamics_head.loss(
            z_states.reshape(B * P, self.z_dim), frames, frame_pad, block,
            z_window[:, 1:P + 1].reshape(B * P, self.z_dim),
            z_goal.reshape(B * P, self.z_dim), self.dyn_goal_cond, valid.reshape(B * P).float())


def build_model(cfg):
    defaults = dict(encoder_size="tiny", encoder_backbone="resnet18sp", encoder_ckpt=None,
                    img_size=224, z_dim=512, d_model=256, n_heads=4, policy_depth=6, dyn_depth=3,
                    dropout=0.1, n_flow_steps=8, num_chunks=2, policy_history_len=3,
                    dyn_history_len=3, goal_conditioning=True, dyn_goal_cond=False, head="flow")
    return LeWAMCrossAttn({**defaults, **cfg})
