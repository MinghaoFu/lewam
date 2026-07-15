"""LeWAM-Seq: one causal Transformer over interleaved [z_t, e_t] state/action tokens.
Read-out is split by token position:
  - state-token (z_t) -> predictor -> next-action-embed (h_e_t) -> action head -> a_t
  - action-token (e_t) -> predictor -> next-state-embed (h_z_{t+1}) -> dynamics head -> z_{t+1}
Both heads take an optional z_goal (zero-filled when absent, to support goal-conditioned 
and goal-agnostic settings via goal_dropout in the training loop.
"""
import torch
from torch import nn

from lewam.models.module import (
    ViTEncoder, Embedder, Transformer, ModalityAdapter, AdaLNBlock, sinusoidal_embedding,
)


class ActionHead(nn.Module):
    """cat[h_act, z_goal] -> AdaLN blocks conditioned on horizon -> action.
    TODO: deterministic MLP-style decoder only for now; add gmm/diffusion decoders later."""

    def __init__(self, embed_dim, z_dim, action_dim, hidden_dim=256, 
                 n_freqs=64, cond_dim=128, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.horizon_mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.block1 = AdaLNBlock(embed_dim + z_dim, hidden_dim, cond_dim, dropout)
        self.block2 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, h_act, z_goal, h_norm):
        cond = self.horizon_mlp(sinusoidal_embedding(h_norm, self.n_freqs))
        x = torch.cat([h_act, z_goal], dim=-1)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        return self.out(x)


class DynamicsHead(nn.Module):
    """cat[h_next, z_goal] -> MLP -> next state latent."""

    def __init__(self, embed_dim, z_dim, mlp_hidden=512):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim + z_dim, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, z_dim),
        )

    def forward(self, h_next, z_goal=None):
        if z_goal is None:
            z_goal = h_next.new_zeros(*h_next.shape[:-1], self.z_dim)
        return self.net(torch.cat([h_next, z_goal], dim=-1))


class LeWAMSeq(nn.Module):
    def __init__(
        self,
        act_dim,
        img_size=224,
        encoder_size="tiny",
        embed_dim=192,
        n_layers=6,
        n_heads=16,
        dim_head=64,
        mlp_dim=384,
        num_frames=8,
        head_hidden=256,
        dropout=0.1,
    ):
        super().__init__()
        self.num_frames = num_frames

        # Encoders
        # NOTE: currently supports CLS latent only - patch-level extension requires block-causal masking
        # TODO: currently supports hf's ViT - can extend to pretrained (e.g. DINO, V-JEPA) by modifying module.py
        self.state_encoder = ViTEncoder(size=encoder_size, output_dim=embed_dim, img_size=img_size)
        self.act_encoder = Embedder(input_dim=act_dim, smoothed_dim=act_dim, emb_dim=embed_dim)

        # Modality adapters + shared per-timestep PE
        self.state_adapter = ModalityAdapter(embed_dim)
        self.act_adapter = ModalityAdapter(embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, embed_dim) * 0.02)

        # Predictor
        self.predictor = Transformer(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=n_layers,
            heads=n_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # Heads
        self.action_head = ActionHead(embed_dim=embed_dim, z_dim=embed_dim, action_dim=act_dim,
                                       hidden_dim=head_hidden, dropout=dropout)
        self.dynamics_head = DynamicsHead(embed_dim=embed_dim, z_dim=embed_dim,
                                           mlp_hidden=head_hidden)

    # -- Encode
    def encode_frames(self, pixels):
        """pixels: (B,T,C,H,W) -> z: (B,T,embed_dim)"""
        b, t = pixels.shape[:2]
        z = self.state_encoder(pixels.reshape(b * t, *pixels.shape[2:]))
        return z.reshape(b, t, -1)

    def encode_actions(self, actions):
        """actions: (B,T,act_dim) -> e: (B,T,embed_dim)"""
        return self.act_encoder(actions)

    # -- Tokenize
    def tokenize(self, z, e):
        """z, e: (B,T,D) same T -> (B,2T,D) interleaved [z_0,e_0,...,z_{T-1},e_{T-1}]."""
        T = z.size(1)
        assert T <= self.num_frames, f"window {T} > num_frames {self.num_frames}"
        pos = self.pos_embedding[:, :T]
        z = self.state_adapter(z) + pos
        e = self.act_adapter(e) + pos
        return torch.stack([z, e], dim=2).flatten(1, 2)

    def _tokenize_pending(self, z, e_hist):
        """z: (B,T,D), e_hist: (B,T-1,D) (a_{T-1} not yet known) -> (B,2T-1,D)
        ending on the unpaired z_{T-1} token."""
        T = z.size(1)
        pos = self.pos_embedding[:, :T]
        zt = self.state_adapter(z) + pos
        if T == 1:
            return zt
        
        assert e_hist.size(1) == T - 1, (e_hist.size(1), T)
        et = self.act_adapter(e_hist) + pos[:, :T - 1]
        paired = torch.stack([zt[:, :-1], et], dim=2).flatten(1, 2)
        return torch.cat([paired, zt[:, -1:]], dim=1)

    # -- Heads
    def _apply_action_head(self, h_act, h_norm, z_goal=None):
        # h_norm is (B,) [one horizon per window, broadcast] OR (B,T) [per-position
        # horizon]. Per-position is the correct training signal: each frame in the
        # window is a different distance from the goal, and the rollout feeds the
        # current frame's decreasing horizon -- a single broadcast value trains the
        # earlier frames on the wrong horizon.
        b, t, d = h_act.shape
        zg = h_act.new_zeros(b, t, d) if z_goal is None else z_goal[:, None, :].expand(b, t, d)
        hn = h_norm[:, None].expand(b, t).reshape(b * t) if h_norm.dim() == 1 else h_norm.reshape(b * t)
        a = self.action_head(h_act.reshape(b * t, d), zg.reshape(b * t, d), hn)
        return a.reshape(b, t, -1)

    def _apply_dynamics_head(self, h_next, z_goal=None):
        zg = None if z_goal is None else z_goal[:, None, :].expand(-1, h_next.size(1), -1)
        return self.dynamics_head(h_next, zg)

    def _dynamics_step(self, z_hist, e_hist, z_goal=None):
        seq = self.tokenize(z_hist, e_hist)
        h_next = self.predictor(seq)[:, 1::2][:, -1:]
        return self._apply_dynamics_head(h_next, z_goal)

    # -- Training (forward)
    def forward(self, pixels, actions, h_norm, z_goal=None, z_goal_dyn=None, return_z=False):
        """pixels: (B,T,C,H,W), actions: (B,T,act_dim), h_norm: (B,) or (B,T).
        z_goal (B,embed_dim) conditions the ACTION head; z_goal_dyn the DYNAMICS head
        (defaults to z_goal). They are separate so goal-dropout can differ per head.
        Returns a_pred (B,T,act_dim), z_pred (B,T,embed_dim) predicting z_{t+1} (drop the
        last one, no target). return_z=True also returns (z, e) so the trainer can build
        dynamics targets (z[:,1:]) and the SIGReg term without re-encoding."""
        z = self.encode_frames(pixels)
        e = self.encode_actions(actions)
        seq = self.tokenize(z, e)
        out = self.predictor(seq)
        h_act, h_next = out[:, 0::2], out[:, 1::2]
        a_pred = self._apply_action_head(h_act, h_norm, z_goal)
        z_pred = self._apply_dynamics_head(h_next, z_goal if z_goal_dyn is None else z_goal_dyn)
        if return_z:
            return a_pred, z_pred, z, e
        return a_pred, z_pred

    # -- Planning / rollout
    @torch.no_grad()
    def get_cost(self, pixels, past_actions, action_candidates, goal_pixels, history_size=None):
        """CEM cost oracle: roll externally supplied action_candidates through the
        dynamics path, score the last latent against the goal latent.
        pixels: (B,T0,C,H,W), past_actions: (B,T0-1,act_dim) (T0==1 -> empty),
        action_candidates: (B,S,Hz,act_dim), goal_pixels: (B,C,H,W).
        Returns cost: (B,S)."""
        device = next(self.parameters()).device
        pixels = pixels.to(device).float()
        goal_pixels = goal_pixels.to(device).float()
        b, t0 = pixels.shape[:2]
        s, hz = action_candidates.shape[1:3]
        HS = history_size or self.num_frames

        z0 = self.encode_frames(pixels)
        e0 = (self.encode_actions(past_actions.to(device).float()) if t0 > 1
              else z0.new_zeros(b, 0, z0.size(-1)))
        z_goal = self.state_encoder(goal_pixels)

        d = z0.size(-1)
        z = z0.unsqueeze(1).expand(b, s, t0, -1).reshape(b * s, t0, d).clone()
        # explicit last dim d: reshape(..., -1) is ambiguous on the 0-element tensor when
        # t0 == 1 (cold start, no past actions) -> RuntimeError. d is always the embed dim.
        e_hist = e0.unsqueeze(1).expand(b, s, t0 - 1, -1).reshape(b * s, t0 - 1, d).clone()
        e_cand = self.encode_actions(action_candidates.reshape(b * s, hz, -1))
        zg = z_goal.unsqueeze(1).expand(b, s, -1).reshape(b * s, -1)

        for step in range(hz):
            e_hist = torch.cat([e_hist, e_cand[:, step:step + 1]], dim=1)
            z_next = self._dynamics_step(z[:, -HS:], e_hist[:, -HS:], zg)
            z = torch.cat([z, z_next], dim=1)

        cost = (z[:, -1] - zg).pow(2).sum(-1)
        return cost.reshape(b, s)

    @torch.no_grad()
    def get_action(self, pixels, past_actions, horizon, z_goal_pixels=None,
                    h_norm0=1.0, H_max=50, history_size=None):
        """Autoregressive policy rollout: reads the action head, embeds its own
        prediction back in. pixels: (B,T0,C,H,W), past_actions: (B,T0-1,act_dim)
        (T0==1 -> empty), z_goal_pixels: (B,C,H,W) or None.
        Returns actions: (B, horizon, act_dim)."""
        device = next(self.parameters()).device
        pixels = pixels.to(device).float()
        b, t0 = pixels.shape[:2]
        HS = history_size or self.num_frames

        z = self.encode_frames(pixels)
        e_hist = (self.encode_actions(past_actions.to(device).float()) if t0 > 1
                  else z.new_zeros(b, 0, z.size(-1)))
        z_goal = (None if z_goal_pixels is None
                  else self.state_encoder(z_goal_pixels.to(device).float()))
        h_norm = torch.full((b,), float(h_norm0), device=device)

        actions = []
        for _ in range(horizon):
            zw = z[:, -HS:]
            tw = zw.size(1)
            ew = e_hist[:, -(tw - 1):] if tw > 1 else e_hist[:, :0]
            pending = self._tokenize_pending(zw, ew)
            h_act = self.predictor(pending)[:, 0::2][:, -1:]
            a = self._apply_action_head(h_act, h_norm, z_goal)[:, -1]
            actions.append(a)

            e_hist = torch.cat([e_hist, self.encode_actions(a.unsqueeze(1))], dim=1)
            z_next = self._dynamics_step(z[:, -HS:], e_hist[:, -HS:], z_goal)
            z = torch.cat([z, z_next], dim=1)
            h_norm = (h_norm - 1.0 / H_max).clamp(min=0.0)

        return torch.stack(actions, dim=1)
