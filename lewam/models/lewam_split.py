"""LeWAM-Split model: a ViT encoder (+ projector) with two separate ("split")
heads on top of its latent -- a goal-conditioned action head (GCHead) and a
goal-conditioned dynamics predictor (GoalCondDynamics). Both heads condition on
the remaining horizon via AdaLN-Zero (`AdaLNBlock`/`sinusoidal_embedding`, shared
with gcidm.py's head via `lewam.models.module`, not redefined here).

Used by `scripts/train_lewam_gc.py`:
  model = LeWAMSplit(embed_dim=192, action_dim=action_block_dim, hidden_dim=...)
  a_pred, z_pred = model(z_t, z_goal, h_norm, a_t)
Anti-collapse regularization on the encoder latent is SIGReg (also from `module`).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from lewam.models.module import AdaLNBlock, sinusoidal_embedding, ViTEncoder

class GCHead(nn.Module):
    """cat[state, z_goal] -> 3 AdaLN blocks -> action head. `state` is one z_dim vector by
    default (the split's z_t); `state_dim` widens it so a caller can pass a concatenated
    state (e.g. the unified model's [z_t, c_t] skip). Defaulting state_dim=z_dim keeps the
    split's head byte-identical (in_dim = 2*z_dim).

    head_type:
      'mse' (default) -- deterministic point action, MSE loss (byte-identical to the old head).
      'gmm' -- a K-component diagonal-Gaussian Mixture Density Network: the final layer emits
        K mixture logits + K means (d each) + K log-sigmas (d each); trained by mixture NLL; a
        SAMPLE draws a component ~ Cat(pi) then reparameterizes within it. Captures multimodal
        expert actions, so sampling yields diverse ON-MANIFOLD candidates the world model can
        verify -- the enabler of policy-proposal planning. `forward` returns the RAW head output;
        consume it with .action_loss / .point / .rsample / .sample (all no-ops passthrough for mse)."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512,
                 n_freqs=64, cond_dim=128, dropout=0.1, state_dim=None,
                 head_type="mse", n_mix=5):
        super().__init__()
        self.n_freqs = n_freqs
        self.action_dim = int(action_dim)
        self.head_type = str(head_type)
        self.n_mix = int(n_mix)
        state_dim = z_dim if state_dim is None else state_dim
        in_dim = state_dim + z_dim
        sin_dim = 2 * n_freqs
        self.horizon_mlp = nn.Sequential(
            nn.Linear(sin_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.block1 = AdaLNBlock(in_dim, hidden_dim, cond_dim, dropout)
        self.block2 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.block3 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        out_dim = self.n_mix * (1 + 2 * self.action_dim) if self.head_type == "gmm" else self.action_dim
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, z_t, z_goal, h_norm):
        x = torch.cat([z_t, z_goal], dim=-1)
        cond = self.horizon_mlp(sinusoidal_embedding(h_norm, self.n_freqs))
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)
        return self.out(x)

    # ---- GMM consumers (all identity/passthrough when head_type == 'mse') ----
    def _gmm_params(self, out):
        """(N, K*(1+2d)) -> logits (N,K), mu (N,K,d), logsig (N,K,d) clamped for stability."""
        N, K, d = out.shape[0], self.n_mix, self.action_dim
        logits = out[:, :K]
        rest = out[:, K:].reshape(N, K, 2 * d)
        return logits, rest[:, :, :d], rest[:, :, d:].clamp(-7.0, 3.0)

    def action_loss(self, out, target):
        """Per-sample loss (N,): MSE mean-over-dims for mse; mixture NLL for gmm."""
        if self.head_type != "gmm":
            return ((out - target) ** 2).mean(-1)
        logits, mu, logsig = self._gmm_params(out)
        logpi = F.log_softmax(logits, dim=-1)                                   # (N,K)
        logcomp = -0.5 * ((((target.unsqueeze(1) - mu) * torch.exp(-logsig)) ** 2)
                          + 2 * logsig + math.log(2 * math.pi)).sum(-1)         # (N,K)
        return -torch.logsumexp(logpi + logcomp, dim=-1)                        # (N,)

    def point(self, out):
        """Deterministic action (N,d): identity for mse; mixture mean sum_k pi_k mu_k for gmm."""
        if self.head_type != "gmm":
            return out
        logits, mu, _ = self._gmm_params(out)
        return (F.softmax(logits, dim=-1).unsqueeze(-1) * mu).sum(1)

    def rsample(self, out):
        """One reparameterized sample (N,d): identity for mse; component ~ Cat(pi) (hard, detached)
        then reparam within it (grad flows to that component's mu/sigma) for gmm."""
        if self.head_type != "gmm":
            return out
        logits, mu, logsig = self._gmm_params(out)
        comp = torch.multinomial(F.softmax(logits, dim=-1), 1)                  # (N,1)
        idx = comp.unsqueeze(-1).expand(-1, 1, self.action_dim)                 # (N,1,d)
        mu_s = mu.gather(1, idx).squeeze(1)
        sig_s = logsig.gather(1, idx).squeeze(1).exp()
        return mu_s + sig_s * torch.randn_like(mu_s)

    def sample(self, out, n):
        """n samples per row (N,n,d) for planning: mse -> mean repeated; gmm -> component~Cat then N."""
        if self.head_type != "gmm":
            return out.unsqueeze(1).expand(-1, n, -1)
        logits, mu, logsig = self._gmm_params(out)
        N, K, d = mu.shape
        comp = torch.multinomial(F.softmax(logits, dim=-1), n, replacement=True)  # (N,n)
        idx = comp.unsqueeze(-1).expand(N, n, d)
        return mu.gather(1, idx) + logsig.gather(1, idx).exp() * torch.randn(N, n, d, device=out.device)


class GoalCondDynamics(nn.Module):
    """cat[z_t, a_t, z_goal] -> z_{t+1}. With goal_cond=False it becomes a PURE forward model
    cat[z_t, a_t] -> z_{t+1} (no goal input) -- the honest world model for planning: a
    goal-conditioned dynamics can drift toward z_goal using the goal input, partly ignoring a_t,
    which makes a CEM/planning cost surface flat and exploitable. forward() keeps the 3-arg
    signature either way (z_goal is simply unused when goal_cond=False) so callers don't change."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512, goal_cond=True):
        super().__init__()
        self.goal_cond = bool(goal_cond)
        in_dim = (2 * z_dim + action_dim) if self.goal_cond else (z_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_t, a_t, z_goal):
        x = torch.cat([z_t, a_t, z_goal], dim=-1) if self.goal_cond else torch.cat([z_t, a_t], dim=-1)
        return self.net(x)


class LeWAMSplit(nn.Module):
    """Shared ViT encoder + two independent goal-conditioned heads on its
    latent: `gc_head` (action) and `dynamics` (next-latent prediction)."""

    def __init__(self, encoder_size="tiny", embed_dim=192, action_dim=25, hidden_dim=512,
                 img_size=224, dropout=0.1, proj_hidden=None):
        super().__init__()
        self.encoder = ViTEncoder(size=encoder_size, output_type="cls",
                                  output_dim=embed_dim, img_size=img_size,
                                  proj_hidden=proj_hidden)
        self.gc_head = GCHead(z_dim=embed_dim, action_dim=action_dim,
                              hidden_dim=hidden_dim, dropout=dropout)
        self.dynamics = GoalCondDynamics(z_dim=embed_dim, action_dim=action_dim,
                                         hidden_dim=hidden_dim)

    def encode(self, pixels):
        """pixels: (N, 3, H, W) -> (N, embed_dim) cls latent."""
        return self.encoder(pixels)

    def forward(self, z_t, z_goal, h_norm, a_t):
        """z_t, z_goal: (B, embed_dim); h_norm, a_t: (B,)/(B, action_dim).
        Returns (a_pred, z_pred) from the action head and the dynamics head."""
        a_pred = self.gc_head(z_t, z_goal, h_norm)
        z_pred = self.dynamics(z_t, a_t, z_goal)
        return a_pred, z_pred
