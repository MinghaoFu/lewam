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
    """cat[state, z_goal] -> 3 AdaLN blocks -> action head. state is one z_dim vector by default (the
    split's z_t); state_dim widens it so a caller can pass a concatenated state (e.g. the unified
    model's [z_t, c_t] skip). state_dim=z_dim keeps the split's head unchanged (in_dim = 2*z_dim).

    head_type:
      'mse' (default) -- deterministic point action, MSE loss.
      'gmm' -- a K-component diagonal-Gaussian mixture density network: the final layer emits K logits
        + K means (d each) + K log-sigmas (d each), trained by mixture NLL; a sample draws a component
        ~ Cat(pi) then reparameterizes within it. Captures multimodal expert actions, so sampling gives
        diverse on-manifold candidates for policy-proposal planning. forward returns the raw head
        output; consume it with .action_loss / .point / .rsample / .sample (passthrough for mse)."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512,
                 n_freqs=64, cond_dim=128, dropout=0.1, state_dim=None,
                 head_type="mse", n_mix=5, num_chunks=1,
                 latent_h="", h_codes=16, h_code_dim=64, h_commit=0.25, h_pred_w=1.0):
        super().__init__()
        self.n_freqs = n_freqs
        self.action_dim = int(action_dim)
        self.head_type = str(head_type)
        self.n_mix = int(n_mix)
        # num_chunks > 1: the head predicts that many consecutive action blocks; block 1 is what
        # point()/sample() return (the executed action), the rest are extra supervision. The
        # chunked loss lives in the trainer; action_loss here assumes width-matched targets.
        self.num_chunks = int(num_chunks)
        assert not (self.head_type == "gmm" and self.num_chunks > 1), \
            "gmm models a joint block distribution; chunked masking is not decomposable"
        state_dim = z_dim if state_dim is None else state_dim
        in_dim = state_dim + z_dim
        sin_dim = 2 * n_freqs
        self.horizon_mlp = nn.Sequential(
            nn.Linear(sin_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        # latent horizon: the conditioning is derived from (state, z_goal) instead of the dataset's
        # h. The true h is used only as a regression target in h_aux_loss, never as an input, so
        # nothing about the eval path needs to know the remaining time.
        self.latent_h = str(latent_h or "")
        assert self.latent_h in ("", "vq", "scalar"), f"latent_h={self.latent_h!r}"
        self.h_codes, self.h_commit, self.h_pred_w = int(h_codes), float(h_commit), float(h_pred_w)
        if self.latent_h:
            u_dim = h_code_dim if self.latent_h == "vq" else 1
            self.h_enc = nn.Sequential(
                nn.Linear(in_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, u_dim))
        if self.latent_h == "vq":
            self.h_codebook = nn.Embedding(self.h_codes, h_code_dim)
            self.h_codebook.weight.data.normal_(0.0, h_code_dim ** -0.5)
            self.h_readout = nn.Linear(h_code_dim, 1)
            self.code_mlp = nn.Sequential(
                nn.Linear(h_code_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
            # usage histogram -> code_stats(); a collapsed codebook makes this arm a no-h arm with
            # extra parameters, which has to be visible in the log rather than inferred from SR.
            self.register_buffer("h_code_count", torch.zeros(self.h_codes))
        self.block1 = AdaLNBlock(in_dim, hidden_dim, cond_dim, dropout)
        self.block2 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        self.block3 = AdaLNBlock(hidden_dim, hidden_dim, cond_dim, dropout)
        out_dim = (self.n_mix * (1 + 2 * self.action_dim) if self.head_type == "gmm"
                   else self.action_dim * self.num_chunks)
        self.out = nn.Linear(hidden_dim, out_dim)

    def _quantize(self, u):
        """Nearest codebook entry. Returns (straight-through e, e, index)."""
        w = self.h_codebook.weight
        d = u.pow(2).sum(-1, keepdim=True) - 2.0 * u @ w.t() + w.pow(2).sum(-1)[None, :]
        idx = d.argmin(-1)
        e = self.h_codebook(idx)
        return u + (e - u).detach(), e, idx

    def h_aux_loss(self, z_t, z_goal, h_target):
        """VQ + horizon-readout loss; 0 when latent_h is off.

        Recomputes u rather than caching it from forward on purpose: there are two training entry
        points (forward_seq, forward_seq_prefix) and the head's output is also consumed by point()
        and sample(), so a cached tensor is easy to read at the wrong time. The cost is one small
        MLP pass."""
        if not self.latent_h:
            return z_t.new_zeros(())
        u = self.h_enc(torch.cat([z_t, z_goal], dim=-1))
        h_t = h_target.float().reshape(-1)
        if self.latent_h == "scalar":
            return self.h_pred_w * F.mse_loss(torch.sigmoid(u).squeeze(-1), h_t)
        e_st, e, _ = self._quantize(u)
        vq = F.mse_loss(e, u.detach()) + self.h_commit * F.mse_loss(u, e.detach())
        return vq + self.h_pred_w * F.mse_loss(self.h_readout(e_st).squeeze(-1), h_t)

    def code_stats(self, reset=True):
        """Codebook usage since the last call: perplexity (effective number of codes in use) and
        the count of codes that fired at all."""
        if self.latent_h != "vq":
            return {}
        c = self.h_code_count
        tot = float(c.sum().item())
        if tot <= 0:
            return {"h_ppl": 0.0, "h_live": 0, "h_n": 0.0}
        p = c / c.sum()
        out = {"h_ppl": float(torch.exp(-(p * (p + 1e-10).log()).sum()).item()),
               "h_live": int((c > 0).sum().item()), "h_n": tot}
        if reset:
            self.h_code_count.zero_()
        return out

    def forward(self, z_t, z_goal, h_norm):
        x = torch.cat([z_t, z_goal], dim=-1)
        if self.latent_h == "vq":
            # h_norm is ignored: the conditioning is a code read off (state, z_goal).
            e_st, _, idx = self._quantize(self.h_enc(x))
            if self.training:
                with torch.no_grad():
                    self.h_code_count.index_add_(
                        0, idx, torch.ones_like(idx, dtype=self.h_code_count.dtype))
            cond = self.code_mlp(e_st)
        elif self.latent_h == "scalar":
            # same sinusoidal path as the horizon arm, fed a predicted h instead of the true one
            cond = self.horizon_mlp(sinusoidal_embedding(
                torch.sigmoid(self.h_enc(x)).squeeze(-1), self.n_freqs))
        else:
            cond = self.horizon_mlp(sinusoidal_embedding(h_norm, self.n_freqs))
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)
        return self.out(x)

    # GMM functions (identity if head_type == "mse")
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
        """Deterministic action (N,d): identity for mse; for gmm the most-likely component's mean
        mu_{argmax pi} (a valid mode), not the mixture mean sum_k pi_k mu_k, which for multimodal data
        lands in the low-density valley between modes."""
        if self.head_type != "gmm":
            return out[..., :self.action_dim] if self.num_chunks > 1 else out
        logits, mu, _ = self._gmm_params(out)
        k = logits.argmax(dim=-1)                                              # (N,) most-likely component
        return mu.gather(1, k[:, None, None].expand(-1, 1, self.action_dim)).squeeze(1)

    def rsample(self, out):
        """One reparameterized sample (N,d): identity for mse; component ~ Cat(pi) (hard, detached)
        then reparam within it (grad flows to that component's mu/sigma) for gmm."""
        if self.head_type != "gmm":
            return out[..., :self.action_dim] if self.num_chunks > 1 else out
        logits, mu, logsig = self._gmm_params(out)
        comp = torch.multinomial(F.softmax(logits, dim=-1), 1)                  # (N,1)
        idx = comp.unsqueeze(-1).expand(-1, 1, self.action_dim)                 # (N,1,d)
        mu_s = mu.gather(1, idx).squeeze(1)
        sig_s = logsig.gather(1, idx).squeeze(1).exp()
        return mu_s + sig_s * torch.randn_like(mu_s)

    def sample(self, out, n, noise=True, generator=None):
        """n samples per row (N,n,d) for planning: mse -> mean repeated; gmm -> component ~ Cat then draw.
        noise=True: full reparam draw mu + sigma*eps (diverse modes and within-mode Gaussian).
        noise=False: the drawn component's mean mu_k, no within-mode Gaussian -- diverse in which mode but
        each candidate is a clean on-manifold mode, not corrupted by the (often inflated) sigma. Measured:
        sigma~0.33 blows sample error 0.25->0.44, so clean modes plan better under WM verification."""
        if self.head_type != "gmm":
            exec_out = out[..., :self.action_dim] if self.num_chunks > 1 else out
            return exec_out.unsqueeze(1).expand(-1, n, -1)
        logits, mu, logsig = self._gmm_params(out)
        N, K, d = mu.shape
        comp = torch.multinomial(F.softmax(logits, dim=-1), n, replacement=True, generator=generator)  # (N,n)
        idx = comp.unsqueeze(-1).expand(N, n, d)
        draw = mu.gather(1, idx)
        if noise:
            draw = draw + logsig.gather(1, idx).exp() * torch.randn(N, n, d, device=out.device, generator=generator)
        return draw


class GoalCondDynamics(nn.Module):
    """cat[z_t, a_t, z_goal] -> z_{t+1}. With goal_cond=False it becomes a pure forward model
    cat[z_t, a_t] -> z_{t+1} (no goal input) -- the forward model for planning: a goal-conditioned
    dynamics can drift toward z_goal and partly ignore a_t, flattening the planning cost surface.
    forward() keeps the 3-arg signature either way (z_goal unused when goal_cond=False) so callers
    don't change."""

    def __init__(self, z_dim=192, action_dim=25, hidden_dim=512, goal_cond=True, action_embed_dim=0,
                 out_dim=0):
        super().__init__()
        self.goal_cond = bool(goal_cond)
        # Prediction width. The trunk emits out_dim (default z_dim = the latent it predicts); when
        # out_dim != z_dim a bias-free readout maps it back so the loss stays MSE against z_{t+1}
        # and every caller keeps its (N, z_dim) contract. forward_wide() exposes the wide vector for
        # consumers that want it (e.g. planning scored in the wider space). out_dim == z_dim makes
        # the readout an Identity, so the module is structurally identical to the pre-flexible
        # version and old checkpoints load strict.
        self.z_dim = int(z_dim)
        self.out_dim = int(out_dim) if out_dim else int(z_dim)
        # action pathway: the raw z-scored action (action_dim) is tiny next to the z_dim latents and gets
        # drowned in the concat. action_embed_dim>0 projects+normalizes it (Linear -> LayerNorm -> GELU)
        # to a comparable width before concat, giving the MLP a real action pathway. 0 = raw-concat
        # baseline. The projection is rank<=action_dim, so it rebalances the concat width without adding
        # action information.
        self.action_embed_dim = int(action_embed_dim)
        if self.action_embed_dim > 0:
            self.a_proj = nn.Sequential(
                nn.Linear(action_dim, self.action_embed_dim),
                nn.LayerNorm(self.action_embed_dim),
                nn.GELU(),
            )
            a_in = self.action_embed_dim
        else:
            self.a_proj = None
            a_in = action_dim
        in_dim = (2 * z_dim + a_in) if self.goal_cond else (z_dim + a_in)
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_dim),
        )
        self.readout = (nn.Identity() if self.out_dim == self.z_dim
                        else nn.Linear(self.out_dim, self.z_dim, bias=False))

    def _trunk(self, z_t, a_t, z_goal):
        a = self.a_proj(a_t) if self.a_proj is not None else a_t
        x = torch.cat([z_t, a, z_goal], dim=-1) if self.goal_cond else torch.cat([z_t, a], dim=-1)
        return self.net(x)

    def forward_wide(self, z_t, a_t, z_goal):
        """(N, out_dim) prediction in the trunk's own width, before the z-space readout."""
        return self._trunk(z_t, a_t, z_goal)

    def forward(self, z_t, a_t, z_goal):
        """(N, z_dim) predicted next latent -- the contract every caller relies on."""
        return self.readout(self._trunk(z_t, a_t, z_goal))


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
