import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class GMMHead(nn.Module):
    """Multimodal action head: K-component diagonal Gaussian mixture (BC-RNN style).

    Drop-in replacement for the deterministic MLP action decoder, to avoid
    mode-averaging on multimodal demos (can/square). Same call signature as MLP
    for inference -- forward(emb) returns one raw-action vector (the highest-weight
    mode's mean, deterministic) -- plus loss(emb, target, weight) = mixture NLL
    used at train time. emb is the intention embedding (B*T, input_dim).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_modes=5,
                 min_std=0.05, max_std=10.0):
        # min_std floor raised from 1e-4 -> 0.05: at 1e-4 the mixture stds collapse
        # to spikes and the NLL diverges (act_loss 3.96 -> 31 on can/40ep). The floor
        # caps the per-sample penalty and stabilises training (standard MDN fix).
        super().__init__()
        self.K = int(n_modes)
        self.adim = int(output_dim)
        self.log_min = math.log(min_std)
        self.log_max = math.log(max_std)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.mean = nn.Linear(hidden_dim, self.K * self.adim)
        self.log_std = nn.Linear(hidden_dim, self.K * self.adim)
        self.logits = nn.Linear(hidden_dim, self.K)

    def _params(self, x):
        h = self.trunk(x)
        n = x.shape[0]
        mean = self.mean(h).view(n, self.K, self.adim)
        log_std = self.log_std(h).clamp(self.log_min, self.log_max).view(n, self.K, self.adim)
        return mean, log_std.exp(), self.logits(h)  # (n,K,adim),(n,K,adim),(n,K)

    def forward(self, x):
        """Deterministic eval action = mean of the most-likely mixture component."""
        mean, _, logits = self._params(x)
        idx = logits.argmax(dim=-1)
        return mean[torch.arange(x.shape[0], device=x.device), idx]  # (n, adim)

    def loss(self, x, target, weight=None):
        """Negative log-likelihood of the GMM. x:(n,in) target:(n,adim);
        weight:(n,adim) per-dim mask for padded action dims (multi-task) or None."""
        mean, std, logits = self._params(x)
        target = target.unsqueeze(1)  # (n,1,adim)
        log_prob = -0.5 * (((target - mean) / std) ** 2) - std.log() - 0.5 * math.log(2 * math.pi)
        if weight is not None:
            log_prob = log_prob * weight.unsqueeze(1)
        log_prob = log_prob.sum(dim=-1)  # (n,K)
        log_mix = torch.log_softmax(logits, dim=-1) + log_prob
        return -torch.logsumexp(log_mix, dim=-1).mean()


class DiffusionHead(nn.Module):
    """Conditional DDPM action head: denoise a raw action conditioned on the
    intention embedding (the A+B design -- our latent intuition conditions a
    Diffusion-Policy-style decoder). Most expressive multimodal head.

    forward(emb) runs reverse diffusion and returns one sampled raw action
    (no_grad, eval). loss(emb, target, weight) = epsilon-prediction MSE (train).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_steps=50,
                 time_dim=64, depth=3, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.adim = int(output_dim)
        self.n_steps = int(n_steps)
        self.time_dim = int(time_dim)
        betas = torch.linspace(beta_start, beta_end, self.n_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("acp", torch.cumprod(alphas, dim=0))
        din = self.adim + self.time_dim + input_dim
        layers = [nn.Linear(din, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, self.adim)]
        self.net = nn.Sequential(*layers)

    def _temb(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)  # (n, time_dim)

    def _eps(self, x_t, t, cond):
        return self.net(torch.cat([x_t, self._temb(t), cond], dim=-1))

    def loss(self, x, target, weight=None):
        """DDPM epsilon-prediction MSE. x=cond:(n,in) target=action:(n,adim)."""
        n = target.shape[0]
        t = torch.randint(0, self.n_steps, (n,), device=target.device)
        noise = torch.randn_like(target)
        acp = self.acp[t].unsqueeze(-1)
        x_t = acp.sqrt() * target + (1 - acp).sqrt() * noise
        err = (self._eps(x_t, t, x) - noise) ** 2
        if weight is not None:
            return (err * weight).sum() / (weight.sum() + 1e-8)
        return err.mean()

    @torch.no_grad()
    def forward(self, x):
        """Reverse-diffusion sample one action chunk conditioned on x (eval)."""
        n = x.shape[0]
        x_t = torch.randn(n, self.adim, device=x.device)
        for t in reversed(range(self.n_steps)):
            tt = torch.full((n,), t, device=x.device, dtype=torch.long)
            eps = self._eps(x_t, tt, x)
            alpha, acp = self.alphas[t], self.acp[t]
            mean = (x_t - (1 - alpha) / (1 - acp).sqrt() * eps) / alpha.sqrt()
            x_t = mean + (self.betas[t].sqrt() * torch.randn_like(x_t) if t > 0 else 0.0)
        return x_t


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        # proprio-as-token: type embedding for [pixel, proprio] tokens (used only when proprio is passed)
        self.type_embedding = nn.Parameter(torch.randn(1, 2, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c, proprio=None):
        """
        x: (B, T, d)           pixel state tokens
        c: (B, T, act_dim)     action conditioning (AdaLN)
        proprio: (B, T, d) or None   separate proprio token per frame
        Returns (B, T, d) pixel preds, or (pixel_preds, proprio_preds) if proprio given.
        """
        T = x.size(1)
        if proprio is None:
            x = x + self.pos_embedding[:, :T]
            x = self.dropout(x)
            return self.transformer(x, c)
        # proprio-as-token: interleave [pix_t, prop_t] per frame -> 2T causal sequence
        pos = self.pos_embedding[:, :T]
        x = x + pos + self.type_embedding[:, 0:1]
        proprio = proprio + pos + self.type_embedding[:, 1:2]
        seq = torch.stack([x, proprio], dim=2).reshape(x.size(0), 2 * T, x.size(-1))
        c2 = torch.stack([c, c], dim=2).reshape(c.size(0), 2 * T, c.size(-1))
        seq = self.dropout(seq)
        out = self.transformer(seq, c2)
        return out[:, 0::2], out[:, 1::2]
