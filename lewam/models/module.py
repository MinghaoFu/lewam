import math

import torch
import torch.nn as nn


def sinusoid(x, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
    a = x[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


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


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=None,
        norm_first=True,
        act_fn=nn.GELU,
    ):
        super().__init__()
        if norm_first:
            norm = norm_fn(input_dim) if norm_fn is not None else nn.Identity()
            self.net = nn.Sequential(
                norm,
                nn.Linear(input_dim, hidden_dim),
                act_fn(),
                nn.Linear(hidden_dim, output_dim or input_dim),
            )
        else:
            norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                norm,
                act_fn(),
                nn.Linear(hidden_dim, output_dim or input_dim),
            )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)
