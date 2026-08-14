"""
FlowPolicyHead: flow-matching goal-conditioned action head.
    - Drop-in for GCHead when head_type='flow'
    - Conditioning = sum of MLP-embedded state, goal, horizon, and flow-timestep
      + adaLN-Zero
    - Rectified flow: x_t = (1 - t) * eps + t * act; target is velocity: act - eps

H (chunk length, in frameskip/block units) tokens are denoised jointly. H = 1 is a drop-in for the
current single-block data path; H > 1 additionally needs the trainer to supply target as 
(N, H, d) plus a beyond-goal mask.
"""
import math
import torch
import torch.nn as nn


def _sinusoid(t, dim):
    """t: (N,) -> (N, dim) sinusoidal embedding (dim even)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None, :]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class DiTBlock(nn.Module):
    """
    AdaLN-Zero transformer block over the H chunk-tokens; global condition c (per row).
    """
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        """x: (B,T,dim)  c: (B,dim)"""
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)  # each (B,1,dim)
        h = self.norm1(x) * (1 + sc1) + sh1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + sc2) + sh2
        x = x + g2 * self.mlp(h)
        return x


class FlowPolicyHead(nn.Module):
    def __init__(self, z_dim=192, action_dim=25, hidden_dim=256, state_dim=None,
                 H=1, depth=3, heads=4, cond_dim=256, n_steps=8, n_freqs=64, dropout=0.0,
                 drop_goal=False):
        super().__init__()
        self.action_dim = int(action_dim)
        self.H = int(H)
        self.drop_goal = bool(drop_goal)
        if self.drop_goal:
            self.null_goal = nn.Parameter(torch.zeros(z_dim))
        self.n_steps = int(n_steps)
        self.n_freqs = int(n_freqs)
        self.head_type = "flow"
        state_dim = z_dim if state_dim is None else state_dim
        sd = 2 * self.n_freqs
        # conditioning: MLP-embed each signal -> cond_dim
        self.emb_state = nn.Sequential(nn.Linear(state_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.emb_goal = nn.Sequential(nn.Linear(z_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.emb_h = nn.Sequential(nn.Linear(sd, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.emb_tau = nn.Sequential(nn.Linear(sd, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.cond_to_hidden = nn.Linear(cond_dim, hidden_dim)
        # velocity net over action-chunk tokens
        self.a_in = nn.Linear(self.action_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.H, hidden_dim))
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout=dropout) for _ in range(depth)])
        self.a_out = nn.Linear(hidden_dim, self.action_dim)
        nn.init.zeros_(self.a_out.weight)        # initial velocity ~ 0 (stable init)
        nn.init.zeros_(self.a_out.bias)

    def forward(self, z_t, z_goal, h_norm):
        if self.drop_goal:
            z_goal = self.null_goal.unsqueeze(0).expand(z_t.shape[0], -1)
        return (self.emb_state(z_t) + self.emb_goal(z_goal)
                + self.emb_h(_sinusoid(h_norm, 2 * self.n_freqs)))

    def _velocity(self, A_tau, cond, tau):
        """A_tau: (N,H,d) cond: (N,cond_dim) tau: (N,)"""
        c = self.cond_to_hidden(cond + self.emb_tau(_sinusoid(tau, 2 * self.n_freqs)))  # (N,hidden)
        x = self.a_in(A_tau) + self.pos
        for blk in self.blocks:
            x = blk(x, c)
        return self.a_out(x)                         # (N,H,d)

    def _as_chunk(self, target):                     # (N,d)->(N,1,d); (N,H,d) passthrough
        if target.dim() == 2:
            target = target.unsqueeze(1)
        assert target.shape[1] == self.H, f"target chunk len {target.shape[1]} != head H={self.H}"
        return target

    def action_loss(self, cond, target, mask=None):  # per-sample (N,)
        A_gt = self._as_chunk(target)
        N = A_gt.shape[0]
        x0 = torch.randn_like(A_gt)                  # eps ~ N(0,1)
        tau = torch.rand(N, device=A_gt.device)      # tau ~ U(0,1)
        A_t = (1 - tau[:, None, None]) * x0 + tau[:, None, None] * A_gt
        v = self._velocity(A_t, cond, tau)
        se = ((v - (A_gt - x0)) ** 2).mean(-1)       # (N,H) velocity MSE per block
        if mask is not None:                         # beyond-goal chunks masked out
            m = mask.float()
            return (se * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        return se.mean(-1)

    @torch.no_grad()
    def _integrate(self, cond, K=1, generator=None):
        """"(N,K,H,d) few-step Euler ODE"""
        N = cond.shape[0]
        cf = cond.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)
        A = torch.randn(N * K, self.H, self.action_dim, device=cond.device, generator=generator)
        for i in range(self.n_steps):
            tau = torch.full((N * K,), i / self.n_steps, device=cond.device)
            A = A + (1.0 / self.n_steps) * self._velocity(A, cf, tau)
        return A.reshape(N, K, self.H, self.action_dim)

    def point(self, cond):
        """(N,d) first block of a *single* ODE draw"""
        return self._integrate(cond, K=1)[:, 0, 0]

    def sample(self, cond, n, noise=None, generator=None):
        """(N,n,d) first block of n ODE draws"""
        return self._integrate(cond, K=n, generator=generator)[:, :, 0]

    def chunk_sample(self, cond, n=1):
        """(N,n,H,d) full chunks (for chunked MPC)"""
        return self._integrate(cond, K=n)
