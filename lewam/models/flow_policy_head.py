"""FlowPolicyHead: flow-matching (rectified-flow) goal-conditioned action-CHUNK head.

Drop-in for GCHead when head_type='flow' -- matches the same duck-typed interface so
LeWAMUnified.forward_seq / gc_action and the trainer's action_loss work unchanged:
    forward(z_t, z_goal, h_norm) -> cond            # the "out": here the conditioning context
    action_loss(cond, target[, mask]) -> (N,)       # flow-matching velocity MSE (masked)
    point(cond) -> action (N, d)                    # few-step Euler ODE, first block
    sample(cond, n, noise=True) -> (N, n, d)        # n ODE draws, first block (policy-proposal)

Design (locked spec, docs/LEWAM_FLOW_DESIGN.md): deterministic dynamics/encoder UNCHANGED; only the
action head becomes generative. Conditioning = ADD of MLP-embedded {state, goal, horizon, flow-time}
+ adaLN-Zero (the proven DiT/SD3/FLUX pattern; per-input LayerNorm is NOT used -- zero-init is the
stabilizer). Rectified flow: straight path x_tau=(1-tau)*noise+tau*A, target velocity A-noise.
References leaned on: UWM adaLN-Zero DiT block, DreamZero flow scheduler + few-step Euler, pi0 flow head.

H (chunk length, in frameskip/block units) tokens are denoised jointly. H=1 is a drop-in for the
current single-block data path; H>1 (the action-chunking lever) additionally needs the trainer to
supply target as (N, H, d) plus a beyond-goal mask.
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


class _DiTBlock(nn.Module):
    """adaLN-Zero transformer block over the H chunk-tokens; global condition c (per row).
    Mirrors module.py's AdaLNBlock (no-affine LN, SiLU->Linear->6*dim modulation zero-init, gated
    attention + MLP residuals)."""

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)          # adaLN-Zero: block boots as identity
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):                         # x: (B,T,dim)  c: (B,dim)
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)  # each (B,1,dim)
        h = self.norm1(x) * (1 + sc1) + sh1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + sc2) + sh2
        x = x + g2 * self.mlp(h)
        return x


class FlowPolicyHead(nn.Module):
    def __init__(self, z_dim=192, action_dim=25, hidden_dim=256, state_dim=None,
                 H=1, depth=3, heads=4, cond_dim=256, n_steps=8, n_freqs=64, dropout=0.0, **_unused):
        super().__init__()
        self.action_dim = int(action_dim)
        self.H = int(H)
        self.n_steps = int(n_steps)
        self.n_freqs = int(n_freqs)
        self.head_type = "flow"
        state_dim = z_dim if state_dim is None else state_dim
        sd = 2 * self.n_freqs
        # conditioning: MLP-embed each signal -> cond_dim, then ADD (tau added in _velocity)
        self.emb_state = nn.Sequential(nn.Linear(state_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.emb_goal = nn.Sequential(nn.Linear(z_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.emb_h = nn.Sequential(nn.Linear(sd, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.emb_tau = nn.Sequential(nn.Linear(sd, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.cond_to_hidden = nn.Linear(cond_dim, hidden_dim)
        # velocity net over the H action-chunk tokens
        self.a_in = nn.Linear(self.action_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.H, hidden_dim))
        self.blocks = nn.ModuleList([_DiTBlock(hidden_dim, heads, dropout=dropout) for _ in range(depth)])
        self.a_out = nn.Linear(hidden_dim, self.action_dim)
        nn.init.zeros_(self.a_out.weight)            # initial velocity ~ 0 (stable start)
        nn.init.zeros_(self.a_out.bias)

    # ---- GCHead-compatible: "forward" returns the conditioning context ----
    def forward(self, z_t, z_goal, h_norm):
        return (self.emb_state(z_t) + self.emb_goal(z_goal)
                + self.emb_h(_sinusoid(h_norm, 2 * self.n_freqs)))          # (N, cond_dim)

    def _velocity(self, A_tau, cond, tau):           # A_tau:(N,H,d) cond:(N,cond_dim) tau:(N,)
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
        x0 = torch.randn_like(A_gt)
        tau = torch.rand(N, device=A_gt.device)
        A_t = (1 - tau[:, None, None]) * x0 + tau[:, None, None] * A_gt     # straight path
        v = self._velocity(A_t, cond, tau)
        se = ((v - (A_gt - x0)) ** 2).mean(-1)       # (N,H) velocity MSE per block
        if mask is not None:                         # beyond-goal chunk positions masked out
            m = mask.float()
            return (se * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        return se.mean(-1)

    @torch.no_grad()
    def _integrate(self, cond, K=1):                 # -> (N,K,H,d) few-step Euler noise->action
        N = cond.shape[0]
        cf = cond.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)
        A = torch.randn(N * K, self.H, self.action_dim, device=cond.device)
        for i in range(self.n_steps):
            tau = torch.full((N * K,), i / self.n_steps, device=cond.device)
            A = A + (1.0 / self.n_steps) * self._velocity(A, cf, tau)
        return A.reshape(N, K, self.H, self.action_dim)

    def point(self, cond):                           # (N,d) first block of a single ODE draw
        return self._integrate(cond, K=1)[:, 0, 0]

    def sample(self, cond, n, noise=True):           # (N,n,d) first block of n ODE draws
        return self._integrate(cond, K=n)[:, :, 0]

    def chunk_sample(self, cond, n=1):               # (N,n,H,d) full chunks (for chunked MPC)
        return self._integrate(cond, K=n)
