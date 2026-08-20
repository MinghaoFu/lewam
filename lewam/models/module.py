import math

import torch
from torch import nn
import torch.nn.functional as F
from stable_pretraining.backbone.utils import vit_hf
from einops import rearrange

def sinusoidal_embedding(h_norm, n_freqs=64):
    """Sinusoidal embedding of a scalar in [0, 1]: log-spaced freqs, concat(sin, cos).
    h_norm: (B,) float tensor. Returns (B, 2*n_freqs)."""
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(n_freqs, device=h_norm.device).float()
        / max(n_freqs - 1, 1)
    )
    ang = h_norm.float()[:, None] * freqs[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


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
        self.embed = MLP(
            input_dim=smoothed_dim,
            hidden_dim=mlp_scale * emb_dim,
            output_dim=emb_dim,
            norm_fn=None,
            norm_first=False,
            act_fn=nn.SiLU
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


class ModalityAdapter(nn.Module):
    """Modality adapter: LayerNorm -> Linear -> add learnable token per modality"""

    def __init__(self, embed_dim, input_dim=None):
        super().__init__()
        input_dim = input_dim or embed_dim
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )
        self.token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D_in) modality latents
        Returns:
            (B, T, D_embed) adapted latents + token
        """
        return self.adapter(x) + self.token


def sinusoid(x, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
    a = x[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class GCHeadMSE(nn.Module):
    """Goal + horizon conditioned readout (GCHead distilled to its MSE path): per-token
    features are concatenated with the goal latent and run through AdaLN blocks conditioned
    on the horizon embedding. A learned null goal stands in for rows whose goal is dropped,
    so the goal-free mode stays trainable."""

    def __init__(self, in_dim, goal_dim, out_dim, hidden_dim=512, cond_dim=128, depth=3,
                 dropout=0.1):
        super().__init__()
        self.cond_dim = cond_dim
        self.null_goal = nn.Parameter(torch.zeros(goal_dim))
        self.h_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(),
                                   nn.Linear(cond_dim, cond_dim))
        dims = [in_dim + goal_dim] + [hidden_dim] * (depth - 1)
        self.blocks = nn.ModuleList(AdaLNBlock(dims[i], hidden_dim, cond_dim, dropout)
                                    for i in range(depth))
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, z_goal=None, h_norm=None, goal_keep=None):
        """x: (B, T, D) token features; z_goal: (B, G); h_norm: (B,) in [0, 1];
        goal_keep: (B,) bool -- False rows use the null goal."""
        B, T = x.shape[0], x.shape[1]
        if z_goal is None:
            z_goal = self.null_goal.expand(B, -1)
        elif goal_keep is not None:
            z_goal = torch.where(goal_keep[:, None], z_goal, self.null_goal.expand(B, -1))
        if h_norm is None:
            h_norm = x.new_zeros(B)
        cond = self.h_mlp(sinusoid(h_norm, self.cond_dim)).unsqueeze(1)
        x = torch.cat([x, z_goal[:, None, :].expand(B, T, -1)], dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        return self.out(x)


class AdaLNBlock(nn.Module):
    """One MLP layer with AdaLN-Zero conditioning: LayerNorm (no affine) -> modulate
    by (scale, shift) from `cond` -> Linear -> GELU -> Dropout. The (scale, shift)
    projection is zero-initialized, so at init this block's conditioning is a no-op
    (AdaLN-Zero)."""

    def __init__(self, in_dim, out_dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False, eps=1e-6)
        self.cond_proj = nn.Linear(cond_dim, 2 * in_dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond):
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.drop(self.act(self.fc(x)))

    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
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

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, causal=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        x : (B, T, D)
        """
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=self.causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal)
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

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal)
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
        causal=True,
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
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, causal)
            )

    def forward(self, x, c=None):

        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)

        return x


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
        # min_std floor 0.05 (not 1e-4): a too-small floor lets mixture stds collapse to
        # spikes and the NLL diverges (standard MDN fix).
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


def _batchnorm_to_groupnorm(module, num_groups=16):
    """Replace every BatchNorm2d in `module` (in place, recursively) with GroupNorm(num_groups).
    ResNet18 channel widths (64/128/256/512) are all divisible by 16."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            _batchnorm_to_groupnorm(child, num_groups)
    return module


class VisionEncoder(nn.Module):
    """Visual encoder"""

    def __init__(self, img_size=224, size="tiny", output_type="cls",
                output_dim=192, proj_mlp_scale=4, proj_hidden=None,
                backbone="scratch", backbone_ckpt=None):
        super().__init__()
        self.output_type = output_type
        self.backbone = backbone

        if backbone == "scratch":
            self.vit = vit_hf(size=size, patch_size=14, image_size=img_size,
                                  pretrained=False, use_mask_token=False)
            mlp_in = self.vit.config.hidden_size
        elif backbone == "dinov3":
            import timm
            size_to_model = {
                "tiny": "vit_small_patch16_dinov3",  # a true "tiny" does not exist
                "small": "vit_small_patch16_dinov3",
                "base": "vit_base_patch16_dinov3",
                "large": "vit_large_patch16_dinov3"
            }
            if backbone_ckpt == "defer":
                # architecture only; caller restores weights from a full-model checkpoint
                self.vit = timm.create_model(size_to_model[size], pretrained=False, num_classes=0)
            elif backbone_ckpt:
                self.vit = timm.create_model(size_to_model[size], pretrained=False, num_classes=0)
                self.vit.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"), strict=True)
            else:
                self.vit = timm.create_model(size_to_model[size], pretrained=True, num_classes=0)

            for p in self.vit.parameters():
                p.requires_grad_(False)
            self.vit.eval()
            mlp_in = self.vit.num_features
        elif backbone == "resnet18sp":
            # ResNet18 conv trunk + SpatialSoftmax: 32 keypoint (x,y) coordinates -> 64-d
            import torchvision
            r18 = torchvision.models.resnet18(weights=None)
            self.trunk = nn.Sequential(*list(r18.children())[:-2])   # (N,512,H/32,W/32)
            self.kp_conv = nn.Conv2d(512, 32, kernel_size=1)
            mlp_in = 64
        elif backbone == "resnet18dp":
            # Diffusion-Policy-faithful obs encoder: ResNet18 with GroupNorm(16) in place of
            # BatchNorm, a 224->202 crop randomizer (random offset in train, center at eval),
            # then the same 32-keypoint SpatialSoftmax head. Matches DP's robomimic vision stack.
            import torchvision
            r18 = torchvision.models.resnet18(weights=None)
            self.trunk = _batchnorm_to_groupnorm(nn.Sequential(*list(r18.children())[:-2]))
            self.kp_conv = nn.Conv2d(512, 32, kernel_size=1)
            self.crop_size = 202
            mlp_in = 64
        else:
            raise ValueError(f"unknown encoder backbone {backbone!r}")
        # maps to representation space using a MLP with Batch Normalization.
        # necessary because the final ViT layer applies Layer Normalization, which prevents
        # SIGReg being optimized effectively.
        self.projector = MLP(input_dim=mlp_in, output_dim=output_dim,
                             hidden_dim=(proj_hidden if proj_hidden else proj_mlp_scale * output_dim),
                             norm_fn=nn.BatchNorm1d, norm_first=False)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone == "dinov3":
            self.vit.eval()
        return self

    def _spatial_softmax(self, pixels):
        """(N,3,H,W) -> (N,64): expected (x,y) image coordinate of each of 32 keypoint maps."""
        fmap = self.kp_conv(self.trunk(pixels))                     # (N,32,h,w)
        N, K, H, W = fmap.shape
        attn = fmap.flatten(2).softmax(-1)                          # (N,32,h*w)
        xs = torch.linspace(-1.0, 1.0, W, device=fmap.device)
        ys = torch.linspace(-1.0, 1.0, H, device=fmap.device)
        grid_x = xs.repeat(H)                                       # row-major flatten: col-fastest
        grid_y = ys.repeat_interleave(W)
        exp_x = (attn * grid_x).sum(-1)                             # (N,32)
        exp_y = (attn * grid_y).sum(-1)
        return torch.cat([exp_x, exp_y], dim=-1)                    # (N,64)

    def _crop(self, pixels):
        """DP CropRandomizer: crop (N,3,H,W) to crop_size, per-sample random offset in train,
        center crop at eval."""
        N, C, H, W = pixels.shape
        s = self.crop_size
        if self.training:
            top = torch.randint(0, H - s + 1, (N,), device=pixels.device)
            left = torch.randint(0, W - s + 1, (N,), device=pixels.device)
        else:
            top = pixels.new_full((N,), (H - s) // 2, dtype=torch.long)
            left = pixels.new_full((N,), (W - s) // 2, dtype=torch.long)
        span = torch.arange(s, device=pixels.device)
        rows = (top[:, None] + span)[:, None, :, None]              # (N,1,s,1)
        cols = (left[:, None] + span)[:, None, None, :]             # (N,1,1,s)
        b = torch.arange(N, device=pixels.device)[:, None, None, None]
        c = torch.arange(C, device=pixels.device)[None, :, None, None]
        return pixels[b, c, rows, cols]                            # (N,C,s,s)

    def forward(self, pixels):
        """pixels: (N, 3, H, W)"""
        if self.backbone == "resnet18sp":
            return self.projector(self._spatial_softmax(pixels))
        if self.backbone == "resnet18dp":
            return self.projector(self._spatial_softmax(self._crop(pixels)))
        if self.backbone == "scratch":
            out = self.vit(pixels, interpolate_pos_encoding=True)
            if self.output_type == "cls":
                return self.projector(out.last_hidden_state[:, 0])  # (N, D)
            return self.projector(out.last_hidden_state[:, 1:])  # (N, n_patch, D)
        # frozen pretrained backbone
        with torch.no_grad():
            feat = self.vit(pixels)  # (N, num_features)
        return self.projector(feat)


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
        # proprio-as-token: type embedding for [pixel, proprio] tokens
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