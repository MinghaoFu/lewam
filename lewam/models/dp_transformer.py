"""Vendored, verbatim, from diffusion_policy (Chi et al.): the transformer denoiser used by
DiffusionTransformerHybridImagePolicy. Kept faithful so a DP-exact policy head can be trained and
eval'd through our pipeline as an isolation test. Only the optimizer-group helpers are dropped."""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TransformerForDiffusion(nn.Module):
    def __init__(self, input_dim, output_dim, horizon, n_obs_steps=None, cond_dim=0,
                 n_layer=12, n_head=12, n_emb=768, p_drop_emb=0.1, p_drop_attn=0.1,
                 causal_attn=False, time_as_cond=True, obs_as_cond=False, n_cond_layers=0):
        super().__init__()
        if n_obs_steps is None:
            n_obs_steps = horizon
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if obs_as_cond else None
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                                               dropout=p_drop_attn, activation='gelu',
                                               batch_first=True, norm_first=True),
                    num_layers=n_cond_layers)
            else:
                self.encoder = nn.Sequential(nn.Linear(n_emb, 4 * n_emb), nn.Mish(),
                                             nn.Linear(4 * n_emb, n_emb))
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                                           dropout=p_drop_attn, activation='gelu',
                                           batch_first=True, norm_first=True),
                num_layers=n_layer)
        else:
            encoder_only = True
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                                           dropout=p_drop_attn, activation='gelu',
                                           batch_first=True, norm_first=True),
                num_layers=n_layer)

        if causal_attn:
            mask = (torch.triu(torch.ones(T, T)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
            self.register_buffer("mask", mask)
            if time_as_cond and obs_as_cond:
                t, s = torch.meshgrid(torch.arange(T), torch.arange(T_cond), indexing='ij')
                m = (t >= (s - 1)).float().masked_fill(t < (s - 1), float('-inf')).masked_fill(t >= (s - 1), 0.0)
                self.register_buffer('memory_mask', m)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']:
                w = getattr(module, name, None)
                if w is not None:
                    torch.nn.init.normal_(w, mean=0.0, std=0.02)
            for name in ['in_proj_bias', 'bias_k', 'bias_v']:
                b = getattr(module, name, None)
                if b is not None:
                    torch.nn.init.zeros_(b)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

    def forward(self, sample, timestep, cond=None):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        input_emb = self.input_emb(sample)
        if self.encoder_only:
            x = self.drop(torch.cat([time_emb, input_emb], dim=1)
                          + self.pos_emb[:, :input_emb.shape[1] + 1, :])
            x = self.encoder(src=x, mask=self.mask)[:, 1:, :]
        else:
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_embeddings = torch.cat([cond_embeddings, self.cond_obs_emb(cond)], dim=1)
            x = self.drop(cond_embeddings + self.cond_pos_emb[:, :cond_embeddings.shape[1], :])
            memory = self.encoder(x)
            x = self.drop(input_emb + self.pos_emb[:, :input_emb.shape[1], :])
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)
        return self.head(self.ln_f(x))
