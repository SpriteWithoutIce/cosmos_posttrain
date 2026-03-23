from __future__ import annotations

import math
import torch
import torch.nn as nn


def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [B] or [B, 1], can be int/float timestep.
    return: [B, dim]
    """
    if t.ndim > 1:
        t = t.view(t.shape[0], -1)[:, 0]
    t = t.float()
    half = dim // 2
    device = t.device
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1))
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(temb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B, T]
        timesteps = timesteps.float()
        bsz, T = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0, device=device)) / max(half_dim, 1)
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)
        if enc.shape[-1] < self.embedding_dim:
            enc = torch.cat([enc, torch.zeros(bsz, T, self.embedding_dim - enc.shape[-1], device=device)], dim=-1)
        return enc


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]  # [B, Din, Dout]
        selected_b = self.b[cat_ids]  # [B, Dout]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """
    Adapted from reasoningVLA flow_matching_action_head:
      actions + timestep encoding -> action token embeddings.
    """

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int = 1):
        super().__init__()
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # actions: [B, T, action_dim], timesteps: [B] (discrete)
        bsz, T, _ = actions.shape
        if timesteps.ndim == 1 and timesteps.shape[0] == bsz:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(f"Expected timesteps shape [B], got {tuple(timesteps.shape)}")

        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x


class ActionDiTBlock(nn.Module):
    """
    DiT-style block:
      - timestep-conditioned AdaLN
      - 1 self-attn
      - 3 delta-v cross-attn
      - 1 state cross-attn with sigma gate
      - FFN
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_norm = AdaLayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.delta_norms = nn.ModuleList([AdaLayerNorm(d_model) for _ in range(3)])
        self.delta_attns = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(3)]
        )

        self.state_norm = AdaLayerNorm(d_model)
        self.state_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ffn_norm = AdaLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        delta_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        sigma: torch.Tensor,
        temb: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_norm(x, temb)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + h

        for ln, attn in zip(self.delta_norms, self.delta_attns):
            h = ln(x, temb)
            h, _ = attn(h, delta_tokens, delta_tokens, need_weights=False)
            x = x + h

        h = self.state_norm(x, temb)
        h, _ = self.state_attn(h, state_tokens, state_tokens, need_weights=False)
        x = x + sigma * h

        x = x + self.ffn(self.ffn_norm(x, temb))
        return x


class ActionMIPHead(nn.Module):
    """
    ReasoningVLA-style DiT action head (adapted):
      - timestep embedding + AdaLN modulation
      - causal self-attention over action sequence
      - cross-attn conditioning from delta-v and state
    """

    def __init__(
        self,
        action_dim: int = 16,
        num_actions: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        n_blocks: int = 8,
        delta_dim: int = 16,
        state_dim: int = 16,
        sigma_k: float = 4.0,
        dropout: float = 0.0,
        timestep_buckets: int = 1000,
        delta_spatial_pool_h: int = 0,
        delta_spatial_pool_w: int = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.sigma_k = sigma_k
        self.timestep_buckets = timestep_buckets
        self.delta_spatial_pool_h = int(delta_spatial_pool_h)
        self.delta_spatial_pool_w = int(delta_spatial_pool_w)
        hidden_mlp = max(128, d_model // 2)
        self.action_encoder = MultiEmbodimentActionEncoder(action_dim=action_dim, hidden_size=d_model, num_embodiments=1)
        self.state_encoder = CategorySpecificMLP(
            num_categories=1, input_dim=state_dim, hidden_dim=hidden_mlp, output_dim=d_model
        )
        self.delta_proj = nn.Linear(delta_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.pos_embedding = nn.Embedding(num_actions, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [ActionDiTBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )

        self.out_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.action_decoder = CategorySpecificMLP(
            num_categories=1, input_dim=d_model, hidden_dim=hidden_mlp, output_dim=action_dim
        )

    @staticmethod
    def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _compute_sigma(self, delta_tokens: torch.Tensor) -> torch.Tensor:
        # delta small -> sigma large; delta large -> sigma small
        delta_mag = torch.norm(delta_tokens, dim=-1, keepdim=True)
        return torch.exp(-self.sigma_k * delta_mag)

    def _prepare_delta_tokens(self, delta_v: torch.Tensor) -> torch.Tensor:
        """
        Support both:
          - [B, 8, C]
          - [B, 8, C, H, W]  (spatially rich delta-v tokens)
        Return: [B, L, D]
        """
        if delta_v.ndim == 3:
            repeat_factor = self.num_actions // delta_v.shape[1]
            tokens = delta_v.repeat_interleave(repeat_factor, dim=1)
            return self.delta_proj(tokens)
        if delta_v.ndim == 5:
            # [B, T, C, H, W]
            bsz, t, c, h, w = delta_v.shape
            x = delta_v
            # Optional spatial pooling to control token count when needed.
            if self.delta_spatial_pool_h > 0 and self.delta_spatial_pool_w > 0:
                x = x.reshape(bsz * t, c, h, w)
                x = torch.nn.functional.adaptive_avg_pool2d(x, (self.delta_spatial_pool_h, self.delta_spatial_pool_w))
                h, w = self.delta_spatial_pool_h, self.delta_spatial_pool_w
                x = x.reshape(bsz, t, c, h, w)
            x = x.permute(0, 1, 3, 4, 2).reshape(bsz, t * h * w, c).contiguous()
            return self.delta_proj(x)
        raise ValueError(f"Unsupported delta_v shape: {tuple(delta_v.shape)}")

    def _build_temb(self, timestep: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if timestep is None:
            t = torch.zeros(batch_size, device=device)
        else:
            t = timestep.to(device=device)
            if t.ndim == 2 and t.shape[1] == 1:
                t = t[:, 0]
        t = t.clamp(min=0).float()
        t = torch.round(t).long()
        t = torch.clamp(t, 0, self.timestep_buckets - 1)
        return self.time_mlp(_sinusoidal_timestep_embedding(t, self.delta_proj.out_features))

    def forward(
        self,
        z_action: torch.Tensor,
        delta_v: torch.Tensor,
        state_vec: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_action: [B, 64, 16]
            delta_v: [B, 8, 16]
            state_vec: [B, 16]
            timestep: [B] or [B,1], optional
        Returns:
            pred_action: [B, 64, 16]
        """
        bsz, L, _ = z_action.shape
        assert L == self.num_actions, f"Expected num_actions={self.num_actions}, got {L}"
        cat_ids = torch.zeros(bsz, dtype=torch.long, device=z_action.device)
        x = self.action_encoder(z_action, timesteps=timestep if timestep is not None else torch.zeros(bsz, device=z_action.device, dtype=torch.long), cat_ids=cat_ids)
        pos_ids = torch.arange(L, dtype=torch.long, device=z_action.device)
        x = x + self.pos_embedding(pos_ids).unsqueeze(0)
        temb = self._build_temb(timestep, batch_size=bsz, device=x.device)

        delta_tokens = self._prepare_delta_tokens(delta_v)
        state_tokens = self.state_encoder(state_vec.unsqueeze(1), cat_ids)
        state_tokens = state_tokens + self.state_proj(state_vec).unsqueeze(1)
        sigma = self._compute_sigma(delta_tokens)

        causal_mask = self._build_causal_mask(L, x.device)
        for block in self.blocks:
            x = block(x, delta_tokens, state_tokens, sigma, temb, causal_mask)

        shift, scale = self.out_mod(temb).chunk(2, dim=-1)
        x = self.out_norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.action_decoder(x, cat_ids)
