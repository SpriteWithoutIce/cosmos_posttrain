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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.sigma_k = sigma_k
        self.timestep_buckets = timestep_buckets

        self.in_proj = nn.Linear(action_dim, d_model)
        self.delta_proj = nn.Linear(delta_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
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
        self.out_proj = nn.Linear(d_model, action_dim)

    @staticmethod
    def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _compute_sigma(self, delta_tokens: torch.Tensor) -> torch.Tensor:
        # delta small -> sigma large; delta large -> sigma small
        delta_mag = torch.norm(delta_tokens, dim=-1, keepdim=True)
        return torch.exp(-self.sigma_k * delta_mag)

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
        return self.time_mlp(_sinusoidal_timestep_embedding(t, self.in_proj.out_features))

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

        x = self.in_proj(z_action)
        temb = self._build_temb(timestep, batch_size=bsz, device=x.device)

        repeat_factor = self.num_actions // delta_v.shape[1]
        delta_tokens = self.delta_proj(delta_v.repeat_interleave(repeat_factor, dim=1))
        state_tokens = self.state_proj(state_vec).unsqueeze(1)
        sigma = self._compute_sigma(delta_tokens)

        causal_mask = self._build_causal_mask(L, x.device)
        for block in self.blocks:
            x = block(x, delta_tokens, state_tokens, sigma, temb, causal_mask)

        shift, scale = self.out_mod(temb).chunk(2, dim=-1)
        x = self.out_norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.out_proj(x)
