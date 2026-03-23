from __future__ import annotations

import torch
import torch.nn as nn


class ActionMIPBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.delta_lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.delta_attns = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(3)]
        )

        self.state_ln = nn.LayerNorm(d_model)
        self.state_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ffn_ln = nn.LayerNorm(d_model)
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
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_ln(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + h

        for ln, attn in zip(self.delta_lns, self.delta_attns):
            h = ln(x)
            h, _ = attn(h, delta_tokens, delta_tokens, need_weights=False)
            x = x + h

        h = self.state_ln(x)
        h, _ = self.state_attn(h, state_tokens, state_tokens, need_weights=False)
        x = x + sigma * h

        x = x + self.ffn(self.ffn_ln(x))
        return x


class ActionMIPHead(nn.Module):
    """
    MIP-style action head with causal self-attention and cross-attention to
    delta-velocity and robot state conditions.
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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.sigma_k = sigma_k

        self.in_proj = nn.Linear(action_dim, d_model)
        self.delta_proj = nn.Linear(delta_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.out_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

        self.blocks = nn.ModuleList([ActionMIPBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)])

    @staticmethod
    def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
        # True means masked for nn.MultiheadAttention
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _compute_sigma(self, delta_tokens: torch.Tensor) -> torch.Tensor:
        # delta small -> sigma large; delta large -> sigma small
        # delta_tokens: [B, L, D]
        delta_mag = torch.norm(delta_tokens, dim=-1, keepdim=True)
        return torch.exp(-self.sigma_k * delta_mag)

    def forward(self, z_action: torch.Tensor, delta_v: torch.Tensor, state_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_action: [B, 64, 16]
            delta_v: [B, 8, 16]
            state_vec: [B, 16]
        Returns:
            pred_action: [B, 64, 16]
        """
        bsz, L, _ = z_action.shape
        assert L == self.num_actions, f"Expected num_actions={self.num_actions}, got {L}"

        x = self.in_proj(z_action)

        # Align 8 delta_v tokens to 64 action tokens (8 actions per latent)
        repeat_factor = self.num_actions // delta_v.shape[1]
        delta_tokens = delta_v.repeat_interleave(repeat_factor, dim=1)
        delta_tokens = self.delta_proj(delta_tokens)

        # One state token broadcast for cross-attention
        state_tokens = self.state_proj(state_vec).unsqueeze(1)
        sigma = self._compute_sigma(delta_tokens)

        causal_mask = self._build_causal_mask(L, device=x.device)
        for block in self.blocks:
            x = block(x, delta_tokens, state_tokens, sigma, causal_mask)

        pred = self.out_proj(self.out_ln(x))
        return pred
