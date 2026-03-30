"""Flow-matching action head.

Standard rectified-flow formulation:
  - Sample t ~ U[0,1] (or beta distribution)
  - Interpolate: z_t = (1-t)*actions + t*noise
  - Predict velocity: v_pred = model(z_t, t, cond)
  - Target velocity: v_target = noise - actions
  - Loss: MSE(v_pred, v_target)

Condition features = last hidden state from video DiT [B, T*H*W, D] (high-dim).
State branch can be disabled via use_state=False.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.action_heads import register_action_head
from models.action_heads.base import BaseActionHead


def _sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal timestep embedding [B] -> [B, dim]."""
    if t.ndim > 1:
        t = t.view(-1)
    t = t.float()
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _build_causal_self_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)


def _build_causal_cross_mask(num_q: int, num_kv: int, actions_per_kv: int, device: torch.device) -> Tensor:
    mask = torch.full((num_q, num_kv), float("-inf"), device=device)
    for j in range(num_q):
        allowed = j // max(actions_per_kv, 1) + 1
        mask[j, : min(allowed, num_kv)] = 0.0
    return mask


class FlowMatchingBlock(nn.Module):
    """Transformer block for flow-matching action head.

    1x self-attention + 1x cross-attention (to video hidden state)
    + optional 1x cross-attention (to state) + FFN.
    """

    def __init__(self, d_model: int, n_heads: int, use_state: bool = True, dropout: float = 0.0):
        super().__init__()
        self.use_state = use_state

        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention to video hidden state
        self.cross_video_q_norm = nn.LayerNorm(d_model)
        self.cross_video_kv_norm = nn.LayerNorm(d_model)
        self.cross_video_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Optional state cross-attention
        if self.use_state:
            self.cross_state_q_norm = nn.LayerNorm(d_model)
            self.cross_state_kv_norm = nn.LayerNorm(d_model)
            self.cross_state_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: Tensor,
        video_tokens: Tensor,
        state_tokens: Tensor | None,
        self_mask: Tensor,
        cross_video_mask: Tensor | None,
        cross_state_mask: Tensor | None,
    ) -> Tensor:
        # Self-attention
        h = self.self_norm(x)
        h, _ = self.self_attn(h, h, h, attn_mask=self_mask, need_weights=False)
        x = x + h

        # Cross-attention to video hidden state
        q = self.cross_video_q_norm(x)
        kv = self.cross_video_kv_norm(video_tokens)
        h, _ = self.cross_video_attn(q, kv, kv, attn_mask=cross_video_mask, need_weights=False)
        x = x + h

        # Optional state cross-attention
        if self.use_state and state_tokens is not None:
            q_s = self.cross_state_q_norm(x)
            kv_s = self.cross_state_kv_norm(state_tokens)
            h_s, _ = self.cross_state_attn(q_s, kv_s, kv_s, attn_mask=cross_state_mask, need_weights=False)
            x = x + h_s

        x = x + self.ffn(self.ffn_norm(x))
        return x


@register_action_head("flow_matching")
class FlowMatchingActionHead(BaseActionHead):
    """Flow-matching action head.

    Receives high-dimensional hidden state from video DiT as condition,
    uses standard rectified-flow loss for action prediction.
    """

    def __init__(
        self,
        action_dim: int = 16,
        num_actions: int = 64,
        d_model: int = 1024,
        n_heads: int = 8,
        n_blocks: int = 8,
        hidden_dim: int = 2048,
        state_dim: int = 16,
        dropout: float = 0.0,
        actions_per_latent: int = 8,
        # flow-matching training params
        timestep_mode: str = "uniform",
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        # state branch control
        use_state: bool = False,
        **_extra,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.d_model = d_model
        self.actions_per_latent = actions_per_latent
        self.timestep_mode = timestep_mode
        self.noise_s = float(noise_s)
        self.use_state = use_state

        if timestep_mode == "beta":
            from torch.distributions import Beta
            self._beta_dist = Beta(float(noise_beta_alpha), float(noise_beta_beta))

        # Action token embedding: action_dim + 1 (for scalar timestep token)
        self.action_in = nn.Sequential(
            nn.Linear(action_dim + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embedding = nn.Embedding(num_actions, d_model)

        # Timestep embedding (global additive)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Project video hidden state to d_model
        self.video_proj = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Optional state encoder
        if self.use_state:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FlowMatchingBlock(d_model=d_model, n_heads=n_heads, use_state=use_state, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Output projection: predict velocity
        self.out_norm = nn.LayerNorm(d_model)
        self.action_out = nn.Linear(d_model, action_dim)

        self._cached_masks = {}

    def _get_masks(self, num_actions: int, num_kv: int, num_state: int, device: torch.device):
        key = (num_actions, num_kv, num_state, str(device))
        if key not in self._cached_masks:
            self_mask = _build_causal_self_mask(num_actions, device)
            cross_video_mask = _build_causal_cross_mask(num_actions, num_kv, self.actions_per_latent, device)
            cross_state_mask = None
            if self.use_state and num_state > 0:
                cross_state_mask = _build_causal_cross_mask(num_actions, num_state, num_actions, device)
            self._cached_masks[key] = (self_mask, cross_video_mask, cross_state_mask)
        return self._cached_masks[key]

    def _sample_timestep(self, batch_size: int, device: torch.device) -> Tensor:
        mode = self.timestep_mode.lower()
        if mode == "uniform":
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        if mode == "beta":
            sample = self._beta_dist.sample([batch_size]).to(device=device, dtype=torch.float32)
            t = (self.noise_s - sample) / self.noise_s
            return torch.clamp(t, 0.0, 1.0)
        raise ValueError(f"Unknown timestep_mode: {mode}")

    def forward(
        self,
        z_action: Tensor,
        condition_features: Tensor,
        timestep: Tensor | None = None,
        state_vec: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Predict velocity given noisy actions and video hidden state.

        Args:
            z_action: [B, L, action_dim] noisy action tokens
            condition_features: [B, N, hidden_dim] video hidden state
            timestep: [B] continuous timestep in [0, 1]
            state_vec: [B, state_dim] robot state (optional)
        """
        bsz, L, _ = z_action.shape
        device = z_action.device

        if timestep is None:
            timestep = torch.zeros(bsz, device=device, dtype=z_action.dtype)
        t = timestep.float()
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(bsz)

        # Scalar timestep token appended to each action token
        t_scalar = t[:, None, None].expand(bsz, L, 1)
        x = self.action_in(torch.cat([z_action, t_scalar.to(dtype=z_action.dtype)], dim=-1))

        # Position + global timestep embedding
        pos_ids = torch.arange(L, dtype=torch.long, device=device)
        temb = _sinusoidal_embedding(t, self.d_model)
        temb = self.timestep_mlp(temb).unsqueeze(1)  # [B, 1, D]
        x = x + self.pos_embedding(pos_ids).unsqueeze(0) + temb

        # Project video hidden state
        video_tokens = self.video_proj(condition_features)  # [B, N, d_model]

        # Optional state tokens
        state_tokens = None
        if self.use_state and state_vec is not None:
            state_tokens = self.state_encoder(state_vec.unsqueeze(1))  # [B, 1, d_model]

        num_state = state_tokens.shape[1] if state_tokens is not None else 0
        self_mask, cross_video_mask, cross_state_mask = self._get_masks(
            L, video_tokens.shape[1], num_state, device
        )

        for block in self.blocks:
            x = block(x, video_tokens, state_tokens, self_mask, cross_video_mask, cross_state_mask)

        x = self.out_norm(x)
        return self.action_out(x)  # [B, L, action_dim] — predicted velocity

    def compute_loss(
        self,
        actions_gt: Tensor,
        condition_features: Tensor,
        state_vec: Tensor | None = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Rectified-flow loss for action prediction.

        z_t = (1-t)*x0 + t*eps
        v_target = eps - x0
        loss = MSE(v_pred, v_target)
        """
        bsz = actions_gt.shape[0]
        device = actions_gt.device

        t = self._sample_timestep(bsz, device)  # [B] in [0, 1]
        noise = torch.randn_like(actions_gt)

        # Interpolation: z_t = (1-t)*actions + t*noise
        t_expand = t[:, None, None]  # [B, 1, 1]
        z_t = (1.0 - t_expand) * actions_gt + t_expand * noise

        # Target velocity: v = noise - actions (direction from data to noise)
        v_target = noise - actions_gt

        v_pred = self.forward(z_t, condition_features, timestep=t, state_vec=state_vec)
        loss = F.mse_loss(v_pred, v_target)

        return {
            "loss": loss,
            "action_loss": loss.detach(),
            "action_timestep_mean": t.detach().mean(),
        }
