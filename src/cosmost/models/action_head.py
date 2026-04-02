"""
Action Head for Video-Action Joint Generation.
Based on FastWAM's action_dit with modifications.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionTransformerBlock(nn.Module):
    """Transformer block for action prediction."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-attention (causal)
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention to video
        self.video_q_norm = nn.LayerNorm(d_model)
        self.video_kv_norm = nn.LayerNorm(d_model)
        self.video_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        video_tokens: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] action tokens
            video_tokens: [B, Lv, D] video tokens
            self_mask: [L, L] causal mask for self-attention
        """
        # Self-attention
        h = self.self_norm(x)
        h, _ = self.self_attn(h, h, h, attn_mask=self_mask, need_weights=False)
        x = x + h
        
        # Cross-attention to video
        q = self.video_q_norm(x)
        kv = self.video_kv_norm(video_tokens)
        h, _ = self.video_attn(q, kv, kv, need_weights=False)
        x = x + h
        
        # Feed-forward
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    
    @staticmethod
    def get_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding(t, self.dim)
        return self.mlp(emb)


class ActionHead(nn.Module):
    """
    Action Head for predicting action sequences from video features.
    
    Architecture:
    - Action embedding
    - Timestep embedding
    - Positional embedding
    - Transformer blocks with cross-attention to video
    - Output projection
    """
    
    def __init__(
        self,
        action_dim: int = 16,
        num_actions: int = 64,
        d_model: int = 1024,
        n_heads: int = 8,
        n_blocks: int = 8,
        video_hidden_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.d_model = d_model
        self.n_blocks = n_blocks
        
        # Action embedding
        self.action_in = nn.Sequential(
            nn.Linear(action_dim + 1, d_model),  # +1 for timestep per token
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Video projection
        self.video_proj = nn.Sequential(
            nn.LayerNorm(video_hidden_dim),
            nn.Linear(video_hidden_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Timestep embedding
        self.timestep_proj = TimestepEmbedder(d_model)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(num_actions, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ActionTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        
        # Output
        self.out_norm = nn.LayerNorm(d_model)
        self.action_out = nn.Linear(d_model, action_dim)
    
    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build causal mask for self-attention."""
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask
    
    def forward(
        self,
        z_action: torch.Tensor,
        video_tokens: List[torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z_action: [B, num_actions, action_dim] noisy actions
            video_tokens: List of [B, N, video_hidden_dim] from video DiT layers
            timestep: [B] timestep
            
        Returns:
            pred_velocity: [B, num_actions, action_dim]
        """
        B, seq_len, _ = z_action.shape
        device = z_action.device
        
        # Validate video_tokens
        if len(video_tokens) != len(self.blocks):
            # If mismatch, use the last layer for all blocks
            video_tokens = [video_tokens[-1]] * len(self.blocks)
        
        # Timestep embedding
        t_emb = self.timestep_proj(timestep)  # [B, d_model]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, d_model]
        
        # Action embedding with timestep
        t_per_token = timestep.view(B, 1, 1).expand(B, seq_len, 1)
        action_input = torch.cat([z_action, t_per_token], dim=-1)
        x = self.action_in(action_input)  # [B, L, D]
        
        # Add positional embedding
        pos_ids = torch.arange(seq_len, device=device)
        x = x + self.pos_embedding(pos_ids).unsqueeze(0)  # [B, L, D]
        
        # Add timestep embedding
        x = x + t_emb  # [B, L, D]
        
        # Build causal mask
        self_mask = self._build_causal_mask(seq_len, device)
        
        # Forward through blocks with layer-aligned video tokens
        for block, video_tokens_layer in zip(self.blocks, video_tokens):
            video_proj = self.video_proj(video_tokens_layer)  # [B, N, d_model]
            x = block(x, video_proj, self_mask)
        
        # Output
        x = self.out_norm(x)
        return self.action_out(x)


def build_action_head(name: str, **kwargs) -> ActionHead:
    """Factory function for action heads."""
    name = name.lower()
    if name == "transformer":
        return ActionHead(**kwargs)
    else:
        raise ValueError(f"Unknown action head: {name}")
