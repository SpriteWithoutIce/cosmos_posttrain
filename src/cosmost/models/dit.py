"""
Simplified DiT (Diffusion Transformer) for Video-Action Joint Generation.
Based on Cosmos-Predict2.5's minimal_v4_dit.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: Optional[int],
        num_heads: int,
        head_dim: int,
        qkv_format: str = "bshd",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim if context_dim is not None else dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim if context_dim is not None else dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context if context is not None else x)
        v = self.to_v(context if context is not None else x)
        
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """
    DiT Block with:
    - Self-attention
    - Cross-attention (for text/action conditioning)
    - Feed-forward
    - AdaLN modulation
    """
    
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = Attention(dim, None, num_heads, head_dim)
        
        # Cross-attention
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, context_dim, num_heads, head_dim)
        
        # Feed-forward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim, int(dim * mlp_ratio))
        
        # AdaLN modulation
        self.use_adaln_lora = use_adaln_lora
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        
        if use_adaln_lora:
            # Additional modulation for action conditioning
            self.adaln_lora_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(adaln_lora_dim, 3 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens
            t_emb: [B, D] timestep embedding
            context: [B, M, C] cross-attention context (text + action)
        """
        # AdaLN parameters
        adaln_params = self.adaln_modulation(t_emb)
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca = adaln_params.chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        h = self.norm1(x) * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        h = self.self_attn(h)
        x = x + gate_sa.unsqueeze(1) * h
        
        # Cross-attention
        h = self.norm3(x)
        h = self.cross_attn(h, context)
        x = x + h
        
        # Feed-forward with AdaLN
        h = self.norm2(x) * (1 + scale_ca.unsqueeze(1)) + shift_ca.unsqueeze(1)
        h = self.ff(h)
        x = x + gate_ca.unsqueeze(1) * h
        
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.freq_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class VideoDiT(nn.Module):
    """
    Video DiT for joint generation.
    Supports action conditioning via cross-attention.
    """
    
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_dim: int = 1024,  # Text embedding dim
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        max_frames: int = 128,
        max_height: int = 240,
        max_width: int = 240,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels,
            model_channels,
            kernel_size=(patch_temporal, patch_spatial, patch_spatial),
            stride=(patch_temporal, patch_spatial, patch_spatial),
        )
        
        # Timestep embedder
        self.t_embedder = TimestepEmbedder(model_channels)
        
        # Cross-attention projection (for text)
        self.crossattn_proj = nn.Linear(crossattn_dim, model_channels)
        
        # Action projection (for action conditioning)
        self.action_proj = nn.Sequential(
            nn.Linear(64 * 16, model_channels),  # 64 actions * 16 dim
            nn.LayerNorm(model_channels),
            nn.GELU(),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=model_channels,
                context_dim=model_channels,  # Text and action projected to same dim
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_adaln_lora=use_adaln_lora,
                adaln_lora_dim=adaln_lora_dim,
            )
            for _ in range(num_blocks)
        ])
        
        # Output layer
        self.final_norm = nn.LayerNorm(model_channels)
        self.final_linear = nn.Linear(model_channels, patch_temporal * patch_spatial * patch_spatial * out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize patch embedding like nn.Linear
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        
        # Initialize timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
    
    def unpatchify(self, x: torch.Tensor, patch_t: int, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Convert from tokens back to video latents.
        x: [B, T*H*W, patch_t*patch_h*patch_w*C]
        output: [B, C, T*patch_t, H*patch_h, W*patch_w]
        """
        B, N, _ = x.shape
        C = self.out_channels
        
        x = rearrange(
            x,
            "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
            t=patch_t,
            h=patch_h,
            w=patch_w,
            pt=self.patch_temporal,
            ph=self.patch_spatial,
            pw=self.patch_spatial,
        )
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """
        Forward pass.
        
        Args:
            x: [B, C, T, H, W] video latents
            timesteps: [B] or [B, 1] timestep
            crossattn_emb: [B, L, D] text embeddings
            action: [B, 64, 16] action sequence (optional)
            return_hidden: whether to return hidden states for action head
            
        Returns:
            output: [B, C, T, H, W] predicted velocity
            hidden_states: list of hidden states if return_hidden=True
        """
        B, C, T, H, W = x.shape
        
        # Patchify
        x = self.patch_embed(x)  # [B, model_channels, T', H', W']
        _, _, patch_t, patch_h, patch_w = x.shape
        x = rearrange(x, "b c t h w -> b (t h w) c")  # [B, N, D]
        
        # Timestep embedding
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb = self.t_embedder(timesteps.squeeze(1))  # [B, D]
        
        # Prepare cross-attention context: [Text, Action]
        # Ensure crossattn_emb is float (not bfloat16) for projection
        if crossattn_emb is not None:
            crossattn_emb = crossattn_emb.float()
        context = self.crossattn_proj(crossattn_emb)  # [B, L, D]
        
        if action is not None:
            # Project and append action to context
            action_flat = rearrange(action, "b n d -> b (n d)")
            action_emb = self.action_proj(action_flat).unsqueeze(1)  # [B, 1, D]
            context = torch.cat([context, action_emb], dim=1)  # [B, L+1, D]
        
        # Forward through blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, t_emb, context)
            if return_hidden:
                hidden_states.append(x)
        
        # Output
        x = self.final_norm(x)
        x = self.final_linear(x)
        output = self.unpatchify(x, patch_t, patch_h, patch_w)
        
        if return_hidden:
            return output, hidden_states
        return output
