from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn


ActionHeadBuilder = Callable[..., nn.Module]
_ACTION_HEAD_REGISTRY: Dict[str, ActionHeadBuilder] = {}


def register_action_head(name: str) -> Callable[[ActionHeadBuilder], ActionHeadBuilder]:
    def decorator(builder: ActionHeadBuilder) -> ActionHeadBuilder:
        key = str(name).lower()
        if key in _ACTION_HEAD_REGISTRY:
            raise ValueError(f"Action head '{name}' is already registered.")
        _ACTION_HEAD_REGISTRY[key] = builder
        return builder

    return decorator


def build_action_head(name: str, **kwargs) -> nn.Module:
    key = str(name).lower()
    if key not in _ACTION_HEAD_REGISTRY:
        available = ", ".join(sorted(_ACTION_HEAD_REGISTRY)) or "<empty>"
        raise ValueError(f"Unknown action head '{name}'. Available: {available}")
    return _ACTION_HEAD_REGISTRY[key](**kwargs)


def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
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


def _build_causal_self_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)


def _build_block_causal_self_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    block_size = max(int(block_size), 1)
    for q_idx in range(seq_len):
        current_block = q_idx // block_size
        allowed_until = min(seq_len, (current_block + 1) * block_size)
        mask[q_idx, :allowed_until] = 0.0
    return mask


def _build_causal_cross_mask(num_q: int, num_kv: int, actions_per_kv: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((num_q, num_kv), float("-inf"), device=device)
    for j in range(num_q):
        allowed = j // max(actions_per_kv, 1) + 1
        mask[j, : min(allowed, num_kv)] = 0.0
    return mask


class ActionTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_state_condition: bool = False):
        super().__init__()
        self.use_state_condition = bool(use_state_condition)

        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.video_q_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.video_kv_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.video_attns = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(3)]
        )

        if self.use_state_condition:
            self.state_q_norm = nn.LayerNorm(d_model)
            self.state_kv_norm = nn.LayerNorm(d_model)
            self.state_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        else:
            self.state_q_norm = None
            self.state_kv_norm = None
            self.state_attn = None

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _cross(self, x: torch.Tensor, cond: torch.Tensor, q_norm: nn.LayerNorm, kv_norm: nn.LayerNorm, attn: nn.Module, attn_mask: torch.Tensor | None) -> torch.Tensor:
        q = q_norm(x)
        kv = kv_norm(cond)
        h, _ = attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        return x + h

    def forward(
        self,
        x: torch.Tensor,
        video_tokens: torch.Tensor,
        state_tokens: torch.Tensor | None,
        self_mask: torch.Tensor,
        cross_video_mask: torch.Tensor,
        cross_state_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,L,D], got {tuple(x.shape)}")
        if video_tokens.ndim != 3:
            raise ValueError(f"Expected video_tokens [B,Lv,D], got {tuple(video_tokens.shape)}")
        if self.use_state_condition and (state_tokens is None or state_tokens.ndim != 3):
            raise ValueError("State conditioning is enabled, but state tokens are missing or invalid.")

        h = self.self_norm(x)
        h, _ = self.self_attn(h, h, h, attn_mask=self_mask, need_weights=False)
        x = x + h

        for q_norm, kv_norm, attn in zip(self.video_q_norms, self.video_kv_norms, self.video_attns):
            x = self._cross(x, video_tokens, q_norm, kv_norm, attn, cross_video_mask)

        if self.use_state_condition:
            x = self._cross(
                x,
                state_tokens,
                self.state_q_norm,
                self.state_kv_norm,
                self.state_attn,
                cross_state_mask,
            )

        x = x + self.ffn(self.ffn_norm(x))
        return x


@register_action_head("flow_matching")
class FlowMatchingActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 16,
        num_actions: int = 64,
        d_model: int = 1024,
        n_heads: int = 8,
        n_blocks: int = 8,
        video_hidden_dim: int = 2048,
        state_dim: int = 16,
        dropout: float = 0.0,
        timestep_buckets: int = 1000,
        actions_per_latent: int = 8,
        use_state_condition: bool = False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_actions = int(num_actions)
        self.video_hidden_dim = int(video_hidden_dim)
        self.state_dim = int(state_dim)
        self.timestep_buckets = int(timestep_buckets)
        self.actions_per_latent = int(actions_per_latent)
        self.use_state_condition = bool(use_state_condition)

        self.action_in = nn.Sequential(
            nn.Linear(self.action_dim + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.video_in = nn.Sequential(
            nn.LayerNorm(self.video_hidden_dim),
            nn.Linear(self.video_hidden_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        if self.use_state_condition:
            self.state_in = nn.Sequential(
                nn.Linear(self.state_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.state_in = None
        self.timestep_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embedding = nn.Embedding(self.num_actions, d_model)
        self.blocks = nn.ModuleList(
            [
                ActionTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    use_state_condition=self.use_state_condition,
                )
                for _ in range(n_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.action_out = nn.Linear(d_model, self.action_dim)
        self._cached_masks: Dict[Tuple[int, int, int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}

    def _build_token_timestep(
        self,
        timestep: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timestep is None:
            t = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            t = timestep.to(device=device, dtype=torch.float32)
            if t.ndim == 2 and t.shape[1] == 1:
                t = t[:, 0]
        t = t.clamp(0.0, 1.0)
        t_scalar = t.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
        temb = _sinusoidal_timestep_embedding(t * max(float(self.timestep_buckets - 1), 1.0), self.pos_embedding.embedding_dim)
        temb = self.timestep_proj(temb).unsqueeze(1)
        return t_scalar, temb

    def _prepare_video_tokens(self, video_tokens: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if video_tokens.ndim == 5:
            bsz, num_frames, h, w, hidden_dim = video_tokens.shape
            if hidden_dim != self.video_hidden_dim:
                raise ValueError(
                    f"Expected video hidden dim {self.video_hidden_dim}, got {hidden_dim}"
                )
            return video_tokens.view(bsz, num_frames, h * w, hidden_dim), num_frames, h * w
        if video_tokens.ndim == 4:
            bsz, num_frames, tokens_per_frame, hidden_dim = video_tokens.shape
            if hidden_dim != self.video_hidden_dim:
                raise ValueError(
                    f"Expected video hidden dim {self.video_hidden_dim}, got {hidden_dim}"
                )
            return video_tokens, num_frames, tokens_per_frame
        if video_tokens.ndim == 3:
            bsz, num_video_tokens, hidden_dim = video_tokens.shape
            if hidden_dim != self.video_hidden_dim:
                raise ValueError(
                    f"Expected video hidden dim {self.video_hidden_dim}, got {hidden_dim}"
                )
            return video_tokens.view(bsz, num_video_tokens, 1, hidden_dim), num_video_tokens, 1
        raise ValueError(
            "Expected video_tokens [B,T,D], [B,T,P,D], or [B,T,H,W,D], "
            f"got {tuple(video_tokens.shape)}"
        )

    def _get_masks(
        self,
        num_actions: int,
        num_frames: int,
        tokens_per_frame: int,
        num_state_tokens: int,
        device: torch.device,
    ):
        key = (num_actions, num_frames, tokens_per_frame, num_state_tokens, str(device))
        if key not in self._cached_masks:
            self_mask = _build_block_causal_self_mask(
                seq_len=num_actions,
                block_size=self.actions_per_latent,
                device=device,
            )
            cross_video_mask = None
            cross_state_mask = None
            if self.use_state_condition:
                cross_state_mask = _build_causal_cross_mask(num_actions, num_state_tokens, 1, device)
            self._cached_masks[key] = (self_mask, cross_video_mask, cross_state_mask)
        return self._cached_masks[key]

    def forward(
        self,
        z_action: torch.Tensor,
        video_tokens: torch.Tensor,
        state_vec: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_action.ndim != 3:
            raise ValueError(f"Expected z_action [B,L,D], got {tuple(z_action.shape)}")

        bsz, seq_len, action_dim = z_action.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action_dim}")
        if seq_len != self.num_actions:
            raise ValueError(f"Expected num_actions {self.num_actions}, got {seq_len}")
        if self.use_state_condition:
            if state_vec is None or state_vec.ndim != 2:
                raise ValueError("State conditioning is enabled, but state_vec is missing or invalid.")
            if state_vec.shape[-1] != self.state_dim:
                raise ValueError(f"Expected state dim {self.state_dim}, got {state_vec.shape[-1]}")

        t_scalar, temb = self._build_token_timestep(timestep, batch_size=bsz, seq_len=seq_len, device=z_action.device)

        x = self.action_in(torch.cat([z_action, t_scalar.to(dtype=z_action.dtype)], dim=-1))
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=z_action.device)
        x = x + self.pos_embedding(pos_ids).unsqueeze(0) + temb.to(dtype=x.dtype)

        video_tokens_grouped, num_frames, tokens_per_frame = self._prepare_video_tokens(video_tokens)
        video_ctx = self.video_in(video_tokens_grouped.to(dtype=x.dtype).view(bsz, num_frames * tokens_per_frame, self.video_hidden_dim))
        state_tokens = None
        num_state_tokens = 0
        if self.use_state_condition:
            state_tokens = self.state_in(state_vec.unsqueeze(1).to(dtype=x.dtype))
            num_state_tokens = state_tokens.shape[1]

        self_mask, cross_video_mask, cross_state_mask = self._get_masks(
            num_actions=seq_len,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            num_state_tokens=num_state_tokens,
            device=x.device,
        )

        for block in self.blocks:
            x = block(
                x,
                video_tokens=video_ctx,
                state_tokens=state_tokens,
                self_mask=self_mask,
                cross_video_mask=cross_video_mask,
                cross_state_mask=cross_state_mask,
            )

        x = self.out_norm(x)
        return self.action_out(x)


__all__ = [
    "FlowMatchingActionHead",
    "build_action_head",
    "register_action_head",
]
