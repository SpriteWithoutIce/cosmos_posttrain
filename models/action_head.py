from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


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
    # float mask for nn.MultiheadAttention: 0 means keep, -inf means masked.
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)


def _build_causal_cross_mask(num_q: int, num_kv: int, actions_per_kv: int, device: torch.device) -> torch.Tensor:
    # action token j can attend to kv 0..j//actions_per_kv
    mask = torch.full((num_q, num_kv), float("-inf"), device=device)
    for j in range(num_q):
        allowed = j // max(actions_per_kv, 1) + 1
        mask[j, : min(allowed, num_kv)] = 0.0
    return mask


def _build_causal_cross_mask_grouped(
    num_q: int,
    num_frames: int,
    tokens_per_frame: int,
    actions_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    # query j can attend to delta_v tokens from frames [0 .. floor(j/actions_per_frame)]
    num_kv = num_frames * tokens_per_frame
    mask = torch.full((num_q, num_kv), float("-inf"), device=device)
    for j in range(num_q):
        allowed_frames = min(num_frames, (j // max(actions_per_frame, 1)) + 1)
        allowed_tokens = allowed_frames * tokens_per_frame
        mask[j, :allowed_tokens] = 0.0
    return mask


def _build_causal_cross_mask_tokenwise(num_q: int, num_kv: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((num_q, num_kv), float("-inf"), device=device)
    for j in range(num_q):
        mask[j, : min(j + 1, num_kv)] = 0.0
    return mask


class ActionDiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.cross_q_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.cross_kv_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.cross_attns = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(4)]
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _cross(self, x: torch.Tensor, cond: torch.Tensor, idx: int, attn_mask: torch.Tensor) -> torch.Tensor:
        q = self.cross_q_norms[idx](x)
        kv = self.cross_kv_norms[idx](cond)
        h, _ = self.cross_attns[idx](q, kv, kv, attn_mask=attn_mask, need_weights=False)
        return x + h

    def forward(
        self,
        x: torch.Tensor,
        delta_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        sigma: torch.Tensor,
        self_mask: torch.Tensor,
        cross_dv_mask: torch.Tensor,
        cross_state_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ActionDiTBlock expects x [B,L,D], got {tuple(x.shape)}")
        if delta_tokens.ndim != 3:
            raise ValueError(f"ActionDiTBlock expects delta_tokens [B,Lkv,D], got {tuple(delta_tokens.shape)}")
        if state_tokens.ndim != 3:
            raise ValueError(f"ActionDiTBlock expects state_tokens [B,Ls,D], got {tuple(state_tokens.shape)}")
        if sigma.ndim != 3:
            raise ValueError(f"ActionDiTBlock expects sigma [B,L,1], got {tuple(sigma.shape)}")

        h = self.self_norm(x)
        h, _ = self.self_attn(h, h, h, attn_mask=self_mask, need_weights=False)
        x = x + h

        x = self._cross(x, delta_tokens, idx=0, attn_mask=cross_dv_mask)
        x = self._cross(x, delta_tokens, idx=1, attn_mask=cross_dv_mask)
        x = self._cross(x, delta_tokens, idx=2, attn_mask=cross_dv_mask)

        h_in = self.cross_q_norms[3](x)
        kv = self.cross_kv_norms[3](state_tokens)
        h_state, _ = self.cross_attns[3](h_in, kv, kv, attn_mask=cross_state_mask, need_weights=False)
        x = x + sigma * h_state

        x = x + self.ffn(self.ffn_norm(x))
        return x


class ActionMIPHead(nn.Module):
    """
    MIP action head aligned with action_expert principle:
      - 1 causal self-attn
      - 3 causal delta_v cross-attn
      - 1 causal state cross-attn with sigma gate
      - FFN
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
        actions_per_latent: int = 8,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_actions = int(num_actions)
        self.delta_dim = int(delta_dim)
        self.sigma_k = float(sigma_k)
        self.timestep_buckets = int(timestep_buckets)
        self.delta_spatial_pool_h = int(delta_spatial_pool_h)
        self.delta_spatial_pool_w = int(delta_spatial_pool_w)
        self.actions_per_latent = int(actions_per_latent)

        self.action_in = nn.Sequential(
            nn.Linear(self.action_dim + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(self.delta_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.timestep_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.pos_embedding = nn.Embedding(self.num_actions, d_model)

        self.blocks = nn.ModuleList(
            [ActionDiTBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.action_out = nn.Linear(d_model, self.action_dim)

        self._cached_masks: Dict[Tuple[int, int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _prepare_delta_tokens(self, delta_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        # Returns:
        #   delta_tokens_raw: [B, T*P, C] (P=tokens_per_frame)
        #   sigma_source: [B, T, C] for sigma gating
        #   num_frames: T
        #   tokens_per_frame: P
        if delta_v.ndim == 3:
            bsz, t, c = delta_v.shape
            if c != self.delta_dim:
                raise ValueError(f"Expected delta token dim={self.delta_dim}, got {c}")
            return delta_v, delta_v, t, 1
        if delta_v.ndim != 5:
            raise ValueError(f"Unsupported delta_v shape: {tuple(delta_v.shape)}")

        # Keep spatial information: [B, T, C, H, W] -> [B, T*P, C]
        # where P is H*W or pooled_H*pooled_W.
        bsz, t, c, h, w = delta_v.shape
        if c != self.delta_dim:
            raise ValueError(f"Expected delta token dim={self.delta_dim}, got {c}")
        x = delta_v
        if self.delta_spatial_pool_h > 0 and self.delta_spatial_pool_w > 0:
            x = x.reshape(bsz * t, c, h, w)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (self.delta_spatial_pool_h, self.delta_spatial_pool_w))
            x = x.reshape(bsz, t, c, self.delta_spatial_pool_h, self.delta_spatial_pool_w)
            h, w = self.delta_spatial_pool_h, self.delta_spatial_pool_w
        sigma_source = x.mean(dim=(-1, -2))  # [B, T, C], only for gating magnitude
        tokens_per_frame = h * w
        delta_tokens_raw = x.permute(0, 1, 3, 4, 2).reshape(bsz, t * tokens_per_frame, c).contiguous()
        return delta_tokens_raw, sigma_source, t, tokens_per_frame

    def _compute_sigma(self, sigma_source: torch.Tensor, num_actions: int) -> torch.Tensor:
        # delta small -> sigma large; delta large -> sigma small.
        sigma_t = torch.exp(-self.sigma_k * torch.norm(sigma_source, dim=-1, keepdim=True))  # [B,T,1]
        t = sigma_t.shape[1]
        repeat = max(1, self.actions_per_latent)
        sigma = sigma_t.repeat_interleave(repeat, dim=1)
        if sigma.shape[1] < num_actions:
            pad = sigma[:, -1:, :].repeat(1, num_actions - sigma.shape[1], 1)
            sigma = torch.cat([sigma, pad], dim=1)
        return sigma[:, :num_actions, :]

    def _get_masks(
        self,
        num_actions: int,
        num_dv: int,
        num_state: int,
        device: torch.device,
        num_frames: int,
        tokens_per_frame: int,
    ):
        key = (num_actions, num_dv, num_state, str(device), num_frames, tokens_per_frame)
        if key not in self._cached_masks:
            self_mask = _build_causal_self_mask(num_actions, device)
            cross_dv_mask = _build_causal_cross_mask_grouped(
                num_q=num_actions,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                actions_per_frame=max(1, self.actions_per_latent),
                device=device,
            )
            cross_state_mask = _build_causal_cross_mask_tokenwise(num_actions, num_state, device)
            self._cached_masks[key] = (self_mask, cross_dv_mask, cross_state_mask)
        return self._cached_masks[key]

    def _build_token_timestep(self, timestep: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device):
        if timestep is None:
            t = torch.zeros(batch_size, device=device)
        else:
            t = timestep.to(device=device)
            if t.ndim == 2 and t.shape[1] == 1:
                t = t[:, 0]
        t = t.clamp(min=0).float()
        # scalar token t in [0,1]
        t_scalar = (t / max(float(self.timestep_buckets - 1), 1.0)).unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)

        # extra global temb injection to stabilize training
        t_disc = torch.round(t).long().clamp_(0, self.timestep_buckets - 1)
        temb = _sinusoidal_timestep_embedding(t_disc, self.pos_embedding.embedding_dim)
        temb = self.timestep_proj(temb).unsqueeze(1)  # [B,1,1]
        return t_scalar, temb

    def forward(
        self,
        z_action: torch.Tensor,
        delta_v: torch.Tensor,
        state_vec: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, L, c = z_action.shape
        if c != self.action_dim:
            raise ValueError(f"Expected z_action last dim={self.action_dim}, got {c}")
        if L != self.num_actions:
            raise ValueError(f"Expected num_actions={self.num_actions}, got {L}")
        if state_vec.ndim != 2:
            raise ValueError(f"Expected state_vec [B,state_dim], got {tuple(state_vec.shape)}")

        t_scalar, temb = self._build_token_timestep(timestep, batch_size=bsz, seq_len=L, device=z_action.device)

        x = self.action_in(torch.cat([z_action, t_scalar.to(dtype=z_action.dtype)], dim=-1))
        pos_ids = torch.arange(L, dtype=torch.long, device=z_action.device)
        x = x + self.pos_embedding(pos_ids).unsqueeze(0) + temb

        delta_tokens_raw, sigma_source, num_frames, tokens_per_frame = self._prepare_delta_tokens(delta_v)
        if delta_tokens_raw.ndim != 3:
            raise ValueError(f"delta_tokens_raw must be [B,Lkv,C], got {tuple(delta_tokens_raw.shape)}")
        delta_tokens = self.delta_encoder(delta_tokens_raw)
        state_tokens = self.state_encoder(state_vec.unsqueeze(1))
        sigma = self._compute_sigma(sigma_source, num_actions=L)

        self_mask, cross_dv_mask, cross_state_mask = self._get_masks(
            num_actions=L,
            num_dv=delta_tokens.shape[1],
            num_state=state_tokens.shape[1],
            device=x.device,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
        )

        for block in self.blocks:
            x = block(
                x,
                delta_tokens=delta_tokens,
                state_tokens=state_tokens,
                sigma=sigma,
                self_mask=self_mask,
                cross_dv_mask=cross_dv_mask,
                cross_state_mask=cross_state_mask,
            )

        x = self.out_norm(x)
        return self.action_out(x)
