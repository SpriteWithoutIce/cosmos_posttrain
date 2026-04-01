from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
from einops import rearrange


VideoActionConditionerBuilder = Callable[..., nn.Module]
_VIDEO_ACTION_CONDITIONER_REGISTRY: Dict[str, VideoActionConditionerBuilder] = {}


def register_video_action_conditioner(name: str) -> Callable[[VideoActionConditionerBuilder], VideoActionConditionerBuilder]:
    def decorator(builder: VideoActionConditionerBuilder) -> VideoActionConditionerBuilder:
        key = str(name).lower()
        if key in _VIDEO_ACTION_CONDITIONER_REGISTRY:
            raise ValueError(f"Video action conditioner '{name}' is already registered.")
        _VIDEO_ACTION_CONDITIONER_REGISTRY[key] = builder
        return builder

    return decorator


def build_video_action_conditioner(name: str, **kwargs) -> nn.Module | None:
    key = str(name).lower()
    if key in {"", "none", "null"}:
        return None
    if key not in _VIDEO_ACTION_CONDITIONER_REGISTRY:
        available = ", ".join(sorted(_VIDEO_ACTION_CONDITIONER_REGISTRY)) or "<empty>"
        raise ValueError(f"Unknown video action conditioner '{name}'. Available: {available}")
    return _VIDEO_ACTION_CONDITIONER_REGISTRY[key](**kwargs)


class _ActionMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_video_action_conditioner("mlp")
class MLPVideoActionConditioner(nn.Module):
    def __init__(
        self,
        action_dim: int,
        num_actions: int,
        model_channels: int,
        hidden_features: int | None = None,
        use_adaln_lora: bool = True,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_actions = int(num_actions)
        self.model_channels = int(model_channels)
        self.hidden_features = int(hidden_features or (4 * model_channels))
        self.use_adaln_lora = bool(use_adaln_lora)
        flat_dim = self.action_dim * self.num_actions
        self.action_embedder_B_D = _ActionMlp(flat_dim, self.hidden_features, self.model_channels)
        self.action_embedder_B_3D = (
            _ActionMlp(flat_dim, self.hidden_features, 3 * self.model_channels) if self.use_adaln_lora else None
        )

    def forward(self, action: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if action is None:
            return None, None
        if action.ndim != 3:
            raise ValueError(f"Expected action [B,L,D], got {tuple(action.shape)}")
        if action.shape[1] != self.num_actions or action.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected action shape [B,{self.num_actions},{self.action_dim}], got {tuple(action.shape)}"
            )
        flat_action = rearrange(action, "b l d -> b (l d)")
        action_emb_B_D = self.action_embedder_B_D(flat_action).unsqueeze(1)
        action_emb_B_3D = None
        if self.action_embedder_B_3D is not None:
            action_emb_B_3D = self.action_embedder_B_3D(flat_action).unsqueeze(1)
        return action_emb_B_D, action_emb_B_3D


__all__ = [
    "MLPVideoActionConditioner",
    "build_video_action_conditioner",
    "register_video_action_conditioner",
]
