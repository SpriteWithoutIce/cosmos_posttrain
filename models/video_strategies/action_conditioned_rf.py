"""Action-conditioned rectified-flow strategy (AC-WorldModel style).

Key differences from standard RF:
  1. Action tokens are injected into the video model's timestep embedding
     via an MLP, so the video model is conditioned on actions.
  2. The action head receives the **last hidden state** from the video DiT
     (high-dimensional, before final projection) — NOT delta-velocity.
  3. Action head gradients are **detached** from the video backbone.
  4. Hidden state is captured via a forward hook on the DiT's final_layer.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from models.video_strategies import register_video_strategy
from models.video_strategies.base import BaseVideoStrategy


class ActionTimestepInjector(nn.Module):
    """MLP that maps action tokens to a bias added to the video timestep embedding.

    Input:  [B, num_actions, action_dim] → pool → MLP → [B, 1, model_channels]
    This is added to the timestep embedding before it enters the DiT blocks.
    """

    def __init__(self, action_dim: int, num_actions: int, model_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim * num_actions, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

    def forward(self, actions: Tensor) -> Tensor:
        """actions: [B, num_actions, action_dim] -> [B, model_channels]"""
        bsz = actions.shape[0]
        flat = actions.reshape(bsz, -1)  # [B, num_actions * action_dim]
        return self.mlp(flat)  # [B, model_channels]


@register_video_strategy("action_conditioned_rf")
class ActionConditionedRFStrategy(BaseVideoStrategy):
    """AC-WorldModel: inject actions into timestep, extract last hidden state.

    This strategy:
      - Owns an ``ActionTimestepInjector`` module (its parameters are trained
        with the video backbone learning rate).
      - Installs a forward hook on ``model.net.final_layer`` to capture the
        last hidden state before the output projection.
      - Detaches the hidden state so action head gradients don't flow back.
    """

    def __init__(
        self,
        action_dim: int = 16,
        num_actions: int = 64,
        model_channels: int = 2048,
        **_extra,
    ):
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.model_channels = model_channels

        self.injector = ActionTimestepInjector(action_dim, num_actions, model_channels)
        self._last_hidden_state: Tensor | None = None
        self._hook_handle = None
        self._hook_installed = False

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------
    def _install_hook(self, model) -> None:
        """Install a forward pre-hook on the DiT's final_layer to capture hidden state."""
        if self._hook_installed:
            return

        def _capture_hook(module, args):
            # final_layer.forward signature: (x_B_T_H_W_D, emb_B_T_D, ...)
            # args[0] is the hidden state before projection
            x = args[0]
            # Flatten spatial: [B, T, H, W, D] -> [B, T*H*W, D]
            B, T, H, W, D = x.shape
            self._last_hidden_state = x.reshape(B, T * H * W, D)

        self._hook_handle = model.net.final_layer.register_forward_pre_hook(_capture_hook)
        self._hook_installed = True

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            self._hook_installed = False

    # ------------------------------------------------------------------
    # BaseVideoStrategy interface
    # ------------------------------------------------------------------
    def prepare_video_forward(self, model, data_batch, output_batch, iteration):
        """Inject action embedding into the video model's timestep embedding."""
        self._install_hook(model)

        actions = data_batch.get("actions")
        if actions is None:
            return {}

        actions = actions.to(device=next(model.parameters()).device, dtype=torch.float32)

        # Compute action bias and store it — the actual injection happens
        # by patching the t_embedder output in the model's forward.
        action_bias = self.injector(actions)  # [B, model_channels]
        return {"action_timestep_bias": action_bias}

    def extract_action_condition(self, model, output_batch, extra):
        """Return the captured last hidden state (detached)."""
        if self._last_hidden_state is None:
            raise RuntimeError(
                "ActionConditionedRFStrategy: no hidden state captured. "
                "Ensure the forward hook is installed and model.net forward was called."
            )
        hidden = self._last_hidden_state.detach()
        self._last_hidden_state = None  # consume to avoid stale reference
        return hidden

    def should_detach_action_grad(self) -> bool:
        return True

    def get_extra_parameters(self) -> list[nn.Parameter]:
        """Return injector parameters so they can be added to the video param group."""
        return list(self.injector.parameters())
