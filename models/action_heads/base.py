"""Abstract base class for all action heads."""

from __future__ import annotations

import abc
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class BaseActionHead(nn.Module, abc.ABC):
    """Base interface for action prediction heads.

    Subclasses must implement:
      - forward():       predict from noisy actions + condition features
      - compute_loss():  full training-time loss computation
    """

    @abc.abstractmethod
    def forward(
        self,
        z_action: Tensor,
        condition_features: Tensor,
        timestep: Tensor | None = None,
        state_vec: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Predict denoised actions (or velocity) given noisy input and conditions.

        Args:
            z_action: noisy action tokens [B, num_actions, action_dim]
            condition_features: conditioning from video model [B, T, D]
            timestep: diffusion timestep [B]
            state_vec: robot state [B, state_dim] (optional)

        Returns:
            prediction [B, num_actions, action_dim]
        """

    @abc.abstractmethod
    def compute_loss(
        self,
        actions_gt: Tensor,
        condition_features: Tensor,
        state_vec: Tensor | None = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Compute training loss.

        Returns:
            dict with at least "loss" key and optional metric keys.
        """
