"""Abstract base class for video training strategies.

A video strategy controls how the video model's training step is augmented —
e.g. whether action embeddings are injected into timesteps, how hidden states
are extracted, and how the action head loss is combined with video loss.
"""

from __future__ import annotations

import abc
from typing import Dict

import torch
from torch import Tensor


class BaseVideoStrategy(abc.ABC):
    """Interface for video training strategies.

    Strategies are *not* nn.Modules — they orchestrate the model's training_step
    but do not own parameters.  They are lightweight policy objects.
    """

    @abc.abstractmethod
    def prepare_video_forward(
        self,
        model,
        data_batch: dict,
        output_batch: dict,
        iteration: int,
    ) -> Dict[str, Tensor]:
        """Called before or during the video forward pass.

        May inject additional conditioning (e.g. action tokens into timesteps).
        Returns a dict of tensors that will be passed to ``extract_action_condition``.
        """

    @abc.abstractmethod
    def extract_action_condition(
        self,
        model,
        output_batch: dict,
        extra: Dict[str, Tensor],
    ) -> Tensor:
        """Extract the condition tensor fed into the action head.

        Returns:
            condition_features: [B, N, D] tensor for the action head.
        """

    @abc.abstractmethod
    def should_detach_action_grad(self) -> bool:
        """Whether action head gradients should be detached from the video backbone."""
