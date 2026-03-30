"""Standard rectified-flow strategy — the original delta-velocity approach.

This corresponds to the legacy behavior where the action head receives
delta_v = v[t] - v[t-1] from the video model's velocity prediction.
Gradients flow from the action head back into the video backbone
(when action_delta_video_t >= 0, a fixed-t denoise pass is used).
"""

from __future__ import annotations

from typing import Dict

import torch
from einops import rearrange
from torch import Tensor

from models.video_strategies import register_video_strategy
from models.video_strategies.base import BaseVideoStrategy


@register_video_strategy("standard_rf")
class StandardRFStrategy(BaseVideoStrategy):
    """Legacy: feed delta-velocity to action head, gradients flow through video net."""

    def __init__(self, delta_video_t: float = 0.5, **_extra):
        self.delta_video_t = float(delta_video_t)

    def prepare_video_forward(self, model, data_batch, output_batch, iteration):
        return {}

    def extract_action_condition(self, model, output_batch, extra):
        num_cond = int(getattr(model.config, "min_num_conditional_frames", 4))
        num_pred = 8
        if self.delta_video_t >= 0.0:
            delta_v = model._compute_delta_v_at_fixed_video_t(output_batch, num_cond=num_cond, num_pred=num_pred)
        else:
            delta_v = model._compute_delta_v(output_batch, num_cond=num_cond, num_pred=num_pred)
        return delta_v

    def should_detach_action_grad(self) -> bool:
        return False
