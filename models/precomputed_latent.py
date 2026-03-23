from __future__ import annotations

import torch
from torch import Tensor

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
)


class IdentityLatentTokenizer(torch.nn.Module):
    """Identity encoder with optional real VAE decoder for visualization callbacks."""

    def __init__(
        self,
        latent_ch: int = 16,
        spatial_compression_factor: int = 8,
        name: str = "identity_latent_tokenizer",
        enable_decode: bool = False,
        vae_pth: str | None = None,
        temporal_window: int = 16,
    ):
        super().__init__()
        self._latent_ch = latent_ch
        self._spatial_compression_factor = spatial_compression_factor
        self.name = name
        self._decoder = None

        if enable_decode:
            from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

            decoder_kwargs = {"temporal_window": temporal_window}
            if vae_pth:
                decoder_kwargs["vae_pth"] = vae_pth
            self._decoder = Wan2pt1VAEInterface(**decoder_kwargs)

    @property
    def latent_ch(self) -> int:
        return self._latent_ch

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self._decoder is None:
            raise RuntimeError(
                "IdentityLatentTokenizer.decode() requested but decoder is disabled. "
                "Set enable_decode=True and provide vae_pth."
            )
        return self._decoder.decode(latent)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if self._decoder is None:
            return num_pixel_frames
        return self._decoder.get_latent_num_frames(num_pixel_frames)

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if self._decoder is None:
            return num_latent_frames
        return self._decoder.get_pixel_num_frames(num_latent_frames)

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor


class PrecomputedLatentVideo2WorldModel(Video2WorldModelRectifiedFlow):
    """Video2World model variant that consumes precomputed latents directly."""

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[Tensor, Tensor, Video2WorldCondition]:
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key

        if input_key not in data_batch:
            raise KeyError(f"Missing required key '{input_key}' in batch")

        # Precomputed latents are already model-ready; skip normalization and VAE encoding.
        latent_state = data_batch[input_key].to(**self.tensor_kwargs).contiguous().float()

        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return latent_state, latent_state, condition
