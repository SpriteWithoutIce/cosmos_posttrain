"""Video training strategy registry and factory.

Usage:
    from models.video_strategies import build_video_strategy

    strategy = build_video_strategy("action_conditioned_rf", ...)
"""

from __future__ import annotations

from typing import Dict, Type

from models.video_strategies.base import BaseVideoStrategy

VIDEO_STRATEGY_REGISTRY: Dict[str, Type[BaseVideoStrategy]] = {}


def register_video_strategy(name: str):
    """Decorator to register a video training strategy."""
    def decorator(cls: Type[BaseVideoStrategy]):
        if name in VIDEO_STRATEGY_REGISTRY:
            raise ValueError(f"Video strategy '{name}' already registered.")
        VIDEO_STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def build_video_strategy(strategy_type: str, **kwargs) -> BaseVideoStrategy:
    """Factory: build a video strategy by type name."""
    if strategy_type not in VIDEO_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown video strategy '{strategy_type}'. "
            f"Available: {sorted(VIDEO_STRATEGY_REGISTRY.keys())}"
        )
    return VIDEO_STRATEGY_REGISTRY[strategy_type](**kwargs)


# Import submodules to trigger registration.
from models.video_strategies import standard_rf as _std  # noqa: F401, E402
from models.video_strategies import action_conditioned_rf as _ac  # noqa: F401, E402

__all__ = ["BaseVideoStrategy", "build_video_strategy", "register_video_strategy", "VIDEO_STRATEGY_REGISTRY"]
