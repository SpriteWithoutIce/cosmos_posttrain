"""Action head registry and factory.

Usage:
    from models.action_heads import build_action_head

    head = build_action_head("flow_matching", action_dim=16, ...)
"""

from __future__ import annotations

from typing import Dict, Type

from models.action_heads.base import BaseActionHead

ACTION_HEAD_REGISTRY: Dict[str, Type[BaseActionHead]] = {}


def register_action_head(name: str):
    """Decorator to register an action head class."""
    def decorator(cls: Type[BaseActionHead]):
        if name in ACTION_HEAD_REGISTRY:
            raise ValueError(f"Action head '{name}' already registered.")
        ACTION_HEAD_REGISTRY[name] = cls
        return cls
    return decorator


def build_action_head(head_type: str, **kwargs) -> BaseActionHead:
    """Factory: build an action head by type name."""
    if head_type not in ACTION_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown action head type '{head_type}'. "
            f"Available: {sorted(ACTION_HEAD_REGISTRY.keys())}"
        )
    return ACTION_HEAD_REGISTRY[head_type](**kwargs)


# Import submodules to trigger registration.
from models.action_heads import mip_head as _mip  # noqa: F401, E402
from models.action_heads import flow_matching_head as _fm  # noqa: F401, E402

__all__ = ["BaseActionHead", "build_action_head", "register_action_head", "ACTION_HEAD_REGISTRY"]
