"""
Backend Factory
===============
Selects and instantiates a TSDBBackend by name.
The only module that imports from timeseries/backends/.
"""

from __future__ import annotations

import os
from typing import Any

from timeseries._registry import resolve_alias
from timeseries.base import TSDBBackend


def create_backend(name: str | None = None, alias: str = "default", **kwargs: Any) -> TSDBBackend:
    """Create a TSDBBackend instance by name.

    Args:
        name: Backend name ("questdb"). Defaults to TSDB_BACKEND env var
              or "questdb" if unset.
        alias: TSDB alias to resolve connection info from (default: "default").
               Resolved config is used as defaults, explicit kwargs override.
        **kwargs: Backend-specific configuration passed to the constructor.

    Returns:
        A concrete TSDBBackend instance.

    Raises:
        ValueError: If the backend name is unknown.
    """
    if name is None:
        name = os.environ.get("TSDB_BACKEND", "questdb")

    # Resolve alias config (cross-process capable) and merge with kwargs
    alias_cfg = resolve_alias(alias) or {}
    merged = {**alias_cfg, **kwargs}  # explicit kwargs win

    if name == "questdb":
        from timeseries.backends.questdb import QuestDBBackend
        return QuestDBBackend(**merged)

    if name == "memory":
        from timeseries.backends.memory import MemoryBackend
        return MemoryBackend(**merged)

    raise ValueError(
        f"Unknown TSDB backend: {name!r}. Available: 'questdb', 'memory'"
    )
