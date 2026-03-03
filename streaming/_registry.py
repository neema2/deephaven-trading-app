"""
Streaming alias registry — maps alias names to streaming server info.
"""

from __future__ import annotations

import threading
from typing import Any

_aliases: dict[str, dict] = {}   # name → {"port": ..., ...}
_lock = threading.Lock()


def register_alias(name: str, **kwargs: Any) -> None:
    """Register a streaming server alias."""
    with _lock:
        _aliases[name] = kwargs


def resolve_alias(name: str) -> dict | None:
    """Resolve a streaming alias to server info."""
    with _lock:
        return _aliases.get(name)
