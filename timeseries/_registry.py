"""
Timeseries alias registry — maps alias names to TSDB connection info.

Cross-process: register_alias sets env vars so subprocesses can resolve.
"""

from __future__ import annotations

import os
import threading
from typing import Any

_aliases: dict[str, dict] = {}   # name → {"host", "http_port", "ilp_port", "pg_port", "data_dir"}
_lock = threading.Lock()

_ENV_KEYS = ("data_dir", "host", "http_port", "ilp_port", "pg_port")


def register_alias(name: str, **kwargs: Any) -> None:
    """Register a TSDB server alias (in-process + env vars for subprocesses)."""
    with _lock:
        _aliases[name] = kwargs
    # Set env vars for cross-process resolution
    prefix = f"TSDB_{name.upper()}_"
    for key in _ENV_KEYS:
        if key in kwargs:
            os.environ[f"{prefix}{key.upper()}"] = str(kwargs[key])


def resolve_alias(name: str) -> dict | None:
    """Resolve a TSDB alias — checks in-process registry, then env vars."""
    with _lock:
        if name in _aliases:
            return _aliases[name]
    # Fallback: read from env vars (subprocess case)
    prefix = f"TSDB_{name.upper()}_"
    env_cfg: dict[str, Any] = {}
    for key in _ENV_KEYS:
        val = os.environ.get(f"{prefix}{key.upper()}")
        if val is not None:
            # Convert port values to int
            if key.endswith("_port"):
                env_cfg[key] = int(val)
            else:
                env_cfg[key] = val
    return env_cfg or None
