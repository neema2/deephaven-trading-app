"""Scheduler alias registry (internal)."""

from __future__ import annotations

import threading

from scheduler.server import SchedulerServer

_aliases: dict[str, dict] = {}
_lock = threading.Lock()


def register_alias(name: str, server: SchedulerServer) -> None:
    """Register a scheduler server under an alias name."""
    with _lock:
        _aliases[name] = {"server": server}


def resolve_alias(name: str) -> dict | None:
    """Resolve an alias to its server reference."""
    with _lock:
        return _aliases.get(name)
