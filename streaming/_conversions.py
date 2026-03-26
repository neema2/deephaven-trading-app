"""
streaming._conversions — DH-specific value conversions (private).

Keeps Deephaven import isolated to the streaming package.

**Dual-mode:** In-process mode converts datetime → java.time.Instant.
Remote (Docker) mode passes values through as-is (serialised via repr()).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from streaming.admin import _needs_docker

_REMOTE = _needs_docker()


def to_streaming_value(value: Any) -> Any:
    """Convert a Python value to a streaming-engine-compatible value.

    In-process mode: datetime → java.time.Instant (via DH JVM).
    Remote mode: pass through (serialised via repr() in RemoteTickingTable).
    """
    if value is None:
        return None
    if isinstance(value, datetime) and not _REMOTE:
        from deephaven.time import to_j_instant
        return to_j_instant(value)
    if isinstance(value, Decimal):
        return float(value)
    return value
