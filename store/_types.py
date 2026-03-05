"""
store._types — Backward-compat re-export.

All types now live in the top-level ``db`` package.
Import from ``db`` directly for new code.
"""

from db import Connection, Cursor, connect

__all__ = ["Connection", "Cursor", "connect"]
