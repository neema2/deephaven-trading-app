"""
streaming — Real-Time Ticking Table Server
============================================
Wraps the streaming engine (currently Deephaven) behind a consistent API.

Public surface::

    from streaming import TickingTable, LiveTable, flush
    from streaming import agg
    from streaming import ticking, get_tables

Platform lifecycle lives in ``streaming.admin``.
"""

from streaming import agg
from streaming.client import StreamingClient
from streaming.decorator import get_tables, get_ticking_tables, ticking
from streaming.table import LiveTable, TickingTable, flush

__all__ = [
    "LiveTable",
    "StreamingClient",
    "TickingTable",
    "agg",
    "flush",
    "get_tables",
    "get_ticking_tables",
    "ticking",
]
