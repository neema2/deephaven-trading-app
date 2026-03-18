"""
streaming — Real-Time Ticking Table Server
============================================
Wraps the streaming engine (currently Deephaven) behind a consistent API.

Public surface::

    from streaming import TickingTable, LiveTable, flush, snapshot
    from streaming import agg
    from streaming import ticking, get_tables

Platform lifecycle lives in ``streaming.admin``.
"""

from streaming import agg
from streaming.client import StreamingClient
from streaming.decorator import clear_stale_tables, get_active_tables, get_tables, get_ticking_tables, ticking
from streaming.port_check import PortInUseError, assert_ports_free, check_ports, preflight_check, probe_ports
from streaming.table import LiveTable, TickingTable, flush, snapshot

__all__ = [
    "LiveTable",
    "PortInUseError",
    "StreamingClient",
    "TickingTable",
    "agg",
    "assert_ports_free",
    "check_ports",
    "clear_stale_tables",
    "flush",
    "get_active_tables",
    "get_tables",
    "get_ticking_tables",
    "preflight_check",
    "probe_ports",
    "snapshot",
    "ticking",
]
