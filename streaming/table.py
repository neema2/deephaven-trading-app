"""
streaming.table — Auto-locked ticking table abstraction.
Hides Deephaven's DynamicTableWriter and update-graph locking behind a
clean Python API.  All derivation operations (last_by, agg_by, sort, …)
auto-acquire the UG shared lock so callers never segfault.
Classes:
    LiveTable     Read-only derived table (all ops auto-locked).
    TickingTable  Writable table (inherits LiveTable, adds write_row/flush).
"""

from __future__ import annotations

import logging
import platform
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture Detection
# ---------------------------------------------------------------------------

def _is_remote() -> bool:
    """Check if we are on ARM Linux (which requires remote Docker for Deephaven)."""
    sys_name = platform.system()
    machine = platform.machine().lower()
    is_arm = sys_name == "Linux" and any(arch in machine for arch in ("aarch64", "arm64", "armv8"))
    if not is_arm and sys_name == "Linux":
        # Extra check: if 'deephaven' is missing and it's Linux, we might be on a mystery ARM
        try:
            import deephaven
        except ImportError:
            # If we are on Linux and 'deephaven' (the x86 server) is missing, 
            # assume we need remote mode.
            return True
    return is_arm

# Shared session for remote ticking tables on ARM
_REMOTE_SESSION = None

def _get_remote_session() -> Any:
    global _REMOTE_SESSION
    if _REMOTE_SESSION is None:
        try:
            from pydeephaven import Session
            _REMOTE_SESSION = Session(host="localhost", port=10000)
        except ImportError:
            logger.error("pydeephaven package not found. On ARM64, please run 'pip install pydeephaven'")
            return None
        except Exception as e:
            logger.debug("Could not connect to Deephaven on localhost:10000: %s", e)
            return None
    return _REMOTE_SESSION


# ---------------------------------------------------------------------------
# Python type → Deephaven type mapping (lazy, avoids import at module level)
# ---------------------------------------------------------------------------

_PY_TO_DH = None

def _type_map() -> dict:
    """Return the Python→DH type dict, building it on first call."""
    global _PY_TO_DH
    if _PY_TO_DH is None:
        if _is_remote():
            import pyarrow as pa
            _PY_TO_DH = {
                str: pa.string(),
                float: pa.float64(),
                int: pa.int64(),
                bool: pa.bool_(),
                Decimal: pa.decimal128(38, 18),
                datetime: pa.timestamp("ns"),
            }
            return _PY_TO_DH
        else:
            from deephaven import dtypes as dht

        _PY_TO_DH = {
            str: dht.string,
            float: dht.double,
            int: dht.int64,
            bool: dht.bool_,
            Decimal: dht.double,
            datetime: dht.Instant,
        }
    return _PY_TO_DH

def _resolve_schema(schema: dict[str, type]) -> dict:
    """Convert a {name: python_type} dict to {name: dh_type}."""
    tm = _type_map()
    resolved = {}
    for name, py_type in schema.items():
        dh_type = tm.get(py_type)
        if dh_type is None:
            raise TypeError(
                f"Unsupported column type {py_type!r} for '{name}'. "
                f"Supported: {list(tm.keys())}"
            )
        resolved[name] = dh_type
    return resolved


# ---------------------------------------------------------------------------
# flush() — module-level helper
# ---------------------------------------------------------------------------

_UG_REF = None  # cached update graph (set on first flush from main thread)


def flush() -> None:
    """Flush the Deephaven update graph so pending writes become visible.

    Safe to call from any thread — the update graph reference is cached
    on the first call (which must happen on the main/DH thread).
    No-op on ARM/Remote where no local JVM server exists.
    """
    if _is_remote():
        return

    global _UG_REF
    if _UG_REF is None:
        from deephaven.execution_context import get_exec_ctx
        _UG_REF = get_exec_ctx().update_graph.j_update_graph
    _UG_REF.requestRefresh()


# ---------------------------------------------------------------------------
# LiveTable — read-only, auto-locked
# ---------------------------------------------------------------------------

class LiveTable:
    """Read-only derived ticking table with auto-locked operations.

    You obtain a ``LiveTable`` from ``TickingTable.last_by()``,
    ``LiveTable.agg_by()``, etc.  You can derive further, snapshot to
    pandas, or publish to the Deephaven query scope — but you cannot
    write rows.

    Every derivation method acquires the UG **shared lock** internally
    so that callers never need to think about Deephaven locking.
    """

    __slots__ = ("_table",)

    def __init__(self, dh_table: Any) -> None:
        self._table = dh_table

    # -- helpers ----------------------------------------------------------

    def _derive(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> LiveTable:
        """Derive a new LiveTable from this table."""
        return LiveTable(fn(*args, **kwargs))

    # -- derivation (all auto-locked) -------------------------------------

    def last_by(self, by: str | Sequence[str]) -> LiveTable:
        """Latest row per group."""
        return self._derive(self._table.last_by, by)

    def first_by(self, by: str | Sequence[str]) -> LiveTable:
        """First row per group."""
        return self._derive(self._table.first_by, by)

    def agg_by(self, aggs: list, by: str | Sequence[str] | None = None) -> LiveTable:
        """Aggregate by group."""
        if by is not None:
            return self._derive(self._table.agg_by, aggs, by)
        return self._derive(self._table.agg_by, aggs)

    def sum_by(self, by: str | Sequence[str] | None = None) -> LiveTable:
        """Sum by group."""
        if by is not None:
            return self._derive(self._table.sum_by, by)
        return self._derive(self._table.sum_by)

    def avg_by(self, by: str | Sequence[str] | None = None) -> LiveTable:
        """Average by group."""
        if by is not None:
            return self._derive(self._table.avg_by, by)
        return self._derive(self._table.avg_by)

    def group_by(self, by: str | Sequence[str] | None = None) -> LiveTable:
        """Group by."""
        if by is not None:
            return self._derive(self._table.group_by, by)
        return self._derive(self._table.group_by)

    def count_by(self, col: str, by: str | Sequence[str] | None = None) -> LiveTable:
        """Count by group."""
        if by is not None:
            return self._derive(self._table.count_by, col, by)
        return self._derive(self._table.count_by, col)

    def sort(self, order_by: str | Sequence[str]) -> LiveTable:
        """Sort ascending."""
        return self._derive(self._table.sort, order_by)

    def sort_descending(self, order_by: str | Sequence[str]) -> LiveTable:
        """Sort descending."""
        return self._derive(self._table.sort_descending, order_by)

    def where(self, filters: str | list[str]) -> LiveTable:
        """Filter rows."""
        return self._derive(self._table.where, filters)

    # -- output -----------------------------------------------------------

    def publish(self, name: str) -> None:
        """Publish this table to the Deephaven query scope.

        After publishing, remote ``pydeephaven`` clients can open the
        table by *name*.
        """
        if _is_remote():
            _get_remote_session().bind_table(name, self._table)
            return

        from deephaven.execution_context import get_exec_ctx
        scope = get_exec_ctx().j_exec_ctx.getQueryScope()
        scope.putParam(name, self._table.j_table)

    def snapshot(self) -> object:
        """Return a pandas DataFrame snapshot of the current table state."""
        if _is_remote():
            return self._table.to_arrow().to_pandas()

        from deephaven import pandas as dhpd
        from deephaven.execution_context import get_exec_ctx
        from deephaven.update_graph import shared_lock

        ug = get_exec_ctx().update_graph
        with shared_lock(ug):
            return dhpd.to_pandas(self._table)

    @property
    def size(self) -> int:
        """Current row count."""
        if _is_remote():
            # pydeephaven.Table doesn't have .size, but can check arrow length
            # Note: this is expensive on remote. Use with caution.
            return self._table.to_arrow().num_rows
        return self._table.size

    def __repr__(self) -> str:
        return "<LiveTable>"


# ---------------------------------------------------------------------------
# TickingTable — writable, inherits LiveTable
# ---------------------------------------------------------------------------

class TickingTable(LiveTable):
    """Writable ticking table backed by a DynamicTableWriter.

    Create with a Python-typed schema::

        prices = TickingTable({"Symbol": str, "Price": float})
        prices.write_row("AAPL", 228.0)
        prices.flush()
        live = prices.last_by("Symbol")   # auto-locked LiveTable

    Inherits all ``LiveTable`` ops: ``last_by``, ``agg_by``,
    ``sort_descending``, ``snapshot``, ``publish``, etc.
    """

    __slots__ = ("_writer", "_session", "_pa_schema")

    def __init__(self, schema: dict[str, type]) -> None:
        resolved_schema = _resolve_schema(schema)
        
        if _is_remote():
            import pyarrow as pa
            self._session = _get_remote_session()
            if self._session is None:
                raise RuntimeError(
                    "Deephaven session not available. On ARM64, ensure the Docker container "
                    "is running (StreamingServer.start() should handle this)."
                )
            # Convert dict schema to pyarrow.Schema
            pa_schema = pa.schema([(name, t) for name, t in resolved_schema.items()])
            self._writer = self._session.input_table(schema=pa_schema)
            self._pa_schema = pa_schema
            super().__init__(self._writer)
        else:
            from deephaven import DynamicTableWriter
            self._writer = DynamicTableWriter(resolved_schema)
            super().__init__(self._writer.table)

    def write_row(self, *values: Any) -> None:
        if _is_remote():
            import pyarrow as pa
            # pydeephaven.InputTable.add takes a Table.
            # Convert row to arrow table and import.
            arrays = [pa.array([v], type=self._pa_schema.field(i).type) for i, v in enumerate(values)]
            batch = pa.RecordBatch.from_arrays(arrays, schema=self._pa_schema)
            table = pa.Table.from_batches([batch])
            self._writer.add(self._session.import_table(table))
        else:
            self._writer.write_row(*values)

    def flush(self) -> None:
        """Flush the update graph so pending writes are visible."""
        flush()

    def close(self) -> None:
        """Close the underlying writer."""
        self._writer.close()

    def __repr__(self) -> str:
        return "<TickingTable>"
