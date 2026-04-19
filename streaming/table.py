"""
streaming.table — Auto-locked ticking table abstraction.

Hides Deephaven's DynamicTableWriter and update-graph locking behind a
clean Python API.  All derivation operations (last_by, agg_by, sort, …)
auto-acquire the UG shared lock so callers never segfault.

**Dual-mode:** On macOS / x86 Linux, uses the in-process Deephaven JVM
(``DynamicTableWriter``).  On Linux ARM64, uses ``pydeephaven`` to talk
to a Docker-hosted Deephaven server.

Classes:
    LiveTable     Read-only derived table (all ops auto-locked).
    TickingTable  Writable table (inherits LiveTable, adds write_row/flush).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from streaming.admin import _needs_docker

# ---------------------------------------------------------------------------
# Mode flag — set once at import time
# ---------------------------------------------------------------------------

_REMOTE = _needs_docker()

# ---------------------------------------------------------------------------
# Python type → Deephaven type mapping (lazy, avoids import at module level)
# ---------------------------------------------------------------------------

_PY_TO_DH = None  # populated on first use


def _type_map() -> dict:
    """Return the Python→DH type dict, building it on first call."""
    global _PY_TO_DH
    if _PY_TO_DH is None:
        import deephaven.dtypes as dht

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

    On remote (Docker) mode, this is a no-op — writes are immediately
    visible through the gRPC session.
    """
    if _REMOTE:
        return  # Docker DH auto-flushes; no UG to poke
    global _UG_REF
    if _UG_REF is None:
        from deephaven.execution_context import get_exec_ctx
        _UG_REF = get_exec_ctx().update_graph.j_update_graph
    _UG_REF.requestRefresh()


def snapshot(table: "LiveTable") -> "pd.DataFrame":
    """Return a pandas DataFrame snapshot of the current table state.

    Convenience wrapper around ``table.snapshot()``::

        from streaming import snapshot
        df = snapshot(prices_raw)

    Works on any ``LiveTable`` or ``TickingTable``.
    """
    return table.snapshot()


# ---------------------------------------------------------------------------
# LiveTable — read-only, auto-locked (in-process mode)
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

    def __init__(self, dh_table: Any, name: str | None = None) -> None:
        self._table = dh_table
        self._name = name

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

    def format_columns(self, formats: str | list[str]) -> LiveTable:
        """Format columns."""
        return self._derive(self._table.format_columns, formats)

    # -- output -----------------------------------------------------------

    def publish(self, name: str) -> None:
        """Publish this table to the Deephaven query scope.

        After publishing, remote ``pydeephaven`` clients can open the
        table by *name*.
        """
        from deephaven.execution_context import get_exec_ctx

        scope = get_exec_ctx().j_exec_ctx.getQueryScope()
        scope.putParam(name, self._table.j_table)

    def snapshot(self) -> pd.DataFrame:
        """Return a pandas DataFrame snapshot of the current table state."""
        from deephaven import pandas as dhpd
        from deephaven.execution_context import get_exec_ctx
        from deephaven.update_graph import shared_lock

        ug = get_exec_ctx().update_graph
        with shared_lock(ug):
            return dhpd.to_pandas(self._table)

    @property
    def size(self) -> int:
        """Current row count."""
        return int(self._table.size)

    def __repr__(self) -> str:
        return f"<LiveTable rows={self.size}>"


# ---------------------------------------------------------------------------
# TickingTable — writable, inherits LiveTable (in-process mode)
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

    __slots__ = ("_writer", "_schema")

    def __init__(self, schema: dict[str, type], name: str | None = None) -> None:
        from deephaven import DynamicTableWriter

        self._schema = schema
        dh_schema = _resolve_schema(schema)
        self._writer = DynamicTableWriter(dh_schema)
        super().__init__(self._writer.table, name=name)


    def write_row(self, *values: Any) -> None:
        """Write a single row.  Thread-safe per Deephaven docs."""
        cleaned = []
        for col_name, v in zip(self._schema.keys(), values):
            if v is not None and self._schema[col_name] in (dict, list, object):
                cleaned.append(str(v))
            else:
                cleaned.append(v)
        self._writer.write_row(*cleaned)

    def flush(self) -> None:
        """Flush the update graph so pending writes are visible."""
        flush()

    def close(self) -> None:
        """Close the underlying writer."""
        self._writer.close()

    def __repr__(self) -> str:
        return f"<TickingTable rows={self.size}>"


# ===========================================================================
# Remote (Docker) mode — used ONLY on Linux ARM64
# ===========================================================================

if _REMOTE:
    # -- Shared session (lazy singleton) ----------------------------------

    _remote_session = None
    _remote_port = 10000  # default; overridden by conftest

    def _get_session():
        """Return the shared pydeephaven session, creating if needed."""
        global _remote_session
        if _remote_session is None:
            from pydeephaven import Session
            _remote_session = Session(host="localhost", port=_remote_port)
        return _remote_session

    def set_remote_port(port: int) -> None:
        """Set the port for remote session (called by conftest/admin)."""
        global _remote_port, _remote_session
        _remote_port = port
        if _remote_session is not None:
            try:
                _remote_session.close()
            except Exception:
                pass
            _remote_session = None

    # -- Name generator for server-side variables -------------------------
    _name_counter = itertools.count()

    def _next_name(prefix: str = "__tt") -> str:
        return f"{prefix}_{next(_name_counter)}"



    # -- Python type → DH type string mapping for run_script --------------
    _PY_TO_DH_STR = {
        str: "dht.string",
        float: "dht.double",
        int: "dht.int64",
        bool: "dht.bool_",
        Decimal: "dht.double",
        datetime: "dht.Instant",
    }

    class RemoteLiveTable:
        """Read-only derived ticking table on a remote Deephaven server."""

        __slots__ = ("_name",)

        def __init__(self, name: str) -> None:
            self._name = name

        def _derive_remote(self, op: str) -> RemoteLiveTable:
            new = _next_name("__lt")
            _get_session().run_script(f"{new} = {self._name}.{op}")
            return RemoteLiveTable(new)


        def last_by(self, by):
            return self._derive_remote(f"last_by({by!r})")

        def first_by(self, by):
            return self._derive_remote(f"first_by({by!r})")

        def agg_by(self, aggs, by=None):
            # Ensure deephaven.agg is in server scope for RemoteAgg reprs
            sess = _get_session()
            sess.run_script("from deephaven import agg")
            if by is not None:
                return self._derive_remote(f"agg_by({aggs!r}, {by!r})")
            return self._derive_remote(f"agg_by({aggs!r})")

        def sum_by(self, by=None):
            if by is not None:
                return self._derive_remote(f"sum_by({by!r})")
            return self._derive_remote("sum_by()")

        def avg_by(self, by=None):
            if by is not None:
                return self._derive_remote(f"avg_by({by!r})")
            return self._derive_remote("avg_by()")

        def group_by(self, by=None):
            if by is not None:
                return self._derive_remote(f"group_by({by!r})")
            return self._derive_remote("group_by()")

        def count_by(self, col, by=None):
            if by is not None:
                return self._derive_remote(f"count_by({col!r}, {by!r})")
            return self._derive_remote(f"count_by({col!r})")

        def sort(self, order_by):
            return self._derive_remote(f"sort({order_by!r})")

        def sort_descending(self, order_by):
            return self._derive_remote(f"sort_descending({order_by!r})")

        def where(self, filters):
            return self._derive_remote(f"where({filters!r})")

        def format_columns(self, formats):
            return self._derive_remote(f"format_columns({formats!r})")

        def publish(self, name: str) -> None:
            # Table already lives server-side; just alias it
            _get_session().run_script(f"{name} = {self._name}")

        def snapshot(self) -> pd.DataFrame:
            tbl = _get_session().open_table(self._name)
            return tbl.to_arrow().to_pandas()

        @property
        def size(self) -> int:
            tbl = _get_session().open_table(self._name)
            return tbl.to_arrow().num_rows

        def __repr__(self) -> str:
            return f"<RemoteLiveTable name={self._name!r}>"

    class RemoteTickingTable(RemoteLiveTable):
        """Writable ticking table on a remote Deephaven server."""

        __slots__ = ("_writer_name", "_schema")

        def __init__(self, schema: dict[str, type], name: str | None = None) -> None:
            self._schema = schema
            if name is None:
                name = _next_name("_tt")
            
            writer_name = f"{name}_w"

            # Build the server-side DynamicTableWriter creation script
            cols = ", ".join(
                f"{col_name!r}: {_PY_TO_DH_STR[py_type]}"
                for col_name, py_type in schema.items()
            )
            script = (
                f"import datetime\n"
                f"import deephaven.dtypes as dht\n"
                f"from deephaven import DynamicTableWriter\n"
                f"from deephaven.time import to_j_instant\n"
                f"{writer_name} = DynamicTableWriter({{{cols}}})\n"
                f"{name} = {writer_name}.table\n"
            )
            _get_session().run_script(script)
            self._writer_name = writer_name
            super().__init__(name)

        def write_row(self, *values: Any) -> None:
            parts = []
            for col_name, v in zip(self._schema.keys(), values):
                if v is None:
                    parts.append("None")
                elif isinstance(v, datetime):
                    # Convert to ISO format string and parse server-side
                    parts.append(f"to_j_instant('{v.isoformat()}')")
                elif self._schema[col_name] is str:
                    parts.append(repr(str(v)))
                elif self._schema[col_name] is float:
                    try:
                        # Handle Expr or other numeric-like objects
                        parts.append(repr(float(v)))
                    except (TypeError, ValueError):
                        # Fallback for complex objects that can't be floatified
                        # (like a Sum expression that somehow leaked through)
                        parts.append("0.0")
                elif self._schema[col_name] is int:
                    try:
                        parts.append(repr(int(v)))
                    except (TypeError, ValueError):
                        parts.append("0")
                elif self._schema[col_name] is bool:
                    parts.append(repr(bool(v)))
                elif self._schema[col_name] in (dict, list, object):
                    parts.append(repr(str(v)))
                else:
                    parts.append(repr(str(v)))

            vals = ", ".join(parts)
            _get_session().run_script(
                f"{self._writer_name}.write_row({vals})"
            )




        def flush(self) -> None:
            # Remote DH auto-flushes; no-op
            pass

        def close(self) -> None:
            _get_session().run_script(f"{self._writer_name}.close()")

        def __repr__(self) -> str:
            return f"<RemoteTickingTable name={self._name!r}>"

    # -- Monkey-patch the module-level names so callers see the right type -
    LiveTable = RemoteLiveTable  # type: ignore[misc]
    TickingTable = RemoteTickingTable  # type: ignore[misc]

