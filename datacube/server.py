"""
Datacube UI server — Tornado + Perspective flat grid.

Runs **in-process** (like ``plt.show()``).  The datacube engine does all
SQL compilation and DuckDB pushdown; Perspective renders the flat result.

Endpoints:

- ``/websocket`` — Perspective Arrow data channel
- ``/cmd``       — datacube command channel (JSON over WebSocket)

On schema change (group_by / pivot adds or removes columns), a new
Perspective table is created with an incremented name (``dc_0``, ``dc_1``, …)
and the client is told to reconnect.

Usage::

    from datacube.server import run
    run(datacube_instance, port=8050)
"""

from __future__ import annotations

import json
import logging
import webbrowser
from pathlib import Path

import pyarrow as pa
import tornado.ioloop
import tornado.web
import tornado.websocket

import perspective
import perspective.handlers.tornado

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ── Arrow helpers ─────────────────────────────────────────────────


def _normalize_arrow(table: pa.Table) -> pa.Table:
    """Cast all numerics to float64 for Perspective compatibility."""
    cols, fields = [], []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_decimal(field.type) or pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
            col = col.cast(pa.float64())
            fields.append(pa.field(field.name, pa.float64()))
        else:
            fields.append(field)
        cols.append(col)
    return pa.table(cols, schema=pa.schema(fields))


def _arrow_to_ipc(table: pa.Table) -> bytes:
    """Normalize and serialize a PyArrow Table to IPC stream bytes."""
    table = _normalize_arrow(table)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()


# ── State helpers ─────────────────────────────────────────────────


def _snapshot_state(dc) -> dict:
    """JSON-serializable datacube state for the client."""
    snap = dc.snapshot
    return {
        "source": snap.source,
        "columns": [
            {
                "name": c.name,
                "type": c.type,
                "kind": c.kind,
                "aggregate_operator": c.aggregate_operator,
                "is_selected": c.is_selected,
            }
            for c in snap.columns
        ],
        "group_by": list(snap.group_by),
        "pivot_by": list(snap.pivot_by),
        "filters": [
            {"field": f.field, "op": f.op, "value": f.value}
            for f in snap.filters
        ],
        "sort": [
            {"field": s.field, "descending": s.descending}
            for s in snap.sort
        ],
        "drill_path": [dict(d) for d in snap.drill_path],
    }


# ── Tree builder ──────────────────────────────────────────────────


MAX_CHILDREN = 200  # limit expanded children to keep UI responsive


def _pad_table(table: pa.Table, target_cols: list[str], target_schema: pa.Schema) -> pa.Table:
    """Pad a table with null columns to match target schema, in target column order."""
    arrays = {}
    n = table.num_rows
    for col_name in target_cols:
        if col_name in table.column_names:
            arrays[col_name] = table.column(col_name)
        else:
            # Add null column with the right type from target schema
            field = target_schema.field(col_name)
            arrays[col_name] = pa.nulls(n, type=field.type)
    return pa.table(arrays)


def _build_tree_result(dc, expanded_keys: set) -> pa.Table:
    """Build a combined Arrow table: grouped parents + expanded children interleaved.

    Uses Arrow-native operations to preserve types (datetimes, etc.).
    Adds ``__tree__`` and ``__depth__`` columns.

    Args:
        dc: Datacube with group_by already set.
        expanded_keys: set of tuples like {("Tech",), ("Finance",)}.

    Returns:
        Arrow table with __tree__, __depth__, then original source columns.
    """
    snap = dc.snapshot
    group_fields = list(snap.group_by)

    parents = _normalize_arrow(dc.query())
    base_dc = dc.set_group_by()  # clear group_by

    # Get source column names in original order (from the ungrouped schema)
    source_schema = _normalize_arrow(base_dc.query().slice(0, 0)).schema
    source_cols = [f.name for f in source_schema]

    # Target column order
    all_cols = ["__tree__", "__depth__"] + source_cols

    # Build target schema (superset of parent + child columns)
    fields = [pa.field("__tree__", pa.string()), pa.field("__depth__", pa.float64())]
    for col_name in source_cols:
        fields.append(source_schema.field(col_name))
    target_schema = pa.schema(fields)

    # Build list of table chunks to concat
    chunks = []

    for i in range(parents.num_rows):
        group_key = tuple(parents.column(f)[i].as_py() for f in group_fields)
        is_expanded = group_key in expanded_keys
        label = " / ".join(str(v) for v in group_key)
        prefix = "▾ " if is_expanded else "▸ "

        # Single parent row — slice(i, i+1) preserves Arrow types
        parent_slice = parents.slice(i, 1)
        tree_col = pa.array([prefix + label], type=pa.string())
        depth_col = pa.array([0.0], type=pa.float64())
        parent_row = parent_slice.append_column("__tree__", tree_col).append_column("__depth__", depth_col)
        parent_row = _pad_table(parent_row, all_cols, target_schema)
        chunks.append(parent_row)

        if is_expanded:
            child_dc = base_dc
            for f, v in zip(group_fields, group_key):
                child_dc = child_dc.add_filter(f, "eq", v)
            children = _normalize_arrow(child_dc.query().slice(0, MAX_CHILDREN))
            if children.num_rows > 0:
                # Build tree labels from first string dimension not in group_fields
                labels = []
                label_col = None
                for col_name in children.column_names:
                    if col_name not in group_fields and children.schema.field(col_name).type == pa.string():
                        label_col = col_name
                        break
                for j in range(children.num_rows):
                    if label_col:
                        val = children.column(label_col)[j].as_py()
                        labels.append(f"    {val}" if val else f"    row {j}")
                    else:
                        labels.append(f"    row {j}")

                tree_arr = pa.array(labels, type=pa.string())
                depth_arr = pa.array([1.0] * children.num_rows, type=pa.float64())
                child_block = children.append_column("__tree__", tree_arr).append_column("__depth__", depth_arr)
                child_block = _pad_table(child_block, all_cols, target_schema)
                chunks.append(child_block)

    if not chunks:
        return pa.table({f.name: pa.array([], type=f.type) for f in target_schema})

    return pa.concat_tables(chunks)


# ── WebSocket command handler ─────────────────────────────────────


class CmdHandler(tornado.websocket.WebSocketHandler):
    """Datacube command channel — receives JSON, mutates engine, refreshes grid."""

    def check_origin(self, origin):
        return True

    def open(self):
        logger.info("Command channel opened")
        self._refresh()

    def on_message(self, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            self.write_message(json.dumps({"error": "Invalid JSON"}))
            return

        cmd = msg.get("cmd", "")
        state = self.application.dc_state
        dc = state["dc"]

        try:
            if cmd == "group_by":
                dc = dc.set_group_by(*msg.get("fields", []))
                state["expanded"] = set()  # clear expansions on group change
            elif cmd == "pivot_by":
                dc = dc.set_pivot_by(*msg.get("fields", []))
            elif cmd == "column":
                name = msg["name"]
                kwargs = {k: v for k, v in msg.items() if k not in ("cmd", "name")}
                dc = dc.set_column(name, **kwargs)
            elif cmd == "filter":
                dc = dc.add_filter(msg["field"], msg["op"], msg.get("value"))
            elif cmd == "clear_filters":
                dc = dc.clear_filters()
            elif cmd == "sort":
                sorts = msg.get("sorts", [])
                dc = dc.set_sort(
                    *[(s["field"], s.get("descending", False)) for s in sorts],
                )
            elif cmd == "expand":
                # Toggle expansion of a group row
                group_fields = list(dc.snapshot.group_by)
                key = tuple(msg.get(f) for f in group_fields)
                expanded = state["expanded"]
                if key in expanded:
                    expanded.discard(key)
                else:
                    expanded.add(key)
                # Don't change dc, just re-render
                self._refresh()
                return
            elif cmd == "collapse_all":
                state["expanded"] = set()
            elif cmd == "reset":
                dc = state["initial"]
                state["expanded"] = set()
            else:
                self.write_message(json.dumps({"error": f"Unknown: {cmd}"}))
                return

            state["dc"] = dc
            self._refresh()

        except Exception as e:
            logger.exception("Command failed: %s", cmd)
            self.write_message(json.dumps({"error": str(e)}))

    def _refresh(self):
        """Re-query datacube, update Perspective table, send state to client."""
        state = self.application.dc_state
        dc = state["dc"]

        try:
            sql = dc.sql()
            expanded = state["expanded"]

            # If group_by is active, build tree with expanded rows
            if dc.snapshot.group_by:
                arrow = _build_tree_result(dc, expanded)
            else:
                arrow = _normalize_arrow(dc.query())

            row_count = arrow.num_rows
            ipc = _arrow_to_ipc(arrow)

            new_cols = sorted(arrow.column_names)
            old_cols = state["last_columns"]

            if new_cols != old_cols:
                # Schema changed → new versioned table
                state["table_version"] += 1
                name = f"dc_{state['table_version']}"
                state["psp_table"] = state["psp_client"].table(ipc, name=name)
                state["last_columns"] = new_cols
                logger.info("Schema changed → %s (%d cols)", name, len(new_cols))
                self.write_message(json.dumps({
                    "type": "reload",
                    "table_name": name,
                    "state": _snapshot_state(dc),
                    "sql": sql,
                    "row_count": row_count,
                    "expanded": [list(k) for k in expanded],
                }))
            else:
                # Same schema → fast in-place replace
                state["psp_table"].replace(ipc)
                self.write_message(json.dumps({
                    "type": "update",
                    "state": _snapshot_state(dc),
                    "sql": sql,
                    "row_count": row_count,
                    "expanded": [list(k) for k in expanded],
                }))

        except Exception as e:
            logger.exception("Query failed")
            self.write_message(json.dumps({
                "error": str(e),
                "state": _snapshot_state(dc),
            }))


# ── Entry point ───────────────────────────────────────────────────


def run(dc, port: int = 8050, open_browser: bool = True):
    """Start the datacube UI server (blocking).

    Args:
        dc: A Datacube instance.
        port: HTTP port (default 8050).
        open_browser: Whether to open a browser tab.
    """
    arrow = dc.query()
    ipc = _arrow_to_ipc(arrow)

    psp_server = perspective.Server()
    psp_client = psp_server.new_local_client()
    psp_table = psp_client.table(ipc, name="dc_0")

    app = tornado.web.Application(
        [
            (
                r"/websocket",
                perspective.handlers.tornado.PerspectiveTornadoHandler,
                {"perspective_server": psp_server},
            ),
            (r"/cmd", CmdHandler),
            (
                r"/(.*)",
                tornado.web.StaticFileHandler,
                {"path": str(STATIC_DIR), "default_filename": "index.html"},
            ),
        ],
        websocket_max_message_size=50 * 1024 * 1024,
    )

    app.dc_state = {
        "dc": dc,
        "initial": dc,
        "psp_table": psp_table,
        "psp_client": psp_client,
        "psp_server": psp_server,
        "last_columns": sorted(_normalize_arrow(arrow).column_names),
        "table_version": 0,
        "expanded": set(),
    }

    app.listen(port)
    url = f"http://localhost:{port}"
    print(f"Datacube UI running at {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open(url)

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nStopped.")
