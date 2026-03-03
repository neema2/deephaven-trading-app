"""
Dashboard Agent — Streaming Joins + Risk Dashboards
=====================================================
Build Deephaven ticking tables, streaming joins, derived views, risk dashboards.

Tools:
    - list_ticking_tables      — current DH tables
    - create_ticking_table     — TickingTable with typed columns
    - create_derived_table     — aggregations, filters, joins
    - setup_store_bridge       — StoreBridge for OLTP event streaming
    - create_reactive_model    — @ticking + @computed Storable
    - publish_table            — publish to DH query scope

Usage::

    from agents._dashboard import create_dashboard_agent

    agent = create_dashboard_agent(ctx)
    result = agent.run("Create a real-time PnL dashboard from trade events")
"""

from __future__ import annotations

import json
import logging

from ai import Agent, tool
from agents._context import _PlatformContext

logger = logging.getLogger(__name__)

DASHBOARD_SYSTEM_PROMPT = """\
You are the Dashboard Agent — a platform specialist that builds real-time \
streaming dashboards using ticking tables.

You can:
1. List existing ticking tables in the streaming server.
2. Create new ticking tables with typed column schemas.
3. Create derived tables using aggregations, filters, joins, and computed columns.
4. Set up the StoreBridge to stream OLTP events into ticking tables.
5. Design reactive models with @computed properties that auto-update.
6. Publish tables for browser-based visualization.

Architecture:
- **TickingTable**: writable table backed by DynamicTableWriter. Python-typed schema.
- **LiveTable**: read-only derived table. Operations: last_by, agg_by, where, select, update, sort.
- **StoreBridge**: streams PG LISTEN/NOTIFY events into ticking tables automatically.
- **@ticking decorator**: auto-creates TickingTable + LiveTable from a Storable dataclass.
- **Aggregations**: sum, avg, count, min, max, first, last, std, var, median.

When building dashboards:
- Use last_by() for latest-value tables (e.g. current positions, live quotes).
- Use agg_by() for real-time aggregations (e.g. sector totals, portfolio risk).
- Chain operations: table.last_by("symbol").agg_by([agg.sum("pnl")], ["sector"]).
- Use StoreBridge to automatically stream OLTP changes into ticking tables.
- Publish tables with meaningful names for the DH web UI.
"""


def create_dashboard_tools(ctx: _PlatformContext) -> list:
    """Create Dashboard agent tools bound to a _PlatformContext."""

    @tool
    def list_ticking_tables() -> str:
        """List all ticking tables currently registered in the streaming server.

        Returns table names, column schemas, and row counts.
        """
        try:
            from streaming import get_tables
            tables = get_tables()
            result = []
            for name, (writer, raw, live) in tables.items():
                result.append({
                    "name": name,
                    "raw_table": f"{name}_raw",
                    "live_table": f"{name}_live",
                })
            return json.dumps({"tables": result, "count": len(result)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def create_ticking_table(name: str, schema_json: str) -> str:
        """Create a new ticking table with a typed column schema.

        Args:
            name: Table name (will create {name}_raw and {name}_live).
            schema_json: JSON object mapping column names to types.
                         Types: "str", "int", "float", "bool", "datetime".
                         Example: {"symbol": "str", "price": "float", "volume": "int"}
        """
        try:
            schema = json.loads(schema_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        type_map = {
            "str": str, "string": str,
            "int": int, "integer": int,
            "float": float, "double": float,
            "bool": bool, "boolean": bool,
        }

        try:
            from streaming import TickingTable
            typed_schema = {}
            for col_name, col_type in schema.items():
                py_type = type_map.get(col_type.lower())
                if py_type is None:
                    return json.dumps({"error": f"Unknown type '{col_type}' for column '{col_name}'. Supported: {list(type_map.keys())}"})
                typed_schema[col_name] = py_type

            table = TickingTable(typed_schema)
            # Create a live table (last_by first column as default key)
            first_col = list(typed_schema.keys())[0]

            return json.dumps({
                "status": "created",
                "name": name,
                "columns": {k: v.__name__ for k, v in typed_schema.items()},
                "key_column": first_col,
                "message": f"TickingTable created. Write rows with table.write_row(...) and derive with .last_by(), .agg_by(), etc.",
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def create_derived_table(source_table: str, operations_json: str) -> str:
        """Create a derived (LiveTable) from an existing ticking table.

        Supports chaining: last_by → agg_by → where → select.

        Args:
            source_table: Name of the source ticking table.
            operations_json: JSON array of operations to apply in order.
                Each operation: {"op": "last_by|agg_by|where|select|sort|update", ...params}
                Examples:
                  [{"op": "last_by", "by": "symbol"}]
                  [{"op": "last_by", "by": "symbol"}, {"op": "where", "filters": "price > 100"}]
                  [{"op": "agg_by", "aggs": [{"type": "sum", "col": "pnl"}], "by": ["sector"]}]
        """
        try:
            operations = json.loads(operations_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        # Describe what would be created (actual execution requires DH running)
        steps = []
        for op in operations:
            op_type = op.get("op", "")
            if op_type == "last_by":
                steps.append(f".last_by('{op.get('by', '')}')")
            elif op_type == "agg_by":
                aggs = op.get("aggs", [])
                by = op.get("by", [])
                agg_strs = [f"agg.{a['type']}('{a['col']}')" for a in aggs]
                steps.append(f".agg_by([{', '.join(agg_strs)}], {by})")
            elif op_type == "where":
                steps.append(f".where('{op.get('filters', '')}')")
            elif op_type == "select":
                steps.append(f".select({op.get('columns', [])})")
            elif op_type == "sort":
                steps.append(f".sort_descending('{op.get('by', '')}')")
            elif op_type == "update":
                steps.append(f".update({op.get('formulas', [])})")

        chain = f"table{' '.join(steps)}"

        return json.dumps({
            "source": source_table,
            "operations": operations,
            "chain_expression": chain,
            "steps": len(operations),
            "message": "Derived table configuration ready. Execute when StreamingServer is running.",
        })

    @tool
    def setup_store_bridge(type_name: str) -> str:
        """Set up a StoreBridge to stream OLTP events into ticking tables.

        The bridge listens to PG NOTIFY events for a Storable type and
        automatically writes changes into a TickingTable.

        Args:
            type_name: Name of the Storable type to bridge (e.g. "Trade", "Position").
        """
        cls = ctx.get_storable_type(type_name)
        if cls is None:
            return json.dumps({"error": f"Type '{type_name}' not found."})

        import dataclasses
        fields = []
        for f in dataclasses.fields(cls):
            if not f.name.startswith("_"):
                fields.append({
                    "name": f.name,
                    "type": f.type.__name__ if isinstance(f.type, type) else str(f.type),
                })

        return json.dumps({
            "type_name": type_name,
            "fields": fields,
            "bridge_config": {
                "code": (
                    f"bridge = StoreBridge('{ctx.store_alias}', "
                    f"user='{ctx.store_user}', password='{ctx.store_password}')\n"
                    f"bridge.register({type_name})\n"
                    f"bridge.start()\n"
                    f"live_table = bridge.table({type_name}).last_by('entity_id')"
                ),
            },
            "message": f"StoreBridge configured for {type_name}. Events will stream into a TickingTable automatically.",
        })

    @tool
    def create_reactive_model(name: str, fields_json: str, computeds_json: str = "[]") -> str:
        """Design a reactive model with @computed properties for a ticking dashboard.

        Generates the code for a @ticking Storable dataclass with reactive computations.

        Args:
            name: PascalCase model name (e.g. "PortfolioRisk").
            fields_json: JSON array of fields: [{"name": str, "type": str}].
            computeds_json: JSON array of computed properties: [{"name": str, "formula": str, "description": str}].
        """
        try:
            fields = json.loads(fields_json)
            computeds = json.loads(computeds_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        # Generate code
        lines = [
            "from dataclasses import dataclass",
            "from store import Storable",
            "from reactive import computed, effect",
            "from streaming import ticking",
            "",
            f"@ticking",
            f"@dataclass",
            f"class {name}(Storable):",
        ]

        # Fields
        type_map = {"str": "str", "int": "int", "float": "float", "bool": "bool"}
        for f in fields:
            py_type = type_map.get(f["type"], "str")
            default = '""' if py_type == "str" else "0" if py_type == "int" else "0.0" if py_type == "float" else "False"
            lines.append(f"    {f['name']}: {py_type} = {default}")

        # Computeds
        if computeds:
            lines.append("")
            for c in computeds:
                lines.append(f"    @computed")
                lines.append(f"    def {c['name']}(self):")
                if c.get("description"):
                    lines.append(f'        """{c["description"]}"""')
                lines.append(f"        return {c['formula']}")

        code = "\n".join(lines)
        return json.dumps({
            "name": name,
            "fields": fields,
            "computeds": computeds,
            "generated_code": code,
            "message": f"Reactive model '{name}' designed. Copy the code to your module and the @ticking decorator will auto-create ticking tables.",
        })

    @tool
    def publish_table(table_name: str, publish_name: str = "") -> str:
        """Publish a ticking table to the Deephaven query scope for browser access.

        Published tables are visible at http://localhost:{streaming_port} in the DH web UI.

        Args:
            table_name: Name of the ticking table to publish.
            publish_name: Name to publish under (defaults to table_name).
        """
        pub_name = publish_name or table_name
        return json.dumps({
            "table_name": table_name,
            "published_as": pub_name,
            "url": f"http://localhost:{ctx.streaming_port}" if ctx.streaming_port else "StreamingServer not configured",
            "code": f'table.publish("{pub_name}")',
            "message": f"Table will be visible as '{pub_name}' in the Deephaven web UI.",
        })

    return [list_ticking_tables, create_ticking_table, create_derived_table,
            setup_store_bridge, create_reactive_model, publish_table]


def create_dashboard_agent(ctx: _PlatformContext, **kwargs) -> Agent:
    """Create a Dashboard Agent bound to a _PlatformContext."""
    tools = create_dashboard_tools(ctx)
    return Agent(
        tools=tools,
        system_prompt=DASHBOARD_SYSTEM_PROMPT,
        name="dashboard",
        **kwargs,
    )
