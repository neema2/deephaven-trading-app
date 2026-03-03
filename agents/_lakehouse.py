"""
Lakehouse Agent — Facts/Dimensions + Datacube
===============================================
Curate OLTP data into star-schema Iceberg tables, build datacube reports.

Tools:
    - list_lakehouse_tables     — Iceberg tables in the catalog
    - describe_lakehouse_table  — column schema for a table
    - design_star_schema        — LLM proposes fact + dim tables from source
    - create_lakehouse_table    — ingest or transform into Iceberg
    - sync_from_store           — OLTP → Iceberg via SyncEngine
    - build_datacube            — create Datacube config with group_by/pivot_by
    - query_lakehouse           — ad-hoc DuckDB SQL

Usage::

    from agents._lakehouse import create_lakehouse_agent

    agent = create_lakehouse_agent(ctx)
    result = agent.run("Build a star schema from the trades dataset")
"""

from __future__ import annotations

import json
import logging

from ai import Agent, tool
from agents._context import _PlatformContext

logger = logging.getLogger(__name__)

LAKEHOUSE_SYSTEM_PROMPT = """\
You are the Lakehouse Curation Agent — a platform specialist that transforms \
operational data into well-structured analytical datasets in Apache Iceberg.

You can:
1. List and describe existing Iceberg tables in the lakehouse.
2. Design star schemas (fact + dimension tables) from OLTP sources.
3. Create new Iceberg tables via SQL transforms or direct ingestion.
4. Build Datacube report configurations with group_by, pivot_by, and measures.
5. Run ad-hoc SQL queries over the lakehouse (DuckDB + Iceberg).

When designing star schemas:
- Separate facts (events/transactions with measures) from dimensions (descriptive attributes).
- Use clear naming: fact_<domain> for fact tables, dim_<entity> for dimensions.
- Ensure proper grain — each row in a fact table should represent one event/measurement.
- Include foreign keys linking facts to dimensions.
- Consider using snapshot or incremental mode for slowly-changing dimensions.

When building datacubes:
- Choose dimensions that enable meaningful drill-down.
- Set appropriate aggregation operators (sum, avg, count, min, max) for each measure.
- Suggest useful pivot_by fields for cross-tabulation.

Tables are accessible as lakehouse.default.<table_name> in SQL.
"""


def create_lakehouse_tools(ctx: _PlatformContext) -> list:
    """Create Lakehouse agent tools bound to a _PlatformContext."""

    def _get_lh():
        """Get the Lakehouse client, raising if not configured."""
        if ctx.lakehouse is None:
            raise RuntimeError("No Lakehouse configured in _PlatformContext")
        return ctx.lakehouse

    @tool
    def list_lakehouse_tables() -> str:
        """List all tables in the Iceberg lakehouse catalog.

        Returns JSON with table names.
        """
        try:
            lh = _get_lh()
            tables = lh.tables()
            return json.dumps({"tables": tables, "count": len(tables)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def describe_lakehouse_table(table_name: str) -> str:
        """Describe the schema of a lakehouse table.

        Args:
            table_name: Table name (short name like 'trades', not fully qualified).
        """
        try:
            lh = _get_lh()
            info = lh.table_info(table_name)
            row_count = lh.row_count(table_name)
            return json.dumps({
                "table_name": table_name,
                "columns": info,
                "row_count": row_count,
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def design_star_schema(source_description: str, source_fields_json: str = "") -> str:
        """Use AI to design a star schema from an OLTP data source.

        The AI analyzes the source data description and field list, then proposes
        fact and dimension tables with proper grain, keys, and relationships.

        Args:
            source_description: Natural language description of the source data
                                (e.g. "Trade records with symbol, quantity, price, side, timestamp, trader, desk").
            source_fields_json: Optional JSON array of source field names and types for more precise design.
        """
        if ctx.ai is None:
            return json.dumps({"error": "No AI configured in _PlatformContext"})

        fields_context = ""
        if source_fields_json:
            try:
                fields = json.loads(source_fields_json)
                fields_context = f"\n\nSource fields:\n{json.dumps(fields, indent=2)}"
            except json.JSONDecodeError:
                fields_context = f"\n\nSource fields (raw): {source_fields_json}"

        prompt = f"""\
Design a star schema for the following data source.

Source: {source_description}{fields_context}

Return a JSON object with:
{{
  "fact_tables": [
    {{
      "name": "fact_<name>",
      "description": "...",
      "grain": "One row per ...",
      "columns": [{{"name": "...", "type": "str|int|float|bool", "role": "key|measure|fk"}}],
      "source_sql": "SELECT ... FROM ..."
    }}
  ],
  "dimension_tables": [
    {{
      "name": "dim_<name>",
      "description": "...",
      "columns": [{{"name": "...", "type": "str|int|float|bool", "role": "key|attribute"}}],
      "source_sql": "SELECT DISTINCT ... FROM ..."
    }}
  ],
  "relationships": [
    {{"fact": "fact_trades", "dimension": "dim_instrument", "join_key": "symbol"}}
  ]
}}

Only return valid JSON."""

        try:
            from ai._types import Message
            response = ctx.ai.generate(
                [Message(role="user", content=prompt)],
                temperature=0.3,
            )
            # Try to parse the JSON from the response
            text = response.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            schema = json.loads(text)
            return json.dumps(schema, indent=2)
        except json.JSONDecodeError:
            # Return the raw text if JSON parsing fails
            return json.dumps({"raw_design": response.content})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def create_lakehouse_table(table_name: str, sql: str, mode: str = "append") -> str:
        """Create or populate a lakehouse table from a SQL transformation.

        Executes the SQL query and writes results into an Iceberg table.
        The table is created automatically if it doesn't exist.

        Args:
            table_name: Target table name (e.g. "fact_trades", "dim_instrument").
            sql: SQL query to produce the data (runs against lakehouse DuckDB).
                 Can reference other lakehouse tables as lakehouse.default.<name>.
            mode: Write mode — "append", "snapshot", "incremental", or "bitemporal".
        """
        try:
            lh = _get_lh()
            row_count = lh.transform(table_name, sql, mode=mode)
            return json.dumps({
                "status": "created",
                "table_name": table_name,
                "mode": mode,
                "rows_written": row_count,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def ingest_to_lakehouse(table_name: str, data_source: str, mode: str = "append") -> str:
        """Ingest data from a file (CSV/Parquet URL) directly into a lakehouse table.

        Data stays in DuckDB — never enters Python memory.

        Args:
            table_name: Target Iceberg table name.
            data_source: URL or file path to CSV/Parquet data, or a SQL SELECT statement.
            mode: Write mode — "append", "snapshot", "incremental", or "bitemporal".
        """
        try:
            lh = _get_lh()
            row_count = lh.ingest(table_name, data_source, mode=mode)
            return json.dumps({
                "status": "ingested",
                "table_name": table_name,
                "source": data_source[:100],
                "mode": mode,
                "rows_written": row_count,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def build_datacube(source_table: str, group_by: str = "", pivot_by: str = "",
                       measures: str = "") -> str:
        """Create a Datacube report configuration for a lakehouse table.

        Returns the datacube SQL and configuration that can be used with dc.show().

        Args:
            source_table: Lakehouse table name (e.g. "fact_trades").
            group_by: Comma-separated dimension fields for GROUP BY (e.g. "sector,symbol").
            pivot_by: Comma-separated fields for horizontal pivot (e.g. "side").
            measures: Comma-separated measure fields (e.g. "quantity,price,pnl").
        """
        try:
            from datacube import Datacube

            lh = _get_lh()
            dc = Datacube(lh, source_name=source_table)

            # Apply group_by
            if group_by:
                fields = [f.strip() for f in group_by.split(",")]
                dc = dc.set_group_by(*fields)

            # Apply pivot_by
            if pivot_by:
                fields = [f.strip() for f in pivot_by.split(",")]
                dc = dc.set_pivot_by(*fields)

            # Get the compiled SQL
            sql = dc.sql()
            available_dims = dc.available_dimensions()
            available_measures = dc.available_measures()

            return json.dumps({
                "source_table": source_table,
                "group_by": group_by.split(",") if group_by else [],
                "pivot_by": pivot_by.split(",") if pivot_by else [],
                "available_dimensions": available_dims,
                "available_measures": available_measures,
                "compiled_sql": sql,
                "message": "Datacube configured. Use dc.show() to launch interactive UI.",
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def query_lakehouse(sql: str) -> str:
        """Execute an ad-hoc SQL query against the Iceberg lakehouse.

        Tables are accessible as lakehouse.default.<table_name>.

        Args:
            sql: SQL query to execute (read-only recommended).
        """
        try:
            lh = _get_lh()
            rows = lh.query(sql)
            return json.dumps({
                "row_count": len(rows),
                "rows": rows[:50],  # Cap at 50 rows for readability
                "truncated": len(rows) > 50,
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return [list_lakehouse_tables, describe_lakehouse_table, design_star_schema,
            create_lakehouse_table, ingest_to_lakehouse, build_datacube,
            query_lakehouse]


# ── Agent factory ──────────────────────────────────────────────────────


def create_lakehouse_agent(ctx: _PlatformContext, **kwargs) -> Agent:
    """Create a Lakehouse Agent bound to a _PlatformContext.

    Args:
        ctx: Platform context with Lakehouse client.
        **kwargs: Extra args forwarded to Agent.

    Returns:
        A configured Agent with Lakehouse tools.
    """
    tools = create_lakehouse_tools(ctx)
    return Agent(
        tools=tools,
        system_prompt=LAKEHOUSE_SYSTEM_PROMPT,
        name="lakehouse",
        **kwargs,
    )
