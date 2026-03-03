"""
OLTP Agent — Dataset Creation
===============================
Design Storable schemas, create OLTP tables, seed data, bulk ingest.

Tools:
    - list_storable_types   — registered types in the store
    - describe_type         — fields, types, column metadata
    - create_dataset        — dynamically create a Storable dataclass
    - insert_records        — seed data via Storable.save()
    - query_dataset         — inspect data via Storable.query()
    - ingest_from_file      — bulk load CSV/Parquet into a Storable type

Usage::

    from agents._oltp import create_oltp_agent

    agent = create_oltp_agent(ctx)
    result = agent.run("Create a trades table with symbol, quantity, price, side")
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Optional

from ai import Agent, tool
from agents._context import _PlatformContext

logger = logging.getLogger(__name__)

OLTP_SYSTEM_PROMPT = """\
You are the OLTP Dataset Agent — a platform specialist that creates and manages \
operational datasets in the object store.

You can:
1. List existing dataset types and describe their schemas.
2. Create new datasets by defining Storable dataclass schemas with proper types.
3. Insert seed records into datasets.
4. Query datasets with filters.
5. Bulk-load data from CSV or Parquet files.

When creating datasets:
- Choose appropriate Python types (str, int, float, bool, datetime).
- Use clear, descriptive field names in snake_case.
- Consider which fields should be used for grouping/filtering.
- Add reasonable defaults where appropriate.

Always confirm what you've created by describing the schema back to the user.
"""

# ── Python type mapping for dynamic Storable creation ──────────────────

_TYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "double": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
}


def _build_storable_class(name: str, fields: list[dict]) -> type:
    """Dynamically create a Storable dataclass from a field spec.

    Args:
        name: Class name (PascalCase).
        fields: List of {"name": str, "type": str, "default": ...} dicts.

    Returns:
        A new dataclass that inherits from Storable.
    """
    from store.base import Storable

    # Build field definitions for dataclasses.make_dataclass
    dc_fields = []
    for f in fields:
        py_type = _TYPE_MAP.get(f["type"].lower(), str)
        default = f.get("default")
        if default is None:
            # Sensible defaults by type
            if py_type is str:
                default = ""
            elif py_type is int:
                default = 0
            elif py_type is float:
                default = 0.0
            elif py_type is bool:
                default = False
        dc_fields.append((f["name"], py_type, dataclasses.field(default=default)))

    # Create an intermediate base that disables column registry validation.
    # Storable.__init_subclass__ checks _registry — setting it to None
    # on our base class prevents validation for dynamic types.
    class _DynamicBase(Storable):
        _registry = None

    cls = dataclasses.make_dataclass(
        name,
        dc_fields,
        bases=(_DynamicBase,),
    )
    return cls


def _serialize_storable(obj) -> dict:
    """Convert a Storable instance to a JSON-safe dict."""
    result = {}
    for f in dataclasses.fields(obj):
        if f.name.startswith("_"):
            continue
        val = getattr(obj, f.name)
        result[f.name] = val
    # Include store metadata
    if obj._store_entity_id:
        result["_entity_id"] = obj._store_entity_id
        result["_version"] = obj._store_version
    return result


# ── Tool factories (closed over _PlatformContext) ───────────────────────


def create_oltp_tools(ctx: _PlatformContext) -> list:
    """Create OLTP agent tools bound to a _PlatformContext."""

    @tool
    def list_storable_types() -> str:
        """List all registered dataset types in the platform.

        Returns JSON with type names and field counts.
        """
        types = ctx.list_storable_types()
        result = []
        for name in types:
            cls = ctx.get_storable_type(name)
            if cls and dataclasses.is_dataclass(cls):
                fields = [
                    {"name": f.name, "type": f.type.__name__ if isinstance(f.type, type) else str(f.type)}
                    for f in dataclasses.fields(cls)
                    if not f.name.startswith("_")
                ]
                result.append({"name": name, "field_count": len(fields), "fields": fields})
            else:
                result.append({"name": name, "field_count": 0, "fields": []})
        return json.dumps(result)

    @tool
    def describe_type(type_name: str) -> str:
        """Describe the schema of a registered dataset type.

        Args:
            type_name: Name of the Storable type to describe.
        """
        cls = ctx.get_storable_type(type_name)
        if cls is None:
            return json.dumps({"error": f"Type '{type_name}' not found. Use list_storable_types() to see available types."})

        fields = []
        for f in dataclasses.fields(cls):
            if f.name.startswith("_"):
                continue
            fields.append({
                "name": f.name,
                "type": f.type.__name__ if isinstance(f.type, type) else str(f.type),
                "default": repr(f.default) if f.default is not dataclasses.MISSING else None,
            })

        return json.dumps({
            "type_name": type_name,
            "class_name": cls.__name__,
            "field_count": len(fields),
            "fields": fields,
        })

    @tool
    def create_dataset(name: str, fields_json: str, description: str = "") -> str:
        """Create a new OLTP dataset type with the given schema.

        This creates a real Storable dataclass that can be persisted to PostgreSQL.

        Args:
            name: PascalCase class name for the dataset (e.g. "Trade", "Instrument").
            fields_json: JSON array of field definitions. Each field: {"name": str, "type": str, "default": optional}.
                         Supported types: str, int, float, bool.
                         Example: [{"name": "symbol", "type": "str"}, {"name": "price", "type": "float"}]
            description: Optional description of the dataset purpose.
        """
        try:
            fields = json.loads(fields_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON for fields: {e}"})

        if not fields:
            return json.dumps({"error": "fields_json must contain at least one field"})

        # Validate field specs
        for f in fields:
            if "name" not in f or "type" not in f:
                return json.dumps({"error": f"Each field must have 'name' and 'type'. Got: {f}"})
            if f["type"].lower() not in _TYPE_MAP:
                return json.dumps({
                    "error": f"Unsupported type '{f['type']}'. Supported: {list(_TYPE_MAP.keys())}"
                })

        # Check for duplicates
        if ctx.get_storable_type(name):
            return json.dumps({"error": f"Type '{name}' already exists."})

        # Build the class
        cls = _build_storable_class(name, fields)
        ctx.register_storable_type(name, cls)

        # Return confirmation
        schema = []
        for f in dataclasses.fields(cls):
            if not f.name.startswith("_"):
                schema.append({
                    "name": f.name,
                    "type": f.type.__name__ if isinstance(f.type, type) else str(f.type),
                })

        result = {
            "status": "created",
            "type_name": name,
            "class_name": cls.__name__,
            "fields": schema,
            "description": description,
            "message": f"Dataset '{name}' created with {len(schema)} fields. Ready for insert_records().",
        }
        logger.info("Created Storable type: %s with %d fields", name, len(schema))
        return json.dumps(result)

    @tool
    def insert_records(type_name: str, records_json: str) -> str:
        """Insert records into an existing dataset.

        Args:
            type_name: Name of the dataset type (as created by create_dataset).
            records_json: JSON array of record objects. Each object's keys must match the dataset fields.
                          Example: [{"symbol": "AAPL", "price": 228.0}, {"symbol": "GOOGL", "price": 192.0}]
        """
        cls = ctx.get_storable_type(type_name)
        if cls is None:
            return json.dumps({"error": f"Type '{type_name}' not found."})

        try:
            records = json.loads(records_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        if not isinstance(records, list):
            return json.dumps({"error": "records_json must be a JSON array."})

        # Need an active store connection
        try:
            from store.connection import get_connection
            conn = get_connection()
        except RuntimeError:
            # Try to connect using context
            if ctx.store_alias:
                conn = ctx.get_store_connection()
            else:
                return json.dumps({"error": "No store connection. Set store_alias in _PlatformContext."})

        inserted = []
        errors = []
        for i, rec in enumerate(records):
            try:
                obj = cls(**rec)
                entity_id = obj.save()
                inserted.append({"index": i, "entity_id": entity_id})
            except Exception as e:
                errors.append({"index": i, "error": str(e)})

        return json.dumps({
            "type_name": type_name,
            "inserted": len(inserted),
            "errors": len(errors),
            "entity_ids": [r["entity_id"] for r in inserted],
            "error_details": errors[:5] if errors else [],
        })

    @tool
    def query_dataset(type_name: str, limit: int = 20) -> str:
        """Query records from a dataset.

        Args:
            type_name: Name of the dataset type.
            limit: Maximum number of records to return (default 20).
        """
        cls = ctx.get_storable_type(type_name)
        if cls is None:
            return json.dumps({"error": f"Type '{type_name}' not found."})

        # Need an active store connection
        try:
            from store.connection import get_connection
            get_connection()
        except RuntimeError:
            if ctx.store_alias:
                ctx.get_store_connection()
            else:
                return json.dumps({"error": "No store connection."})

        try:
            result = cls.query(limit=limit)
            rows = [_serialize_storable(obj) for obj in result]
            return json.dumps({
                "type_name": type_name,
                "count": len(rows),
                "rows": rows,
            }, default=str)
        except Exception as e:
            return json.dumps({"error": f"Query failed: {e}"})

    @tool
    def ingest_from_file(type_name: str, file_path: str) -> str:
        """Bulk-load records from a CSV or Parquet file into a dataset.

        Reads the file with DuckDB, maps columns to dataset fields, and inserts
        all rows via Storable.save(). Supports local paths and URLs.

        Args:
            type_name: Name of the dataset type.
            file_path: Path or URL to CSV/Parquet file.
        """
        cls = ctx.get_storable_type(type_name)
        if cls is None:
            return json.dumps({"error": f"Type '{type_name}' not found."})

        # Need an active store connection
        try:
            from store.connection import get_connection
            get_connection()
        except RuntimeError:
            if ctx.store_alias:
                ctx.get_store_connection()
            else:
                return json.dumps({"error": "No store connection."})

        try:
            import duckdb
            conn = duckdb.connect()

            # Detect file type
            path_lower = file_path.lower()
            if path_lower.endswith(".parquet"):
                sql = f"SELECT * FROM read_parquet('{file_path}')"
            elif path_lower.endswith(".csv"):
                sql = f"SELECT * FROM read_csv_auto('{file_path}')"
            else:
                # Try as generic — DuckDB is smart about detection
                sql = f"SELECT * FROM '{file_path}'"

            rows = conn.execute(sql).fetchdf().to_dict(orient="records")
            conn.close()
        except Exception as e:
            return json.dumps({"error": f"Failed to read file: {e}"})

        # Map file columns to dataset fields
        field_names = {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}
        inserted = 0
        errors = []
        for i, row in enumerate(rows):
            try:
                # Only include fields that exist in the dataset
                filtered = {k: v for k, v in row.items() if k in field_names}
                obj = cls(**filtered)
                obj.save()
                inserted += 1
            except Exception as e:
                if len(errors) < 5:
                    errors.append({"index": i, "error": str(e)})

        return json.dumps({
            "type_name": type_name,
            "file": file_path,
            "total_rows_in_file": len(rows),
            "inserted": inserted,
            "errors": len(rows) - inserted,
            "error_samples": errors,
        })

    return [list_storable_types, describe_type, create_dataset,
            insert_records, query_dataset, ingest_from_file]


# ── Agent factory ──────────────────────────────────────────────────────


def create_oltp_agent(ctx: _PlatformContext, **kwargs) -> Agent:
    """Create an OLTP Agent bound to a _PlatformContext.

    Args:
        ctx: Platform context with store connection info.
        **kwargs: Extra args forwarded to Agent (e.g. temperature, model).

    Returns:
        A configured Agent with OLTP tools.
    """
    tools = create_oltp_tools(ctx)
    return Agent(
        tools=tools,
        system_prompt=OLTP_SYSTEM_PROMPT,
        name="oltp",
        **kwargs,
    )
