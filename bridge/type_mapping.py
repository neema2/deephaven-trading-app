"""
Type mapping between Python @dataclass fields and streaming column types.

Provides:
- infer_schema(storable_cls) → OrderedDict of {column_name: python_type}
- extract_row(obj, column_names) → tuple of values in column order
"""

import dataclasses
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal
from typing import Any, get_args, get_origin, get_type_hints

# Supported Python types for columns
_SUPPORTED_TYPES = {str, int, float, bool, Decimal, datetime}


def _unwrap_type(python_type: type) -> type:
    """Unwrap Optional[X] → X, and normalise to a supported Python type."""
    origin = get_origin(python_type)
    if origin is not None:
        args = get_args(python_type)
        # Optional[X] is Union[X, None]
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            python_type = non_none[0]

    return python_type if python_type in _SUPPORTED_TYPES else str


# Standard metadata columns prepended to every schema
_METADATA_COLUMNS = [
    ("EntityId", str),
    ("Version", int),
    ("EventType", str),
    ("State", str),
    ("UpdatedBy", str),
    ("TxTime", datetime),
]


def infer_schema(storable_cls: type) -> OrderedDict:
    """Auto-generate a column schema from a @dataclass Storable.

    Returns an OrderedDict of {column_name: python_type}.

    Metadata columns (EntityId, Version, EventType, State, UpdatedBy, TxTime)
    are always prepended. Domain columns follow from the dataclass fields.

    When a ColumnRegistry is available on the class, uses ColumnDef.python_type
    for canonical type resolution instead of raw annotation inference.
    """
    schema = OrderedDict()

    # Metadata columns
    for col_name, py_type in _METADATA_COLUMNS:
        schema[col_name] = py_type

    # Domain columns from dataclass fields
    if dataclasses.is_dataclass(storable_cls):
        registry = getattr(storable_cls, '_registry', None)
        hints = get_type_hints(storable_cls)
        for field in dataclasses.fields(storable_cls):
            # Prefer registry's canonical type when available
            if registry is not None:
                try:
                    col_def, _ = registry.resolve(field.name)
                    py_type = col_def.python_type
                except Exception:
                    py_type = hints.get(field.name, str)
            else:
                py_type = hints.get(field.name, str)
            schema[field.name] = _unwrap_type(py_type)

    return schema


# Backward compat alias
infer_dh_schema = infer_schema


def _to_dh_value(value: Any) -> Any:
    """Convert a Python value to a Deephaven-compatible value.

    Uses DH's own SDK for type conversions. The only type that
    DynamicTableWriter.write_row() can't auto-convert is
    datetime → java.time.Instant.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        import platform
        if platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"):
            # On ARM/pydeephaven, we use native Python datetimes or Arrow timestamps
            return value
        from deephaven.time import to_j_instant
        return to_j_instant(value)
    if isinstance(value, Decimal):
        return float(value)
    return value


# Metadata column definitions: (column_name, store_attr, python_type)
_META_COLUMNS = [
    ("EntityId", "_store_entity_id", str),
    ("Version", "_store_version", int),
    ("EventType", "_store_event_type", str),
    ("State", "_store_state", str),
    ("UpdatedBy", "_store_updated_by", str),
    ("TxTime", "_store_tx_time", datetime),
]

_META_ATTR_MAP = {name: attr for name, attr, _ in _META_COLUMNS}


def extract_row(obj: Any, column_names: list[str]) -> tuple:
    """Extract values from a Storable instance in the given column order.

    Args:
        obj: A Storable instance with _store_* metadata populated.
        column_names: Iterable of column names matching the schema.

    Returns:
        Tuple of DH-compatible values in the same order as column_names.
    """
    values = []
    for col in column_names:
        if col in _META_ATTR_MAP:
            raw = getattr(obj, _META_ATTR_MAP[col], None)
        else:
            raw = getattr(obj, col, None)
        values.append(_to_dh_value(raw))
    return tuple(values)
