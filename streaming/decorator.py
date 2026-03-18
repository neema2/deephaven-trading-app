"""
streaming.decorator — @ticking class decorator.

Auto-creates a TickingTable from a Storable dataclass, deriving column
schema from dataclass fields and @computed properties.

Usage::

    @ticking
    @dataclass
    class FXSpot(Storable):
        __key__ = "pair"
        pair: str = ""
        bid: float = 0.0
        ...

    @ticking(exclude={"base_rate", "sensitivity"})
    @dataclass
    class YieldCurvePoint(Storable):
        __key__ = "label"
        ...

Adds to the class:
    cls._ticking_table   TickingTable instance
    cls._ticking_live    LiveTable (last_by __key__)
    cls._ticking_cols    [(col_name, attr_name, python_type), ...]
    cls._ticking_name    snake_case name derived from class name
    self.tick()          instance method — writes all column values
"""

import re
from typing import Any

from streaming.table import LiveTable, TickingTable

# Global registry: table_name → (TickingTable, LiveTable, write_count_list)
# write_count_list is a single-element list [int] so it can be mutated via
# closure in _tick without needing to attach state to TickingTable (which uses __slots__).
_registry: dict[str, tuple[TickingTable, LiveTable, list]] = {}

# Primitive types that map to ticking table columns
_PRIMITIVE_TYPES = {str, float, int, bool}


def _to_snake_case(name: str) -> str:
    """Convert CamelCase class name to snake_case table name.

    FXSpot           → fx_spot
    YieldCurvePoint  → yield_curve_point
    IRSwapFixedFloatApprox → interest_rate_swap
    SwapPortfolio    → swap_portfolio
    """
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _resolve_column_specs(cls: type, exclude: set | None = None) -> list[tuple[str, str, type]]:
    """Pure-Python column resolution — no DH imports needed.

    Returns list of (col_name, attr_name, python_type).
    Skips non-primitive fields (object, list, etc.) and anything in exclude.
    """
    from reactive.computed import ComputedProperty

    exclude = set(exclude) if exclude else set()
    specs = []

    # 1. Dataclass fields (in definition order)
    for fname, fobj in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if fname in exclude or fname.startswith("_"):
            continue
        py_type = fobj.type
        if isinstance(py_type, str):
            py_type = {"str": str, "float": float, "int": int, "bool": bool}.get(py_type)
        if py_type not in _PRIMITIVE_TYPES:
            continue  # skip object, list, etc.
        specs.append((fname, fname, py_type))

    # 2. @computed properties (sorted for deterministic order)
    computed_names = sorted(
        name
        for name in dir(cls)
        if not name.startswith("_")
        and name not in exclude
        and isinstance(getattr(cls, name, None), ComputedProperty)
    )
    for name in computed_names:
        cp = getattr(cls, name)
        ret = getattr(cp.fn, "__annotations__", {}).get("return", float)
        
        # Handle "from __future__ import annotations" stringified types
        if isinstance(ret, str):
            ret = {"str": str, "float": float, "int": int, "bool": bool}.get(ret)

        if ret not in _PRIMITIVE_TYPES:
            # Skip non-primitive return types (list, dict, object) 
            # instead of defaulting to float.
            continue
            
        specs.append((name, name, ret))

    return specs


def _tick(self: Any) -> None:
    """Write all column values to the ticking table. Added to decorated classes."""
    cls = type(self)
    entry = _registry.get(cls._ticking_name)
    try:
        cls._ticking_table.write_row(*(getattr(self, attr) for _, attr, _ in cls._ticking_cols))
        if entry is not None:
            entry[2][0] += 1  # increment write counter
    except RuntimeError as e:
        if "Deephaven session not available" in str(e):
            # Silent fallback if DH is not running (e.g. offline tests)
            pass
        else:
            raise


def _apply_ticking(cls: type, exclude: set | None = None) -> type:
    """Core logic: create TickingTable, derive live table, attach to class."""
    # Require __key__
    key = getattr(cls, "__key__", None)
    if key is None:
        raise ValueError(
            f"@ticking on {cls.__name__} requires a __key__ class variable "
            f"(e.g. __key__ = 'symbol')"
        )

    # Resolve columns (pure Python types)
    col_specs = _resolve_column_specs(cls, exclude)
    if not col_specs:
        raise ValueError(f"@ticking on {cls.__name__}: no columns resolved")

    # Table name from class name
    table_name = _to_snake_case(cls.__name__)

    # Create TickingTable with Python-typed schema
    schema = {col_name: py_type for col_name, _, py_type in col_specs}
    tt = TickingTable(schema)

    # Derive live table (auto-locked via TickingTable.last_by)
    live = tt.last_by(key)

    # Attach to class
    cls._ticking_table = tt  # type: ignore[attr-defined]
    cls._ticking_live = live  # type: ignore[attr-defined]
    cls._ticking_cols = col_specs  # type: ignore[attr-defined]
    cls._ticking_name = table_name  # type: ignore[attr-defined]
    cls.tick = _tick  # type: ignore[attr-defined]

    # Register — write_count is a mutable single-element list so _tick can
    # increment it without needing to store state on TickingTable itself.
    _registry[table_name] = (tt, live, [0])

    return cls


def ticking(cls: type | None = None, *, exclude: set | None = None) -> type:
    """Class decorator: auto-create TickingTable + live table from Storable fields.

    Supports both bare and parameterized usage::

        @ticking                          # auto-infer all columns
        @ticking(exclude={"internal"})    # skip specific fields
    """
    if cls is not None:
        # Bare @ticking (no parentheses)
        return _apply_ticking(cls)
    # Parameterized @ticking(exclude=...)
    def decorator(cls: type) -> type:
        return _apply_ticking(cls, exclude=exclude)
    return decorator  # type: ignore[return-value]


def get_tables() -> dict:
    """Return dict of all registered tables: {name_raw: TickingTable, name_live: LiveTable}.

    Returns wrapped tables so all ops are auto-locked.
    """
    tables: dict[str, LiveTable] = {}
    for name, (tt, live, _wc) in _registry.items():
        tables[f"{name}_raw"] = tt          # TickingTable (inherits LiveTable)
        tables[f"{name}_live"] = live       # LiveTable from last_by
    return tables


def get_active_tables() -> dict:
    """Return only tables that have had at least one row written.

    The global registry accumulates an entry for every ``@ticking``-decorated
    class that is *imported*, even if no instances of that class are ever
    created in the current run.  This function filters to only the tables
    where ``.tick()`` was called at least once, keeping the DH panel list
    clean and limited to classes that are actively in use.

    Use instead of ``get_tables()`` when publishing to Deephaven::

        tables = get_active_tables()   # only live classes
        for name, tbl in tables.items():
            tbl.publish(name)
    """
    tables: dict[str, LiveTable] = {}
    for name, (tt, live, wc) in _registry.items():
        if wc[0] > 0:
            tables[f"{name}_raw"] = tt
            tables[f"{name}_live"] = live
    return tables


def get_ticking_tables() -> dict:
    """Return dict of all registered TickingTable instances: {name: TickingTable}."""
    return {name: tt for name, (tt, _live, _wc) in _registry.items()}


def clear_stale_tables(extra_names: list[str] | None = None) -> list[str]:
    """Remove from the Deephaven session any registered tables not written this run.

    When a Deephaven server persists between demo restarts (e.g. a Docker
    container left running), table names from the previous session remain
    bound in the query scope.  This function actively unbinds those names
    so the panel list reflects only the current run's active tables.

    Parameters
    ----------
    extra_names:
        Additional table name stems to clear (without ``_raw``/``_live`` suffix).
        Useful for clearing hand-crafted aggregates like ``swap_summary``.

    Returns
    -------
    list[str]
        The names that were successfully unbound.
    """
    from streaming.table import _is_remote, _get_remote_session

    cleared: list[str] = []

    # Collect all registered table names (raw + live variants) that are INACTIVE
    stale: list[str] = []
    for name, (tt, _live, wc) in _registry.items():
        if wc[0] == 0:
            stale.append(f"{name}_raw")
            stale.append(f"{name}_live")

    # Add any caller-supplied extras
    for stem in (extra_names or []):
        stale.append(stem)

    if not stale:
        return cleared

    if _is_remote():
        session = _get_remote_session()
        if session is None:
            return cleared
        for name in stale:
            try:
                # pydeephaven: remove a binding by pushing None-equivalent.
                # The cleanest approach is via the session's publish_table with
                # an empty / null binding, but the public API uses bind_table.
                # We use the internal console execute to delete the variable.
                session.run_script(f"if '{name}' in globals(): del {name}")
                cleared.append(name)
            except Exception:
                pass  # variable may not exist — safe to ignore
    else:
        # Local JVM (x86 embedded Deephaven)
        try:
            from deephaven.execution_context import get_exec_ctx
            scope = get_exec_ctx().j_exec_ctx.getQueryScope()
            for name in stale:
                try:
                    scope.removeParam(name)
                    cleared.append(name)
                except Exception:
                    pass
        except Exception:
            pass

    return cleared
