"""
Column catalog — the single source of truth for all Storable field definitions.

Exports REGISTRY, the global ColumnRegistry instance populated with all
approved columns from trading, finance, and general domains.
"""

from store.registry import ColumnRegistry

REGISTRY = ColumnRegistry()

# Import domain modules to populate the registry
from store.columns import (
    finance,  # noqa: F401
    general,  # noqa: F401
    media,  # noqa: F401
    scheduler,  # noqa: F401
    trading,  # noqa: F401
)


# Auto-import agent-generated column modules (no agent dependency)
def _load_agent_columns() -> None:
    import importlib.util
    import logging
    import sys
    from pathlib import Path
    d = Path(__file__).parent / "agent_generated"
    if not d.exists():
        return
    for f in sorted(d.glob("*.py")):
        if f.name.startswith("_"):
            continue
        mod_name = f"_agent_col_{f.stem}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, f)
        if spec and spec.loader:
            try:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
            except Exception as e:
                logging.getLogger(__name__).error(
                    "Failed to load agent column module %s: %s", f, e)

_load_agent_columns()
del _load_agent_columns
