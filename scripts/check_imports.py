#!/usr/bin/env python3
"""
Import boundary checker — enforces public API usage across modules.

Detects cross-module imports that bypass __init__.py public APIs and reach
directly into private submodules.

Usage:
    python scripts/check_imports.py          # check all
    python scripts/check_imports.py --fix    # show suggested fixes

Exit code 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# ── Module boundary rules ────────────────────────────────────────────────
# Keys are module names.  Values list the ONLY allowed import paths.
# Any import from a submodule not in this list is a violation (when
# imported from OUTSIDE the module).

ALLOWED_IMPORTS: dict[str, set[str]] = {
    "store": {
        "store",              # from store import Storable, connect, ...
        "store.admin",        # from store.admin import StoreServer
    },
    "bridge": {
        "bridge",             # from bridge import StoreBridge, EventSink, ...
    },
    "lakehouse": {
        "lakehouse",          # from lakehouse import Lakehouse, LakehouseQuery
        "lakehouse.admin",    # from lakehouse.admin import LakehouseServer, ...
    },
    "timeseries": {
        "timeseries",         # from timeseries import Timeseries, TSDBBackend, ...
        "timeseries.admin",   # from timeseries.admin import TsdbServer
    },
}

# Modules whose internals should never be imported by others
PROTECTED_MODULES = set(ALLOWED_IMPORTS.keys())

# Allowed exceptions: (file_module, import_path) pairs where cross-module
# access to private submodules is intentional by design.
EXCEPTIONS: set[tuple[str, str]] = set()

# Files/directories to skip (tests deliberately test internals; use --include-tests)
SKIP_DIRS = {"_reference", "__pycache__", ".git", "node_modules", "venv", "tests"}

# Suggested fixes for common violations
FIXES: dict[str, str] = {
    "store.base.Storable": "from store import Storable",
    "store.base.Embedded": "from store import Embedded",
    "store.subscriptions.ChangeEvent": "from store import ChangeEvent",
    "store.subscriptions.EventListener": "from store import EventListener",
    "store.connection.connect": "from store import connect",
    "store.connection.active_connection": "from store import active_connection",
    "store.connection.UserConnection": "from store import UserConnection",
    "store.state_machine.StateMachine": "from store import StateMachine",
    "store.state_machine.Transition": "from store import Transition",
    "store.columns.REGISTRY": "from store import REGISTRY",
    "store.registry.ColumnDef": "from store import ColumnDef",
    "store.server.StoreServer": "from store.admin import StoreServer",
    "timeseries.base.TSDBBackend": "from timeseries import TSDBBackend",
}


def _module_of(filepath: Path) -> str | None:
    """Return the top-level module a file belongs to, or None."""
    rel = filepath.relative_to(ROOT)
    parts = rel.parts
    if parts and parts[0] in PROTECTED_MODULES:
        return parts[0]
    return None


def _is_type_checking_block(node: ast.AST) -> bool:
    """Check if a node is inside an `if TYPE_CHECKING:` block."""
    # We'll handle this at the caller level
    return False


def check_file(filepath: Path) -> list[dict[str, Any]]:
    """Check a single Python file for import boundary violations."""
    violations: list[dict[str, Any]] = []
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return violations

    file_module = _module_of(filepath)

    for node in ast.walk(tree):
        # from X.Y import Z
        if isinstance(node, ast.ImportFrom) and node.module:
            module_path = node.module
            top_level = module_path.split(".")[0]

            # Only check cross-module imports into protected modules
            if top_level not in PROTECTED_MODULES:
                continue
            if top_level == file_module:
                continue  # Same module — internal import, OK

            # Check if this import path is allowed
            allowed = ALLOWED_IMPORTS.get(top_level, set())
            if module_path in allowed:
                continue

            # Check exceptions (intentional cross-module access)
            if file_module and (file_module, module_path) in EXCEPTIONS:
                continue

            names = [alias.name for alias in (node.names or [])]
            for name in names:
                full_ref = f"{module_path}.{name}"
                fix = FIXES.get(full_ref)
                violations.append({
                    "file": str(filepath.relative_to(ROOT)),
                    "line": node.lineno,
                    "import": f"from {module_path} import {name}",
                    "fix": fix,
                })

    return violations


def main() -> int:
    show_fix = "--fix" in sys.argv

    all_violations = []
    for pyfile in sorted(ROOT.rglob("*.py")):
        # Skip excluded directories
        rel = pyfile.relative_to(ROOT)
        if any(d in rel.parts for d in SKIP_DIRS):
            continue
        # Skip tests — they may need internal access for testing
        # (uncomment to also check tests)
        # if "test" in rel.parts[0] if rel.parts else "":
        #     continue

        violations = check_file(pyfile)
        all_violations.extend(violations)

    if not all_violations:
        print("✅ All cross-module imports use public APIs.")
        return 0

    print(f"❌ Found {len(all_violations)} import boundary violation(s):\n")
    for v in all_violations:
        print(f"  {v['file']}:{v['line']}")
        print(f"    {v['import']}")
        if show_fix and v.get("fix"):
            print(f"    → {v['fix']}")
        print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
