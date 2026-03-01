"""
Tests for datacube server-side functions — tree builder, schema helpers, Arrow utils.

These test the server.py functions directly with real DuckDB data,
without starting a Tornado server or Perspective.
"""

import pytest
import pyarrow as pa
import duckdb

from datacube.config import PIVOT_COLUMN_NAME_SEPARATOR
from datacube.engine import Datacube
from datacube.server import (
    _normalize_arrow,
    _arrow_to_ipc,
    _pad_table,
    _get_source_schema,
    _build_tree_result,
    _snapshot_state,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def trades_conn():
    """DuckDB connection with sample trades data."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE trades (
            sector VARCHAR,
            symbol VARCHAR,
            side VARCHAR,
            rate_code VARCHAR,
            quantity INTEGER,
            price DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO trades VALUES
        ('Tech',    'AAPL',  'BUY',  'Standard', 100, 150.0),
        ('Tech',    'AAPL',  'SELL', 'Standard',  50, 155.0),
        ('Tech',    'GOOGL', 'BUY',  'JFK',      200, 2800.0),
        ('Tech',    'GOOGL', 'SELL', 'Standard',  100, 2850.0),
        ('Finance', 'JPM',   'BUY',  'Standard',  300, 160.0),
        ('Finance', 'JPM',   'SELL', 'JFK',       150, 162.0),
        ('Finance', 'GS',    'BUY',  'JFK',       250, 380.0),
        ('Finance', 'GS',    'SELL', 'Standard',   100, 385.0),
        ('Energy',  'XOM',   'BUY',  'Standard',  400, 95.0),
        ('Energy',  'XOM',   'SELL', 'JFK',       200, 97.0)
    """)
    return conn


@pytest.fixture
def dc(trades_conn):
    """Base datacube on trades table."""
    return Datacube(trades_conn, source_name="trades")


# ═══════════════════════════════════════════════════════════════════════
# 1. Arrow Utility Tests
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizeArrow:

    def test_int32_promoted_to_float64(self):
        t = pa.table({"x": pa.array([1, 2, 3], type=pa.int32())})
        result = _normalize_arrow(t)
        assert result.schema.field("x").type == pa.float64()

    def test_int64_promoted_to_float64(self):
        t = pa.table({"x": pa.array([1, 2, 3], type=pa.int64())})
        result = _normalize_arrow(t)
        assert result.schema.field("x").type == pa.float64()

    def test_float64_unchanged(self):
        t = pa.table({"x": pa.array([1.0, 2.0], type=pa.float64())})
        result = _normalize_arrow(t)
        assert result.schema.field("x").type == pa.float64()

    def test_string_unchanged(self):
        t = pa.table({"s": pa.array(["a", "b"], type=pa.string())})
        result = _normalize_arrow(t)
        assert result.schema.field("s").type == pa.string()

    def test_values_preserved(self):
        t = pa.table({"x": pa.array([10, 20], type=pa.int32())})
        result = _normalize_arrow(t)
        assert result.column("x").to_pylist() == [10.0, 20.0]


class TestArrowToIpc:

    def test_round_trip(self):
        t = pa.table({"x": [1.0, 2.0], "s": ["a", "b"]})
        ipc = _arrow_to_ipc(t)
        assert isinstance(ipc, bytes)
        assert len(ipc) > 0
        # Read back
        reader = pa.ipc.open_stream(ipc)
        restored = reader.read_all()
        assert restored.num_rows == 2
        assert restored.column_names == ["x", "s"]


class TestPadTable:

    def test_pads_missing_columns(self):
        t = pa.table({"a": [1.0], "b": ["x"]})
        target_cols = ["a", "b", "c"]
        target_schema = pa.schema([
            pa.field("a", pa.float64()),
            pa.field("b", pa.string()),
            pa.field("c", pa.float64()),
        ])
        result = _pad_table(t, target_cols, target_schema)
        assert "c" in result.column_names
        assert result.column("c")[0].as_py() is None

    def test_preserves_existing_columns(self):
        t = pa.table({"a": [1.0], "b": ["x"]})
        target_cols = ["a", "b"]
        target_schema = pa.schema([
            pa.field("a", pa.float64()),
            pa.field("b", pa.string()),
        ])
        result = _pad_table(t, target_cols, target_schema)
        assert result.column("a")[0].as_py() == 1.0
        assert result.column("b")[0].as_py() == "x"

    def test_column_order_matches_target(self):
        t = pa.table({"b": ["x"], "a": [1.0]})
        target_cols = ["a", "b"]
        target_schema = pa.schema([
            pa.field("a", pa.float64()),
            pa.field("b", pa.string()),
        ])
        result = _pad_table(t, target_cols, target_schema)
        assert result.column_names == ["a", "b"]


# ═══════════════════════════════════════════════════════════════════════
# 2. Source Schema Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGetSourceSchema:

    def test_flat_schema(self, dc):
        schema = _get_source_schema(dc)
        assert "sector" in [f.name for f in schema]
        assert "quantity" in [f.name for f in schema]

    def test_flat_schema_no_tree_columns(self, dc):
        schema = _get_source_schema(dc)
        names = [f.name for f in schema]
        assert "__tree__" not in names

    def test_vpivot_only_returns_flat_schema(self, dc):
        """VPivot-only: source schema should be flat (unpivoted)."""
        dc2 = dc.set_group_by("sector")
        schema = _get_source_schema(dc2)
        names = [f.name for f in schema]
        assert "sector" in names
        assert "quantity" in names
        # No pivoted columns
        sep = PIVOT_COLUMN_NAME_SEPARATOR
        assert not any(sep in n for n in names)

    def test_hpivot_with_vpivot_returns_pivoted_schema(self, dc):
        """VPivot+HPivot: source schema should include pivoted columns."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        names = [f.name for f in schema]
        sep = PIVOT_COLUMN_NAME_SEPARATOR
        # Should have pivoted columns
        pivot_cols = [n for n in names if sep in n]
        assert len(pivot_cols) > 0
        assert f"BUY{sep}quantity" in names
        assert f"SELL{sep}quantity" in names

    def test_hpivot_schema_includes_group_dimension(self, dc):
        """Pivoted schema should include the group_by dimension."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        names = [f.name for f in schema]
        assert "sector" in names


# ═══════════════════════════════════════════════════════════════════════
# 3. Tree Builder Tests
# ═══════════════════════════════════════════════════════════════════════


class TestBuildTreeResult:

    def test_single_level_tree(self, dc):
        """Group by sector → 3 group rows."""
        dc2 = dc.set_group_by("sector")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        assert "__tree__" in result.column_names
        assert result.num_rows == 3  # Tech, Finance, Energy
        # All rows should have ▸ prefix (collapsed)
        trees = result.column("__tree__").to_pylist()
        assert all("▸" in t for t in trees)

    def test_tree_no_depth_column(self, dc):
        """Tree builder should NOT include __depth__ column."""
        dc2 = dc.set_group_by("sector")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        assert "__depth__" not in result.column_names

    def test_expand_one_level(self, dc):
        """Expand Tech → shows Tech children + other collapsed groups."""
        dc2 = dc.set_group_by("sector", "symbol")
        schema = _get_source_schema(dc2)
        expanded = {("Tech",)}
        result = _build_tree_result(dc2, expanded, schema)
        trees = result.column("__tree__").to_pylist()
        # Should have: Tech (expanded ▾), AAPL, GOOGL, Finance (▸), Energy (▸)
        assert any("▾" in t and "Tech" in t for t in trees)
        assert any("AAPL" in t for t in trees)
        assert any("GOOGL" in t for t in trees)
        assert any("▸" in t and "Finance" in t for t in trees)
        assert any("▸" in t and "Energy" in t for t in trees)

    def test_expand_to_leaf(self, dc):
        """Expand Tech → AAPL → leaf rows."""
        dc2 = dc.set_group_by("sector", "symbol")
        schema = _get_source_schema(dc2)
        expanded = {("Tech",), ("Tech", "AAPL")}
        result = _build_tree_result(dc2, expanded, schema)
        trees = result.column("__tree__").to_pylist()
        # Should have leaf rows (contain "row")
        assert any("row" in t for t in trees)

    def test_collapse_removes_children(self, dc):
        """After collapsing Tech, only top-level groups remain."""
        dc2 = dc.set_group_by("sector", "symbol")
        schema = _get_source_schema(dc2)
        # All collapsed
        result = _build_tree_result(dc2, set(), schema)
        assert result.num_rows == 3  # Just the 3 sectors

    def test_tree_indentation(self, dc):
        """Child rows have more non-breaking space indent than parents."""
        dc2 = dc.set_group_by("sector", "symbol")
        schema = _get_source_schema(dc2)
        expanded = {("Tech",)}
        result = _build_tree_result(dc2, expanded, schema)
        trees = result.column("__tree__").to_pylist()
        # Tech row (depth 0) should have no indent
        tech_row = [t for t in trees if "Tech" in t and "▾" in t][0]
        assert not tech_row.startswith("\u00a0")
        # AAPL row (depth 1) should have indent
        aapl_row = [t for t in trees if "AAPL" in t][0]
        assert aapl_row.startswith("\u00a0")

    def test_tree_aggregation_values(self, dc):
        """Group rows should have aggregated values."""
        dc2 = dc.set_group_by("sector")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        # Tech: 100+50+200+100 = 450 quantity
        sectors = result.column("sector").to_pylist()
        quantities = result.column("quantity").to_pylist()
        tech_idx = next(i for i, s in enumerate(sectors) if s == "Tech")
        assert quantities[tech_idx] == 450.0

    def test_empty_tree(self, dc):
        """Group by with no matching data → empty table with correct schema."""
        dc2 = dc.set_group_by("sector").add_filter("sector", "eq", "NonExistent")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        assert result.num_rows == 0
        assert "__tree__" in result.column_names


# ═══════════════════════════════════════════════════════════════════════
# 4. Tree Builder + HPivot Tests
# ═══════════════════════════════════════════════════════════════════════


class TestTreeBuilderWithHPivot:

    def test_vpivot_hpivot_produces_pivoted_columns(self, dc):
        """VPivot + HPivot: tree rows should have pivoted column names."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        sep = PIVOT_COLUMN_NAME_SEPARATOR
        pivot_cols = [c for c in result.column_names if sep in c]
        assert len(pivot_cols) > 0
        assert f"BUY{sep}quantity" in result.column_names
        assert f"SELL{sep}quantity" in result.column_names

    def test_vpivot_hpivot_has_tree_column(self, dc):
        """Combined pivot still has __tree__ column."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        assert "__tree__" in result.column_names

    def test_vpivot_hpivot_row_count(self, dc):
        """VPivot by sector + HPivot by side → 3 rows (one per sector)."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        assert result.num_rows == 3

    def test_vpivot_hpivot_aggregation_values(self, dc):
        """Pivoted aggregation values are correct."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        sep = PIVOT_COLUMN_NAME_SEPARATOR
        sectors = result.column("sector").to_pylist()
        buy_qty = result.column(f"BUY{sep}quantity").to_pylist()
        tech_idx = next(i for i, s in enumerate(sectors) if s == "Tech")
        # Tech BUY: AAPL 100 + GOOGL 200 = 300
        assert buy_qty[tech_idx] == 300.0

    def test_hpivot_skips_leaf_rows(self, dc):
        """With HPivot active, leaf rows should NOT be emitted."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        expanded = {("Tech",)}
        result = _build_tree_result(dc2, expanded, schema)
        trees = result.column("__tree__").to_pylist()
        # With single-level group_by, expand goes to leaf.
        # But with HPivot, leaf rows are skipped → only group rows remain.
        assert not any("row" in t for t in trees)

    def test_hpivot_multi_level_expand(self, dc):
        """Multi-level VPivot + HPivot: expanding shows sub-groups with pivoted data."""
        dc2 = dc.set_group_by("sector", "symbol").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        expanded = {("Tech",)}
        result = _build_tree_result(dc2, expanded, schema)
        trees = result.column("__tree__").to_pylist()
        sep = PIVOT_COLUMN_NAME_SEPARATOR
        # Should have Tech expanded, AAPL/GOOGL as children, Finance/Energy collapsed
        assert any("▾" in t and "Tech" in t for t in trees)
        assert any("AAPL" in t for t in trees)
        assert f"BUY{sep}quantity" in result.column_names

    def test_hpivot_total_column(self, dc):
        """Total statistic column included in pivoted tree."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        schema = _get_source_schema(dc2)
        result = _build_tree_result(dc2, set(), schema)
        sep = PIVOT_COLUMN_NAME_SEPARATOR
        total_cols = [c for c in result.column_names if c.startswith(f"Total{sep}")]
        assert len(total_cols) > 0


# ═══════════════════════════════════════════════════════════════════════
# 5. Snapshot State Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSnapshotState:

    def test_basic_state(self, dc):
        state = _snapshot_state(dc)
        assert "source" in state
        assert "columns" in state
        assert "group_by" in state
        assert "pivot_by" in state
        assert "filters" in state
        assert "sort" in state

    def test_state_with_group_by(self, dc):
        dc2 = dc.set_group_by("sector")
        state = _snapshot_state(dc2)
        assert state["group_by"] == ["sector"]

    def test_state_with_pivot_by(self, dc):
        dc2 = dc.set_pivot_by("side")
        state = _snapshot_state(dc2)
        assert state["pivot_by"] == ["side"]

    def test_state_columns_have_required_fields(self, dc):
        state = _snapshot_state(dc)
        for col in state["columns"]:
            assert "name" in col
            assert "type" in col
            assert "kind" in col
            assert "aggregate_operator" in col
            assert "is_selected" in col


# ═══════════════════════════════════════════════════════════════════════
# 6. Separator Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPivotSeparator:

    def test_separator_is_safe_for_perspective(self):
        """Separator should not contain | which crashes Perspective 4.x."""
        assert "|" not in PIVOT_COLUMN_NAME_SEPARATOR

    def test_separator_in_column_names(self, dc):
        """Pivoted column names use the configured separator."""
        dc2 = dc.set_group_by("sector").set_pivot_by("side")
        df = dc2.query_df()
        sep = PIVOT_COLUMN_NAME_SEPARATOR
        pivot_cols = [c for c in df.columns if sep in c]
        assert len(pivot_cols) > 0
        # Verify format: "VALUE / measure"
        for col in pivot_cols:
            parts = col.split(sep)
            assert len(parts) == 2
