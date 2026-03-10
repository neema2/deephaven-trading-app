"""
Tests for the Column Registry — enforced schema catalog.

Covers:
- Column definition and retrieval
- Enforcement rules (role, description, unit for measures)
- Prefix resolution (allowed_prefixes)
- Class validation (__init_subclass__ enforcement)
- Instance validation (runtime constraint checks)
- Introspection (entities, columns_for, entities_with)
- Type mismatch detection
"""

from dataclasses import dataclass

import pytest
from store.registry import ColumnDef, ColumnRegistry, RegistryError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reg():
    """Fresh registry for each test."""
    return ColumnRegistry()


@pytest.fixture
def trading_reg():
    """Registry pre-loaded with common trading columns."""
    r = ColumnRegistry()
    r.define("symbol", str,
        description="Ticker symbol", role="dimension",
        max_length=12, pattern=r"^[A-Z0-9./]+$",
    )
    r.define("price", float,
        description="Trade price", role="measure",
        unit="USD", min_value=0,
    )
    r.define("quantity", int,
        description="Number of shares", role="measure",
        unit="shares", min_value=0,
    )
    r.define("side", str,
        description="Buy or sell", role="dimension",
        enum=["BUY", "SELL"],
    )
    r.define("name", str,
        description="Person name", role="dimension",
        allowed_prefixes=["trader", "salesperson", "client"],
    )
    r.define("notes", str,
        description="Free text", role="attribute",
        nullable=True,
    )
    return r


# ===========================================================================
# A. Column Definition
# ===========================================================================

class TestDefine:

    def test_define_basic(self, reg):
        col = reg.define("symbol", str, description="Ticker", role="dimension")
        assert col.name == "symbol"
        assert col.python_type is str
        assert col.role == "dimension"

    def test_define_returns_columndef(self, reg):
        col = reg.define("price", float,
            description="Price", role="measure", unit="USD")
        assert isinstance(col, ColumnDef)

    def test_define_duplicate_raises(self, reg):
        reg.define("x", str, description="X", role="dimension")
        with pytest.raises(RegistryError, match="already defined"):
            reg.define("x", str, description="X again", role="dimension")

    def test_define_requires_role(self, reg):
        with pytest.raises(RegistryError, match=r"role.*required"):
            reg.define("x", str, description="X")

    def test_define_invalid_role(self, reg):
        with pytest.raises(RegistryError, match=r"role must be"):
            reg.define("x", str, description="X", role="bogus")

    def test_define_requires_description(self, reg):
        with pytest.raises(RegistryError, match=r"description.*required"):
            reg.define("x", str, role="dimension")

    def test_measure_requires_unit(self, reg):
        with pytest.raises(RegistryError, match=r"measures require.*unit"):
            reg.define("val", float, description="Value", role="measure")

    def test_measure_with_unit_ok(self, reg):
        col = reg.define("val", float,
            description="Value", role="measure", unit="USD")
        assert col.unit == "USD"

    def test_dimension_no_unit_ok(self, reg):
        col = reg.define("sym", str, description="Symbol", role="dimension")
        assert col.unit is None

    def test_all_metadata_fields(self, reg):
        col = reg.define("price", float,
            description="Trade price",
            role="measure",
            unit="USD",
            nullable=False,
            enum=None,
            min_value=0.0,
            max_value=1e9,
            max_length=None,
            pattern=None,
            synonyms=["px", "execution price"],
            sample_values=[100.0, 228.50],
            semantic_type="currency_amount",
            aggregation="last",
            display_name="Price",
            format=",.2f",
            category="trading",
            sensitivity="public",
            deprecated=False,
            tags=["front-office"],
            legend_type="Float",
            dh_type_override=None,
            allowed_prefixes=None,
        )
        assert col.synonyms == ["px", "execution price"]
        assert col.sample_values == [100.0, 228.50]
        assert col.semantic_type == "currency_amount"
        assert col.aggregation == "last"
        assert col.display_name == "Price"
        assert col.format == ",.2f"
        assert col.category == "trading"
        assert col.sensitivity == "public"
        assert col.tags == ["front-office"]
        assert col.legend_type == "Float"


# ===========================================================================
# B. Lookup
# ===========================================================================

class TestLookup:

    def test_get_existing(self, trading_reg):
        col = trading_reg.get("symbol")
        assert col.name == "symbol"
        assert col.python_type is str

    def test_get_missing_raises(self, trading_reg):
        with pytest.raises(RegistryError, match="not defined"):
            trading_reg.get("nonexistent")

    def test_has(self, trading_reg):
        assert trading_reg.has("symbol")
        assert not trading_reg.has("nonexistent")

    def test_all_columns(self, trading_reg):
        cols = trading_reg.all_columns()
        assert "symbol" in cols
        assert "price" in cols
        assert len(cols) == 6  # symbol, price, quantity, side, name, notes


# ===========================================================================
# C. Prefix Resolution
# ===========================================================================

class TestPrefixResolution:

    def test_resolve_direct_column(self, trading_reg):
        col, prefix = trading_reg.resolve("symbol")
        assert col.name == "symbol"
        assert prefix is None

    def test_resolve_prefixed_column(self, trading_reg):
        col, prefix = trading_reg.resolve("trader_name")
        assert col.name == "name"
        assert prefix == "trader"

    def test_resolve_different_prefix(self, trading_reg):
        col, prefix = trading_reg.resolve("client_name")
        assert col.name == "name"
        assert prefix == "client"

    def test_resolve_unapproved_prefix_raises(self, trading_reg):
        with pytest.raises(RegistryError, match=r"not defined.*allowed prefix"):
            trading_reg.resolve("random_name")

    def test_resolve_unknown_column_raises(self, trading_reg):
        with pytest.raises(RegistryError, match="not defined"):
            trading_reg.resolve("foo")

    def test_is_prefixed(self, trading_reg):
        assert trading_reg.is_prefixed("trader_name")
        assert not trading_reg.is_prefixed("symbol")
        assert not trading_reg.is_prefixed("nonexistent")

    def test_prefixed_columns(self, trading_reg):
        variants = trading_reg.prefixed_columns("name")
        assert "trader_name" in variants
        assert "salesperson_name" in variants
        assert "client_name" in variants
        assert len(variants) == 3

    def test_prefixed_columns_no_prefixes(self, trading_reg):
        assert trading_reg.prefixed_columns("symbol") == []

    def test_multi_underscore_prefix(self, reg):
        """Column names with underscores should resolve correctly."""
        reg.define("id", str, description="Identifier", role="dimension",
                   allowed_prefixes=["account", "trade_desk"])
        col, prefix = reg.resolve("trade_desk_id")
        assert col.name == "id"
        assert prefix == "trade_desk"

    def test_multi_underscore_base(self, reg):
        """Base column names with underscores should resolve correctly."""
        reg.define("full_name", str, description="Full name", role="dimension",
                   allowed_prefixes=["trader", "client"])
        col, prefix = reg.resolve("trader_full_name")
        assert col.name == "full_name"
        assert prefix == "trader"


# ===========================================================================
# D. Class Validation
# ===========================================================================

class TestClassValidation:

    def test_valid_class(self, trading_reg):
        @dataclass
        class Trade:
            __annotations__ = {"symbol": str, "price": float, "quantity": int}
        trading_reg.validate_class(Trade)
        assert Trade in trading_reg.entities()

    def test_unknown_field_raises(self, trading_reg):
        @dataclass
        class Bad:
            __annotations__ = {"foo": str}
        with pytest.raises(RegistryError, match=r"column 'foo'.*not defined"):
            trading_reg.validate_class(Bad)

    def test_type_mismatch_raises(self, trading_reg):
        @dataclass
        class Bad:
            __annotations__ = {"price": str}  # should be float
        with pytest.raises(RegistryError, match=r"type str.*does not match.*float"):
            trading_reg.validate_class(Bad)

    def test_prefixed_field_accepted(self, trading_reg):
        @dataclass
        class WithTrader:
            __annotations__ = {"trader_name": str, "symbol": str}
        trading_reg.validate_class(WithTrader)
        assert WithTrader in trading_reg.entities()

    def test_unapproved_prefix_rejected(self, trading_reg):
        @dataclass
        class Bad:
            __annotations__ = {"random_name": str}
        with pytest.raises(RegistryError, match="not defined"):
            trading_reg.validate_class(Bad)

    def test_optional_type_unwrapped(self, trading_reg):
        """Optional[str] should match a str column."""
        @dataclass
        class WithNotes:
            __annotations__ = {"notes": str | None}
        trading_reg.validate_class(WithNotes)

    def test_private_fields_skipped(self, trading_reg):
        """Fields starting with _ should be ignored."""
        @dataclass
        class WithPrivate:
            __annotations__ = {"symbol": str, "_internal": int}
        trading_reg.validate_class(WithPrivate)


# ===========================================================================
# E. __init_subclass__ Integration
# ===========================================================================

class TestInitSubclass:

    def test_storable_subclass_validated(self):
        """Storable subclasses with the global registry are validated."""
        from store.base import Storable
        # This should work — symbol, price, quantity are all registered
        @dataclass
        class TestTrade(Storable):
            symbol: str = ""
            price: float = 0.0
            quantity: int = 0
        assert TestTrade in Storable._registry.entities()

    def test_bad_storable_subclass_rejected(self):
        """Storable subclass with unregistered field should fail."""
        from store.base import Storable
        with pytest.raises(RegistryError, match="not defined"):
            @dataclass
            class BadStorable(Storable):
                unregistered_field_xyz: str = ""

    def test_type_mismatch_in_storable(self):
        """Storable subclass with wrong type should fail."""
        from store.base import Storable
        with pytest.raises(RegistryError, match="does not match"):
            @dataclass
            class BadType(Storable):
                price: str = ""  # should be float


# ===========================================================================
# F. Instance Validation
# ===========================================================================

class TestInstanceValidation:

    def test_valid_instance(self, trading_reg):
        @dataclass
        class Trade:
            symbol: str = ""
            price: float = 0.0
            side: str = "BUY"
        trading_reg.validate_class(Trade)
        errors = trading_reg.validate_instance(Trade(symbol="AAPL", price=228.5, side="BUY"))
        assert errors == []

    def test_enum_violation(self, trading_reg):
        @dataclass
        class Trade:
            side: str = ""
        trading_reg.validate_class(Trade)
        errors = trading_reg.validate_instance(Trade(side="HOLD"))
        assert any("not in allowed values" in e for e in errors)

    def test_min_value_violation(self, trading_reg):
        @dataclass
        class Trade:
            price: float = 0.0
        trading_reg.validate_class(Trade)
        errors = trading_reg.validate_instance(Trade(price=-1.0))
        assert any("min_value" in e for e in errors)

    def test_max_length_violation(self, trading_reg):
        @dataclass
        class Trade:
            symbol: str = ""
        trading_reg.validate_class(Trade)
        errors = trading_reg.validate_instance(Trade(symbol="A" * 20))
        assert any("max_length" in e for e in errors)

    def test_pattern_violation(self, trading_reg):
        @dataclass
        class Trade:
            symbol: str = ""
        trading_reg.validate_class(Trade)
        errors = trading_reg.validate_instance(Trade(symbol="lowercase"))
        assert any("pattern" in e for e in errors)

    def test_nullable_allows_none(self, trading_reg):
        @dataclass
        class WithNotes:
            notes: str | None = None
        trading_reg.validate_class(WithNotes)
        errors = trading_reg.validate_instance(WithNotes(notes=None))
        assert errors == []

    def test_non_nullable_rejects_none(self, trading_reg):
        @dataclass
        class Trade:
            symbol: str = None
        trading_reg.validate_class(Trade)
        errors = trading_reg.validate_instance(Trade(symbol=None))
        assert any("not nullable" in e for e in errors)

    def test_multiple_violations(self, trading_reg):
        @dataclass
        class Trade:
            symbol: str = ""
            price: float = 0.0
            side: str = ""
        trading_reg.validate_class(Trade)
        errors = trading_reg.validate_instance(
            Trade(symbol="toolongsymbolname123", price=-5.0, side="HOLD")
        )
        assert len(errors) == 4  # max_length, pattern, min_value, enum


# ===========================================================================
# G. Introspection
# ===========================================================================

class TestIntrospection:

    def test_entities(self, trading_reg):
        @dataclass
        class A:
            symbol: str = ""
        @dataclass
        class B:
            price: float = 0.0
        trading_reg.validate_class(A)
        trading_reg.validate_class(B)
        entities = trading_reg.entities()
        assert A in entities
        assert B in entities

    def test_columns_for(self, trading_reg):
        @dataclass
        class Trade:
            symbol: str = ""
            price: float = 0.0
        trading_reg.validate_class(Trade)
        cols = trading_reg.columns_for(Trade)
        names = [c.name for c in cols]
        assert "symbol" in names
        assert "price" in names

    def test_columns_for_unregistered_class(self, trading_reg):
        class Unknown:
            pass
        assert trading_reg.columns_for(Unknown) == []

    def test_entities_with(self, trading_reg):
        @dataclass
        class Trade:
            symbol: str = ""
            price: float = 0.0
        @dataclass
        class Signal:
            symbol: str = ""
        trading_reg.validate_class(Trade)
        trading_reg.validate_class(Signal)
        entities = trading_reg.entities_with("symbol")
        assert Trade in entities
        assert Signal in entities

    def test_entities_with_prefixed(self, trading_reg):
        """entities_with should find classes using prefixed variants."""
        @dataclass
        class Trade:
            trader_name: str = ""
        trading_reg.validate_class(Trade)
        entities = trading_reg.entities_with("name")
        assert Trade in entities


# ===========================================================================
# H. Global Registry Integration
# ===========================================================================

class TestGlobalRegistry:

    def test_global_registry_loaded_and_wired(self):
        from store.base import Storable
        from store.columns import REGISTRY
        from store.models import Order, Signal, Trade
        cols = REGISTRY.all_columns()
        assert len(cols) >= 40
        assert Storable._registry is not None
        entities = Storable._registry.entities()
        assert Trade in entities and Order in entities and Signal in entities

    def test_registry_has_expected_columns(self):
        from store.columns import REGISTRY
        expected = ["symbol", "price", "quantity", "side", "pnl",
                    "bid", "ask", "strike", "volatility", "notional",
                    "name", "label", "title", "status", "notes"]
        for col_name in expected:
            assert REGISTRY.has(col_name), f"Missing column: {col_name}"

    def test_all_columns_well_formed(self):
        """Every column must have role, description; measures must have unit."""
        from store.columns import REGISTRY
        for name, col in REGISTRY.all_columns().items():
            assert col.role in ("dimension", "measure", "attribute"), \
                f"Column '{name}' has invalid role: {col.role}"
            assert col.description, f"Column '{name}' missing description"
            if col.role == "measure":
                assert col.unit, f"Measure '{name}' missing unit"
