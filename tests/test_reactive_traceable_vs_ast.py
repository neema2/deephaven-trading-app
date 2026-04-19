import pytest
from dataclasses import dataclass, field
from store.base import Storable
from reactive.computed import computed
from reactive.traceable import traceable
from reactive.expr import Const, BinOp, Field

import store.registry
store.registry.ColumnRegistry.validate_class = lambda self, cls: None

# -----------------------------------------------------------------------------
# Test cases inspired from AST testing in test_reactive.py
# Covering a smaller set here for direct AST vs Traceable demonstration.
# -----------------------------------------------------------------------------

@dataclass
class ComputedPosition(Storable):
    qty: float
    price: float

    @computed
    def mv(self) -> float:
        return self.qty * self.price


@dataclass
class ComputedPortfolio(Storable):
    """Cross-entity aggregation via @computed."""
    positions: list = field(default_factory=list)

    @computed
    def two_mv(self):
        return self.positions[0].mv + self.positions[1].mv

    @computed
    def total_mv(self):
        return sum(p.mv for p in self.positions) if self.positions else 0


@dataclass
class TraceablePosition(Storable):
    qty: float
    price: float

    @traceable
    def mv(self) -> float:
        return self.qty * self.price


@dataclass
class TraceablePortfolio(Storable):
    """Cross-entity aggregation via @traceable."""
    positions: list = field(default_factory=list)

    @traceable
    def two_mv(self):
        return self.positions[0].mv + self.positions[1].mv

    @traceable
    def total_mv(self):
        return sum(p.mv for p in self.positions) if self.positions else 0


def test_ast_vs_traceable_position():
    """Verify that Traceable constructs the math matching old AST Walking."""
    #########################################
    # Verify standard computation behaves identically
    #########################################
    pos_computed = ComputedPosition(qty=100.0, price=150.0)
    pos_standard = TraceablePosition(qty=100.0, price=150.0)
    
    assert pos_computed.mv == 15000.0
    assert pos_standard.mv == 15000.0

    #########################################
    # 1. Original AST parsed @computed
    #########################################
    ast_tree = ComputedPosition.mv.expr
    
    assert isinstance(ast_tree, BinOp)
    assert ast_tree.op == "*"
    assert ast_tree.left.name == "qty"
    assert ast_tree.right.name == "price"

    #########################################
    # 2. Alternative @traceable that can trace
    #########################################
    trace_tree = TraceablePosition.mv.expr
    
    assert isinstance(trace_tree, BinOp)
    assert trace_tree.op == "*"
    assert trace_tree.left.name == "qty"
    assert trace_tree.right.name == "price"


def test_ast_vs_traceable_portfolio():
    """Verify both paradigms extend identically across class boundaries."""
    port_c = ComputedPortfolio(positions=[
        ComputedPosition(qty=10.0, price=150.0),
        ComputedPosition(qty=20.0, price=300.0)
    ])
    assert port_c.total_mv == 7500.0

    port_t = TraceablePortfolio(positions=[
        TraceablePosition(qty=10.0, price=150.0),
        TraceablePosition(qty=20.0, price=300.0)
    ])
    assert port_t.total_mv == 7500.0
    
    # Prove that both static AST parsing architectures correctly detect complex cross-entity mappings and identically fallback gracefully to None at the class level!
    tree_c = ComputedPortfolio.total_mv.expr
    tree_t = TraceablePortfolio.total_mv.expr
    
    assert tree_c is None
    assert tree_t is None


def test_quantity_recomputes_on_update():
    """Verify reactive calculation of a change in underlying Position triggers the recalc in Portfolio total."""
    # Test Original AST @computed
    p1 = ComputedPosition(qty=100.0, price=228.0)
    p2 = ComputedPosition(qty=50.0, price=192.0)
    book_c = ComputedPortfolio(positions=[p1, p2])
    
    assert book_c.total_mv == 100.0 * 228.0 + 50.0 * 192.0
    p1.qty = 200.0
    assert book_c.total_mv == 200.0 * 228.0 + 50.0 * 192.0

    # Test Alternative @traceable
    p1_t = TraceablePosition(qty=100.0, price=228.0)
    p2_t = TraceablePosition(qty=50.0, price=192.0)
    book_t = TraceablePortfolio(positions=[p1_t, p2_t])
    
    assert book_t.total_mv == 100.0 * 228.0 + 50.0 * 192.0
    p1_t.qty = 200.0
    assert book_t.total_mv == 200.0 * 228.0 + 50.0 * 192.0


def test_two_mv_class_level_expr_trees():
    """
    Verify that both architectures elegantly bypass producing expressions dynamically on the raw Class-level when mapping matrix subsets.
    Neither AST nor Tracing magically fabricates graphs targeting undefined unknown arrays structurally without populated variables to hook onto.
    """
    tree_c = ComputedPortfolio.two_mv.expr
    tree_t = TraceablePortfolio.two_mv.expr
    
    assert tree_c is None
    assert tree_t is None


def test_instance_level_expr_trees():
    """
    Showcase the critical difference in capabilities between AST extraction vs Tracing dynamically on instances.
    AST inherently acts strictly isolated to the class 'formula' unpopulated (returning None for cross-entities).
    Tracing seamlessly walks instantiated values dynamically tracking variables natively to generate specific graphs.
    """
    from reactive.expr import Field, BinOp
    from reactive.sum_expr import Sum

    p1_t = TraceablePosition(qty=Field("qty1"), price=Field("price1"))
    p2_t = TraceablePosition(qty=Field("qty2"), price=Field("price2"))
    book_t = TraceablePortfolio(positions=[p1_t, p2_t])

    # Tracing is capable of generating mathematical trees cleanly evaluating cross-boundary array sum operators:
    assert isinstance(book_t.total_mv, Sum)
    
    # Tracing fully maps explicit index referencing structurally into simple mathematical bins natively traversing down into the Position instances:
    instance_tree = book_t.two_mv
    assert isinstance(instance_tree, Sum)
    assert len(instance_tree.terms) == 2
    assert instance_tree.terms[0].op == "*"  # Inner Position 1 MVP BinOp mapping
    assert instance_tree.terms[1].op == "*"  # Inner Position 2 MVP BinOp mapping

    # Conversely, ComputedPortfolio natively resolves to simple floats and lacks a functional tree-generating tracing mechanism
    p1_c = ComputedPosition(qty=10.0, price=150.0)
    p2_c = ComputedPosition(qty=20.0, price=300.0)
    book_c = ComputedPortfolio(positions=[p1_c, p2_c])

    with pytest.raises(TypeError):
        book_c.two_mv()
