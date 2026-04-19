import pytest
from dataclasses import dataclass, field
from store.base import Storable
from reactive.traceable import traceable
from reactive.expr import Const, BinOp, Variable
from reactive.calculus import diff
from reactive.evaluation import eval_cached

import store.registry
store.registry.ColumnRegistry.validate_class = lambda self, cls: None

# -----------------------------------------------------------------------------
# Test cases similar to test_reactive_ast_vs_traceable.py but extended
# to systematically mark the price as explicitly governed mapping natively enabling automatic analytical differentiation.
# -----------------------------------------------------------------------------

@dataclass
class Price(Storable):
    name: str
    value: float

    @traceable(is_variable=True)
    def quote(self) -> float:
        """Behaves cleanly as a regular float property natively, but structurally drops a tracked Variable dynamically."""
        return self.value


@dataclass
class TraceablePosition(Storable):
    qty: float
    price_obj: Price

    @traceable
    def mv(self) -> float:
        return self.qty * self.price_obj.quote


def test_variable_tracing():
    p_quote = Price(name="AAPL", value=150.0)
    pos_traced = TraceablePosition(qty=100.0, price_obj=p_quote)
    
    assert pos_traced.mv == 15000.0

    trace_tree = pos_traced.mv()
    assert trace_tree.op == "*"
    assert trace_tree.left.value == 100.0
    # Price is seamlessly mapped as the embedded abstract analytic leaf because of @variable
    assert trace_tree.right.name == "AAPL"


def test_portfolio_differentiation():
    """Show @traceable handles iteration and tracks analytical variables perfectly."""
    p1_quote = Price(name="AAPL", value=150.0)
    p2_quote = Price(name="MSFT", value=300.0)

    @dataclass
    class TraceablePortfolio(Storable):
        positions: list = field(default_factory=list)

        @traceable
        def total_mv(self) -> float:
            return sum(pos.mv for pos in self.positions) if self.positions else 0

    port = TraceablePortfolio(positions=[
        TraceablePosition(qty=10.0, price_obj=p1_quote),
        TraceablePosition(qty=20.0, price_obj=p2_quote),
    ])

    # Natively calculate via plain floats with natural Python
    assert port.total_mv == 1500.0 + 6000.0

    # Differentiate using the dynamically captured graph
    port_tree = port.total_mv()
    
    # Derivative w.r.t AAPL should just be exactly the quantity multiplier 10.0
    deriv_aapl = diff(port_tree, "AAPL")
    assert eval_cached(deriv_aapl, {}) == 10.0

    # Derivative w.r.t MSFT should be exactly 20.0
    deriv_msft = diff(port_tree, "MSFT")
    assert eval_cached(deriv_msft, {}) == 20.0
