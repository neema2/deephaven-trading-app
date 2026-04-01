"""
Mirror test for demo_ir_risk.py
=================================
Verifies Risk_IR_DV01 bump-and-reprice logic:

  - Central difference: DV01 = (P_up - P_down) / (2 * shock)
  - Forward difference: DV01 = (P_up - P_base) / shock
  - Curve restoration after compute
  - Shock-size scaling (linearity for small bumps)
"""

from dataclasses import dataclass

import pytest

from reactive.computed import computed, effect
from streaming import flush, get_tables, ticking
from store import Storable


# ── Reactive domain models (same pattern as demo) ────────────────────────

@ticking
@dataclass
class DV01FXSpot(Storable):
    __key__ = "pair"
    pair: str = ""
    bid: float = 0.0
    ask: float = 0.0
    currency: str = ""

    @computed
    def mid(self):
        return (self.bid + self.ask) / 2

    @effect("mid")
    def on_mid(self, value):
        self.tick()


@ticking(exclude={"base_rate", "sensitivity", "fx_base_mid"})
@dataclass
class DV01CurvePoint(Storable):
    __key__ = "label"
    label: str = ""
    tenor_years: float = 0.0
    base_rate: float = 0.0
    sensitivity: float = 0.5
    currency: str = "USD"
    fx_ref: object = None
    fx_base_mid: float = 0.0

    @computed
    def rate(self):
        if self.fx_ref is None:
            return self.base_rate
        fx_base = self.fx_base_mid
        if fx_base == 0.0:
            return self.base_rate
        pct_move = (self.fx_ref.mid - fx_base) / fx_base
        return max(0.0001, self.base_rate + self.sensitivity * pct_move)

    @computed
    def discount_factor(self):
        return 1.0 / (1.0 + self.rate) ** self.tenor_years

    @effect("rate")
    def on_rate(self, value):
        self.tick()


@ticking
@dataclass
class DV01Swap(Storable):
    __key__ = "symbol"
    symbol: str = ""
    notional: float = 0.0
    fixed_rate: float = 0.0
    tenor_years: float = 0.0
    currency: str = "USD"
    curve_ref: object = None

    @computed
    def float_rate(self):
        if self.curve_ref is None:
            return 0.0
        return self.curve_ref.rate

    @computed
    def npv(self):
        float_df = 1.0 / (1.0 + self.float_rate) ** self.tenor_years
        fixed_df = 1.0 / (1.0 + self.fixed_rate) ** self.tenor_years
        float_pv = self.notional * self.float_rate * self.tenor_years * float_df
        fixed_pv = self.notional * self.fixed_rate * self.tenor_years * fixed_df
        return float_pv - fixed_pv

    @effect("npv")
    def on_npv(self, value):
        self.tick()


# ── Import Risk_IR_DV01 from the demo ───────────────────────────────────

from ir.risk import Risk_IR_DV01


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def risk_graph(streaming_server):
    """Build a reactive graph for risk testing."""
    fx = DV01FXSpot(pair="EUR/USD", bid=1.0850, ask=1.0855, currency="USD")

    usd_5y = DV01CurvePoint(
        label="DV01_USD_5Y", tenor_years=5.0, base_rate=0.0410,
        sensitivity=0.5, currency="USD",
        fx_ref=fx, fx_base_mid=fx.mid,
    )
    usd_10y = DV01CurvePoint(
        label="DV01_USD_10Y", tenor_years=10.0, base_rate=0.0395,
        sensitivity=0.5, currency="USD",
        fx_ref=fx, fx_base_mid=fx.mid,
    )

    swap_5y = DV01Swap(
        symbol="DV01-USD-5Y", notional=50_000_000, fixed_rate=0.0400,
        tenor_years=5.0, currency="USD", curve_ref=usd_5y,
    )
    swap_10y = DV01Swap(
        symbol="DV01-USD-10Y", notional=100_000_000, fixed_rate=0.0395,
        tenor_years=10.0, currency="USD", curve_ref=usd_10y,
    )

    tables = get_tables()
    for name, tbl in tables.items():
        tbl.publish(f"dv01_{name}")
    flush()

    return {
        "fx": fx,
        "usd_5y": usd_5y, "usd_10y": usd_10y,
        "swap_5y": swap_5y, "swap_10y": swap_10y,
    }


# ── Tests ────────────────────────────────────────────────────────────────

class TestRiskIRDV01:
    """Tests for Risk_IR_DV01 bump-and-reprice."""

    def test_central_dv01_nonzero(self, risk_graph) -> None:
        """Central difference DV01 should be non-zero for a swap with rate risk."""
        risk = Risk_IR_DV01(risk_graph["swap_5y"], risk_graph["usd_5y"])
        result = risk.compute_central()
        assert result["dv01"] != 0.0
        assert "p_up" in result
        assert "p_down" in result
        assert "base_npv" in result

    def test_forward_dv01_nonzero(self, risk_graph) -> None:
        """Forward difference DV01 should be non-zero."""
        risk = Risk_IR_DV01(risk_graph["swap_5y"], risk_graph["usd_5y"])
        result = risk.compute_forward()
        assert result["dv01"] != 0.0
        assert "p_up" in result
        assert "base_npv" in result

    def test_central_vs_forward_close(self, risk_graph) -> None:
        """Central and forward DV01 should be close for small shocks."""
        risk = Risk_IR_DV01(risk_graph["swap_5y"], risk_graph["usd_5y"], shock_bps=1.0)
        central = risk.compute_central()
        forward = risk.compute_forward()
        # They should agree to within a few percent for 1bp shock
        ratio = central["dv01"] / forward["dv01"]
        assert 0.95 < ratio < 1.05

    def test_curve_restored_after_central(self, risk_graph) -> None:
        """Curve point rate should be restored after compute_central()."""
        cp = risk_graph["usd_5y"]
        rate_before = cp.rate
        risk = Risk_IR_DV01(risk_graph["swap_5y"], cp)
        risk.compute_central()
        assert abs(cp.rate - rate_before) < 1e-12

    def test_curve_restored_after_forward(self, risk_graph) -> None:
        """Curve point rate should be restored after compute_forward()."""
        cp = risk_graph["usd_5y"]
        rate_before = cp.rate
        risk = Risk_IR_DV01(risk_graph["swap_5y"], cp)
        risk.compute_forward()
        assert abs(cp.rate - rate_before) < 1e-12

    def test_shock_size_scaling(self, risk_graph) -> None:
        """DV01 should be approximately constant for different small shocks."""
        risk_1bp = Risk_IR_DV01(risk_graph["swap_5y"], risk_graph["usd_5y"], shock_bps=1.0)
        risk_2bp = Risk_IR_DV01(risk_graph["swap_5y"], risk_graph["usd_5y"], shock_bps=2.0)
        dv01_1 = risk_1bp.compute_central()["dv01"]
        dv01_2 = risk_2bp.compute_central()["dv01"]
        # For small linear shocks, DV01 should be nearly identical
        ratio = dv01_1 / dv01_2
        assert 0.95 < ratio < 1.05

    def test_different_curve_points_give_different_dv01(self, risk_graph) -> None:
        """DV01 w.r.t. different curve points should differ."""
        risk_5y = Risk_IR_DV01(risk_graph["swap_5y"], risk_graph["usd_5y"])
        risk_10y = Risk_IR_DV01(risk_graph["swap_10y"], risk_graph["usd_10y"])
        dv01_5y = risk_5y.compute_central()["dv01"]
        dv01_10y = risk_10y.compute_central()["dv01"]
        # Different swaps/tenors → different DV01
        assert abs(dv01_5y - dv01_10y) > 1.0

    def test_p_up_greater_than_p_down(self, risk_graph) -> None:
        """For a receive-float swap, P_up > P_down (higher rates → higher float PV)."""
        risk = Risk_IR_DV01(risk_graph["swap_5y"], risk_graph["usd_5y"])
        result = risk.compute_central()
        # Float rate rises → float leg PV rises → NPV rises (receive-float)
        assert result["p_up"] > result["p_down"]

    def test_dv01_monotonic_with_tenor(self, risk_graph) -> None:
        """Longer tenor → larger absolute DV01 (notional held constant)."""
        cp_5y = risk_graph["usd_5y"]
        cp_10y = risk_graph["usd_10y"]
        swap_short = DV01Swap(
            symbol="MONO-T-5Y", notional=50_000_000, fixed_rate=0.0400,
            tenor_years=5.0, currency="USD", curve_ref=cp_5y,
        )
        swap_long = DV01Swap(
            symbol="MONO-T-10Y", notional=50_000_000, fixed_rate=0.0400,
            tenor_years=10.0, currency="USD", curve_ref=cp_10y,
        )
        flush()
        dv01_short = Risk_IR_DV01(swap_short, cp_5y).compute_central()["dv01"]
        dv01_long = Risk_IR_DV01(swap_long, cp_10y).compute_central()["dv01"]
        assert dv01_long > dv01_short

    def test_dv01_monotonic_with_notional(self, risk_graph) -> None:
        """Larger notional → larger absolute DV01 (tenor held constant)."""
        cp = risk_graph["usd_5y"]
        swap_small = DV01Swap(
            symbol="MONO-N-SM", notional=25_000_000, fixed_rate=0.0400,
            tenor_years=5.0, currency="USD", curve_ref=cp,
        )
        swap_large = DV01Swap(
            symbol="MONO-N-LG", notional=100_000_000, fixed_rate=0.0400,
            tenor_years=5.0, currency="USD", curve_ref=cp,
        )
        flush()
        dv01_small = Risk_IR_DV01(swap_small, cp).compute_central()["dv01"]
        dv01_large = Risk_IR_DV01(swap_large, cp).compute_central()["dv01"]
        assert dv01_large > dv01_small

