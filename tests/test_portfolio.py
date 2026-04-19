import pytest
# streaming - force conftest.py to start Deephaven for recursive deps
import numpy as np
import pandas as pd
from pricing.instruments.portfolio import Portfolio
from pricing.instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox
from pricing.instruments.ir_swap_fixed_float import IRSwapFixedFloat
from pricing.instruments.ir_swap_float_float import IRSwapFloatFloat
from pricing.marketmodels.ir_curve_yield import LinearTermDiscountCurve, YieldCurvePoint
from reactive.basis_extractor import BasisExtractor
from pricing.engines import PythonEngineExpr, SkinnyEngineDuckDB

# ─── Helpers for creating market data ────────────────────────────────────────

def build_usd_ois():
    tenors = [1.0, 5.0, 10.0, 30.0]
    rates = [0.03, 0.04, 0.05, 0.06]
    points = [
        YieldCurvePoint(name=f"USD_OIS_{t}Y", tenor_years=t, fitted_rate=r, is_fitted=True)
        for t, r in zip(tenors, rates)
    ]
    return LinearTermDiscountCurve(name="USD_OIS", points=points)

def build_jpy_ois():
    tenors = [1.0, 5.0, 10.0, 30.0]
    rates = [0.01, 0.015, 0.02, 0.025]
    points = [
        YieldCurvePoint(name=f"JPY_OIS_{t}Y", tenor_years=t, fitted_rate=r, is_fitted=True)
        for t, r in zip(tenors, rates)
    ]
    return LinearTermDiscountCurve(name="JPY_OIS", points=points)

def build_jpy_tibor():
    # Different curve for projection
    tenors = [1.0, 5.0, 10.0, 30.0]
    rates = [0.02, 0.025, 0.03, 0.035]
    points = [
        YieldCurvePoint(name=f"JPY_TIBOR_{t}Y", tenor_years=t, fitted_rate=r, is_fitted=True)
        for t, r in zip(tenors, rates)
    ]
    return LinearTermDiscountCurve(name="JPY_TIBOR", points=points)

# ─── Portfolio Tests ────────────────────────────────────────────────────────

class TestPortfolioFeatures:
    
    def test_empty_portfolio(self):
        port = Portfolio()
        assert port.names == []
        assert port.pillar_names == []
        assert port.pillar_context() == {}

    def test_pillar_aggregation_multi_currency(self):
        usd_curve = build_usd_ois()
        jpy_curve = build_jpy_ois()
        
        port = Portfolio()
        # Add a USD swap
        s1 = IRSwapFixedFloatApprox(symbol="USD_10Y", notional=1e6, fixed_rate=0.05, tenor_years=10.0, curve=usd_curve)
        port.add_instrument("USD_S1", s1)
        
        # Add a JPY swap
        s2 = IRSwapFixedFloatApprox(symbol="JPY_10Y", notional=100e6, fixed_rate=0.02, tenor_years=10.0, curve=jpy_curve)
        port.add_instrument("JPY_S2", s2)
        
        # USD 10Y uses 3 pillars (1, 5, 10)
        # JPY 10Y uses 3 pillars (1, 5, 10)
        # Total = 6
        pillars = port.pillar_names
        assert len(pillars) == 6
        assert all(p.startswith("USD_OIS_") or p.startswith("JPY_OIS_") for p in pillars)
        
        # Verify context gathering
        ctx = port.pillar_context()
        assert len(ctx) == 8
        assert ctx["USD_OIS_10.0Y"] == 0.05
        assert ctx["JPY_OIS_10.0Y"] == 0.02

    def test_heterogeneous_instruments_and_explicit_multi_curve(self):
        usd_ois = build_usd_ois()
        jpy_ois = build_jpy_ois()
        jpy_tibor = build_jpy_tibor()
        
        port = Portfolio()
        
        # Add Approximate swap (Standard USD)
        s_approx = IRSwapFixedFloatApprox(
            symbol="USD_APPROX", notional=1.0e6, fixed_rate=0.045, tenor_years=5.0, curve=usd_ois
        )
        port.add_instrument("USD_APPROX", s_approx)
        
        # Add Explicit Multi-Curve swap (JPY Tibor v JPY OIS)
        s_explicit = IRSwapFixedFloat(
            symbol="JPY_EXPLICIT", 
            notional=100.0e6, 
            fixed_rate=0.025, 
            tenor_years=10.0, 
            discount_curve=jpy_ois, 
            projection_curve=jpy_tibor
        )
        port.add_instrument("JPY_EXPLICIT", s_explicit)
        
        assert port.names == ["USD_APPROX", "JPY_EXPLICIT"]
        
        # JPY Explicit depends on both OIS and TIBOR pillars (3+3 = 6)
        # USD Approx depends on USD OIS (2 used)
        # Total pillars = 8 (Discovery is now precise via Expr.variables)
        assert len(port.pillar_names) == 8
        
        # Numeric Evaluation
        engine = PythonEngineExpr()
        ctx = port.pillar_context()
        npvs = engine.npvs(port, ctx)
        assert "USD_APPROX" in npvs
        assert "JPY_EXPLICIT" in npvs
        assert isinstance(npvs["USD_APPROX"], float)
        assert isinstance(npvs["JPY_EXPLICIT"], float)

    def test_total_portfolio_risk(self):
        # Using a standard Linear Curve
        curve = build_usd_ois()
        
        port = Portfolio()
        for t in [5.0, 10.0, 20.0]:
            s = IRSwapFixedFloatApprox(symbol=f"S{t:.0f}", notional=1e6, fixed_rate=0.04, tenor_years=t, curve=curve)
            port.add_instrument(f"S{t:.0f}", s)
            
        engine = PythonEngineExpr()
        ctx = port.pillar_context()
        total_risk = engine.total_risk(port, ctx)
        
        # Check we have risk for all pillars
        assert set(total_risk.keys()) == set(port.pillar_names)
        # Risk should be non-zero for pillars relevant to these tenors
        assert any(abs(v) > 1e-5 for v in total_risk.values())

    def test_skinny_table_logic(self):
        curve = build_usd_ois()
        port = Portfolio()
        s = IRSwapFixedFloatApprox(symbol="S5", notional=1e6, fixed_rate=0.04, tenor_years=5.0, curve=curve)
        port.add_instrument("S5", s)
        
        extractor = BasisExtractor()
        engine = SkinnyEngineDuckDB(extractor)
        # 1. Component Extraction
        comps = engine.to_components(port, per_swap=True)
        assert len(comps) > 0
        df_comps = pd.DataFrame(comps)
        assert "Swap_Id" in df_comps.columns
        assert "Component_Class" in df_comps.columns
        assert "Component_Type" in df_comps.columns
        
        # 2. SQL Generation check
        sql = engine.generate_sql(port)
        assert "SELECT" in sql
        assert "SUM(" in sql
        assert "dNPV_dUSD_OIS_5.0Y" in sql # Check dynamic column naming

    def test_scaling_small_and_jacobian_consistency(self):
        """Mini scaling test (10 swaps) - ensures Jacobian expr trees are valid."""
        curve = build_usd_ois()
        port = Portfolio()
        num_swaps = 10
        for i in range(num_swaps):
            s = IRSwapFixedFloatApprox(
                symbol=f"S{i}", 
                notional=1_000_000, 
                fixed_rate=0.03 + (i * 0.001), 
                tenor_years=1.0 + i, 
                curve=curve
            )
            port.add_instrument(f"S{i}", s)
            
        engine = PythonEngineExpr()
        ctx = port.pillar_context()
        
        # Evaluate Jacobian
        jac = engine.instrument_risk(port, ctx)
        
        assert len(jac) == num_swaps
        for name in jac:
            # Sparse discovery: only used pillars are returned.
            assert len(jac[name]) >= 1
            assert len(jac[name]) <= len(port.pillar_names)
            
        # Verify specific sensitivity: 1Y swap risk to 1Y pillar should be dominant
        s0_risk = jac["S0"]
        p1y_risk = s0_risk["USD_OIS_1.0Y"]
        assert abs(p1y_risk) > 1e-10


    def test_cross_currency_float_float(self):
        """Verify cross-currency basis swap with 3 curve dependencies."""
        usd_ois = build_usd_ois()
        jpy_ois = build_jpy_ois()
        jpy_tibor = build_jpy_tibor()
        
        port = Portfolio()
        
        # XCCY Basis Swap: Leg1 JPY Tibor, Leg2 USD OIS
        s_xccy = IRSwapFloatFloat(
            symbol="JPYUSD_XCCY",
            leg1_notional=100.0e6, # JPY 100M
            leg1_currency="JPY",
            leg2_currency="USD",
            initial_fx=0.0065, # 1 JPY = 0.0065 USD
            float_spread=0.0010, # 10 bps
            tenor_years=10.0,
            leg1_discount_curve=jpy_ois,
            leg1_projection_curve=jpy_tibor,
            leg2_discount_curve=usd_ois,
            leg2_projection_curve=usd_ois,
            exchange_notional=True
        )
        port.add_instrument("XCCY", s_xccy)
        
        engine = PythonEngineExpr()
        ctx = port.pillar_context()
        npvs = engine.npvs(port, ctx)
        
        assert "XCCY" in npvs
        assert isinstance(npvs["XCCY"], float)
        
        # Verify risk columns exist for all 3 curves
        jac = engine.instrument_risk(port, ctx)
        pillars_found = set(jac["XCCY"].keys())
        # Check that we have pillars from different curves
        assert any("USD_OIS" in p for p in pillars_found)
        assert any("JPY_OIS" in p for p in pillars_found)
        assert any("JPY_TIBOR" in p for p in pillars_found)

if __name__ == "__main__":
    pytest.main([__file__])
