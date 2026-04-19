import pytest
import numpy as np
import pandas as pd
from pricing.instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox
from pricing.marketmodels.ir_curve_yield import LinearTermDiscountCurve, YieldCurvePoint
from pricing.risk.firstorder_analytic import FirstOrderAnalyticRisk
from pricing.risk.firstorder_numeric import FirstOrderNumericalRisk
from pricing.engines.python_float import PythonEngineFloat
from pricing.engines.python_expr import PythonEngineExpr
from pricing.engines.skinny import SkinnyEngineNumPy
from pricing.instruments.portfolio import Portfolio

def build_test_curve():
    tenors = [1.0, 5.0, 10.0, 30.0]
    rates = [0.03, 0.04, 0.05, 0.06]
    points = [
        YieldCurvePoint(name=f"USD_OIS_{t}Y", tenor_years=t, fitted_rate=r, is_fitted=True)
        for t, r in zip(tenors, rates)
    ]
    return LinearTermDiscountCurve(name="USD_OIS", points=points)

def test_engine_cross_comparison():
    """Verify all engine/risk combinations agree on DV01."""
    curve = build_test_curve()
    notional = 10_000_000
    swap = IRSwapFixedFloatApprox(
        symbol="S10Y", notional=notional, fixed_rate=0.045, tenor_years=10.0, curve=curve
    )
    
    port = Portfolio()
    port.add_instrument("S10Y", swap)
    ctx = port.pillar_context()
    
    configs = [
        ("PythonFloat", PythonEngineFloat(), FirstOrderNumericalRisk),
        ("PythonExpr (Analytic)", PythonEngineExpr(), FirstOrderAnalyticRisk),
        ("PythonExpr (Numerical)", PythonEngineExpr(), FirstOrderNumericalRisk),
        ("SkinnyNumPy (Analytic)", SkinnyEngineNumPy(), FirstOrderAnalyticRisk),
        ("SkinnyNumPy (Numerical)", SkinnyEngineNumPy(), FirstOrderNumericalRisk),
    ]
    
    all_results = {}
    
    print("\nEngine/Risk Cross-Comparison (S10Y DV01):")
    print(f"{'Config':<26} | {'USD_OIS_10.0Y':>15}")
    print("-" * 46)
    
    for label, engine, method in configs:
        # High precision central difference for numerical paths
        risk_kwargs = {"method": "central"} if method == FirstOrderNumericalRisk else {}

        if isinstance(engine, (PythonEngineFloat, PythonEngineExpr)):
            risk = engine.total_risk(port, ctx, risk_method=method, **risk_kwargs)
        else:
            df = engine.evaluate(port, ctx, risk_method=method, **risk_kwargs)
            # Find the column containing USD_OIS_10.0Y but not exactly 'NPV'
            col = [c for c in df.columns if "USD_OIS_10.0Y" in c and "NPV" in str(c)]
            if col:
                risk = { "USD_OIS_10.0Y": df.loc["S10Y", col[0]] }
            else:
                risk = {}
        
        dv01_10y = risk.get("USD_OIS_10.0Y", 0.0) * 0.0001
        all_results[label] = dv01_10y
        print(f"{label:<26} | {dv01_10y:15.4f}")

    baseline = all_results["PythonExpr (Analytic)"]
    for label, val in all_results.items():
        diff = abs(val - baseline)
        # Tolerance: Numerical vs Analytic is ~1.3e-3 due to discretization error (h=1e-5 on 10M).
        # We accept this as mathematical agreement.
        assert diff < 2e-3, f"Engine mismatch for {label}: {val} vs {baseline} (diff={diff})"

if __name__ == "__main__":
    pytest.main(["-s", __file__])
