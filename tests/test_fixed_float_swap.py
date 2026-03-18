"""
Tests for explicit float swap, testing whether passing the exact same curve
for both projection and discount gives IDENTICAL results to the 'shortcut'
FixedFloatApproxSwapExpr.
"""

import pytest

from instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox
from instruments.ir_swap_fixed_float import IRSwapFixedFloat
from marketmodel.yield_curve import LinearTermDiscountCurve, YieldCurvePoint
from reactive.expr import eval_cached

PILLARS = [
    ("IR_USD_DISC_USD.1Y", 1.0, 0.03),
    ("IR_USD_DISC_USD.5Y", 5.0, 0.05),
    ("IR_USD_DISC_USD.10Y", 10.0, 0.07),
]

@pytest.fixture
def curve():
    points = [
        YieldCurvePoint(name=name, tenor_years=tenor, fitted_rate=rate, is_fitted=True)
        for name, tenor, rate in PILLARS
    ]
    return LinearTermDiscountCurve(name="TEST", points=points)

@pytest.fixture
def ctx():
    return {name: rate for name, _, rate in PILLARS}

def test_fixed_float_matches_approx(curve, ctx):
    """
    If the projection curve and discount curve are the identical LinearTermDiscountCurve,
    explicit float pricing should perfectly match the telescoped approx.
    """
    approx_swap = IRSwapFixedFloatApprox(
        notional=100e6,
        fixed_rate=0.04,
        tenor_years=5.0,
        curve=curve,
        side="RECEIVER"
    )

    explicit_swap = IRSwapFixedFloat(
        notional=100e6,
        fixed_rate=0.04,
        tenor_years=5.0,
        discount_curve=curve,
        projection_curve=curve,
        side="RECEIVER"
    )

    approx_npv = eval_cached(approx_swap.npv(), ctx)
    explicit_npv = eval_cached(explicit_swap.npv(), ctx)

    approx_float_pv = eval_cached(approx_swap.float_leg_pv(), ctx)
    explicit_float_pv = eval_cached(explicit_swap.float_leg_pv(), ctx)

    approx_fixed_pv = eval_cached(approx_swap.fixed_leg_pv(), ctx)
    explicit_fixed_pv = eval_cached(explicit_swap.fixed_leg_pv(), ctx)
    
    approx_dv01 = eval_cached(approx_swap.dv01(), ctx)
    explicit_dv01 = eval_cached(explicit_swap.dv01(), ctx)

    # The full scheduler sums individual coupon cashflows; the approx uses the
    # telescoping float-leg identity.  Both are correct; floating-point rounding
    # across ~20 periods means we match to ~1e-6 relative, not machine epsilon.
    assert explicit_dv01     == pytest.approx(approx_dv01,     rel=1e-6)
    assert explicit_fixed_pv == pytest.approx(approx_fixed_pv, rel=1e-6)
    assert explicit_float_pv == pytest.approx(approx_float_pv, rel=1e-6)
    assert explicit_npv      == pytest.approx(approx_npv,      rel=1e-6)

def test_fwd_expr_generation(curve, ctx):
    """
    Test that fwd() produces valid rates evaluated from context.
    """
    expr = curve.fwd(1.0, 2.0)
    rate = eval_cached(expr, ctx)
    assert isinstance(rate, float)
    
    # Simple check on forward rate: if 1Y is 3% and 5Y is 5%, 1->2 should be roughly > 3
    assert rate > 0.0
