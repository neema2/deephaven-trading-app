import datetime
import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

# 1. Advanced Mock System for CurveFitter Testing
# We need real numpy/scipy but mocked platform imports
m = MagicMock()

def mock_computed(fn): return property(fn)
def mock_computed_expr(fn): return fn

mock_reactive = MagicMock()
mock_reactive.computed = mock_computed
mock_reactive.computed_expr = mock_computed_expr
mock_reactive.Sum = lambda it: sum(it) if it else 0.0

for mod in ["reactive", "reactive.computed", "reactive.computed_expr", "reactive.expr"]:
    sys.modules[mod] = mock_reactive

m.ticking = lambda x=None, **kwargs: (x if callable(x) else (lambda f: f))
m.Storable = type("DummyStorable", (), {})

for mod in ["streaming", "store", "store.base", "reaktiv"]:
    sys.modules[mod] = m

# Now import our components
import instruments.ir_scheduling as sched
from instruments.ir_swap_fixed_ois import IRSwapFixedOIS
from instruments.ir_swap_xccy_ois import IRSwapXCCYOIS

# Minimal Mock Curve that supports fitting
class MockPillar:
    def __init__(self, name, tenor, rate):
        self.name, self.tenor, self.rate = name, tenor, rate
        
class MockCurve:
    def __init__(self, pillars):
        self.points = sorted(pillars, key=lambda p: p.tenor)
    @property
    def pillar_names(self): return [p.name for p in self.points]
    def _sorted_points(self): return self.points
    def df(self, t):
        # Piecewise linear zero rate interpolation for testing
        ts = [p.tenor for p in self.points]
        rates = [p.rate for p in self.points]
        if t <= ts[0]: r = rates[0]
        elif t >= ts[-1]: r = rates[-1]
        else: r = np.interp(t, ts, rates)
        return np.exp(-r * t)
    def pillar_context(self):
        return {p.name: p.rate for p in self.points}

# 2. Mock Portfolio and Fitter (Minimal subset for testing)
class MockPortfolio:
    def __init__(self, swaps): self.swaps = swaps
    def npv(self, ctx):
        # Manually evaluate each swap NPV using the provided pillar context
        res = []
        for s in self.swaps:
            # We must monkeypatch the curves with the context rates
            for p in s.discount_curve.points:
                if p.name in ctx: p.rate = ctx[p.name]
            if hasattr(s, "leg1_discount_curve") and s.leg1_discount_curve:
                for p in s.leg1_discount_curve.points:
                    if p.name in ctx: p.rate = ctx[p.name]
            if hasattr(s, "leg2_discount_curve") and s.leg2_discount_curve:
                for p in s.leg2_discount_curve.points:
                    if p.name in ctx: p.rate = ctx[p.name]
            res.append(float(s.npv()))
        return np.array(res)

def test_usd_sofr_bootstrap():
    """Verify that IRSwapFixedOIS can drive a single-curve USD SOFR bootstrap."""
    print("\nUSD SOFR Bootstrap Test...")
    eval_date = datetime.date(2026, 1, 1)
    tenors = [1.0, 2.0, 3.0, 5.0, 10.0]
    mkt_quotes = [0.045, 0.042, 0.041, 0.040, 0.039]
    
    # Setup Curve
    pillars = [MockPillar(f"USD_SOFR_{int(t)}Y", t, 0.05) for t in tenors]
    curve = MockCurve(pillars)
    
    # Setup Target Swaps
    swaps = [IRSwapFixedOIS(
        symbol=f"USD_OIS_{int(t)}Y",
        currency="USD", notional=1_000_000.0,
        fixed_rate=q, effective_date=eval_date,
        termination_date=eval_date + datetime.timedelta(days=int(t*365.25)),
        discount_curve=curve, evaluation_date_override=eval_date
    ) for t, q in zip(tenors, mkt_quotes)]
    
    port = MockPortfolio(swaps)
    
    def objective(x):
        ctx = {p.name: x[i] for i, p in enumerate(pillars)}
        return port.npv(ctx)

    from scipy.optimize import least_squares
    x0 = np.array([0.05] * len(tenors))
    res = least_squares(objective, x0, jac='2-point', method='lm')
    
    print(f"USD Fit Status: {res.status}, Cost: {res.cost:,.2e}")
    # Update curve points
    for i, p in enumerate(pillars): p.rate = res.x[i]
    
    # Final check: NPVs should be zero
    final_npvs = port.npv({p.name: p.rate for p in pillars})
    print(f"Final USD NPVs: {final_npvs}")
    assert np.allclose(final_npvs, 0, atol=1e-5)

def test_eur_multi_curve_bootstrap():
    """Verify EUR OIS curve fitting using BOTH XCCY Basis and Single-CCY Swaps."""
    print("\nEUR Multi-Curve Bootstrap Test (XCCY + Single CCY)...")
    eval_date = datetime.date(2026, 1, 1)
    tenors = [1.0, 2.0, 5.0, 10.0]
    
    # 1. FIXED USD SOFR Curve (Simulated result from first test)
    usd_pillars = [MockPillar(f"USD_SOFR_{int(t)}Y", t, 0.04) for t in tenors]
    usd_curve = MockCurve(usd_pillars)
    
    # 2. EUR MARKET DATA
    # EUR ESTR Single-CCY OIS quotes
    eur_ois_quotes = [0.035, 0.033, 0.031, 0.030]
    # EUR/USD XCCY Basis quotes (EUR ESTR + Basis vs USD SOFR Flat)
    # Usually basis is negative (EUR is 'cheaper' to fund)
    xccy_basis_quotes = [-0.0015, -0.0020, -0.0025, -0.0030] # -15bp to -30bp
    
    # Setup EUR Curve to solve
    eur_pillars = [MockPillar(f"EUR_ESTR_{int(t)}Y", t, 0.03) for t in tenors]
    eur_curve = MockCurve(eur_pillars)
    
    # Setup Targets: We mix both instrument types!
    targets = []
    # (a) EUR Single-CCY OIS Swaps (solve for EUR projection/discounting)
    for t, q in zip(tenors, eur_ois_quotes):
        targets.append(IRSwapFixedOIS(
            symbol=f"EUR_OIS_{int(t)}Y",
            currency="EUR", notional=1_000_000.0,
            fixed_rate=q, effective_date=eval_date,
            termination_date=eval_date + datetime.timedelta(days=int(t*365.25)),
            discount_curve=eur_curve, evaluation_date_override=eval_date
        ))
    
    # (b) EUR/USD XCCY Basis Swaps (solve for EUR vs USD discount relation)
    for t, b in zip(tenors, xccy_basis_quotes):
        targets.append(IRSwapXCCYOIS(
            symbol=f"EURUSD_XCCY_{int(t)}Y",
            leg1_currency="EUR", leg1_notional=1_000_000.0, leg1_discount_curve=eur_curve,
            leg2_currency="USD", leg2_notional=1_100_000.0, leg2_discount_curve=usd_curve,
            basis_spread=b, initial_fx=1.10,
            effective_date=eval_date,
            termination_date=eval_date + datetime.timedelta(days=int(t*365.25)),
            exchange_notional=True, evaluation_date_override=eval_date
        ))
        
    port = MockPortfolio(targets)
    
    def objective(x):
        # x is the vector of EUR pillar rates
        ctx = {p.name: x[i] for i, p in enumerate(eur_pillars)}
        # Add fixed USD pillars to context
        for p in usd_pillars: ctx[p.name] = p.rate
        return port.npv(ctx)

    from scipy.optimize import least_squares
    x0 = np.array([0.03] * len(eur_pillars))
    # This is an over-determined system (8 targets, 4 pillars) if they aren't consistent.
    # But in reality, Basis and OIS Swaps are used to build separate curves.
    # If we want to solve EUR Discounting from basis AND EUR projection from OIS,
    # we usually need two separate EUR curves or a spread curve.
    # For this test, we demonstrate the FITTER can handle both types simultaneously.
    res = least_squares(objective, x0, jac='2-point', method='lm')
    
    print(f"EUR Fit Status: {res.status}, Cost: {res.cost:,.2e}")
    # Final check residuals
    final_npvs = port.npv({p.name: p.rate for p in eur_pillars + usd_pillars})
    print(f"Final EUR/XCCY NPVs: {final_npvs}")
    
    # In an over-determined consistent system, cost should be low.
    # If they are inconsistent, cost will be high but the "best fit" is found.
    assert res.cost < 1.0 # Should be very low if basis is small relative to principal
    
if __name__ == "__main__":
    test_usd_sofr_bootstrap()
    test_eur_multi_curve_bootstrap()
