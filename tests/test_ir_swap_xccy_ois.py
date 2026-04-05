import datetime
import pytest
from unittest.mock import MagicMock
import sys

# Master mock system for offline testing
m = MagicMock()
mock_reactive = MagicMock()
def mock_computed(fn): return property(fn)
def mock_computed_expr(fn): return fn
mock_reactive.computed = mock_computed
mock_reactive.computed_expr = mock_computed_expr
mock_reactive.Sum = lambda it: sum(it) if it else 0.0

for mod in ["reactive", "reactive.computed", "reactive.computed_expr", "reactive.expr"]:
    sys.modules[mod] = mock_reactive

m.ticking = lambda x=None, **kwargs: (x if callable(x) else (lambda f: f))
m.Storable = type("DummyStorable", (), {})
for mod in ["streaming", "store", "store.base"]:
    sys.modules[mod] = m

# Imports
import instruments.ir_scheduling as sched
from instruments.ir_swap_xccy_ois import IRSwapXCCYOIS

class MockCurve:
    def __init__(self, rate: float = 0.05, today=datetime.date(2026, 1, 1)):
        self.rate, self.today = rate, today
        self.pillar_names = ["R1", "R2"]
    def df(self, date: datetime.date) -> float:
        t = float(date) if isinstance(date, (float, int)) else (date - self.today).days / 365.2425
        import math
        return math.exp(-self.rate * t)

def test_xccy_ois_index_auto_resolution():
    """Verify that both legs of XCCY swap resolve correct indices."""
    curve1 = MockCurve(rate=0.04)
    curve2 = MockCurve(rate=0.05)
    
    # EUR/USD -> ESTR / SOFR
    swap = IRSwapXCCYOIS(
        leg1_currency="EUR", leg1_discount_curve=curve1,
        leg2_currency="USD", leg2_discount_curve=curve2
    )
    assert swap.resolved_leg1_index == "ESTR"
    assert swap.resolved_leg2_index == "SOFR"
    
    # GBP/JPY -> SONIA / TONAR
    swap2 = IRSwapXCCYOIS(
        leg1_currency="GBP", leg1_discount_curve=curve1,
        leg2_currency="JPY", leg2_discount_curve=curve2
    )
    assert swap2.resolved_leg1_index == "SONIA"
    assert swap2.resolved_leg2_index == "TONAR"

def test_xccy_ois_npv_parity():
    """Verify XCCY OIS NPV calculation with notional exchange."""
    curve1 = MockCurve(rate=0.04) # EUR
    curve2 = MockCurve(rate=0.03) # USD
    eval_date = datetime.date(2026, 1, 1)
    
    # EUR/USD XCCY Basis Swap
    # Receives EUR (+ Basis Spread), Pays USD
    swap = IRSwapXCCYOIS(
        symbol="EURUSD_XCCY",
        leg1_currency="EUR", leg1_notional=1_000_000.0, leg1_discount_curve=curve1,
        leg2_currency="USD", leg2_notional=1_100_000.0, leg2_discount_curve=curve2,
        basis_spread=0.0020, # +20bps
        initial_fx=1.10,
        effective_date=datetime.date(2026, 6, 1),
        termination_date=datetime.date(2027, 6, 1), # 1Y
        frequency_months=12,
        exchange_notional=True,
        side="RECEIVER",
        evaluation_date_override=eval_date
    )
    
    npv = swap.npv()
    print(f"XCCY OIS NPV (USD): {npv}")
    
    # Expected Leg 1 NPV (EUR):
    sch = swap.schedule
    tau = sched.year_fraction(sch[0], sch[1], sched.DayCountConvention.Act360)
    df_end1 = curve1.df(sch[1])
    # Telescopic Forward Rate: (P_start / P_end - 1) / tau
    fwd1 = (curve1.df(sch[0]) / df_end1 - 1.0) / tau
    leg1_accrual = 1_000_000.0 * (fwd1 + 0.002) * tau * df_end1
    leg1_nx = 1_000_000.0 * (df_end1 - 1.0)
    leg1_npv_total = leg1_accrual + leg1_nx
    
    # Expected Leg 2 NPV (USD):
    df_end2 = curve2.df(sch[1])
    fwd2 = (curve2.df(sch[0]) / df_end2 - 1.0) / tau
    leg2_accrual = 1_100_000.0 * fwd2 * tau * df_end2
    leg2_nx = 1_100_000.0 * (df_end2 - 1.0)
    leg2_npv_total = leg2_accrual + leg2_nx
    
    expected_npv_usd = leg1_npv_total * 1.10 - leg2_npv_total
    assert abs(npv - expected_npv_usd) < 1.0

if __name__ == "__main__":
    test_xccy_ois_index_auto_resolution()
    test_xccy_ois_npv_parity()
    print("XCCY OIS tests passed!")
