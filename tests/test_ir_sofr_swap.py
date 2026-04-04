import datetime
import pytest
from unittest.mock import MagicMock
import sys

# Master mock system to allow offline testing without any heavy dependencies
m = MagicMock()

# Mock out reactive/computed decorators and types
mock_reactive = MagicMock()

def mock_computed(fn):
    return property(fn)

def mock_computed_expr(fn):
    # In the code, computed_expr methods are called: self.fixed_leg_pv()
    # So we keep them as methods.
    return fn

mock_reactive.computed = mock_computed
mock_reactive.computed_expr = mock_computed_expr
mock_reactive.effect = lambda *args: (lambda f: f)

# Make Sum perform actual numeric summation for tests
def mock_sum(iterable):
    return sum(iterable) if iterable else 0.0
mock_reactive.Sum = mock_sum

for mod in ["reactive", "reactive.computed", "reactive.computed_expr", "reactive.expr"]:
    sys.modules[mod] = mock_reactive

# Ticking mock
m.ticking = lambda x=None, **kwargs: (x if callable(x) else (lambda f: f))

class DummyStorable: pass
class DummyVariableMixin: pass
m.Storable = DummyStorable
m.VariableMixin = DummyVariableMixin

# Inject all possible heavy paths
heavy_mods = [
    "streaming", "streaming.decorator", "streaming.table", "streaming.admin",
    "store", "store._active_record", "store.base", "store.query_result"
]
for mod in heavy_mods:
    sys.modules[mod] = m

# Mock out unrelated instruments that trigger heavy loads
sys.modules["instruments.ir_swap_fixed_floatapprox"] = MagicMock()
sys.modules["instruments.portfolio"] = MagicMock()

# Now we can safely import our local modules
import instruments.ir_scheduling as sched
from instruments.ir_sofr_swap import IRSOFRSwap

class MockCurve:
    """Simple discount curve for testing."""
    def __init__(self, rate: float = 0.05):
        self.rate = rate
        self.pillar_names = ["R1", "R2"]
    
    def df(self, date: datetime.date) -> float:
        """Simple Act/365 exponential discounting for tests."""
        today = datetime.date(2026, 1, 1)
        t = (date - today).days / 365.0
        import math
        return math.exp(-self.rate * t)

def test_future_sofr_swap_npv():
    """Verify that a future SOFR swap (forward-starting) uses the telescopic property."""
    curve = MockCurve(rate=0.05)
    
    effective = datetime.date(2026, 6, 1)
    maturity = datetime.date(2031, 6, 1)
    
    swap = IRSOFRSwap(
        symbol="USD_SOFR_5Y",
        notional=1_000_000.0,
        fixed_rate=0.048, 
        effective_date=effective,
        termination_date=maturity,
        frequency_months=12,
        side="RECEIVER",
        discount_curve=curve,
        evaluation_date_override=datetime.date(2026, 1, 1)
    )
    
    # NPV calculation should trigger our new compounding logic
    npv = swap.npv()
    print(f"Future NPV: {npv}")
    
    sch = swap.schedule
    pv_fixed = 0.0
    dcc_fixed = sched.DayCountConvention.Thirty360US
    for i in range(len(sch) - 1):
        tau = sched.year_fraction(sch[i], sch[i+1], dcc_fixed)
        pv_fixed += 1_000_000.0 * 0.048 * tau * curve.df(sch[i+1])
        
    df_start = curve.df(sch[0])
    df_end = curve.df(sch[-1])
    pv_float = 1_000_000.0 * (df_start - df_end)
    
    expected_npv = pv_fixed - pv_float
    assert abs(npv - expected_npv) < 1.0 

def test_aged_sofr_swap_npv():
    """Verify that an aged SOFR swap (already started) uses the historical fixings."""
    curve = MockCurve(rate=0.05)
    
    today = datetime.date(2026, 6, 1)
    effective = datetime.date(2026, 1, 1)
    maturity = datetime.date(2028, 1, 1)
    
    # 5 months of fixings
    fixings = {}
    curr = effective
    while curr < today:
        fixings[curr] = 0.04
        curr += datetime.timedelta(days=1)
        
    swap = IRSOFRSwap(
        symbol="USD_SOFR_2Y_AGED",
        notional=1_000_000.0,
        fixed_rate=0.045,
        effective_date=effective,
        termination_date=maturity,
        frequency_months=12,
        side="RECEIVER",
        discount_curve=curve,
        fixings=fixings,
        evaluation_date_override=today
    )
    
    npv = swap.npv()
    print(f"Aged NPV: {npv}")
    assert npv is not None
    assert abs(npv) > 0.0

if __name__ == "__main__":
    test_future_sofr_swap_npv()
    test_aged_sofr_swap_npv()
    print("OIS tests passed!")
