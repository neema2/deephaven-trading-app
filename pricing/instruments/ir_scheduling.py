"""
ir_scheduling — Minimal fractional-year date arithmetic for PR-B swaps.

Provides decimal-year helpers for the core demo IRS swaps.
"""
from typing import List

def rack_dates(tenor_years: float) -> List[float]:
    """Complete schedule rack including 0.0, walking back from tenor."""
    if tenor_years < 0:
        raise ValueError(f"tenor_years must be >= 0, got {tenor_years}")
    dates = []
    t = float(tenor_years)
    while t > 1e-9:
        dates.append(t)
        t -= 1.0
    dates.append(0.0)
    return sorted(dates)

def payment_dates(tenor_years: float) -> List[float]:
    """Payment dates = rack minus the first (0.0)."""
    return rack_dates(tenor_years)[1:]

def reset_dates(tenor_years: float) -> List[float]:
    """Float reset dates = rack minus the last (tenor_years)."""
    return rack_dates(tenor_years)[:-1]

def day_count_fraction(t1: float, t2: float) -> float:
    """ACT/365 year fraction between two decimal-year values."""
    return float(t2 - t1)
