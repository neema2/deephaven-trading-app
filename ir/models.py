"""
ir.models — Shared reactive domain models for IR demos.

Defines the ``@ticking`` classes used by both ``demo_ir_swap.py`` and
``demo_ir_risk.py``.  These classes create Deephaven ticking tables
automatically via the ``@ticking`` decorator.

.. note:: Because ``@ticking`` derives table names from class names and
   registers them globally, only one process should import this module.
   Tests define their own uniquely-named copies to avoid collisions.

Usage::

    from ir.models import FXSpot, YieldCurvePoint, InterestRateSwap
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from reactive.computed import computed, effect
from streaming import ticking
from store import Storable


# Optional publish queue — set by the caller so CurvePoint @effects
# can enqueue CurveTick dicts for republishing to the market data hub.
curve_publish_queue: deque = deque()


@ticking
@dataclass
class FXSpot(Storable):
    """Live FX spot rate.  @effect pushes every mid change to DH."""
    __key__ = "pair"

    pair: str = ""
    bid: float = 0.0
    ask: float = 0.0
    currency: str = ""

    @computed
    def mid(self):
        return (self.bid + self.ask) / 2

    @computed
    def spread_pips(self):
        return (self.ask - self.bid) * 10000

    @effect("mid")
    def on_mid(self, value):
        self.tick()


@ticking(exclude={"base_rate", "sensitivity", "fx_base_mid"})
@dataclass
class YieldCurvePoint(Storable):
    """Single curve point.  rate is @computed from fx_ref.mid (cross-entity)."""
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
        pct_move = (self.fx_ref.mid - fx_base) / fx_base  # type: ignore[attr-defined]
        return max(0.0001, self.base_rate + self.sensitivity * pct_move)

    @computed
    def discount_factor(self):
        return 1.0 / (1.0 + self.rate) ** self.tenor_years

    @effect("rate")
    def on_rate(self, value):
        self.tick()
        curve_publish_queue.append({
            "type": "curve",
            "label": self.label,
            "tenor_years": self.tenor_years,
            "rate": self.rate,
            "discount_factor": self.discount_factor,
            "currency": self.currency,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


@ticking
@dataclass
class InterestRateSwap(Storable):
    """IRS pricing.  float_rate is @computed from curve_ref.rate (cross-entity)."""
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
        return self.curve_ref.rate  # type: ignore[attr-defined]

    @computed
    def fixed_leg_pv(self):
        df = 1.0 / (1.0 + self.fixed_rate) ** self.tenor_years
        return self.notional * self.fixed_rate * self.tenor_years * df

    @computed
    def float_leg_pv(self):
        df = 1.0 / (1.0 + self.float_rate) ** self.tenor_years
        return self.notional * self.float_rate * self.tenor_years * df

    @computed
    def npv(self):
        float_df = 1.0 / (1.0 + self.float_rate) ** self.tenor_years
        fixed_df = 1.0 / (1.0 + self.fixed_rate) ** self.tenor_years
        float_pv = self.notional * self.float_rate * self.tenor_years * float_df
        fixed_pv = self.notional * self.fixed_rate * self.tenor_years * fixed_df
        return float_pv - fixed_pv

    @computed
    def dv01(self):
        return self.notional * self.tenor_years * 0.0001

    @computed
    def pnl_status(self) -> str:
        float_df = 1.0 / (1.0 + self.float_rate) ** self.tenor_years
        fixed_df = 1.0 / (1.0 + self.fixed_rate) ** self.tenor_years
        float_pv = self.notional * self.float_rate * self.tenor_years * float_df
        fixed_pv = self.notional * self.fixed_rate * self.tenor_years * fixed_df
        npv = float_pv - fixed_pv
        if npv > 0:
            return "PROFIT"
        if npv < 0:
            return "LOSS"
        return "FLAT"

    @effect("npv")
    def on_npv(self, value):
        self.tick()


@ticking
@dataclass
class SwapPortfolio(Storable):
    """Aggregate portfolio.  @computed reads child swap NPVs (cross-entity)."""
    __key__ = "name"

    name: str = ""
    swaps: list = field(default_factory=list)

    @computed
    def total_npv(self):
        return sum(s.npv for s in self.swaps) if self.swaps else 0.0

    @computed
    def total_dv01(self):
        return sum(s.dv01 for s in self.swaps) if self.swaps else 0.0

    @computed
    def max_npv(self):
        return max(s.npv for s in self.swaps) if self.swaps else 0.0

    @computed
    def min_npv(self):
        return min(s.npv for s in self.swaps) if self.swaps else 0.0

    @computed
    def swap_count(self) -> int:
        return len(self.swaps) if self.swaps else 0

    @effect("total_npv")
    def on_total_npv(self, value):
        self.tick()
