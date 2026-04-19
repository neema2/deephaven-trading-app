"""
ir_swap_fixed_float.py — Explicit Float Leg Interest Rate Swap

Uses separate projection and discount curves to price each floating coupon
explicitly using forward rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pydantic import ConfigDict, Field
from dataclasses import field
from typing import Any

from store import Storable
from reactive.traceable import traceable
from reactive.computed import effect
from reactive.expr import diff, Expr
from pricing.instruments.base import Instrument
import pricing.marketmodels.ir_curve_fitter
from streaming import ticking
from pricing.instruments.ir_scheduling import payment_dates, reset_dates


@ticking(exclude={"discount_curve", "projection_curve", "risk_ladder", "pillar_names"})
@dataclass
class IRSwapFixedFloat(Instrument):
    """IRS with explicit float leg tracking fwd inputs.

    Two execution paths from the same object:
      1. Reactive
      2. Expr DAG compiled to SQL
    """
    __key__ = "symbol"
    
    symbol: str = ""
    notional: float = 0.0
    fixed_rate: float = 0.0
    float_spread: float = 0.0
    tenor_years: float = 0.0
    currency: str = "USD"
    collateral_currency: str = "USD"
    discount_curve: object = field(default=None, repr=False)
    projection_curve: object = field(default=None, repr=False)
    side: str = "RECEIVER"  # RECEIVER = receive fixed, PAYER = pay fixed
    is_target: bool = False

    def target_dates(self) -> list[float]:
        return payment_dates(self.tenor_years)

    def reset_dates(self) -> list[float]:
        return reset_dates(self.tenor_years)

    # pillar_names is inherited and automated via Instrument

    def _safe_dt(self, t1: float, t2: float) -> float:
        # Simple accrual assumption for demo: actual time in years
        return float(t2 - t1)

    @traceable
    def dv01(self) -> Expr:
        """Sum of payment periods * discount(T) * notional * 0.0001."""
        if not self.discount_curve:
            return 0.0
        
        targets = self.target_dates()
        resets = self.reset_dates()
        terms = []
        for i in range(len(targets)):
            dt = self._safe_dt(resets[i], targets[i])
            df = self.discount_curve.df(targets[i])
            terms.append(df * (dt * 0.0001 * self.notional))
            
        return sum(terms) if terms else 0.0

    @traceable
    def fixed_leg_pv(self):
        """PV of fixed leg = dv01 * fixed_rate * 10000.0"""
        dv01_val = self.dv01
        if dv01_val is None:
            return 0.0
        return dv01_val * self.fixed_rate * 10000.0

    @traceable
    def float_leg_pv(self) -> Expr:
        """Explicit floating PV for multi-curve pricing.
        For each period: PV = notional * rate * dt * df_end
        """
        if not self.discount_curve or not self.projection_curve:
            return 0.0
            
        targets = self.target_dates()
        resets = self.reset_dates()
        terms = []
        
        for i in range(len(targets)):
            start = resets[i]
            end = targets[i]
            dt = self._safe_dt(start, end)
            
            rate = self.projection_curve.fwd(start, end) + self.float_spread
            df = self.discount_curve.df(end)
            
            terms.append(rate * df * self.notional * dt)
            
        return sum(terms) if terms else 0.0

    @traceable
    def npv(self) -> Expr:
        """NPV: RECEIVER = fixed - float, PAYER = float - fixed."""
        if self.side == "PAYER":
            return self.float_leg_pv - self.fixed_leg_pv
        return self.fixed_leg_pv - self.float_leg_pv

    @traceable
    def par_rate(self):
        """Par rate: the fixed_rate at which NPV = 0."""
        dv01_val = self.dv01
        if dv01_val is None:
            return 0.0
        # If float evaluator runs, prevent div by 0
        try:
            return self.float_leg_pv / (dv01_val * 10000.0)
        except ZeroDivisionError:
            return 0.0

    @traceable
    def pnl_status(self) -> str:
        val = self.npv
        if val > 0:
            return "PROFIT"
        elif val < 0:
            return "LOSS"
        return "FLAT"

    @effect("npv")
    def on_npv(self, value):
        if self.is_target:
            return
        self.tick()

    @traceable
    def risk_ladder(self) -> dict[str, Expr]:
        """∂npv/∂pillar_rate via symbolic differentiation."""
        expr = self.npv()
        if expr is None: return {}
        return {
            name: diff(expr, name)
            for name in self.pillar_names
        }
