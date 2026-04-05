"""
ir_swap_fixed_float.py — Explicit Float Leg Interest Rate Swap

Uses separate projection and discount curves to price each floating coupon
explicitly using forward rates.
"""

from __future__ import annotations

from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field
from dataclasses import field
from typing import Any

from store import Storable
from reactive.computed import computed, effect
from reactive.computed_expr import computed_expr
from reactive.expr import diff, Expr
import marketmodel.curve_fitter
from streaming import ticking
from instruments.ir_scheduling import payment_dates, reset_dates


@ticking(exclude={"discount_curve", "projection_curve", "risk"})
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class IRSwapFixedFloat(Storable):
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

    @property
    def pillar_names(self) -> list[str]:
        names1 = self.discount_curve.pillar_names if self.discount_curve else []
        names2 = self.projection_curve.pillar_names if self.projection_curve else []
        merged = set(names1) | set(names2)
        return sorted(list(merged))

    def _safe_dt(self, t1: float, t2: float) -> float:
        # Simple accrual assumption for demo: actual time in years
        return float(t2 - t1)

    @computed_expr
    def dv01(self) -> Expr:
        """Sum of payment periods * discount(T) * notional * 0.0001."""
        from reactive.expr import Sum
        if not self.discount_curve:
            return 0.0
        
        targets = self.target_dates()
        resets = self.reset_dates()
        terms = []
        for i in range(len(targets)):
            dt = self._safe_dt(resets[i], targets[i])
            df = self.discount_curve.df(targets[i])
            terms.append(df * (dt * 0.0001 * self.notional))
            
        return Sum(terms) if terms else 0.0

    @computed_expr
    def fixed_leg_pv(self) -> Expr:
        """PV of fixed leg = sum of fixed_rate * notional * dt * df"""
        from reactive.expr import Sum
        if not self.discount_curve:
            return 0.0
            
        targets = self.target_dates()
        resets = self.reset_dates()
        terms = []
        for i in range(len(targets)):
            dt = self._safe_dt(resets[i], targets[i])
            df = self.discount_curve.df(targets[i])
            terms.append(df * (dt * self.fixed_rate * self.notional))
            
        return Sum(terms) if terms else 0.0

    @computed_expr
    def float_leg_pv(self) -> Expr:
        """Explicit floating PV for multi-curve pricing.
        For each period: PV = notional * rate * dt * df_end
        """
        from reactive.expr import Sum
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
            
        return Sum(terms) if terms else 0.0

    @computed_expr
    def npv(self) -> Expr:
        """NPV: RECEIVER = fixed - float, PAYER = float - fixed."""
        if self.side == "PAYER":
            return self.float_leg_pv() - self.fixed_leg_pv()
        return self.fixed_leg_pv() - self.float_leg_pv()

    @computed_expr
    def par_rate(self) -> Expr:
        """Par rate: the fixed_rate at which NPV = 0."""
        dv01_val = self.dv01()
        if dv01_val is None:
            return 0.0
        return self.float_leg_pv() / (dv01_val * 10000.0)

    @computed
    def pnl_status(self) -> str:
        val = self.npv
        if val > 0:
            return "PROFIT"
        elif val < 0:
            return "LOSS"
        return "FLAT"

    @effect("npv")
    def on_npv(self, value):
        if marketmodel.curve_fitter.IS_SOLVING or self.is_target:
            return
        self.tick()

    @computed_expr
    def risk(self) -> dict[str, Expr]:
        """∂npv/∂pillar_rate via symbolic differentiation."""
        expr = self.npv()
        if expr is None: return {}
        return {
            name: diff(expr, name)
            for name in self.pillar_names
        }

    def pillar_context(self) -> dict[str, float]:
        """Build a context dict from the curve's current pillar rates."""
        ctx = {}
        if self.discount_curve:
            pts = self.discount_curve._sorted_points()
            for p in pts:
                ctx[p.name] = p.rate
        if self.projection_curve:
            pts = self.projection_curve._sorted_points()
            for p in pts:
                ctx[p.name] = p.rate
        return ctx
