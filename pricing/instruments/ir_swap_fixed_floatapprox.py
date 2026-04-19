"""
ir_swap — Interest Rate Swap pricing.instruments.

Two swap types with self-contained payoff logic:

  IRSwapFixedFloatApprox     — "Shortcut" float leg: N × (1 - DF_T).
                         Uses the telescoping identity so the float leg
                         needs only the maturity discount factor.

  IRSwapFixedFloat           — Explicit float leg: looks up fwd_at() at each
                          reset date, prices each floating coupon individually.
                          Supports separate projection_curve (for forwards)
                          and discount_curve (for discounting).

Both share:
  - The same rack_dates() → payment_dates() / reset_dates() schedule
  - The same fixed_leg logic (annuity × fixed_rate × notional)
  - @computed for reactivity, @ticking for streaming

Risk sensitivities are computed via the Expr tree architecture:
  - FixedFloatApproxSwapExpr builds the full pricing formula as an Expr tree
  - diff(npv_expr, pillar_label) → symbolic derivative Expr
  - Same tree compiles to Python (.eval) and SQL (.to_sql)

The swap does NOT know about YieldCurvePoints, pillar names, or how
the curve interpolates.  It only depends on the curve via:
    curve.fwd_at(tenor)
    curve.df_at(tenor)
    curve.df_array(tenors)
"""

from __future__ import annotations

from dataclasses import dataclass
from pydantic import ConfigDict, Field
from dataclasses import field

from store import Storable
from reactive.computed import computed, effect
from reactive.expr import Const, diff, eval_cached, If, Expr
import pricing.marketmodels.ir_curve_fitter
from streaming import ticking


from pricing.instruments.ir_scheduling import rack_dates, payment_dates, reset_dates, day_count_fraction

from reactive.computed import computed, effect
from reactive.traceable import traceable


# ═══════════════════════════════════════════════════════════════════════════
# IRSwapFixedFloatApprox  (shortcut float leg)
# ═══════════════════════════════════════════════════════════════════════════

from pricing.instruments.base import Instrument

@ticking(exclude={"curve", "risk_ladder", "pillar_names"})
@dataclass
class IRSwapFixedFloatApprox(Instrument):
    """IRS with telescoping float leg — single class for both reactive and Expr.

    Used for single-curve USD OIS swaps. Uses PR1 stabilization patterns
    but adds Pydantic validation for static definition.

    Two execution paths from the same object:
      1. Reactive: @computed properties use curve.df_at() for live ticking
      2. Expr tree: npv_expr built at construction, compilable to SQL/Pure

    The Expr tree is built via __post_init__ when the curve supports
    df() (i.e. has pillar points).  Otherwise Expr fields are None.
    """
    __key__ = "symbol"

    symbol: str = ""
    notional: float = 0.0
    fixed_rate: float = 0.0
    tenor_years: float = 0.0
    currency: str = "USD"
    collateral_currency: str = "USD"
    side: str = "RECEIVER"    # "RECEIVER" (Receive Fixed, Pay Float) or "PAYER"
    curve: object = None      # anything with fwd_at(), df_at(), df_array()
    is_target: bool = False   # If True, hide from live portfolio ticking

    def __post_init__(self):
        super().__post_init__()
        if self.currency != self.collateral_currency:
            raise ValueError(f"IRSwapFixedFloatApprox cannot have differing currency ({self.currency}) and collateral_currency ({self.collateral_currency}). Please use IRSwapFixedFloat instead.")

    def coupon_payment_dates(self) -> list[float]:
        """Payment dates for this swap (short front stub, no 0.0)."""
        return payment_dates(self.tenor_years)

    # ── Traceable Pricing Properties ──────────────────────────────────
    #
    # @traceable: same code runs on plain floats (debug) or TracedFloat
    # (for symbolic Expr trees).  Default is float — zero overhead,
    # transparent in debugger watch windows.
    #
    # swap.dv01   → 4523.17   (float, fast)
    # swap.dv01() → Expr tree (lazy-traced, cached)
    #

    @traceable
    def dv01(self) -> float:
        """DV01 = Σ (notional × period_i × df_i × 0.0001)."""
        dates = self.coupon_payment_dates()
        pv = 0.0
        prev_t = 0.0
        for t in dates:
            df = self.curve.df(t)
            pv += df * day_count_fraction(prev_t, t) * self.notional * 0.0001
            prev_t = t
        return pv

    @traceable
    def fixed_leg_pv(self):
        """PV of fixed leg = dv01 * fixed_rate * 10000.0"""
        dv01_val = self.dv01
        if dv01_val is None:
            return 0.0
        return dv01_val * self.fixed_rate * 10000.0

    @traceable
    def float_leg_pv(self) -> float:
        """PV of floating leg = notional × (1 - DF_maturity)."""
        df_T = self.curve.df(float(self.tenor_years))
        return self.notional * (1.0 - df_T)

    @traceable
    def npv(self) -> float:
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
        if self.is_target:
            return
        self.tick()
