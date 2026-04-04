"""
ir_sofr_swap.py — USD SOFR Overnight Indexed Swap (OIS)

Implements the standard USD SOFR swap using daily compounding on the floating leg.
Supports:
1.  Aged periods (uses historical fixings).
2.  Future periods (uses telescopic property/approximation).
3.  Calendar-aware scheduling via ir_scheduling.py.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from dataclasses import field

from store import Storable
from reactive.computed import computed, effect
from reactive.computed_expr import computed_expr
from reactive.expr import diff, Expr
from streaming import ticking
import instruments.ir_scheduling as sched


@ticking(exclude={"discount_curve", "risk", "fixings"})
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class IRSOFRSwap(Storable):
    """USD SOFR Overnight Indexed Swap (OIS).
    
    Attributes
    ----------
    symbol: str
        Unique identifier.
    notional: float
        Principal amount.
    fixed_rate: float
        Fixed coupon rate (decimal, e.g. 0.05).
    effective_date: datetime.date
        Start of the first accrual period.
    termination_date: datetime.date
        Maturity of the swap.
    frequency_months: int
        Payment frequency (standard OIS is 12).
    side: str
        "RECEIVER" (receives fixed) or "PAYER" (pays fixed).
    discount_curve: object
        Curve providing .df(target_date) for discounting and SOFR projection.
    fixings: dict[datetime.date, float]
        Historical daily SOFR rates for aged coupons.
    """
    __key__ = "symbol"
    
    symbol: str = ""
    notional: float = 0.0
    fixed_rate: float = 0.0
    effective_date: Optional[datetime.date] = None
    termination_date: Optional[datetime.date] = None
    frequency_months: int = 12
    side: str = "RECEIVER"
    currency: str = "USD"
    
    discount_curve: Any = field(default=None, repr=False)
    fixings: dict[datetime.date, float] = field(default_factory=dict, repr=False)
    
    evaluation_date_override: Optional[datetime.date] = None

    @computed
    def tenor_years(self) -> float:
        """Tenor of the swap in years (used by fitter)."""
        if not self.effective_date or not self.termination_date:
            return 0.0
        return (self.termination_date - self.effective_date).days / 365.2425

    @computed
    def evaluation_date(self) -> datetime.date:
        if self.evaluation_date_override:
            return self.evaluation_date_override
        # Fallback to today if no override
        return datetime.date.today()

    @computed
    def schedule(self) -> list[datetime.date]:
        """Generate the calendar-aware date schedule."""
        if not self.effective_date or not self.termination_date:
            return []
        
        return sched.swap_schedule(
            self.effective_date,
            self.termination_date,
            freq_months=self.frequency_months,
            currency=self.currency,
            end_of_month=True
        )

    @property
    def pillar_names(self) -> list[str]:
        if hasattr(self.discount_curve, "pillar_names"):
            return self.discount_curve.pillar_names
        return []

    def _tenor(self, date: datetime.date) -> float:
        """Helper to get tenor in years from evaluation date."""
        if not self.evaluation_date:
            return 0.0
        return (date - self.evaluation_date).days / 365.2425

    @computed_expr
    def fixed_leg_pv(self) -> Expr:
        """PV of fixed leg = Σ [notional * rate * tau * df_end]"""
        from reactive.expr import Sum
        if not self.discount_curve or not self.schedule:
            return 0.0
        
        sch = self.schedule
        dcc = sched.DayCountConvention.Thirty360US
        pvs = []
        for i in range(len(sch) - 1):
            end = sch[i+1]
            tau = sched.year_fraction(sch[i], end, dcc)
            # Use tenor in years (float) for the curve
            df = self.discount_curve.df(self._tenor(end))
            pvs.append(self.notional * self.fixed_rate * tau * df)
        return Sum(pvs)

    @computed_expr
    def float_leg_pv(self) -> Expr:
        """PV of floating leg using OIS compounding (with telescopic approx)."""
        from reactive.expr import Sum
        if not self.discount_curve or not self.schedule:
            return 0.0
        
        sch = self.schedule
        pvs = []
        for i in range(len(sch) - 1):
            start = sch[i]
            end = sch[i+1]
            tau = sched.year_fraction(start, end, sched.DayCountConvention.Act360)
            
            rate = sched.compounded_rate(
                start, 
                end, 
                evaluation_date=self.evaluation_date,
                fixings=self.fixings,
                discount_curve=self.discount_curve,
                telescopic=True,
                day_counter=sched.DayCountConvention.Act360
            )
            df_end = self.discount_curve.df(self._tenor(end))
            pvs.append(self.notional * rate * tau * df_end)
        return Sum(pvs)

    @computed_expr
    def npv(self) -> Expr:
        """NPV = Fixed - Float (RECEIVER) or Float - Fixed (PAYER)."""
        if self.side == "PAYER":
            return self.float_leg_pv() - self.fixed_leg_pv()
        return self.fixed_leg_pv() - self.float_leg_pv()

    def pillar_context(self) -> dict[str, Any]:
        """Context for solver: helps resolve cross-curve dependencies."""
        if hasattr(self.discount_curve, 'pillar_context'):
            return self.discount_curve.pillar_context()
        return {}

    @computed_expr
    def dv01(self) -> Expr:
        """DV01: Approximation via fixed leg annuity."""
        from reactive.expr import Sum
        if not self.discount_curve or not self.schedule:
            return 0.0
        
        sch = self.schedule
        dcc = sched.DayCountConvention.Thirty360US
        
        terms = []
        for i in range(len(sch) - 1):
            start, end = sch[i], sch[i+1]
            tau = sched.year_fraction(start, end, dcc)
            df = self.discount_curve.df(self._tenor(end))
            terms.append(self.notional * tau * df * 0.0001)
            
        return Sum(terms) if terms else 0.0

    @computed_expr
    def par_rate(self) -> Expr:
        """The fixed rate that would make the current NPV zero."""
        # PV_float / Annuity
        if not self.schedule:
            return 0.0
            
        # Annuity = fixed_leg_pv / fixed_rate
        # par_rate = float_leg_pv / (Annuity)
        
        sch = self.schedule
        dcc = sched.DayCountConvention.Thirty360US
        annuity_terms = []
        for i in range(len(sch) - 1):
            start, end = sch[i], sch[i+1]
            tau = sched.year_fraction(start, end, dcc)
            df = self.discount_curve.df(self._tenor(end))
            annuity_terms.append(self.notional * tau * df)
            
        from reactive.expr import Sum
        annuity = Sum(annuity_terms)
        return self.float_leg_pv() / annuity if annuity_terms else 0.0

    @computed_expr
    def risk(self) -> dict[str, Expr]:
        """∂npv/∂pillar_rate."""
        expr = self.npv()
        if expr is None: return {}
        return {
            name: diff(expr, name)
            for name in self.pillar_names
        }

    def tick(self):
        """Manual tick if needed for dashboard/streaming."""
        pass
