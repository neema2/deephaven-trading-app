"""
ir_future_ois.py — OIS/SOFR Future Instrument

Models a standard overnight index future (like the 3-Month SOFR Future).
Price = 100 - [100 * (CompoundedRate + ConvexityAdj)]
"""

from __future__ import annotations

import datetime
from typing import Any, Optional
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from dataclasses import field

from store import Storable
from reactive.computed import computed
from reactive.computed_expr import computed_expr
from reactive.expr import Expr
from streaming import ticking
import instruments.ir_scheduling as sched


@ticking(exclude={"discount_curve", "fixings"})
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class IRFutureOIS(Storable):
    """3-month SOFR/OIS Future.
    
    Settles against the daily compounded SOFR over a 3-month reference period
    starting on an IMM date.
    """
    __key__ = "symbol"
    
    symbol: str = ""
    currency: str = "USD"
    
    imm_month: int = 3   # e.g. 3, 6, 9, 12
    imm_year: int = 2026
    
    # Optional curve and fixings
    discount_curve: Any = field(default=None, repr=False)
    fixings: dict[datetime.date, float] = field(default_factory=dict, repr=False)
    
    convexity_adj: float = 0.0 # Spread in bps, e.g. 0.0001
    evaluation_date_override: Optional[datetime.date] = None

    @computed
    def reference_period(self) -> tuple[datetime.date, datetime.date]:
        """Reference period for compounding (approx 3 months from IMM date)."""
        start = sched.get_imm_date(self.imm_month, self.imm_year)
        # End is the next IMM date
        end_month = self.imm_month + 3
        end_year = self.imm_year
        if end_month > 12:
            end_month -= 12
            end_year += 1
        end = sched.get_imm_date(end_month, end_year)
        return (start, end)

    @computed
    def evaluation_date(self) -> datetime.date:
        if self.evaluation_date_override:
            return self.evaluation_date_override
        return datetime.date.today()

    @computed_expr
    def implied_rate(self) -> Expr:
        """The daily compounded rate over the reference period."""
        if not self.discount_curve:
            return 0.0
            
        start, end = self.reference_period
        return sched.compounded_rate(
            start, end, 
            evaluation_date=self.evaluation_date, 
            fixings=self.fixings, 
            discount_curve=self.discount_curve,
            telescopic=True,
            day_counter=sched.DayCountConvention.Act360
        ) + self.convexity_adj

    @computed_expr
    def npv(self) -> Expr:
        """NPV of the future relative to a 'target_price' (e.g. 96.50).
        
        This is for use in a CurveFitter: solve Price(curve) = MarketPrice.
        Target NPV = MarketPrice - (100 - implied_rate * 100)
        """
        # We need a price input, but Fitter usually assumes target.npv = 0.
        # So we'll define a target_price externally and subtract.
        # But for simplicity, we provide a property.
        return 0.0 # Placeholder for solver integration

    @computed_expr
    def price(self) -> Expr:
        """Future Price = 100 - Implied Rate * 100."""
        return 100.0 - (self.implied_rate * 100.0)
