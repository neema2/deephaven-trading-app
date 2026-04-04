"""
ir_swap_xccy_ois.py — Cross-Currency Overnight Indexed Basis Swap (XCCY OIS)

Implements a floating-floating cross-currency swap where both legs use 
daily compounding (OIS/RFR). Supports historical fixings and the 
telescopic property for forward projections.
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
from reactive.expr import diff, Expr
from streaming import ticking
import instruments.ir_scheduling as sched


@ticking(exclude={
    "leg1_discount_curve", "leg2_discount_curve", 
    "risk", "fixings1", "fixings2"
})
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class IRSwapXCCYOIS(Storable):
    """Cross-Currency OIS Basis Swap.
    
    Attributes
    ----------
    symbol: str
        Unique identifier.
    leg1_currency: str
        Currency of the first leg (e.g., "EUR").
    leg1_notional: float
        Notional amount in leg 1 currency.
    leg1_discount_curve: object
        Curve for discounting and OIS projection for Leg 1.
    leg1_index_name: Optional[str]
        Manual override for RFR index name (e.g., "ESTR").
    
    leg2_currency: str
        Currency of the second leg (e.g., "USD").
    leg2_notional: float
        Notional amount in leg 2 currency.
    leg2_discount_curve: object
        Curve for discounting and OIS projection for Leg 2.
    leg2_index_name: Optional[str]
        Manual override for RFR index name (e.g., "SOFR").
        
    basis_spread: float
        Spread added to Leg 1 floating rate (decimal, e.g. 0.0010 for 10bps).
    initial_fx: float
        FX rate used to convert Leg 1 to Leg 2 at initiation (Leg2 = Leg1 * FX).
    
    effective_date: datetime.date
    termination_date: datetime.date
    frequency_months: int = 12
    exchange_notional: bool = True
    
    fixings1: dict[datetime.date, float] = field(default_factory=dict, repr=False)
    fixings2: dict[datetime.date, float] = field(default_factory=dict, repr=False)
    
    evaluation_date_override: Optional[datetime.date] = None
    """
    __key__ = "symbol"
    
    symbol: str = ""
    
    # Leg 1
    leg1_currency: str = "EUR"
    leg1_notional: float = 0.0
    leg1_discount_curve: Any = field(default=None, repr=False)
    leg1_index_name: Optional[str] = None
    
    # Leg 2
    leg2_currency: str = "USD"
    leg2_notional: float = 0.0
    leg2_discount_curve: Any = field(default=None, repr=False)
    leg2_index_name: Optional[str] = None
    
    basis_spread: float = 0.0
    initial_fx: float = 1.0  # e.g. EURUSD = 1.08
    
    effective_date: Optional[datetime.date] = None
    termination_date: Optional[datetime.date] = None
    frequency_months: int = 12
    exchange_notional: bool = True
    side: str = "RECEIVER"  # RECEIVER means receive Leg 1, pay Leg 2
    
    fixings1: dict[datetime.date, float] = field(default_factory=dict, repr=False)
    fixings2: dict[datetime.date, float] = field(default_factory=dict, repr=False)
    
    evaluation_date_override: Optional[datetime.date] = None

    def __post_init__(self):
        # Auto-initialize notionals if only one is provided
        if self.leg2_notional == 0.0 and self.leg1_notional != 0.0:
            object.__setattr__(self, 'leg2_notional', self.leg1_notional * self.initial_fx)
        elif self.leg1_notional == 0.0 and self.leg2_notional != 0.0:
            object.__setattr__(self, 'leg1_notional', self.leg2_notional / self.initial_fx)

    @computed
    def rfr_indices(self) -> dict[str, str]:
        """Mapping table for RFR indices."""
        return {
            "USD": "SOFR", "EUR": "ESTR", "GBP": "SONIA", "AUD": "AONIA",
            "CAD": "CORRA", "SGD": "SORA", "SGP": "SORA", "CHF": "SARON", "JPY": "TONAR"
        }

    @computed
    def resolved_leg1_index(self) -> str:
        if self.leg1_index_name: return self.leg1_index_name
        return self.rfr_indices.get(self.leg1_currency.upper(), f"{self.leg1_currency.upper()}_OIS")

    @computed
    def resolved_leg2_index(self) -> str:
        if self.leg2_index_name: return self.leg2_index_name
        return self.rfr_indices.get(self.leg2_currency.upper(), f"{self.leg2_currency.upper()}_OIS")

    @computed
    def evaluation_date(self) -> datetime.date:
        if self.evaluation_date_override:
            return self.evaluation_date_override
        return datetime.date.today()

    @computed
    def schedule(self) -> list[datetime.date]:
        if not self.effective_date or not self.termination_date:
            return []
        # Usually XCCY swaps use Leg 1's calendar or a joint calendar
        return sched.swap_schedule(
            self.effective_date, self.termination_date, 
            freq_months=self.frequency_months, 
            currency=self.leg1_currency, 
            end_of_month=True
        )

    @property
    def pillar_names(self) -> list[str]:
        names = set()
        if hasattr(self.leg1_discount_curve, "pillar_names"):
            names.update(self.leg1_discount_curve.pillar_names)
        if hasattr(self.leg2_discount_curve, "pillar_names"):
            names.update(self.leg2_discount_curve.pillar_names)
        return sorted(list(names))

    def _tenor(self, date: datetime.date) -> float:
        return (date - self.evaluation_date).days / 365.2425

    def _calc_leg_pv(self, notional: float, curve: Any, fixings: dict, spread: float = 0.0) -> Expr:
        """Calculate the PV of a single OIS leg."""
        from reactive.expr import Sum
        if not curve or not self.schedule:
            return 0.0
            
        sch = self.schedule
        pvs = []
        # Accrual Terms
        for i in range(len(sch) - 1):
            start, end = sch[i], sch[i+1]
            tau = sched.year_fraction(start, end, sched.DayCountConvention.Act360)
            
            rate = sched.compounded_rate(
                start, end, 
                evaluation_date=self.evaluation_date, 
                fixings=fixings, 
                discount_curve=curve, 
                telescopic=True, 
                day_counter=sched.DayCountConvention.Act360
            )
            df_end = curve.df(self._tenor(end))
            pvs.append(notional * (rate + spread) * tau * df_end)
            
        # Notional Exchange
        if self.exchange_notional:
            # - Notional at start (df=1.0 assumed at settlement)
            # + Notional at maturity
            df_end = curve.df(self._tenor(sch[-1]))
            pvs.append(notional * (df_end - 1.0))
            
        return Sum(pvs)

    @computed_expr
    def leg1_pv(self) -> Expr:
        return self._calc_leg_pv(self.leg1_notional, self.leg1_discount_curve, self.fixings1, self.basis_spread)

    @computed_expr
    def leg2_pv(self) -> Expr:
        return self._calc_leg_pv(self.leg2_notional, self.leg2_discount_curve, self.fixings2)

    @computed_expr
    def npv(self) -> Expr:
        """NPV in Leg 2 Currency (usually USD)."""
        # NPV = Leg 1 * FX - Leg 2 (RECEIVER)
        leg1_val = self.leg1_pv() * self.initial_fx
        if self.side == "PAYER":
            return self.leg2_pv() - leg1_val
        return leg1_val - self.leg2_pv()

    @computed_expr
    def risk(self) -> dict[str, Expr]:
        expr = self.npv()
        if expr is None: return {}
        return {name: diff(expr, name) for name in self.pillar_names}

    def pillar_context(self) -> dict[str, Any]:
        ctx = {}
        if hasattr(self.leg1_discount_curve, 'pillar_context'):
            ctx.update(self.leg1_discount_curve.pillar_context())
        if hasattr(self.leg2_discount_curve, 'pillar_context'):
            ctx.update(self.leg2_discount_curve.pillar_context())
        return ctx
