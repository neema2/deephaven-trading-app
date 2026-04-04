"""
jump_curve.py — Additive Step-Function Spreads (MPC Meetings, Turn-of-Year)

A decorator/layer that adds discrete steps or jumps to a base yield curve's
forward rates. This is used to model rate hikes at meeting dates or
liquidity spreads over the New Year (Turn-of-Year knots).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

from store import Storable
from reactive.computed import computed
from reactive.computed_expr import computed_expr
from reactive.expr import Expr, If, Const
from marketmodel.curve_base import CurveBase


@dataclass
class RateJump(Storable):
    """A discrete jump in the forward rate starting at a specific date."""
    name: str
    effective_date: datetime.date
    spread: float  # e.g., 0.0025 for a 25bps hike or seasonal spread
    tenor_years: float = 0.0 # Cache for fast lookup

    def __post_init__(self):
        # We assume eval_date is start of year for simplicity in this demo
        # or we'll pass an eval_date to a setup method
        pass


@dataclass
class JumpCurveLayer(CurveBase):
    """Curve layer that adds discrete forward rate jumps to a base curve.
    
    Forward(t) = BaseForward(t) + Σ Jump_i * 1(t > jump_t_i)
    Zero(T) = BaseZero(T) + Σ Jump_i * (T - jump_t_i)/T * 1(T > jump_t_i)
    DF(T) = BaseDF(T) * exp(-Σ Jump_i * (T - jump_t_i) * 1(T > jump_t_i))
    """
    base_curve: CurveBase
    jumps: list[RateJump] = field(default_factory=list)
    evaluation_date: datetime.date = field(default_factory=datetime.date.today)

    def _t(self, d: datetime.date) -> float:
        return (d - self.evaluation_date).days / 365.2425

    @computed_expr
    def df(self, t: float) -> Expr:
        """Discount factor including integrated step-function jumps."""
        from reactive.expr import Exp, Sum
        
        base_df = self.base_curve.df(t)
        
        jump_impacts = []
        for jump in self.jumps:
            # Impact on log-discount: -spread * max(0, t - jump_t)
            jt = self._t(jump.effective_date)
            # using If(t > jt, -(t-jt)*spread, 0)
            impact = If(t > jt, -(t - jt) * jump.spread, 0.0)
            jump_impacts.append(impact)
            
        if not jump_impacts:
            return base_df
            
        return base_df * Exp(Sum(jump_impacts))

    @computed_expr
    def fwd(self, t1: float, t2: float) -> Expr:
        """Forward rate including discrete jumps."""
        from reactive.expr import Sum
        
        base_fwd = self.base_curve.fwd(t1, t2)
        
        # Continuous fwd = (log(DF1) - log(DF2)) / (t2 - t1)
        # For a single jump at tj between t1 and t2:
        # contribution is spread * (t2 - max(t1, tj)) / (t2 - t1)
        
        jump_contributions = []
        dt = t2 - t1
        for jump in self.jumps:
            jt = self._t(jump.effective_date)
            # if t2 > jt:
            #    overlap = t2 - max(t1, jt)
            #    contribution = spread * overlap / dt
            overlap = If(t2 > jt, t2 - If(t1 > jt, t1, jt), 0.0)
            jump_contributions.append(jump.spread * (overlap / dt))
            
        if not jump_contributions:
            return base_fwd
            
        return base_fwd + Sum(jump_contributions)

    @property
    def pillar_names(self) -> list[str]:
        return self.base_curve.pillar_names

    def pillar_context(self) -> dict[str, Any]:
        return self.base_curve.pillar_context()
