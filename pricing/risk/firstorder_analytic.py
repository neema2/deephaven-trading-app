from __future__ import annotations
from typing import Any, TYPE_CHECKING
from reactive.expr import diff, eval_cached, Const
from pricing.risk.base import FirstOrderRiskBase

if TYPE_CHECKING:
    from pricing.instruments.base import Instrument

class FirstOrderAnalyticRisk(FirstOrderRiskBase):
    """Calculates first order sensitivities (e.g., Delta, PV01, Rho) via symbolic differentiation.
    
    Returns symbolic expression trees (Expr) for differentiation.
    """
    def __init__(self, instrument: Instrument, regex: str = None, name: str = "Risk"):
        super().__init__(instrument, regex, name)
        self._memo: dict = {} # Shared differentiation memo

    def total_risk(self) -> dict[str, Any]:
        """Builds the symbolic expression tree for each relevant pillar: ∂npv / ∂pillar."""
        npv_expr = self.instrument.npv()
        pillars = self._pillars()
        return {
            p: diff(npv_expr, p, _memo=self._memo)
            for p in pillars
        }

    def instrument_risk(self) -> dict[str, dict[str, Any]]:
        """Per-instrument risk: {inst_name: {pillar: ∂npv/∂pillar Expr}}."""
        instruments = self._get_instruments()
        res = {}
        for name, inst in instruments.items():
            calc = FirstOrderAnalyticRisk(inst, regex=self.regex)
            res[name] = calc.total_risk()
        return res

    def jacobian(self) -> dict[str, dict[str, Any]]:
        """Alias for instrument_risk, used by the solver.
        
        The solver now expects unscaled derivatives (∂npv/∂pillar) because
        it uses a unit-notional 'fit_basket' for residual calculation.
        """
        return self.instrument_risk()

    def evaluate(self, ctx: dict) -> dict[str, float]:
        """Evaluates the sensitivities against the provided market context."""
        exprs = self.total_risk()
        cache: dict = {} # Local evaluation cache
        return {
            p: eval_cached(expr, ctx, _cache=cache)
            for p, expr in exprs.items()
        }
