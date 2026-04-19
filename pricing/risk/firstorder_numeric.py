from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pricing.risk.base import FirstOrderRiskBase

if TYPE_CHECKING:
    from pricing.instruments.base import Instrument

class NumericalRiskBumper:
    """Core logic for finite-difference bumping.
    
    Returns the raw derivative (delta): dNPV / dRate.
    """
    def __init__(self, bump_size: float = 0.0001):
        self.bump_size = bump_size

    def _get_npv(self, instrument: Any) -> float:
        """Helper to get numeric NPV, evaluating if it's an Expr."""
        val = instrument.npv()
        if not isinstance(val, (float, int)):
            from reactive.expr import eval_cached
            # We use the instrument's current pillar context for evaluation
            ctx = instrument.pillar_context()
            return float(eval_cached(val, ctx))
        return float(val)

    def _bump_and_reprice(self, instrument: Any, pillar: Any, delta: float, **kwargs) -> float:
        """Bump a single pillar, flush the graph, and return instrument NPV."""
        original = pillar.fitted_rate
        pillar.fitted_rate = original + delta
        try:
            from streaming import flush
            flush()
            return self._get_npv(instrument)
        finally:
            pillar.fitted_rate = original
            from streaming import flush
            flush()

    def compute_forward(self, instrument: Any, pillar: Any, **kwargs) -> float:
        """Calculate forward difference derivative: (P_up - P_base) / bump_size."""
        base_npv = self._get_npv(instrument)
        p_up = self._bump_and_reprice(instrument, pillar, self.bump_size, **kwargs)
        return (p_up - base_npv) / self.bump_size

    def compute_central(self, instrument: Any, pillar: Any, **kwargs) -> float:
        """Calculate central difference derivative: (P_up - P_down) / (2 * bump_size)."""
        p_up = self._bump_and_reprice(instrument, pillar, self.bump_size, **kwargs)
        p_down = self._bump_and_reprice(instrument, pillar, -self.bump_size, **kwargs)
        return (p_up - p_down) / (2 * self.bump_size)

class Risk_IR_DV01(NumericalRiskBumper):
    """DV01 calculator for a specific interest rate swap and curve point."""
    def __init__(self, swap: Any, curve_point: Any, shock_bps: float = 1.0):
        super().__init__(bump_size=shock_bps * 0.0001)
        self.swap = swap
        self.curve_point = curve_point

    def evaluate(self, method: str = "forward", **kwargs) -> float:
        delta = self.compute_forward(self.swap, self.curve_point, **kwargs) if method == "forward" else \
                self.compute_central(self.swap, self.curve_point, **kwargs)
        return delta * 0.0001

class FirstOrderNumericalRisk(FirstOrderRiskBase, NumericalRiskBumper):
    """Numerical first-order risk helper. 
    
    Provides same interface as FirstOrderAnalyticRisk but uses bump-and-reprice.
    """
    def __init__(self, instrument: Instrument, regex: str = None, name: str = "Risk", bump_size: float = 0.0001):
        FirstOrderRiskBase.__init__(self, instrument, regex, name)
        NumericalRiskBumper.__init__(self, bump_size)

    def evaluate(self, ctx: dict = None, method: str = "forward", **kwargs) -> dict[str, float]:
        """Numerical evaluation. ctx is optionally applied to the pillars before bumping."""
        if ctx:
            points_map = self.instrument.pillar_points()
            for k, v in ctx.items():
                if k in points_map:
                    points_map[k].set_fitted_rate(float(v))
            from streaming import flush
            flush()
        return self.total_risk(method=method, **kwargs)

    def total_risk(self, method: str = "forward", **kwargs) -> dict[str, float]:
        """Total risk for the aggregate instrument: {pillar: delta}."""
        points_map = self.instrument.pillar_points()
        pillars = self._pillars()
        res = {}
        for p_name in pillars:
            p_obj = points_map.get(p_name)
            if not p_obj: continue
            res[p_name] = self.compute_forward(self.instrument, p_obj, **kwargs) if method == "forward" else \
                          self.compute_central(self.instrument, p_obj, **kwargs)
        return res

    def instrument_risk(self, method: str = "forward", **kwargs) -> dict[str, dict[str, float]]:
        """Per-instrument risk mapping."""
        instruments = self._get_instruments()
        res = {}
        for name, inst in instruments.items():
            calc = FirstOrderNumericalRisk(inst, regex=self.regex, bump_size=self.bump_size)
            res[name] = calc.total_risk(method=method, **kwargs)
        return res

    def jacobian(self, method: str = "forward", **kwargs) -> dict[str, dict[str, float]]:
        """Fitter Jacobian: ∂(npv / notional) / ∂pillar for each instrument."""
        instruments = self._get_instruments()
        result = {}
        for name, inst in instruments.items():
            notional = getattr(inst, "notional", getattr(inst, "leg1_notional", 1.0))
            calc = FirstOrderNumericalRisk(inst, regex=self.regex, bump_size=self.bump_size)
            risk = calc.total_risk(method=method, **kwargs)
            result[name] = {p: val / notional for p, val in risk.items()}
        return result
