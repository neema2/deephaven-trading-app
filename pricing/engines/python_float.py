from __future__ import annotations
from typing import Any
from pricing.engines.base import ExecutionEngine
from pricing.risk.firstorder_numeric import FirstOrderNumericalRisk

class PythonEngineFloat(ExecutionEngine):
    """Simple float-based execution (No Expr trees, just @computed).
    
    Ideal for debugging core instrument logic without symbolic overhead.
    """
    def npvs(self, portfolio: Any, ctx: dict) -> dict[str, float]:
        """Evaluation of NPVs using the live graph."""
        # 1. Apply ctx to graph
        points_map = portfolio.pillar_points()
        for k, v in ctx.items():
            if k in points_map:
                points_map[k].fitted_rate = float(v)
        
        # 2. Flush and read
        from streaming import flush
        flush()
        return {name: inst.npv for name, inst in portfolio._instruments.items()}

    def instrument_risk(self, portfolio: Any, ctx: dict, risk_method: Any = None, regex: str = None, **kwargs) -> dict[str, dict[str, float]]:
        """Evaluation of risk via numerical bump-and-reprice."""
        from pricing.risk.firstorder_analytic import FirstOrderAnalyticRisk
        if risk_method and issubclass(risk_method, FirstOrderAnalyticRisk):
            raise ValueError("PythonEngineFloat does not support Analytic risk. Use FirstOrderNumericalRisk.")
        
        method = risk_method or FirstOrderNumericalRisk
        calc = method(portfolio, regex=regex)
        return calc.instrument_risk(**kwargs)

    def total_risk(self, portfolio: Any, ctx: dict, risk_method: Any = None, regex: str = None, **kwargs) -> dict[str, float]:
        from pricing.risk.firstorder_analytic import FirstOrderAnalyticRisk
        if risk_method and issubclass(risk_method, FirstOrderAnalyticRisk):
            raise ValueError("PythonEngineFloat does not support Analytic risk. Use FirstOrderNumericalRisk.")

        method = risk_method or FirstOrderNumericalRisk
        calc = method(portfolio, regex=regex)
        return calc.evaluate(ctx, **kwargs)
