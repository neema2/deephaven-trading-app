from __future__ import annotations
from typing import Any
from reactive.expr import eval_cached, diff
from pricing.engines.base import ExecutionEngine
from pricing.risk.firstorder_analytic import FirstOrderAnalyticRisk

class PythonEngineExpr(ExecutionEngine):
    """Native Python execution using symbolic expression evaluation."""
    def npvs(self, portfolio: Any, ctx: dict) -> dict[str, float]:
        """Evaluate all NPVs."""
        return {name: eval_cached(expr, ctx) for name, expr in portfolio.npv_exprs.items()}

    def residuals(self, portfolio: Any, ctx: dict) -> dict[str, float]:
        """Evaluate residual vector (what the fitter minimizes)."""
        return self.npvs(portfolio, ctx)

    def instrument_risk(self, portfolio: Any, ctx: dict, risk_method: Any = None, regex: str = None, **kwargs) -> dict[str, dict[str, float]]:
        """Evaluate the risk per instrument."""
        method = risk_method or FirstOrderAnalyticRisk
        calc = method(portfolio, regex=regex)
        
        # Unified evaluate call (returns floats)
        res = calc.instrument_risk(**kwargs)
        
        processed = {}
        cache: dict = {}
        for name, row in res.items():
            processed[name] = {
                p: (eval_cached(val, ctx, _cache=cache) if not isinstance(val, (float, int)) else val)
                for p, val in row.items()
            }
        return processed

    def total_risk(self, portfolio: Any, ctx: dict, risk_method: Any = None, regex: str = None, **kwargs) -> dict[str, float]:
        """Aggregate ∂(Σnpv)/∂pillar across all pricing.instruments."""
        method = risk_method or FirstOrderAnalyticRisk
        calc = method(portfolio, regex=regex)
        return calc.evaluate(ctx, **kwargs)
