from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
from reactive.expr import Const
from reactive.basis_extractor import BasisExtractor
from pricing.engines.base import ExecutionEngine

class SkinnyEngineBase(ExecutionEngine):
    """Base engine for 'Skinny Table' component extraction.
    
    Splits portfolio expressions into (BasisFunction, Weights) components.
    Subclasses provide specific execution/projection for these components.
    """
    def __init__(self, extractor: BasisExtractor | None = None):
        self.extractor = extractor or BasisExtractor()

    def to_components(self, portfolio: Any, per_swap=True, risk_method: Any = None, **kwargs) -> list[dict]:
        """Extract the flat table of components."""
        agg_map: dict[tuple, float] = {}

        def _add_to_agg(trade_name, comp_class, expr):
            trade_components = self.extractor.extract_components(expr)
            for bf, vars, params, weight in trade_components:
                key_prefix = (trade_name if per_swap else None, comp_class, bf.component_type)
                full_key = key_prefix + tuple(vars) + tuple(params)
                agg_map[full_key] = agg_map.get(full_key, 0.0) + weight

        from pricing.risk.firstorder_analytic import FirstOrderAnalyticRisk
        from pricing.risk.firstorder_numeric import FirstOrderNumericalRisk
        
        # If numerical risk is requested here, it means we want the result of bumping 
        # but expressed in terms of components. This is only possible if we re-extract 
        # or if the risk helper returns Exprs. Numerical risk helper returns floats,
        # so we'd have to wrap them as Const Exprs if we wanted to stick to the component path.
        if risk_method and issubclass(risk_method, FirstOrderNumericalRisk):
            # For simplicity, we'll extract NPV components and then also 
            # add components for the numerical sensitivities as 'Const' weights.
            # (Warning: inefficient, better to just use Analytic for skinny)
            calc = risk_method(portfolio)
            risk_results = calc.instrument_risk(**kwargs)
            
            # 1. Add normal NPV components
            for name, expr in portfolio.npv_exprs.items():
                _add_to_agg(name, "NPV", expr)
            
            # 2. Add Numerical risk as static component weights
            for name, row in risk_results.items():
                for pillar, val in row.items():
                    _add_to_agg(name, f"dNPV_d{pillar}", Const(val))
        else:
            method = risk_method or FirstOrderAnalyticRisk
            calc = method(portfolio)
            risk_map = calc.instrument_risk(**kwargs)

            for name, expr in portfolio.npv_exprs.items():
                _add_to_agg(name, "NPV", expr)
            for name, row_exprs in risk_map.items():
                for pillar, expr in row_exprs.items():
                    if not (isinstance(expr, Const) and expr.value == 0.0):
                        _add_to_agg(name, f"dNPV_d{pillar}", expr)

        res = []
        for key, total_weight in agg_map.items():
            trade_name, comp_class, comp_type = key[:3]
            bf = self.extractor.registry_by_type[comp_type]
            row = {"Component_Class": comp_class, "Component_Type": comp_type, "Weight": total_weight}
            if per_swap: row["Swap_Id"] = trade_name
            for j, v in enumerate(key[3:3 + bf.num_vars]): row[f"X{j+1}"] = v
            for j, p in enumerate(key[3 + bf.num_vars:]): row[f"p{j+1}"] = p
            res.append(row)
        return res
