from typing import Any
import pandas as pd
import numpy as np
from pricing.engines.skinny_base import SkinnyEngineBase

class SkinnyEngineNumPy(SkinnyEngineBase):
    """Execute components via NumPy vectorized arrays."""
    def evaluate(self, portfolio: Any, ctx: dict, risk_method: Any = None, **kwargs) -> pd.DataFrame:
        comps = self.to_components(portfolio, per_swap=True, risk_method=risk_method, **kwargs)
        df = pd.DataFrame(comps)
        return self._evaluate_internal(df, ctx)

    def _evaluate_internal(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        type_map = {bf.component_type: bf for bf in self.extractor.registry.values()}
        for bf in type_map.values():
            if not hasattr(bf, "_compiled_np"):
                py_code = bf.dh_template.replace("Math.pow", "np.power").replace("Math.exp", "np.exp")
                bf._compiled_np = compile(py_code, f"<basis_{bf.component_type}>", "eval")
        
        groups = df.groupby(["Component_Class", "Component_Type"])
        num_groups = len(groups)
        print(f"    [NumPy] Processing {num_groups} vectorized basis groups...")
        
        df["Evaluated_Value"] = 0.0
        for (c_class, c_type), group in groups:
            bf = type_map[c_type]
            # Vectorized mapping: p1..pn and X1..Xn are now arrays across all swaps in the group
            params = {f"p{j}": group[f"p{j}"].values for j in range(1, bf.num_params + 1)}
            vars = {f"X{j}": np.array([ctx.get(k, 0.04) for k in group[f"X{j}"]]) for j in range(1, bf.num_vars + 1)}
            ns = {"np": np, **params, **vars}
            
            vals = eval(bf._compiled_np, {"np": np}, ns) * group["Weight"].values
            df.loc[group.index, "Evaluated_Value"] = vals
            
        # Aggregate back to Swap_Id and Metric
        res_df = df.groupby(["Swap_Id", "Component_Class"])["Evaluated_Value"].sum().reset_index()
        res_df.columns = ["Swap_Id", "Metric", "Value"]
        return res_df.pivot(index="Swap_Id", columns="Metric", values="Value")
