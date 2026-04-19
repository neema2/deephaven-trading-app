from typing import Any
from pricing.engines.skinny_base import SkinnyEngineBase

class SkinnyEngineDuckDB(SkinnyEngineBase):
    """Project components into DuckDB SQL."""
    def generate_sql(self, portfolio: Any, per_swap=True, risk_method: Any = None, **kwargs) -> str:
        comps = self.to_components(portfolio, per_swap=per_swap, risk_method=risk_method, **kwargs)
        max_vars = max((bf.num_vars for bf in self.extractor.registry.values()), default=1)
        select_cols = []
        if per_swap: select_cols.append("c.Swap_Id")
        
        math_case_lines = []
        for bf in self.extractor.registry.values():
            s_expr = bf.sql_template.replace("^", "**")
            for j in range(1, bf.num_vars + 1):
                s_expr = s_expr.replace(f"X{j}", f"s{j}.Knot_Value")
            math_case_lines.append(f"                WHEN {bf.component_type} THEN {s_expr}")
        
        math_case = "CASE c.Component_Type\n" + "\n".join(math_case_lines) + "\n            END"
        select_cols.append(f"SUM(CASE WHEN c.Component_Class = 'NPV' THEN c.Weight * ({math_case}) ELSE 0.0 END) as NPV")
        
        for pillar in portfolio.pillar_names:
            select_cols.append(f"SUM(CASE WHEN c.Component_Class = 'dNPV_d{pillar}' THEN c.Weight * ({math_case}) ELSE 0.0 END) as \"{pillar}\"")
        
        group_by = "sc.Scenario_Id" + (", c.Swap_Id" if per_swap else "")
        joins = ["FROM t_components c", "    CROSS JOIN (SELECT DISTINCT Scenario_Id FROM t_scenarios) sc"]
        for j in range(1, max_vars + 1):
            joins.append(f"    LEFT JOIN t_scenarios s{j} ON c.X{j} = s{j}.Knot_Id AND sc.Scenario_Id = s{j}.Scenario_Id")
            
        return f"SELECT\n    {group_by},\n    {', '.join(select_cols)}\n{' '.join(joins)}\nGROUP BY {group_by}"
