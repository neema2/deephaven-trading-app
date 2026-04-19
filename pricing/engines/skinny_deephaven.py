from typing import Any
from pricing.engines.skinny_base import SkinnyEngineBase

class SkinnyEngineDeephaven(SkinnyEngineBase):
    """Project components into Deephaven streaming scripts."""
    def generate_script(self, portfolio: Any, per_swap=True) -> str:
        dh_lines = []
        for bf in self.extractor.registry.values():
            tmpl = bf.dh_template.replace("Math.pow", "pow").replace("Math.exp", "exp")
            for j in range(1, bf.num_vars + 1):
                tmpl = tmpl.replace(f"X{j}", f"X{j}_Val")
            dh_lines.append(f"Component_Type == {bf.component_type} ? {tmpl} :")
        full_ternary = " ".join(dh_lines) + " 0.0"

        # Note: This is a template script that assumes t_c (components) 
        # and t_p (pillars) are already defined in the Deephaven session.
        script = f"""
t_mapped = t_c.natural_join(t_p, on=['X1=Knot_Id'], joins=['X1_Val=Knot_Value'])
t_mapped = t_mapped.natural_join(t_p, on=['X2=Knot_Id'], joins=['X2_Val=Knot_Value'])
t_mapped = t_mapped.update(["X1_Val = (X1_Val == null) ? 0.04 : X1_Val", "X2_Val = (X2_Val == null) ? 0.04 : X2_Val"])

t_evaluated = t_mapped.update(["Out = (double)(Weight * ({full_ternary}))"])
t_filtered = t_evaluated.view(["Swap_Id", "Component_Class", "Out"])

t_npv_res = t_filtered.where(["Component_Class == `NPV`"]).agg_by([agg.sum_("Out")], ["Swap_Id"])
t_risk_swap_res = t_filtered.agg_by([agg.sum_("Out")], ["Swap_Id", "Component_Class"])
t_risk_total_res = t_filtered.agg_by([agg.sum_("Out")], ["Component_Class"])
"""
        return script
