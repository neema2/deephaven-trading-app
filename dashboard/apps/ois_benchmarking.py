import datetime
import pandas as pd
import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

from marketmodel.yield_curve import LinearTermDiscountCurve, YieldCurvePoint, _interp
from marketmodel.jump_curve import JumpCurveLayer, RateJump
from instruments.ir_swap_fixed_ois import IRSwapFixedOIS
from instruments.ir_future_ois import IRFutureOIS

def create_bench_dashboard():
    pn.extension(sizing_mode="stretch_width")
    
    val_dt = datetime.date(2026, 1, 1)
    
    # --- 1. MOCK DATA: USD OIS ---
    tenors = [1, 2, 3, 5, 7, 10]
    quotes = [0.045, 0.042, 0.041, 0.040, 0.039, 0.038] # Par Rates
    
    # --- 2. MPC JUMPS (Hypothetical Meeting Dates) ---
    # We add 25bps at the next 3 meetings
    jumps_list = [
        RateJump("FOMC_Jan", datetime.date(2026, 1, 31), 0.0025),
        RateJump("FOMC_Mar", datetime.date(2026, 3, 20), 0.0025),
        RateJump("FOMC_May", datetime.date(2026, 5, 2), 0.0025),
        # Turn-of-Year Spread (usually negative spread over the year end)
        RateJump("ToY_2026", datetime.date(2026, 12, 31), -0.0050),
        RateJump("ToY_End", datetime.date(2027, 1, 1), 0.0050) # Pulse ends
    ]
    
    # --- 3. CURVE SETUP ---
    # Curve A: Smooth Linear Zero
    pillars_a = [YieldCurvePoint(name=f"P_{t}Y", tenor_years=t, fitted_rate=q, is_fitted=True) for t, q in zip(tenors, quotes)]
    curve_a = LinearTermDiscountCurve(name="Smooth", points=pillars_a)
    
    # Curve B: Linear Zero + MPC Jumps
    # Same base rates (in a real fitter, Curve B's base would be solved DIFFRENTLY)
    pillars_b = [YieldCurvePoint(name=f"PB_{t}Y", tenor_years=t, fitted_rate=q-0.001, is_fitted=True) for t, q in zip(tenors, quotes)]
    curve_base_b = LinearTermDiscountCurve(name="BaseB", points=pillars_b)
    curve_b = JumpCurveLayer(base_curve=curve_base_b, jumps=jumps_list, evaluation_date=val_dt)
    
    # --- 4. ANALYTICS ---
    plot_tenors = np.linspace(0.01, 2.0, 400) # Short end focus for jumps
    
    def get_fwd_data():
        fw_a = [float(curve_a.fwd(t, t+0.0833)) for t in plot_tenors] # 1M Fwds
        fw_b = [float(curve_b.fwd(t, t+0.0833)) for t in plot_tenors]
        
        # QuantLib Mock: Piecewise flat between quotes
        # For simplicity, we just jitter one of them
        ql_fwd = [fw_a[i] + 0.0005 * np.sin(t*5) for i, t in enumerate(plot_tenors)]
        
        return pd.DataFrame({
            "Tenor": plot_tenors,
            "Standard (%)": np.array(fw_a) * 100,
            "MPC_Jump (%)": np.array(fw_b) * 100,
            "QuantLib (%)": np.array(ql_fwd) * 100
        })

    df = get_fwd_data()
    source = ColumnDataSource(df)
    
    p = figure(
        title="OIS Forward Curve: Smoothing vs Discrete Jumps (MPC/ToY)", 
        height=600, sizing_mode="stretch_width",
        x_axis_label="Tenor (Years)", y_axis_label="Forward Rate (%)"
    )
    
    l1 = p.line("Tenor", "Standard (%)", source=source, color="#3b82f6", line_width=2.5, legend_label="Python: Smooth Linear")
    l2 = p.line("Tenor", "MPC_Jump (%)", source=source, color="#ef4444", line_width=3, legend_label="Python: MPC/ToY Jump Layer")
    l3 = p.line("Tenor", "QuantLib (%)", source=source, color="#6b7280", line_width=1.5, line_dash="dashed", legend_label="QuantLib (Piecewise Flat)")
    
    p.add_tools(HoverTool(renderers=[l2], tooltips=[("Tenor", "@Tenor{0.00}Y"), ("MPC Jump", "@{MPC_Jump (%)}{0.000}%")]))
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    
    # Dashboard Grid
    main = pn.Column(
        pn.pane.Markdown("# 📉 OIS Curve Benchmarking: Jumps & Steps"),
        pn.pane.Markdown("Showing the short-end (0-2Y) focus. Note the **Step Jumps** on MPC dates and the **Turn-of-Year Pulse** in the red curve."),
        pn.Row(
            pn.Card(p, title="Forward Rate Projections", sizing_mode="stretch_width", min_height=650),
            pn.Card(
                pn.widgets.DataFrame(df.head(20), name="Data Snippet"), 
                title="Numerical Values", width=400
            )
        )
    )
    
    return main

if __name__ == "__main__":
    app = create_bench_dashboard()
    app.show()
