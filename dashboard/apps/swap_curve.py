import asyncio
import logging
import threading
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend

from dashboard.core import Dashboard
from marketdata.bus import TickBus
from marketdata.client import MarketDataClient
from marketmodel.swap_curve import SwapQuote
from marketmodel.yield_curve import YieldCurvePoint, LinearTermDiscountCurve
from marketmodel.curve_fitter import CurveFitter
from instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox

logger = logging.getLogger(__name__)

# ── Market Data Consumer ───────────────────────────────────────────────────

def _start_md_consumer(state: 'SwapCurveState'):
    """Daemon thread to consume ticks from the MarketDataServer."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def _consume():
        client = MarketDataClient("ws://localhost:8000")
        async with client.connect() as conn:
            async for tick in conn.subscribe("IR_USD_OIS_QUOTE.*"):
                state.on_tick(tick)
    
    try:
        loop.run_until_complete(_consume())
    except Exception as e:
        logger.error(f"MD Consumer error: {e}")

# ── Business State ──────────────────────────────────────────────────────────

class SwapCurveState(param.Parameterized):
    """Reactive state for the Swap Curve app. Handles solver + DF snapshots."""
    
    last_tick_time = param.Date(default=None)
    tick_count = param.Integer(default=0)
    
    # DataFrames for UI
    quote_df = param.DataFrame()
    curve_df = param.DataFrame()
    forward_df = param.DataFrame()
    
    def __init__(self, **params):
        super().__init__(**params)
        self._lock = threading.Lock()
        self._setup_fitter()
        
    def _setup_fitter(self):
        """Build the curve, pillars, and fitter."""
        # 1. Curve & Points
        self.points = [
            YieldCurvePoint(name=f"USD_OIS_{t}Y", tenor_years=t, is_fitted=True)
            for t in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 30.0]
        ]
        self.curve = LinearTermDiscountCurve(name="USD_OIS", points=self.points)
        
        # 2. Quotes & Target Swaps
        self.quotes = {
            f"IR_USD_OIS_QUOTE.{int(t)}Y": SwapQuote(symbol=f"IR_USD_OIS_QUOTE.{int(t)}Y", tenor=t)
            for t in [1, 2, 3, 5, 7, 10, 12, 15, 20, 30]
        }
        for i, pt in enumerate(self.points):
            pt.quote_ref = self.quotes[f"IR_USD_OIS_QUOTE.{int(pt.tenor_years)}Y"]
            
        self.target_swaps = [
            IRSwapFixedFloatApprox(
                symbol=f"OIS_{int(t)}Y", 
                tenor_years=t, 
                curve=self.curve,
                notional=10e6
            ) for t in [1, 2, 3, 5, 7, 10, 12, 15, 20, 30]
        ]
        
        # 3. Fitter
        self.fitter = CurveFitter(
            name="USD_OIS_FITTER", 
            curve=self.curve, 
            points=self.points,
            quotes=list(self.quotes.values()),
            target_swaps=self.target_swaps
        )
        
    def on_tick(self, tick_data: dict):
        """Update quotes and trigger re-solve."""
        sym = tick_data.get("symbol")
        if sym in self.quotes:
            with self._lock:
                q = self.quotes[sym]
                q.rate = float(tick_data["rate"])
                q.bid = float(tick_data["bid"])
                q.ask = float(tick_data["ask"])
                
                self.fitter.solve()
                self.tick_count += 1
                self.last_tick_time = datetime.now(timezone.utc)
                self._update_dfs()

    def _update_dfs(self):
        """Generate snapshots for the UI components."""
        # 1. Quote DF
        q_rows = []
        for q in self.quotes.values():
            q_rows.append({
                "Symbol": q.symbol,
                "Tenor": f"{int(q.tenor)}Y",
                "Bid (%)": q.bid * 100,
                "Ask (%)": q.ask * 100,
                "Mid (%)": (q.bid + q.ask) * 50,
                "Market (%)": q.rate * 100,
            })
        self.quote_df = pd.DataFrame(q_rows)
        
        # 2. Curve DF (Pillars)
        c_rows = []
        for pt in self.points:
            c_rows.append({
                "Pillar": pt.name,
                "Tenor": pt.tenor_years,
                "Fitted Zero (%)": pt.fitted_rate * 100,
                "DF": pt.discount_factor
            })
        self.curve_df = pd.DataFrame(c_rows)
        
        # 3. Forward Rate Comparison (Monthly Knots)
        # 12 months * 30 years = 360 knots
        tenors = np.linspace(0.0833, 30.0, 360) 
        
        # Fitted curve forwards
        fitted_fwds = self.curve.fwd_array(tenors.tolist(), period=0.25)
        
        # "Original" forwards (piecewise flat from par quotes)
        # We simulate this by using the curve's .interp with raw quote rates
        # (Linear Zero interpolation from par quotes)
        orig_rates = [q.rate for q in self.quotes.values()]
        orig_tenors = [q.tenor for q in self.quotes.values()]
        from marketmodel.yield_curve import _interp
        orig_zeros = [_interp(orig_tenors, orig_rates, t) for t in tenors]
        
        # For simplicity in this demo, original forwards use a naive DF ratio
        orig_dfs = [(1.0 + r)**(-t) for r, t in zip(orig_zeros, tenors)]
        orig_dfs_plus = [(1.0 + _interp(orig_tenors, orig_rates, t+0.25))**(-(t+0.25)) for t in tenors]
        orig_fwds = [(s/e - 1.0)/0.25 for s, e in zip(orig_dfs, orig_dfs_plus)]

        self.forward_df = pd.DataFrame({
            "Tenor": tenors,
            "Fitted Forward (%)": np.array(fitted_fwds) * 100,
            "Original Forward (%)": np.array(orig_fwds) * 100
        })

# ── UI App ──────────────────────────────────────────────────────────────────

def create_app():
    state = SwapCurveState()
    
    # Start background consumer
    thread = threading.Thread(target=_start_md_consumer, args=(state,), daemon=True)
    thread.start()
    
    db = Dashboard("Swap Curve Monitor", subtitle="USD OIS • Live Fitting")
    
    # --- Page 1: Quote Monitor ---
    p1 = db.add_page("Quotes", icon="📈")
    
    # KPI Strip
    kpi_row = pn.Row(sizing_mode="stretch_width")
    for t in [1, 5, 10, 30]:
        kpi = pn.pane.HTML("", width=200, css_classes=["card", "kpi-card"])
        def _update_kpi(target=kpi, tenor=t):
            df = state.quote_df
            if df is not None and not df.empty:
                val = df[df["Tenor"] == f"{tenor}Y"]["Market (%)"].iloc[0]
                target.object = f'<div class="kpi-title">{tenor}Y Par Rate</div><div class="kpi-value">{val:.3f}%</div>'
        state.param.watch(lambda e: _update_kpi(), "quote_df")
        kpi_row.append(kpi)
    
    p1.add_widget(pn.pane.Markdown("### Real-time Quotes"), span=12)
    p1.add_table(state.param.quote_df, title="OIS Market Quotes", span=12, height=400)
    p1.components.append(pn.Column(kpi_row, sizing_mode="stretch_width"))

    # --- Page 2: Fitted Curve ---
    p2 = db.add_page("Analytics", icon="📐")
    
    # Forward Chart
    def _create_chart(df):
        if df is None or df.empty:
            return figure(title="Awaiting Data...")
        
        source = ColumnDataSource(df)
        p = figure(
            title="Forward Rate Structure (3M Fwds)", 
            height=450, 
            sizing_mode="stretch_width",
            x_axis_label="Tenor (Years)",
            y_axis_label="Rate (%)",
            tools="pan,wheel_zoom,reset,save"
        )
        
        l1 = p.line("Tenor", "Fitted Forward (%)", source=source, color="#3b82f6", line_width=3, legend_label="Fitted Forward")
        l2 = p.line("Tenor", "Original Forward (%)", source=source, color="#6b7280", line_width=1.5, line_dash="dashed", legend_label="Original Forward")
        
        p.add_tools(HoverTool(renderers=[l1], tooltips=[("Tenor", "@Tenor{0.0}Y"), ("Fitted", "@{Fitted Forward (%)}{0.000}%"), ("Original", "@{Original Forward (%)}{0.000}%")]))
        
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.6
        
        return p

    chart_pane = pn.pane.Bokeh(sizing_mode="stretch_both")
    def _update_chart(event):
        chart_pane.object = _create_chart(event.new)
    
    state.param.watch(_update_chart, "forward_df")
    
    p2.add_chart(chart_pane, title="Forward Rates: Fitted vs Observed", span=8, height=500)
    p2.add_table(state.param.curve_df, title="Curve Pillars", span=4, height=500)
    p2.add_table(state.param.forward_df, title="Forward Rate Data (Monthly Knots)", span=12, height=300)

    return db

if __name__ == "__main__":
    app = create_app()
    app.serve(port=8050)
