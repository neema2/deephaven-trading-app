#!/usr/bin/env python3
"""
Demo: Interest Rate Swap — Fully Reactive Grid via Market Data Hub
===================================================================

Consumes USD OIS swap rate ticks from the Market Data Server (port 8000).
Prices a 7Y IRS using IRSwapFixedFloat (full coupon scheduling) against
an IntegratedShortRateCurve (C²-smooth, EXP-based discount factors).

Usage:
  python3 demo_ir_swap.py
  Open http://localhost:10000 in your browser

Required ports:
  10000 — Deephaven streaming server (Web IDE)
    8000 — Market data WebSocket server
"""

import asyncio
import json
import logging
import os
import sys
import threading
from collections import deque
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

# ── 0. Pre-flight: check required ports are free ─────────────────────────
# Done before importing streaming.admin so stale processes are identified
# immediately with actionable kill advice rather than an opaque bind error.
from streaming.port_check import preflight_check

preflight_check({
    10000: "Deephaven Web IDE / streaming server",
    8000:  "Market data WebSocket server",
})


# ── 1. Start streaming server ──────────────────────────────────────────────
print("=" * 70)
print("  Interest Rate Swap — Reactive Ticking Demo")
print("=" * 70)
print()
print("  Starting streaming server...")

from streaming.admin import StreamingServer

streaming_server = StreamingServer(port=10000)
streaming_server.start()
streaming_server.register_alias("demo")
print(f"  Streaming server started on {streaming_server.url}")

# Activate ticking tables — materialises TickingTable + LiveTable for all
# @ticking-decorated classes. Until this call, @ticking is pure metadata
# and .tick() is a no-op (compute-library mode).
from streaming import activate
activate()
print("  Streaming tables activated.")

print("  Starting market data server...")
from marketdata.admin import MarketDataServer

_md_server = MarketDataServer(port=8000)
asyncio.run(_md_server.start())
print(f"  Market data server started on port {_md_server.port}")

# ── 2. Import instrument models from instruments/ ─────────────────────────
from streaming import agg, flush, get_active_tables, get_tables, clear_stale_tables

from pricing.marketmodels.ir_curve_yield import YieldCurvePoint, LinearTermDiscountCurve
from pricing.marketmodels.ir_curve_fitter import CurveFitter
from pricing.marketmodels.symbols import fit_symbol, tenor_name
from pricing.marketmodels.ir_curve_swap import SwapQuote, SwapQuoteRisk
from pricing.instruments.ir_swap_fixed_float import IRSwapFixedFloat

# ── 3. Build reactive objects — cross-entity refs wired at construction ───
print("  Building reactive objects...")

# 1. Market Quotes (Inputs)
swap_quotes = {
    "IR_USD_OIS_QUOTE.1Y":  SwapQuote(symbol="IR_USD_OIS_QUOTE.1Y",  tenor=1.0,  rate=0.011),
    "IR_USD_OIS_QUOTE.2Y":  SwapQuote(symbol="IR_USD_OIS_QUOTE.2Y",  tenor=2.0,  rate=0.012),
    "IR_USD_OIS_QUOTE.5Y":  SwapQuote(symbol="IR_USD_OIS_QUOTE.5Y",  tenor=5.0,  rate=0.015),
    "IR_USD_OIS_QUOTE.10Y": SwapQuote(symbol="IR_USD_OIS_QUOTE.10Y", tenor=10.0, rate=0.020),
    "IR_USD_OIS_QUOTE.20Y": SwapQuote(symbol="IR_USD_OIS_QUOTE.20Y", tenor=20.0, rate=0.030),
}

# 2. Linear Term Discount Curve pillars
curve_points = {}
for q_sym, q in swap_quotes.items():
    t_label = tenor_name(q.tenor)
    label = fit_symbol("USD", t_label)
    curve_points[label] = YieldCurvePoint(
        name=label, symbol=label, tenor_years=q.tenor,
        currency="USD", quote_ref=q, is_fitted=True,
    )

# 3. Integrated short rate curve (C² smooth, EXP-based discount factors)
# 3. Linear Term Discount Curve builds Expr trees that compile directly TO_SQL.
usd_curve = LinearTermDiscountCurve(
    name="USD_OIS", currency="USD",
    points=list(curve_points.values())
)

# 4. Target Swaps for the Fitter — use the full IRSwapFixedFloat pricer.
# No more approx shortcut: these use the real coupon scheduling so the Jacobian
# that the fitter solves is consistent with portfolio pricing.
target_swaps = []
for q in swap_quotes.values():
    target_swaps.append(IRSwapFixedFloat(
        symbol=f"FIT.{q.symbol}", notional=50_000_000,
        fixed_rate=q.rate, tenor_years=q.tenor,
        discount_curve=usd_curve, projection_curve=usd_curve,
        is_target=True,
    ))

# 5. Global Fitter — the writer that publishes fitted rates
fitter = CurveFitter(
    name="USD_OIS_FITTER", currency="USD",
    target_swaps=target_swaps,
    quotes=list(swap_quotes.values()),
    curve=usd_curve,
    points=list(curve_points.values())
)
# Initial solve
print("  Solving initial curve...")
fitter.solve()
print("  Curve solved.")


# 6. Portfolio — one 7Y IRS to test interpolation risk split across 5Y/10Y pillars
swaps = {
    "USD-7Y": IRSwapFixedFloat(
        symbol="USD-7Y", notional=100_000_000, fixed_rate=0.0175,
        tenor_years=7.0, 
        discount_curve=usd_curve, projection_curve=usd_curve,
    )
}

if hasattr(swaps["USD-7Y"], "tick"):
    swaps["USD-7Y"].tick()


# ── 4. Publish @ticking tables to DH global scope ────────────────────────
# Step 1: clear any stale tables from a previous session on this DH server.
# When the Docker container persists between restarts, old table names linger
# in the query scope. clear_stale_tables() removes any @ticking-registered
# name that has write_count==0 this run (i.e. an imported-but-unused class).
cleared = clear_stale_tables()
if cleared:
    print(f"  Cleared {len(cleared)} stale table(s) from previous session.")

# Step 2: publish only the tables written in this run.
tables = get_active_tables()

# Aggregates and curated views
swap_summary = IRSwapFixedFloat._ticking_live.agg_by([  # type: ignore[attr-defined]
    agg.sum(["TotalNPV=npv", "TotalDV01=dv01"]),
    agg.count("NumSwaps"),
    agg.avg(["AvgNPV=npv"]),
], by=[])
tables["swap_summary"]           = swap_summary
tables["swap_risk_ladder"]       = SwapQuoteRisk._ticking_live._derive_remote(
    "update_view('equiv_notional = equiv_notional / 1000000').format_columns(['risk = Decimal(\"0.00\")', 'equiv_notional = Decimal(\"0.0 M\")'])"
)
tables["interest_rate_swap_live"] = IRSwapFixedFloat._ticking_live   # type: ignore[attr-defined]
tables["yield_curve_live"]       = usd_curve._ticking_live           # type: ignore[attr-defined]

print(f"  Publishing {len(tables)} tables to Deephaven...")
for name, tbl in tables.items():
    tbl.publish(name)
    print(f"    [DH] Published table: {name}")

# All initial DH pushes happened automatically via @effects during construction.
# Just flush the DH update graph once.
flush()

print(f"  Built: {len(swap_quotes)} OIS quotes, {len(curve_points)} knots, "
      f"1 LinearTermDiscountCurve, {len(target_swaps)} fitter swaps, {len(swaps)} portfolio swaps")
print("  All initial state pushed to DH via @effect (no manual push needed)")


# ── 5. Market Data Server WebSocket consumer ─────────────────────────────

MD_SERVER_URL = "ws://localhost:8000/md/subscribe"
RECONNECT_DELAY = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_log = logging.getLogger("irs-md-consumer")


async def _consume_and_publish():
    """Connect to the Market Data Server, consume OIS swap ticks.

    Each OIS tick triggers the FULL reactive cascade via a single batch_update():
      batch_update(rate=...)
        → @effect on_rate        → DH write
        → @computed curve rate   → @effect → DH write
        → @computed float_rate   → @computed npv → @effect → DH write
        → @computed total_npv    → @effect → DH write

    No imperative push functions — every DH write is an @effect side-effect.
    """
    import websockets

    tick_count = 0

    while True:
        try:
            _log.info("Connecting to Market Data Server at %s ...", MD_SERVER_URL)
            async with websockets.connect(MD_SERVER_URL) as ws:
                # Subscribe to swap ticks only
                await ws.send(json.dumps({"types": ["swap"]}))
                _log.info("Connected — consuming USD OIS swap ticks, full reactive cascade active")

                # Publish initial CurveTicks back to hub
                for label, pt in curve_points.items():
                    ct = {
                        "type": "curve", "symbol": pt.symbol, "label": label,
                        "tenor_years": pt.tenor_years,
                        "rate": pt.rate,
                        "discount_factor": pt.discount_factor,
                        "currency": pt.currency,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await ws.send(json.dumps(ct))

                # Publish initial JacobianTicks back to hub
                for entry in usd_curve.jacobian:
                    jt = {
                        "type": "jacobian", "symbol": entry.symbol,
                        "output_tenor": entry.output_tenor,
                        "input_tenor": entry.input_tenor,
                        "value": entry.value,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await ws.send(json.dumps(jt))

                async for msg_str in ws:
                    tick = json.loads(msg_str)
                    if tick.get("type") == "batch":
                        for t in tick.get("ticks", []):
                            if t.get("type") != "swap":
                                continue
                            sq = swap_quotes.get(t["symbol"])
                            if sq:
                                sq.batch_update(rate=t["rate"])
                    else:
                        if tick.get("type") != "swap":
                            continue

                        sq = swap_quotes.get(tick["symbol"])
                        if sq is None:
                            continue

                        # ── THE ONLY IMPERATIVE CALL ──────────────────────
                        # Everything else is reactive: @computed recalcs +
                        # @effect DH pushes all fire inside batch_update().
                        sq.batch_update(rate=tick["rate"])
                    
                    # Trigger the global fitting 
                    fitter.solve()
                    
                    # === Calculate and publish Risk Ladder ===
                    try:
                        from pricing.instruments.portfolio import Portfolio
                        from pricing.engines.python_expr import PythonEngineExpr
                        from pricing.marketmodels.ir_curve_swap import SwapQuoteRisk
                        
                        port = Portfolio()
                        for s_name, s in swaps.items():
                            port.add_instrument(s_name, s)
                            
                        engine = PythonEngineExpr()
                        ctx = port.pillar_context()
                        total_risk = engine.total_risk(port, ctx)
                        
                        # Create map of target benchmark swaps to compute exact DV01 per unit notional
                        target_swap_map = {ts.symbol.replace("FIT.", ""): ts for ts in target_swaps}
                        
                        for q_sym, quote in swap_quotes.items():
                            quote_risk = 0.0
                            for entry in usd_curve.jacobian:
                                if entry.quote_symbol == quote.symbol:
                                    pillar_name = next((pt.name for pt in curve_points.values() if abs(pt.tenor_years - entry.output_tenor) < 1e-4), None)
                                    if pillar_name and pillar_name in total_risk:
                                        quote_risk += total_risk[pillar_name] * entry.value
                            
                            # Convert risk from ∂Portfolio/∂Rate to DV01 (dollar value of 1bp move)
                            quote_dv01 = quote_risk * 0.0001
                            
                            if abs(quote_dv01) > 1e-8:
                                ts = target_swap_map.get(quote.symbol)
                                unit_dv01 = float(ts.dv01) / ts.notional if ts and ts.notional else 0.0
                                
                                sqr = SwapQuoteRisk(
                                    symbol=f"PORT.{quote.symbol}",
                                    portfolio="PORT",
                                    quote=quote.symbol,
                                    risk=quote_dv01,
                                    equiv_notional=quote_dv01 / unit_dv01 if unit_dv01 else 0.0
                                )
                                sqr.tick()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error calculating risk ladder: {e}")

                    flush()

                    tick_count += 1
                    if tick.get("type") == "batch":
                        print(f"  [Tick] Processed batch of {len(tick.get('ticks', []))} ticks #{tick_count}")
                    else:
                        print(f"  [Tick] Processed {tick['symbol']} #{tick_count}")
                    # Publish derived CurveTicks back to the hub
                    for label, pt in curve_points.items():
                        ct = {
                            "type": "curve", "symbol": pt.symbol, "label": label,
                            "tenor_years": pt.tenor_years,
                            "rate": pt.rate,
                            "discount_factor": pt.discount_factor,
                            "currency": pt.currency,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        await ws.send(json.dumps(ct))

                    # Publish derived JacobianTicks back to the hub
                    for entry in usd_curve.jacobian:
                        jt = {
                            "type": "jacobian", "symbol": entry.symbol,
                            "output_tenor": entry.output_tenor,
                            "input_tenor": entry.input_tenor,
                            "value": entry.value,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        await ws.send(json.dumps(jt))

                    # Print summary every tick (was every 5th)
                    if True:
                        # Use the new structured labels for lookup
                        lbl_5y = fit_symbol("USD", "5Y")
                        lbl_10y = fit_symbol("USD", "10Y")
                        
                        usd_5y = curve_points[lbl_5y].rate * 100
                        usd_10y = curve_points[lbl_10y].rate * 100

                        print("\n" + "-"*60)
                        if tick.get("type") == "batch":
                            print(f"  SUMMARY TICK #{tick_count} | Batch of {len(tick.get('ticks', []))} ticks")
                        else:
                            print(f"  SUMMARY TICK #{tick_count} | {tick['symbol']} {tick['rate']:.4%}")
                        print(f"  USD 5Y: {usd_5y:.4f}%  |  USD 10Y: {usd_10y:.4f}%")
                        print("-"*60)
                        
                        try:
                            risk_df = tables["swap_risk_ladder"].snapshot()
                            swap_df = tables["interest_rate_swap_live"].snapshot()
                            
                            print("\n  [Live Swap NPV Breakdown]")
                            cols = ["symbol", "fixed_leg_pv", "float_leg_pv", "npv", "dv01"]
                            print(swap_df[cols].to_string(index=False))

                            print("\n  [Live Risk Ladder Snapshot]")
                            if not risk_df.empty:
                                ladder_cols = ["quote", "risk", "equiv_notional"]
                                active_risk = risk_df[risk_df['risk'] != 0].copy()
                                active_risk['equiv_notional'] = active_risk['equiv_notional'].apply(lambda x: f"{x:.1f}M")
                                print(active_risk[ladder_cols].to_string(index=False))
                            else:
                                print("  (empty ladder)")
                            print("-" * 60 + "\n")
                            sys.stdout.flush()
                        except Exception as snapshot_err:
                            print(f"  (Failed to take console snapshot: {snapshot_err})")

        except Exception as e:
            _log.warning(
                "Market Data connection lost (%s). Retrying in %ds...",
                e, RECONNECT_DELAY,
            )
            await asyncio.sleep(RECONNECT_DELAY)


def _start_md_consumer():
    """Run the market data consumer in a background thread with its own loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_consume_and_publish())


print()
print("=" * 70)
print("  DEMO READY — Fully Reactive IRS Grid via USD OIS Market Data")
print("  Web UI:  http://localhost:10000")
print()
print(f"  Built: {len(swap_quotes)} OIS quotes, {len(curve_points)} knots, "
      f"1 LinearTermDiscountCurve, {len(target_swaps)} fitter swaps, {len(swaps)} portfolio swaps")
print("  All initial state pushed to DH via @effect (no manual push needed)")
print()
print("  Published tables (open in DH web IDE):")
print("    swap_quote_live         — USD OIS par rates (from market data server)")
print("    yield_curve_point_live  — curve knots (Zero Rates)")
print("    interest_rate_swap_live — IRS pricing: NPV, DV01 (@computed, basic IRS)")
print("    swap_summary            — aggregate NPV + DV01 (ticking summary)")
print("    swap_risk_ladder        — portfolio risk ladder: ∂Portfolio / ∂Quote")
print("    yield_curve_live        — LinearTermDiscountCurve (Linear Interpolation)")
print()
print("  Reactive chain (all from one batch_update):")
print("    OIS rate → @computed curve rate → @computed float_rate")
print("    → @computed npv → @computed total_npv")
print("    Every @effect fires automatically: DH push")
print()
print("  Press Ctrl+C to stop.")
print("=" * 70)
print()

# Start the WS consumer in a daemon thread
_md_thread = threading.Thread(
    target=_start_md_consumer, daemon=True, name="irs-md-consumer",
)
_md_thread.start()

try:
    # Keep main thread alive
    while True:
        import time
        time.sleep(1)
except KeyboardInterrupt:
    print("\n  Shutting down...")
    asyncio.run(_md_server.stop())
    print("  Done!")
