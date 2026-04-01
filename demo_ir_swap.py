#!/usr/bin/env python3
"""
Demo: Interest Rate Swap — Fully Reactive Grid via Market Data Hub
===================================================================

Consumes FX ticks from the Market Data Server (port 8000).  Every
computation AND every Deephaven push is expressed as @computed or
@effect — the WS consumer loop is a single batch_update() call.

The reactive chain (triggered by each FX tick):
  FXSpot.batch_update(bid, ask)
    → @computed mid
    → @effect on_mid          → fx_writer.write_row(...)
    → @computed YieldCurvePoint.rate   (cross-entity: reads fx_ref.mid)
      → @computed discount_factor
      → @effect on_rate        → curve_writer.write_row(...) + queue CurveTick
    → @computed InterestRateSwap.float_rate  (cross-entity: reads curve_ref.rate)
      → @computed npv, dv01, pnl_status
      → @effect on_npv         → swap_writer.write_row(...)
    → @computed SwapPortfolio.total_npv      (cross-entity: reads swaps[].npv)
      → @effect on_total_npv   → portfolio_writer.write_row(...)

Usage:
  python3 demo_ir_swap.py
  Open http://localhost:10000 in your browser
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── 1. Start streaming server ──────────────────────────────────────────────
print("=" * 70)
print("  Interest Rate Swap — Reactive Ticking Demo")
print("=" * 70)
print()
print("  Starting streaming server...")

from streaming.admin import StreamingServer

streaming = StreamingServer(port=10000)
streaming.start()
streaming.register_alias("demo")
print(f"  Streaming server started on {streaming.url}")

print("  Starting market data server...")
from marketdata.admin import MarketDataServer

_md_server = MarketDataServer(port=8000)
asyncio.run(_md_server.start())
print(f"  Market data server started on port {_md_server.port}")

# ── 2. Build reactive graph using shared ir/ package ──────────────────────
from streaming import agg, flush, get_tables

from demo_ir_config import CURVE_CONFIGS, FX_CONFIG, SWAP_CONFIGS
from ir.consumer import MarketDataConsumer
from ir.graph import build_ir_graph
from ir.models import SwapPortfolio, curve_publish_queue

print("  Building reactive objects...")
graph = build_ir_graph(
    fx_config=FX_CONFIG,
    curve_configs=CURVE_CONFIGS,
    swap_configs=SWAP_CONFIGS,
)
fx_spots = graph["fx_spots"]
curve_points = graph["curve_points"]
swaps = graph["swaps"]

# Convenience accessors for summary printing
usd_curve_points = {k: v for k, v in curve_points.items() if k.startswith("USD")}
jpy_curve_points = {k: v for k, v in curve_points.items() if k.startswith("JPY")}

# Swap portfolios — @computed aggregates react to child swap changes
usd_swaps = [s for s in swaps.values() if s.currency == "USD"]
jpy_swaps = [s for s in swaps.values() if s.currency == "JPY"]

portfolios = {
    "ALL": SwapPortfolio(name="ALL", swaps=list(swaps.values())),
    "USD": SwapPortfolio(name="USD", swaps=usd_swaps),
    "JPY": SwapPortfolio(name="JPY", swaps=jpy_swaps),
}

# ── 3. Publish @ticking tables to DH global scope ────────────────────────
from ir.models import InterestRateSwap

tables = get_tables()

# Aggregates
swap_summary = InterestRateSwap._ticking_live.agg_by([  # type: ignore[attr-defined]
    agg.sum(["TotalNPV=npv", "TotalDV01=dv01"]),
    agg.count("NumSwaps"),
    agg.avg(["AvgNPV=npv"]),
])
tables["swap_summary"] = swap_summary

for name, tbl in tables.items():
    tbl.publish(name)

flush()

# Drain initial CurveTick publishes
_initial_curve_ticks = list(curve_publish_queue)
curve_publish_queue.clear()

print(f"  Built: {len(fx_spots)} FX spots, "
      f"{len(graph['all_curve_points'])} curve points, "
      f"{len(swaps)} swaps, {len(portfolios)} portfolios")
print("  All initial state pushed to DH via @effect (no manual push needed)")

# ── 4. Market Data consumer ──────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

tick_count = 0


def _on_tick(count: int) -> None:
    global tick_count
    tick_count = count
    flush()
    if count % 10 == 0:
        usdjpy = fx_spots["USD/JPY"].mid
        usd_5y = usd_curve_points["USD_5Y"].rate * 100
        jpy_10y = jpy_curve_points["JPY_10Y"].rate * 100
        p_all = portfolios["ALL"]
        print(
            f"  [{count:4d}] "
            f"USD/JPY {usdjpy:.2f}  |  "
            f"USD 5Y: {usd_5y:.2f}%  |  "
            f"JPY 10Y: {jpy_10y:.3f}%  |  "
            f"Portfolio NPV: ${p_all.total_npv:+,.0f}"
        )


consumer = MarketDataConsumer(
    fx_spots=fx_spots,
    on_tick=_on_tick,
    curve_publish_queue=curve_publish_queue,
    initial_curve_ticks=_initial_curve_ticks,
)

print()
print("=" * 70)
print("  DEMO READY — Fully Reactive IRS Grid via Market Data Hub")
print("  Web UI:  http://localhost:10000")
print()
print("  Published tables (open in DH web IDE):")
print("    fx_spot_live            — FX spot rates (from market data server)")
print("    yield_curve_point_live  — yield curve points (@computed from FX)")
print("    interest_rate_swap_live — IRS pricing: NPV, DV01, PnL (@computed cascade)")
print("    swap_summary            — aggregate NPV + DV01 (ticking)")
print("    swap_portfolio_live     — portfolio breakdown: ALL / USD / JPY (@computed)")
print()
print("  Raw (append-only) tables: fx_spot_raw, yield_curve_point_raw,")
print("    interest_rate_swap_raw, swap_portfolio_raw")
print()
print("  Reactive chain (all from one batch_update):")
print("    FX bid/ask → @computed mid → @computed rate → @computed float_rate")
print("    → @computed npv → @computed total_npv")
print("    Every @effect fires automatically: DH push + CurveTick publish")
print()
print("  Press Ctrl+C to stop.")
print("=" * 70)
print()

consumer.start_background(name="irs-md-consumer")

try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n  Shutting down...")
    asyncio.run(_md_server.stop())
    print("  Done!")
