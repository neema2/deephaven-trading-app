#!/usr/bin/env python3
"""
Demo: IR Risk — Live DV01 Ticking Grid via Market Data Hub
============================================================

Consumes FX ticks from the Market Data Server (port 8000).  On each
tick the reactive cascade updates curve rates and swap NPVs, then
``Risk_IR_DV01`` recomputes DV01 via bump-and-reprice for every swap
and pushes the results to a ticking Deephaven table.

``Risk_IR_DV01`` supports two finite-difference methods:

  * **Central difference** — bump ±shock, DV01 = (P_up − P_down) / (2 × shock_bps)
  * **Forward difference** — bump +shock,  DV01 = (P_up − P_base) / shock_bps

Usage:
  python3 demo_ir_risk.py
  Open http://localhost:10000 in your browser
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── 1. Start streaming server ──────────────────────────────────────────────
print("=" * 70)
print("  IR Risk — Live DV01 Ticking Grid via Market Data Hub")
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
from dataclasses import dataclass

from reactive.computed import effect
from streaming import agg, flush, get_tables, ticking
from store import Storable

from demo_ir_config import CURVE_CONFIGS, FX_CONFIG, SWAP_CONFIGS
from ir.consumer import MarketDataConsumer
from ir.graph import build_ir_graph
from ir.models import curve_publish_queue
from ir.risk import Risk_IR_DV01

# ── RiskDV01Result (demo-specific @ticking class) ────────────────────────

@ticking
@dataclass
class RiskDV01Result(Storable):
    """Per-swap DV01 risk result.  Updated after each market data tick."""
    __key__ = "swap_symbol"

    swap_symbol: str = ""
    curve_point: str = ""
    currency: str = "USD"
    notional: float = 0.0
    tenor_years: float = 0.0
    base_npv: float = 0.0
    central_dv01: float = 0.0
    forward_dv01: float = 0.0
    p_up: float = 0.0
    p_down: float = 0.0
    shock_bps: float = 1.0

    @effect("central_dv01")
    def on_update(self, value):
        self.tick()

# ── Build reactive graph ─────────────────────────────────────────────────

print("  Building reactive objects...")
graph = build_ir_graph(
    fx_config=FX_CONFIG,
    curve_configs=CURVE_CONFIGS,
    swap_configs=SWAP_CONFIGS,
)
fx_spots = graph["fx_spots"]
curve_points = graph["curve_points"]
swaps = graph["swaps"]
swap_curve_map = graph["swap_curve_map"]

# Convenience accessor
usd_curve_points = {k: v for k, v in curve_points.items() if k.startswith("USD")}

# Risk result objects (one per swap)
risk_results = {}
for sym, swap in swaps.items():
    cp_label = swap_curve_map[sym]
    risk_results[sym] = RiskDV01Result(
        swap_symbol=sym,
        curve_point=cp_label,
        currency=swap.currency,
        notional=swap.notional,
        tenor_years=swap.tenor_years,
    )

# ── Risk computation helper ──────────────────────────────────────────────

def _compute_all_risk():
    """Recompute DV01 for every swap via bump-and-reprice."""
    for sym, swap in swaps.items():
        cp = curve_points[swap_curve_map[sym]]
        risk = Risk_IR_DV01(swap, cp, shock_bps=1.0)
        central = risk.compute_central()
        forward = risk.compute_forward()
        risk_results[sym].batch_update(
            base_npv=central["base_npv"],
            central_dv01=central["dv01"],
            forward_dv01=forward["dv01"],
            p_up=central["p_up"],
            p_down=central["p_down"],
        )

# Initial risk computation
_compute_all_risk()

# ── 3. Publish @ticking tables to DH global scope ────────────────────────

tables = get_tables()

# Aggregate risk summary
risk_summary = RiskDV01Result._ticking_live.agg_by([  # type: ignore[attr-defined]
    agg.sum(["TotalDV01=central_dv01"]),
    agg.count("NumSwaps"),
    agg.avg(["AvgDV01=central_dv01"]),
])
tables["risk_dv01_summary"] = risk_summary

for name, tbl in tables.items():
    tbl.publish(name)
flush()

# Drain initial CurveTick publishes
_initial_curve_ticks = list(curve_publish_queue)
curve_publish_queue.clear()

print(f"  Built: {len(fx_spots)} FX spots, "
      f"{len(curve_points)} curve points, "
      f"{len(swaps)} swaps, {len(risk_results)} risk results")
print("  All initial state pushed to DH via @effect")

# ── 4. Market Data consumer with risk recompute ──────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def _on_tick(count: int) -> None:
    _compute_all_risk()
    flush()
    if count % 10 == 0:
        usdjpy = fx_spots["USD/JPY"].mid
        usd_5y_rate = usd_curve_points["USD_5Y"].rate * 100
        total_dv01 = sum(r.central_dv01 for r in risk_results.values())
        print(
            f"  [{count:4d}] "
            f"USD/JPY {usdjpy:.2f}  |  "
            f"USD 5Y: {usd_5y_rate:.2f}%  |  "
            f"Total DV01: {total_dv01:+,.0f}"
        )

consumer = MarketDataConsumer(
    fx_spots=fx_spots,
    on_tick=_on_tick,
    curve_publish_queue=curve_publish_queue,
    initial_curve_ticks=_initial_curve_ticks,
)

print()
print("=" * 70)
print("  DEMO READY — Live DV01 Risk Grid via Market Data Hub")
print(f"  Web UI:  http://localhost:10000")
print()
print("  Published tables (open in DH web IDE):")
print("    fx_spot_live                 — FX spot rates")
print("    yield_curve_point_live       — yield curve points (@computed from FX)")
print("    interest_rate_swap_live      — IRS pricing: NPV (@computed cascade)")
print("    risk_dv01_result_live        — DV01 risk per swap (bump-and-reprice)")
print("    risk_dv01_summary            — aggregate DV01 across book")
print()
print("  Press Ctrl+C to stop.")
print("=" * 70)
print()

consumer.start_background(name="ir-risk-consumer")

try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n  Shutting down...")
    asyncio.run(_md_server.stop())
    print("  Done!")
