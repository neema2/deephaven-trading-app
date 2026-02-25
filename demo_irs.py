#!/usr/bin/env python3
"""
Demo: Interest Rate Swap Reactive Grid in Deephaven
====================================================

Starts an in-process Deephaven server and pushes a live IRS portfolio
through the reactive framework into ticking DH tables.

The reactive graph:
  FX spots → yield curve points → yield curves → IRS pricing → portfolio

Every market-data tick recalculates all the way up through swap NPV to
portfolio aggregates — and you see it live in the Deephaven web UI.

Open http://localhost:10000 in your browser to see the ticking tables!

Usage:  python3 demo_irs.py
"""

import os
import sys
import time
import math
import random

sys.path.insert(0, os.path.dirname(__file__))

# ── 1. Start Deephaven server ──────────────────────────────────────────────
print("=" * 70)
print("  Interest Rate Swap — Reactive Ticking Demo")
print("=" * 70)
print()
print("  Starting Deephaven server...")

from deephaven_server import Server

dh = Server(
    port=10000,
    jvm_args=[
        "-Xmx1g",
        "-Dprocess.info.system-info.enabled=false",
        "-DAuthHandlers=io.deephaven.auth.AnonymousAuthenticationHandler",
    ],
    default_jvm_args=[
        "-XX:+UseG1GC",
        "-XX:MaxGCPauseMillis=100",
        "-XX:+UseStringDeduplication",
    ],
)
dh.start()
print("  Deephaven started on http://localhost:10000")

from deephaven.execution_context import get_exec_ctx
from deephaven import DynamicTableWriter, agg
import deephaven.dtypes as dht

# ── 2. Domain models (from test_reactive_irs.py) ──────────────────────────
from dataclasses import dataclass, field
from store.base import Storable
from reactive.computed import computed, effect


@dataclass
class FXSpot(Storable):
    pair: str = ""
    bid: float = 0.0
    ask: float = 0.0
    currency: str = ""

    @computed
    def mid(self):
        return (self.bid + self.ask) / 2

    @computed
    def spread_pips(self):
        return (self.ask - self.bid) * 10000


@dataclass
class YieldCurvePoint(Storable):
    label: str = ""
    tenor_years: float = 0.0
    rate: float = 0.0
    currency: str = "USD"

    @computed
    def discount_factor(self):
        return 1.0 / (1.0 + self.rate) ** self.tenor_years


@dataclass
class InterestRateSwap(Storable):
    symbol: str = ""
    notional: float = 0.0
    fixed_rate: float = 0.0
    float_rate: float = 0.0
    tenor_years: float = 0.0
    currency: str = "USD"

    @computed
    def fixed_leg_pv(self):
        df = 1.0 / (1.0 + self.fixed_rate) ** self.tenor_years
        return self.notional * self.fixed_rate * self.tenor_years * df

    @computed
    def float_leg_pv(self):
        df = 1.0 / (1.0 + self.float_rate) ** self.tenor_years
        return self.notional * self.float_rate * self.tenor_years * df

    @computed
    def npv(self):
        float_df = 1.0 / (1.0 + self.float_rate) ** self.tenor_years
        fixed_df = 1.0 / (1.0 + self.fixed_rate) ** self.tenor_years
        float_pv = self.notional * self.float_rate * self.tenor_years * float_df
        fixed_pv = self.notional * self.fixed_rate * self.tenor_years * fixed_df
        return float_pv - fixed_pv

    @computed
    def dv01(self):
        return self.notional * self.tenor_years * 0.0001

    @computed
    def pnl_status(self):
        float_df = 1.0 / (1.0 + self.float_rate) ** self.tenor_years
        fixed_df = 1.0 / (1.0 + self.fixed_rate) ** self.tenor_years
        float_pv = self.notional * self.float_rate * self.tenor_years * float_df
        fixed_pv = self.notional * self.fixed_rate * self.tenor_years * fixed_df
        npv = float_pv - fixed_pv
        if npv > 0:
            return "PROFIT"
        if npv < 0:
            return "LOSS"
        return "FLAT"


# ── 3. DynamicTableWriters → Deephaven ticking tables ─────────────────────
print("  Creating DynamicTableWriters...")

fx_writer = DynamicTableWriter({
    "pair": dht.string,
    "bid": dht.double,
    "ask": dht.double,
    "mid": dht.double,
    "spread_pips": dht.double,
    "currency": dht.string,
})

curve_writer = DynamicTableWriter({
    "label": dht.string,
    "tenor_years": dht.double,
    "rate": dht.double,
    "discount_factor": dht.double,
    "currency": dht.string,
})

swap_writer = DynamicTableWriter({
    "symbol": dht.string,
    "notional": dht.double,
    "fixed_rate": dht.double,
    "float_rate": dht.double,
    "tenor_years": dht.double,
    "fixed_leg_pv": dht.double,
    "float_leg_pv": dht.double,
    "npv": dht.double,
    "dv01": dht.double,
    "pnl_status": dht.string,
    "currency": dht.string,
})

portfolio_writer = DynamicTableWriter({
    "portfolio": dht.string,
    "total_npv": dht.double,
    "total_dv01": dht.double,
    "swap_count": dht.int32,
    "max_npv": dht.double,
    "min_npv": dht.double,
})

# Get raw tables and create live (last-by) views
fx_raw = fx_writer.table
fx_live = fx_raw.last_by("pair")

curve_raw = curve_writer.table
curve_live = curve_raw.last_by("label")

swap_raw = swap_writer.table
swap_live = swap_raw.last_by("symbol")

portfolio_raw = portfolio_writer.table
portfolio_live = portfolio_raw.last_by("portfolio")

# Aggregates
swap_summary = swap_live.agg_by([
    agg.sum_(["TotalNPV=npv", "TotalDV01=dv01"]),
    agg.count_("NumSwaps"),
    agg.avg(["AvgNPV=npv"]),
])

# Publish to DH global scope
for name, tbl in {
    "fx_raw": fx_raw,
    "fx_live": fx_live,
    "curve_raw": curve_raw,
    "curve_live": curve_live,
    "swap_raw": swap_raw,
    "swap_live": swap_live,
    "swap_summary": swap_summary,
    "portfolio_raw": portfolio_raw,
    "portfolio_live": portfolio_live,
}.items():
    globals()[name] = tbl


# ── 4. Build initial reactive objects ──────────────────────────────────────
print("  Building reactive objects...")

# FX spots
fx_spots = {
    "USD/JPY": FXSpot(pair="USD/JPY", bid=149.50, ask=149.60, currency="JPY"),
    "EUR/USD": FXSpot(pair="EUR/USD", bid=1.0850, ask=1.0855, currency="USD"),
    "GBP/USD": FXSpot(pair="GBP/USD", bid=1.2700, ask=1.2710, currency="USD"),
}

# Yield curve points (USD)
usd_curve_data = [
    ("USD_3M",  0.25, 0.0525),
    ("USD_1Y",  1.0,  0.0490),
    ("USD_2Y",  2.0,  0.0445),
    ("USD_5Y",  5.0,  0.0410),
    ("USD_10Y", 10.0, 0.0395),
    ("USD_30Y", 30.0, 0.0420),
]
usd_curve_points = {}
for label, tenor, rate in usd_curve_data:
    usd_curve_points[label] = YieldCurvePoint(
        label=label, tenor_years=tenor, rate=rate, currency="USD",
    )

# JPY curve points
jpy_curve_data = [
    ("JPY_1Y",  1.0,  0.001),
    ("JPY_5Y",  5.0,  0.005),
    ("JPY_10Y", 10.0, 0.008),
]
jpy_curve_points = {}
for label, tenor, rate in jpy_curve_data:
    jpy_curve_points[label] = YieldCurvePoint(
        label=label, tenor_years=tenor, rate=rate, currency="JPY",
    )

# Interest rate swaps
swap_configs = [
    ("USD-5Y-A", 50_000_000,  0.0400, 0.0525, 5.0,  "USD"),
    ("USD-5Y-B", 25_000_000,  0.0380, 0.0525, 5.0,  "USD"),
    ("USD-10Y",  100_000_000, 0.0395, 0.0490, 10.0, "USD"),
    ("USD-2Y",   75_000_000,  0.0450, 0.0445, 2.0,  "USD"),
    ("JPY-5Y",   5_000_000_000, 0.005, 0.001, 5.0,  "JPY"),
    ("JPY-10Y",  10_000_000_000, 0.008, 0.005, 10.0, "JPY"),
]

swaps = {}
for sym, notl, fixed, flt, tenor, ccy in swap_configs:
    swaps[sym] = InterestRateSwap(
        symbol=sym, notional=notl, fixed_rate=fixed,
        float_rate=flt, tenor_years=tenor, currency=ccy,
    )


# ── 5. Push helpers ───────────────────────────────────────────────────────

def push_fx():
    for fx in fx_spots.values():
        fx_writer.write_row(
            fx.pair, fx.bid, fx.ask, fx.mid, fx.spread_pips, fx.currency,
        )

def push_curve():
    for pts in (usd_curve_points, jpy_curve_points):
        for pt in pts.values():
            curve_writer.write_row(
                pt.label, pt.tenor_years, pt.rate,
                pt.discount_factor, pt.currency,
            )

def push_swaps():
    for s in swaps.values():
        swap_writer.write_row(
            s.symbol, s.notional, s.fixed_rate, s.float_rate,
            s.tenor_years, s.fixed_leg_pv, s.float_leg_pv,
            s.npv, s.dv01, s.pnl_status, s.currency,
        )

def push_portfolio():
    usd_swaps = [s for s in swaps.values() if s.currency == "USD"]
    jpy_swaps = [s for s in swaps.values() if s.currency == "JPY"]
    all_swaps = list(swaps.values())

    for name, subset in [("ALL", all_swaps), ("USD", usd_swaps), ("JPY", jpy_swaps)]:
        if not subset:
            continue
        total_npv = sum(s.npv for s in subset)
        total_dv01 = sum(s.dv01 for s in subset)
        max_npv = max(s.npv for s in subset)
        min_npv = min(s.npv for s in subset)
        portfolio_writer.write_row(
            name, total_npv, total_dv01, len(subset), max_npv, min_npv,
        )


# ── 6. Initial push ──────────────────────────────────────────────────────
push_fx()
push_curve()
push_swaps()
push_portfolio()
get_exec_ctx().update_graph.j_update_graph.requestRefresh()

print()
print("=" * 70)
print("  DEMO READY!")
print("  Web UI:  http://localhost:10000")
print()
print("  Published tables (open in DH web IDE):")
print("    fx_live         — FX spot rates (ticking)")
print("    curve_live      — yield curve points (ticking)")
print("    swap_live       — IRS pricing: NPV, DV01, PnL status (ticking)")
print("    swap_summary    — aggregate NPV + DV01 (ticking)")
print("    portfolio_live  — portfolio breakdown: ALL / USD / JPY (ticking)")
print()
print("  Raw (append-only) tables: fx_raw, curve_raw, swap_raw, portfolio_raw")
print()
print("  Simulating market-data ticks every 1.5 seconds...")
print("  Press Ctrl+C to stop.")
print("=" * 70)
print()


# ── 7. Market-data simulation loop ───────────────────────────────────────

def bump_fx():
    """Random walk on FX spots."""
    for fx in fx_spots.values():
        spread = fx.ask - fx.bid
        mid_move = random.gauss(0, 0.0003) * fx.mid
        new_mid = fx.mid + mid_move
        fx.batch_update(
            bid=round(new_mid - spread / 2, 5),
            ask=round(new_mid + spread / 2, 5),
        )

def bump_curve():
    """Random walk on yield curve rates (+/- a few bps)."""
    for pts in (usd_curve_points, jpy_curve_points):
        for pt in pts.values():
            bp_move = random.gauss(0, 1.5) * 0.0001  # ~1.5bp σ
            new_rate = max(0.0001, pt.rate + bp_move)
            pt.rate = round(new_rate, 6)

def update_swap_rates():
    """Propagate curve changes to swap float rates."""
    for s in swaps.values():
        if s.currency == "USD":
            # Match tenor to nearest curve point
            if s.tenor_years <= 2:
                ref = usd_curve_points["USD_2Y"]
            elif s.tenor_years <= 5:
                ref = usd_curve_points["USD_5Y"]
            else:
                ref = usd_curve_points["USD_10Y"]
            s.float_rate = ref.rate
        elif s.currency == "JPY":
            if s.tenor_years <= 5:
                ref = jpy_curve_points["JPY_5Y"]
            else:
                ref = jpy_curve_points["JPY_10Y"]
            s.float_rate = ref.rate


try:
    tick = 0
    while True:
        tick += 1

        # 1. Bump market data (reactive graph recalculates automatically)
        bump_fx()
        bump_curve()

        # 2. Propagate curve rates → swap float rates
        update_swap_rates()

        # 3. Push updated state to DH tables
        push_fx()
        push_curve()
        push_swaps()
        push_portfolio()

        # 4. Flush DH update graph
        get_exec_ctx().update_graph.j_update_graph.requestRefresh()

        # 5. Print summary
        usd_npv = sum(s.npv for s in swaps.values() if s.currency == "USD")
        jpy_npv = sum(s.npv for s in swaps.values() if s.currency == "JPY")
        usdjpy = fx_spots["USD/JPY"].mid
        usd_5y = usd_curve_points["USD_5Y"].rate * 100
        jpy_10y = jpy_curve_points["JPY_10Y"].rate * 100

        print(
            f"  [{tick:4d}] "
            f"USD/JPY {usdjpy:.2f}  |  "
            f"USD 5Y: {usd_5y:.2f}%  |  "
            f"JPY 10Y: {jpy_10y:.3f}%  |  "
            f"USD NPV: ${usd_npv:+,.0f}  |  "
            f"JPY NPV: ¥{jpy_npv:+,.0f}"
        )

        time.sleep(1.5)

except KeyboardInterrupt:
    print("\n  Shutting down...")
    print("  Done!")
