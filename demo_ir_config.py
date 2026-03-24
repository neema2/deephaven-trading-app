"""
Demo IR configuration — static data shared by demo_ir_swap and demo_ir_risk.

This file defines the FX spots, yield curve points, and swap book
used by both IR demos.  It is NOT part of the ``ir/`` library — the
library is fully generic and accepts any configuration.
"""

# ── FX Spots ─────────────────────────────────────────────────────────────

FX_CONFIG = {
    "USD/JPY": {"bid": 149.50, "ask": 149.60, "currency": "JPY"},
    "EUR/USD": {"bid": 1.0850, "ask": 1.0855, "currency": "USD"},
    "GBP/USD": {"bid": 1.2700, "ask": 1.2710, "currency": "USD"},
}

# ── Yield Curve Points ───────────────────────────────────────────────────

CURVE_CONFIGS = [
    # USD
    {"label": "USD_3M",  "tenor_years": 0.25, "base_rate": 0.0525, "sensitivity": 0.5, "currency": "USD", "fx_pair": "EUR/USD"},
    {"label": "USD_1Y",  "tenor_years": 1.0,  "base_rate": 0.0490, "sensitivity": 0.5, "currency": "USD", "fx_pair": "EUR/USD"},
    {"label": "USD_2Y",  "tenor_years": 2.0,  "base_rate": 0.0445, "sensitivity": 0.5, "currency": "USD", "fx_pair": "EUR/USD"},
    {"label": "USD_5Y",  "tenor_years": 5.0,  "base_rate": 0.0410, "sensitivity": 0.5, "currency": "USD", "fx_pair": "EUR/USD"},
    {"label": "USD_10Y", "tenor_years": 10.0, "base_rate": 0.0395, "sensitivity": 0.5, "currency": "USD", "fx_pair": "EUR/USD"},
    {"label": "USD_30Y", "tenor_years": 30.0, "base_rate": 0.0420, "sensitivity": 0.5, "currency": "USD", "fx_pair": "EUR/USD"},
    # JPY
    {"label": "JPY_1Y",  "tenor_years": 1.0,  "base_rate": 0.001, "sensitivity": 0.5, "currency": "JPY", "fx_pair": "USD/JPY"},
    {"label": "JPY_5Y",  "tenor_years": 5.0,  "base_rate": 0.005, "sensitivity": 0.5, "currency": "JPY", "fx_pair": "USD/JPY"},
    {"label": "JPY_10Y", "tenor_years": 10.0, "base_rate": 0.008, "sensitivity": 0.5, "currency": "JPY", "fx_pair": "USD/JPY"},
]

# ── Swap Book ────────────────────────────────────────────────────────────

SWAP_CONFIGS = [
    {"symbol": "USD-5Y-A", "notional": 50_000_000,     "fixed_rate": 0.0400, "tenor_years": 5.0,  "currency": "USD", "curve_label": "USD_5Y"},
    {"symbol": "USD-5Y-B", "notional": 25_000_000,     "fixed_rate": 0.0380, "tenor_years": 5.0,  "currency": "USD", "curve_label": "USD_5Y"},
    {"symbol": "USD-10Y",  "notional": 100_000_000,    "fixed_rate": 0.0395, "tenor_years": 10.0, "currency": "USD", "curve_label": "USD_10Y"},
    {"symbol": "USD-2Y-A", "notional": 75_000_000,     "fixed_rate": 0.0450, "tenor_years": 2.0,  "currency": "USD", "curve_label": "USD_2Y"},
    {"symbol": "USD-2Y-B", "notional": 40_000_000,     "fixed_rate": 0.0420, "tenor_years": 2.0,  "currency": "USD", "curve_label": "USD_2Y"},
    {"symbol": "USD-2Y-C", "notional": 120_000_000,    "fixed_rate": 0.0475, "tenor_years": 2.0,  "currency": "USD", "curve_label": "USD_2Y"},
    {"symbol": "USD-2Y-D", "notional": 30_000_000,     "fixed_rate": 0.0390, "tenor_years": 2.0,  "currency": "USD", "curve_label": "USD_2Y"},
    {"symbol": "JPY-5Y",   "notional": 5_000_000_000,  "fixed_rate": 0.005,  "tenor_years": 5.0,  "currency": "JPY", "curve_label": "JPY_5Y"},
    {"symbol": "JPY-10Y",  "notional": 10_000_000_000, "fixed_rate": 0.008,  "tenor_years": 10.0, "currency": "JPY", "curve_label": "JPY_10Y"},
]
