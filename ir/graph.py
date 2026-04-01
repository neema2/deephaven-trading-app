"""
ir.graph — Generic reactive IR graph builder.

Accepts configuration data (FX spots, curve points, swap book) as
parameters and constructs wired reactive objects.  No static data
lives here — callers provide their own configuration.

Usage::

    from ir.graph import build_ir_graph

    graph = build_ir_graph(
        fx_config={"EUR/USD": {"bid": 1.08, "ask": 1.09, "currency": "USD"}},
        curve_configs=[
            {"label": "USD_5Y", "tenor_years": 5.0, "base_rate": 0.041,
             "sensitivity": 0.5, "currency": "USD", "fx_pair": "EUR/USD"},
        ],
        swap_configs=[
            {"symbol": "USD-5Y", "notional": 50e6, "fixed_rate": 0.04,
             "tenor_years": 5.0, "currency": "USD", "curve_label": "USD_5Y"},
        ],
    )
"""

from __future__ import annotations

from ir.models import FXSpot, InterestRateSwap, YieldCurvePoint


def build_ir_graph(
    fx_config: dict[str, dict],
    curve_configs: list[dict],
    swap_configs: list[dict],
) -> dict:
    """Build a reactive IR graph from configuration data.

    Parameters
    ----------
    fx_config : dict[str, dict]
        ``{pair: {"bid": ..., "ask": ..., "currency": ...}}``
    curve_configs : list[dict]
        Each dict needs: ``label``, ``tenor_years``, ``base_rate``,
        ``sensitivity``, ``currency``, ``fx_pair`` (key into fx_config).
    swap_configs : list[dict]
        Each dict needs: ``symbol``, ``notional``, ``fixed_rate``,
        ``tenor_years``, ``currency``, ``curve_label`` (key into curves).

    Returns
    -------
    dict with keys:
        fx_spots : dict[str, FXSpot]
        curve_points : dict[str, YieldCurvePoint]
        swaps : dict[str, InterestRateSwap]
        swap_curve_map : dict[str, str]   (symbol → curve_label)
    """
    # FX spots
    fx_spots: dict[str, FXSpot] = {}
    for pair, cfg in fx_config.items():
        fx_spots[pair] = FXSpot(pair=pair, **cfg)

    # Curve points
    curve_points: dict[str, YieldCurvePoint] = {}
    for cfg in curve_configs:
        fx_pair = cfg["fx_pair"]
        fx_ref = fx_spots[fx_pair]
        curve_points[cfg["label"]] = YieldCurvePoint(
            label=cfg["label"],
            tenor_years=cfg["tenor_years"],
            base_rate=cfg["base_rate"],
            sensitivity=cfg.get("sensitivity", 0.5),
            currency=cfg["currency"],
            fx_ref=fx_ref,
            fx_base_mid=fx_ref.mid,
        )

    # Swaps
    swap_curve_map: dict[str, str] = {}
    swaps: dict[str, InterestRateSwap] = {}
    for cfg in swap_configs:
        sym = cfg["symbol"]
        curve_label = cfg["curve_label"]
        swap_curve_map[sym] = curve_label
        swaps[sym] = InterestRateSwap(
            symbol=sym,
            notional=cfg["notional"],
            fixed_rate=cfg["fixed_rate"],
            tenor_years=cfg["tenor_years"],
            currency=cfg["currency"],
            curve_ref=curve_points[curve_label],
        )

    return {
        "fx_spots": fx_spots,
        "curve_points": curve_points,
        "swaps": swaps,
        "swap_curve_map": swap_curve_map,
    }
