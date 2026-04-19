"""
pricing.scenarios.builder — Generic reactive IR graph builder.

Accepts configuration data (FX spots, curve points, swap book) as
parameters and constructs wired reactive objects using the native pricing domain.
"""

from __future__ import annotations

from pricing.marketmodels.fx_spot import FXSpot, FXLinkedSwapQuote
from pricing.marketmodels.ir_curve_integrated_rate import IntegratedRatePoint, IntegratedShortRateCurve
from pricing.instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox

def build_ir_graph(
    fx_config: dict[str, dict],
    curve_configs: list[dict],
    swap_configs: list[dict],
) -> dict:
    """Build a reactive IR graph combining FX, Integrated curves, and IRS swaps."""
    # 1. FX spots
    fx_spots: dict[str, FXSpot] = {}
    for pair, cfg in fx_config.items():
        fx_spots[pair] = FXSpot(pair=pair, **cfg)

    # 2. Curve points backed by FX-Linked Quotes
    quotes: dict[str, FXLinkedSwapQuote] = {}
    curve_points: dict[str, IntegratedRatePoint] = {}
    
    for cfg in curve_configs:
        lbl = cfg["label"]
        fx_pair = cfg["fx_pair"]
        fx_ref = fx_spots[fx_pair]
        
        # Build a live quote linked to the cross-entity FX tick
        quote = FXLinkedSwapQuote(
            symbol=lbl,
            tenor=cfg["tenor_years"],
            currency=cfg["currency"],
            fx_ref=fx_ref,
            fx_base_mid=fx_ref.mid,
            base_rate=cfg["base_rate"],
            sensitivity=cfg.get("sensitivity", 0.5)
        )
        quotes[lbl] = quote

        # Map to an integrated rate pillar
        curve_points[lbl] = IntegratedRatePoint(
            name=lbl,
            symbol=lbl,
            tenor_years=cfg["tenor_years"],
            currency=cfg["currency"],
            quote_ref=quote, 
            is_fitted=False
        )

    # 3. Create the robust C-2 Smooth Integrated Short Rate Curve
    curve = IntegratedShortRateCurve(
        name="SCENARIO_CURVE",
        currency="USD",
        points=list(curve_points.values())
    )

    # 4. Swaps measured against the robust curve
    swap_curve_map: dict[str, str] = {}
    swaps: dict[str, IRSwapFixedFloatApprox] = {}
    
    for cfg in swap_configs:
        sym = cfg["symbol"]
        swap_curve_map[sym] = "SCENARIO_CURVE"
        swaps[sym] = IRSwapFixedFloatApprox(
            symbol=sym,
            notional=cfg["notional"],
            fixed_rate=cfg["fixed_rate"],
            tenor_years=cfg["tenor_years"],
            discount_curve=curve,
            projection_curve=curve
        )

    return {
        "fx_spots": fx_spots,
        "quotes": quotes,
        "curve_points": curve_points,
        "curve": curve,
        "swaps": swaps,
        "swap_curve_map": swap_curve_map,
    }
