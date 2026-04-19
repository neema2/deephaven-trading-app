"""
marketmodel — Market model domain models.

Market models describe the state of the market (yield curves, vol surfaces, etc.)
These are NOT instruments — instruments (swaps, bonds, etc.) reference market models
via abstract interfaces (fwd_at, df_at, df_array) defined by CurveBase.

    marketmodel/
        curve_base.py           — CurveBase ABC (pluggable curve interface)
        yield_curve.py          — LinearTermDiscountCurve (alias: LinearTermDiscountCurve)
        integrated_rate_curve.py— IntegratedRatePoint, IntegratedShortRateCurve
        curve_fitter.py         — CurveFitter (Simultaneous Solve)
        swap_curve.py           — SwapQuote, SwapQuoteRisk
        symbols.py              — Naming convention helpers
"""

from pricing.marketmodels.curve_base import CurveBase
from pricing.marketmodels.ir_curve_yield import (
    YieldCurvePoint, LinearTermDiscountCurve, CurveJacobianEntry,
)
from pricing.marketmodels.ir_curve_fitter import CurveFitter
from pricing.marketmodels.ir_curve_swap import SwapQuote, SwapQuoteRisk
from pricing.marketmodels.symbols import (
    quote_symbol, fit_symbol, jacobian_symbol,
    tenor_name, parse_symbol,
)

__all__ = [
    "CurveBase",
    "YieldCurvePoint",
    "LinearTermDiscountCurve",
    "CurveJacobianEntry",
    "CurveFitter",
    "SwapQuote",
    "SwapQuoteRisk",
    "quote_symbol",
    "fit_symbol",
    "jacobian_symbol",
    "tenor_name",
    "parse_symbol",
]

