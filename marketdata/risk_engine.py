"""
Risk Engine — Black-Scholes Greeks Calculator
Used by the market data simulator to compute per-position risk.
"""

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def calculate_greeks(price: float, strike: float | None = None, T: float = 0.25, r: float = 0.05, sigma: float = 0.25) -> tuple[float, float, float, float]:
    """Return (delta, gamma, theta, vega) for a European call option."""
    S = price
    K = strike or price * 1.05
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    norm_factor = math.exp(-d1 ** 2 / 2) / math.sqrt(2 * math.pi)

    delta = _norm_cdf(d1)
    gamma = norm_factor / (S * sigma * sqrt_T)
    theta = -(S * norm_factor * sigma) / (2 * sqrt_T) \
            - r * K * math.exp(-r * T) * _norm_cdf(d2)
    vega = S * norm_factor * sqrt_T
    return delta, gamma, theta, vega
