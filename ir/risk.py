"""
ir.risk — DV01 via bump-and-reprice.

Pure Python — no Deephaven or streaming dependency at import time.
Only ``flush()`` is called at runtime during bump-and-reprice.

Usage::

    from ir.risk import Risk_IR_DV01

    risk = Risk_IR_DV01(swap, curve_point, shock_bps=1.0)
    central = risk.compute_central()
    forward = risk.compute_forward()
"""

from __future__ import annotations

from reactive.computed import computed
from streaming import flush


class Risk_IR_DV01:
    """DV01 calculator via bump-and-reprice on a single curve point.

    Parameters
    ----------
    swap
        The interest rate swap to reprice (needs ``.npv`` property).
    curve_point
        The yield curve point to shock (needs ``.base_rate`` attribute).
    shock_bps : float
        Shock size in basis points (default 1.0 = 1bp = 0.0001 in rate).
    """

    def __init__(self, swap, curve_point, shock_bps=1.0):
        self.swap = swap
        self.curve_point = curve_point
        self.shock_bps = shock_bps
        self._shock = shock_bps * 0.0001  # convert bps to rate

    def _bump_and_reprice(self, delta):
        """Bump curve_point.base_rate by *delta*, flush, return swap.npv."""
        original = self.curve_point.base_rate
        self.curve_point.base_rate = original + delta
        flush()
        npv = self.swap.npv
        self.curve_point.base_rate = original
        flush()
        return npv

    def compute_central(self):
        """Central difference: DV01 = (P_up - P_down) / (2 * shock_bps).

        Bumps curve point +shock and -shock symmetrically.
        Result is in dollars per basis point (conventional DV01).

        Returns
        -------
        dict with keys: dv01, p_up, p_down, base_npv, shock_bps
        """
        base_npv = self.swap.npv
        p_up = self._bump_and_reprice(+self._shock)
        p_down = self._bump_and_reprice(-self._shock)
        dv01 = (p_up - p_down) / (2 * self.shock_bps)
        return {
            "dv01": dv01,
            "p_up": p_up,
            "p_down": p_down,
            "base_npv": base_npv,
            "shock_bps": self.shock_bps,
        }

    def compute_forward(self):
        """Forward difference: DV01 = (P_up - P_base) / shock_bps.

        Bumps curve point +shock only.
        Result is in dollars per basis point (conventional DV01).

        Returns
        -------
        dict with keys: dv01, p_up, base_npv, shock_bps
        """
        base_npv = self.swap.npv
        p_up = self._bump_and_reprice(+self._shock)
        dv01 = (p_up - base_npv) / self.shock_bps
        return {
            "dv01": dv01,
            "p_up": p_up,
            "base_npv": base_npv,
            "shock_bps": self.shock_bps,
        }
