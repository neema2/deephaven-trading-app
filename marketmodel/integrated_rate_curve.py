"""
integrated_rate_curve — Integrated short rate interpolation curve.

Parameterizes the curve by R_i = (1/t_i) ∫₀ᵗⁱ r(s) ds  (average short rate)
so R_i has rate-like units.  Discount factors use  DF(T) = exp(-R(T) × T).

Interpolation uses a piecewise-linear instantaneous short rate r(t) that:
  - Matches level and gradient from the previous interval (causal, forward-only)
  - Preserves the integral area: ∫_{t_i}^{t_{i+1}} r(s)ds = I_{i+1} - I_i
  - Produces C¹ continuity in I(t) and C² in DF(t)
  - Creates tridiagonal (bandwidth ≤ 3) Jacobians, not dense

Extends CurveBase — drop-in replacement for LinearTermDiscountCurve on any
currency or curve purpose (discount, projection, etc.).

SQL compilation: all math compiles to EXP() of polynomials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from store import Storable
from reactive.computed import computed, effect
from reactive.expr import VariableMixin
from streaming import ticking
from marketmodel.curve_base import CurveBase, _point_tenor_key


# ── Domain model for the integrated rate point ───────────────────────────

@ticking(exclude={"quote_ref", "fitted_rate"})
@dataclass
class IntegratedRatePoint(Storable, VariableMixin):
    """Single curve knot storing the average short rate R_i = (1/t_i) ∫₀ᵗⁱ r(s)ds.

    R_i has rate-like units (e.g. ~5% for 5Y), making pillar_rates directly
    comparable to zero rates.  The cumulative integral is I_i = R_i × t_i.

    Discount factor: DF(t_i) = exp(-R_i × t_i)

    This is the Expr leaf node — diff(expr, point.name) differentiates
    with respect to R_i.
    """
    __key__ = "name"

    name: str = ""
    symbol: str = ""           # e.g. IR_USD_OIS_FIT.R.5Y
    tenor_years: float = 0.0
    fitted_rate: float = 0.0   # R_i — set by CurveFitter
    currency: str = "USD"
    quote_ref: object = None
    is_fitted: bool = False

    @computed
    def rate(self):
        """The average short rate R_i = (1/t_i) ∫₀ᵗⁱ r(s) ds.

        Has the same units as a zero rate, making it easy to reason about.
        The cumulative integral is R_i × t_i.
        """
        if self.is_fitted:
            return self.fitted_rate

        if self.quote_ref is None:
            return 0.0
        # Default: use quoted rate directly (it's already in rate units)
        r = getattr(self.quote_ref, 'rate', 0.0)
        if r is None:
            return 0.0
        return float(r)

    def set_fitted_rate(self, value: float):
        """Update R_i from a solver."""
        self.fitted_rate = value
        import marketmodel.curve_fitter
        if not marketmodel.curve_fitter.IS_SOLVING:
            self.tick()

    def initial_guess(self) -> float:
        """Initial guess for the fitter: R_i ≈ quote_rate.

        Since R_i is now in rate units (average short rate), the par swap
        rate is a natural starting point without any scaling.
        """
        if self.quote_ref is None:
            return self.fitted_rate
        r = getattr(self.quote_ref, 'rate', 0.0)
        if r is None:
            return 0.0
        return float(r)

    @computed
    def zero_rate(self):
        """Equivalent zero rate — for continuous compounding, R_i IS the zero rate."""
        return self.rate

    @computed
    def discount_factor(self):
        """DF(t_i) = exp(-R_i × t_i)."""
        return math.exp(-self.rate * self.tenor_years)

    @effect("rate")
    def on_rate(self, value):
        import marketmodel.curve_fitter
        if marketmodel.curve_fitter.IS_SOLVING:
            return
        self.tick()


# ── Integrated short rate curve ──────────────────────────────────────────

@ticking(exclude={"points", "jacobian"})
@dataclass
class IntegratedShortRateCurve(Storable, CurveBase):
    """Yield curve using average short rate parameterization.

    Parameters: R_i = (1/t_i) ∫₀ᵗⁱ r(s) ds  — the average short rate at each knot.
                R_i has rate-like units (same magnitude as zero rates).

    Derived:    I_i = R_i × t_i  — the cumulative integral at each knot.
                DF(T) = exp(-I(T)) = exp(-R(T) × T)

    Interpolation: on [t_i, t_{i+1}], the instantaneous short rate is
    piecewise-linear:
        r(τ) = r_left + m·τ       where τ = t - t_i

    Constraints:
        - r_left = short rate at end of previous interval (slope continuity)
        - ∫₀ʰ r(τ)dτ = I_{i+1} - I_i  (area preservation)

    This gives:
        - C⁰ in r(t) (continuous short rate)
        - C¹ in I(t) (continuous first derivative of cumulative integral)
        - C² in DF(t) (very smooth discount factors)
        - Tridiagonal Jacobian: ∂DF(t)/∂R_j = 0 for |j - i| > 1

    SQL: compiles to EXP() of polynomial argument.
    """
    __key__ = "name"

    name: str = ""
    currency: str = "USD"
    points: list = field(default_factory=list)
    jacobian: list = field(default_factory=list)

    @computed
    def point_count(self) -> int:
        return len(self.points)

    def _sorted_points(self):
        """Sort points by tenor."""
        return sorted(self.points, key=_point_tenor_key)

    @computed
    def pillar_tenors(self) -> list:
        pts = self._sorted_points()
        return [p.tenor_years for p in pts]

    @computed
    def pillar_rates(self) -> list:
        """Average short rates R_i at each knot (rate-like units)."""
        pts = self._sorted_points()
        return [p.rate for p in pts]

    @computed
    def pillar_names(self) -> list:
        pts = self._sorted_points()
        return [p.name for p in pts]

    # ── Numerical helpers ─────────────────────────────────────────────────

    def _compute_slopes(self) -> list[float]:
        """Compute the short rate r_left at the start of each interval.

        Works with I_i = R_i × t_i (cumulative integrals) internally.

        Returns r_left[i] for i = 0, 1, ..., n-1  where:
          - r_left[0] = R_0  (flat short rate assumption: r(s)=R_0 for s<t_0)
          - r_left[i] = r_left[i-1] + m_{i-1} * h_{i-1}  for i > 0
        """
        pts = self._sorted_points()
        n = len(pts)
        if n == 0:
            return []

        R = [p.rate for p in pts]
        T = [p.tenor_years for p in pts]
        I = [R[i] * T[i] for i in range(n)]

        r_left = [0.0] * n
        r_left[0] = R[0]  # flat short rate from 0 to t_0

        for i in range(1, n):
            h_prev = T[i] - T[i - 1]
            if h_prev <= 0:
                r_left[i] = r_left[i - 1]
                continue
            dI_prev = I[i] - I[i - 1]
            m_prev = 2.0 * (dI_prev / h_prev - r_left[i - 1]) / h_prev
            r_left[i] = r_left[i - 1] + m_prev * h_prev

        return r_left

    def _I_at(self, t: float) -> float:
        """Numerical cumulative integral I(t) = ∫₀ᵗ r(s) ds at any tenor."""
        pts = self._sorted_points()
        n = len(pts)
        if n == 0:
            return 0.0

        R = [p.rate for p in pts]
        T = [p.tenor_years for p in pts]
        I = [R[i] * T[i] for i in range(n)]

        if t <= 0:
            return 0.0
        if n == 1:
            return R[0] * t

        r_left = self._compute_slopes()

        if t <= T[0]:
            return R[0] * t

        if t >= T[-1]:
            if n >= 2:
                h_last = T[-1] - T[-2]
                dI_last = I[-1] - I[-2]
                m_last = 2.0 * (dI_last / h_last - r_left[-1]) / h_last if h_last > 0 else 0.0
                r_end = r_left[-1] + m_last * h_last
            else:
                r_end = R[-1]
            return I[-1] + r_end * (t - T[-1])

        for i in range(n - 1):
            if T[i] <= t <= T[i + 1]:
                tau = t - T[i]
                h = T[i + 1] - T[i]
                dI = I[i + 1] - I[i]
                m = 2.0 * (dI / h - r_left[i]) / h if h > 0 else 0.0
                return I[i] + r_left[i] * tau + m * tau * tau / 2.0

        return I[-1]

    def _R_at(self, t: float) -> float:
        """Backward-compat alias for _I_at (cumulative integral)."""
        return self._I_at(t)

    def df_at(self, tenor: float) -> float:
        """DF(t) = exp(-I(t)) = exp(-R(t) × t)."""
        return math.exp(-self._I_at(tenor))

    def df_array(self, tenors: list[float]) -> list[float]:
        """Batch discount factors."""
        return [self.df_at(t) for t in tenors]

    def fwd_at(self, tenor: float, period: float = 1.0) -> float:
        """Forward rate: (I(t+p) - I(t)) / p."""
        I_start = self._I_at(tenor)
        I_end = self._I_at(tenor + period)
        if period <= 0:
            return 0.0
        return (I_end - I_start) / period

    # ── Symbolic Expr builders ────────────────────────────────────────────

    def interp(self, t: float) -> "Expr":
        """Build an Expr tree for the cumulative integral I(t) = ∫₀ᵗ r(s) ds.

        I(t) on interval [t_i, t_{i+1}] is a quadratic in τ = t - t_i:
            I(t) = I_i + r_left·τ + m·τ²/2

        where:
            I_i = R_i × t_i  (symbolic: Variable(R_i) × Const(t_i))
            r_left is baked as Const
            m = 2*(ΔI/h - r_left)/h

        Each I(t) Expr depends on at most 2 Variables: R_i and R_{i+1}.
        """
        cache = getattr(self, '_interp_cache', None)
        if cache is None:
            cache = {}
            object.__setattr__(self, '_interp_cache', cache)
        if t in cache:
            return cache[t]

        from reactive.expr import Const, Variable

        pts = self._sorted_points()
        n = len(pts)

        if n == 0:
            expr = Const(0.0)
            cache[t] = expr
            return expr

        T = [p.tenor_years for p in pts]
        R_vals = [p.rate for p in pts]
        r_lefts = self._compute_slopes()

        def _leaf(point):
            return Variable(point.name)

        # Before/at first knot: I(t) = R_0 × t
        if t <= T[0]:
            expr = _leaf(pts[0]) * Const(t)
            cache[t] = expr
            return expr

        # After/at last knot: flat extrapolation
        if t >= T[-1]:
            if n >= 2:
                h_last = T[-1] - T[-2]
                I_last = R_vals[-1] * T[-1]
                I_prev = R_vals[-2] * T[-2]
                dI_last = I_last - I_prev
                r_left_last = r_lefts[-1]
                m_last = 2.0 * (dI_last / h_last - r_left_last) / h_last if h_last > 0 else 0.0
                r_end = r_left_last + m_last * h_last
                # I(t) = R_n × T_n + r_end × (t - T_n)
                expr = _leaf(pts[-1]) * Const(T[-1]) + Const(r_end * (t - T[-1]))
            else:
                expr = _leaf(pts[-1]) * Const(t)
            cache[t] = expr
            return expr

        # Interior: find interval
        for i in range(n - 1):
            t_i, t_ip1 = T[i], T[i + 1]
            if t_i <= t <= t_ip1:
                tau = t - t_i
                h = t_ip1 - t_i
                if h <= 0:
                    expr = _leaf(pts[i]) * Const(T[i])
                    cache[t] = expr
                    return expr

                r_left_val = r_lefts[i]

                # R_i and R_{i+1} as Variables (average rates)
                Ri = _leaf(pts[i])
                Rip1 = _leaf(pts[i + 1])

                # I_i = R_i × t_i,  I_{i+1} = R_{i+1} × t_{i+1}  (symbolic)
                Ii = Ri * Const(t_i)
                Iip1 = Rip1 * Const(t_ip1)

                # ΔI = I_{i+1} - I_i
                dI = Iip1 - Ii

                # m = 2/(h²) × ΔI - 2×r_left/h
                m = (Const(2.0 / (h * h)) * dI) - Const(2.0 * r_left_val / h)

                # I(t) = I_i + r_left×τ + m×τ²/2
                expr = Ii + Const(r_left_val * tau) + m * Const(tau * tau / 2.0)

                cache[t] = expr
                return expr

        expr = _leaf(pts[-1]) * Const(T[-1])
        cache[t] = expr
        return expr

    def df(self, t: float) -> "Expr":
        """Build an Expr tree for DF(t) = exp(-I(t)) = exp(-R(t) × t).

        Uses EXP() — compiles to a single hardware instruction in SQL.
        """
        cache = getattr(self, '_df_cache', None)
        if cache is None:
            cache = {}
            object.__setattr__(self, '_df_cache', cache)
        if t in cache:
            return cache[t]

        from reactive.expr import Exp
        I_expr = self.interp(t)
        expr = Exp(-I_expr)
        cache[t] = expr
        return expr

    def fwd(self, start: float, end: float) -> "Expr":
        """Build an Expr tree for the forward rate.

        fwd(start, end) = (I(end) - I(start)) / (end - start)
        """
        from reactive.expr import Const
        dt = end - start
        if dt <= 0:
            return Const(0.0)

        I_start = self.interp(start)
        I_end = self.interp(end)
        return (I_end - I_start) / Const(dt)

    # ── Cache invalidation ────────────────────────────────────────────────

    def invalidate_caches(self):
        """Clear Expr caches when curve parameters change."""
        if hasattr(self, '_interp_cache'):
            object.__setattr__(self, '_interp_cache', {})
        if hasattr(self, '_df_cache'):
            object.__setattr__(self, '_df_cache', {})

    @effect("pillar_rates")
    def on_rates_change(self, value):
        self.invalidate_caches()
        import marketmodel.curve_fitter
        if marketmodel.curve_fitter.IS_SOLVING:
            return
        self.tick()


