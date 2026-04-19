"""
yield_curve — LinearTermDiscountCurve: linear zero-rate interpolation curve.

This is the simplest curve type: stores zero rates at knot points,
interpolates linearly, and computes DF = (1 + r)^(-t) using POWER.

Extends CurveBase — instruments and the fitter depend only on the
abstract interface, so this curve type is interchangeable with
IntegratedShortRateCurve or any other CurveBase implementation.

Backward compatibility: `LinearTermDiscountCurve` is a module-level alias for
`LinearTermDiscountCurve`, so existing imports continue to work.

SQL compilation traces through this file — all math here must remain
SQL-translatable: only +, -, *, /, POWER, CASE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from store import Storable
from reactive.traceable import traceable
from reactive.computed import effect
from reactive.expr import VariableMixin
from streaming import ticking
from pricing.marketmodels.curve_base import CurveBase


# ── Interpolation (SQL-translatable: only +, -, *, /) ────────────────────

def _interp(tenors: list[float], rates: list[float], t: float) -> float:
    """Linear interpolation with flat extrapolation.

    All operations are +, -, *, / — directly translatable to SQL.
    """
    n = len(tenors)
    if n == 0:
        return 0.0
    if t <= tenors[0]:
        return rates[0]
    if t >= tenors[n - 1]:
        return rates[n - 1]
    for i in range(n - 1):
        t1, t2 = tenors[i], tenors[i + 1]
        if t1 <= t <= t2:
            w = (t - t1) / (t2 - t1)
            return rates[i] + w * (rates[i + 1] - rates[i])
    return rates[n - 1]


# ── Domain Models ────────────────────────────────────────────────────────

def _point_tenor_key(p):
    """Sort key for YieldCurvePoint — avoids lambda in @computed."""
    return p.tenor_years

@ticking(exclude={"quote_ref", "fitted_rate"})
@dataclass
class YieldCurvePoint(Storable, VariableMixin):
    """Single yield curve pillar point — also an Expr leaf via VariableMixin.

    rate is @computed from quote_ref (typically a SwapQuote) — this is
    the curve fitting step.  Today it's a pass-through; in future it may
    be a complex solver (bootstrap, etc.).
    """
    __key__ = "name"

    name: str = ""
    symbol: str = ""       # IR_USD_YC_FIT.5Y
    tenor_years: float = 0.0
    fitted_rate: float = 0.0  # Set by CurveFitter
    currency: str = "USD"
    quote_ref: object = None      # The input quote (e.g. SwapQuote)
    is_fitted: bool = False       # If True, rate is set by a CurveFitter

    @traceable
    def rate(self):
        """The pillar rate. Pass-through from quote unless is_fitted=True."""
        if self.is_fitted:
            return self.fitted_rate
        
        if self.quote_ref is None:
            return 0.0
        r = getattr(self.quote_ref, 'rate', 0.0)
        if r is None:
            return 0.0
        return float(r)

    def set_fitted_rate(self, value: float):
        """Update the rate from a solver."""
        self.fitted_rate = value
        # During batch solving, we don't want to tick every intermediate update.
        # This instance-level check handles the final apply_results correctly.
        # if not getattr(self, "_is_solving", False):
        #     self.tick()
        # Actually, for PR-B, we assume the caller handles the final tick via flush() or batch.
        # Let's keep it simple: just update the value.

    def initial_guess(self) -> float:
        """Initial guess for the fitter: the quoted zero rate."""
        if self.quote_ref is None:
            return self.fitted_rate
        r = getattr(self.quote_ref, 'rate', 0.0)
        return float(r) if r is not None else 0.0

    @traceable
    def discount_factor(self):
        # Clip rate to avoid overflow/underflow
        # Floors at -0.5 (50% negative) for extreme numerical safety.
        r = float(self.rate)
        r = max(-0.5, min(5.0, r))
        return 1.0 / (1.0 + r) ** self.tenor_years

    @effect("rate")
    def on_rate(self, value):
        # Effects are for real-time propagation. During solving, 
        # we bypass these to avoid re-triggering risk ladders.
        self.tick()


@dataclass
class CurveJacobianEntry(Storable):
    """One entry in the fitter's jacobian: ∂fit_rate[output] / ∂quote_rate[input].

    Symbol convention: IR_USD_YC_JACOBIAN.<output_tenor>.<input_tenor>

    For a pass-through fitter, the jacobian is identity:
        JACOBIAN.5Y.5Y = 1.0, JACOBIAN.5Y.10Y = 0.0

    For a bootstrap/solver fitter, off-diagonals can be non-zero.
    """
    __key__ = "symbol"

    symbol: str = ""                # IR_USD_YC_JACOBIAN.5Y.10Y
    output_tenor: float = 0.0       # the fitted point's tenor
    input_tenor: float = 0.0        # the quote's tenor
    value: float = 0.0              # ∂fit_rate / ∂quote_rate
    quote_symbol: str = ""          # the quote's symbol (e.g. IR_USD_OIS_QUOTE.5Y)


@ticking(exclude={"points", "jacobian", "pillar_names", "pillar_rates", "pillar_tenors"})
@dataclass
class LinearTermDiscountCurve(Storable, CurveBase):
    """Linear zero-rate interpolation curve.

    Parameters: zero rates r_i at knot points.
    Interpolation: linear in r between knots, flat extrapolation.
    Discount factor: DF(t) = (1 + r(t))^(-t)  →  POWER() in SQL.

    This is the original/simplest curve type.  For smoother curves
    with sparser Jacobians, see IntegratedShortRateCurve.

    Implements CurveBase — can be used interchangeably with any other
    curve type by instruments and the fitter.
    """
    __key__ = "name"

    name: str = ""
    currency: str = "USD"
    points: list = field(default_factory=list)
    jacobian: list = field(default_factory=list)  # list of CurveJacobianEntry

    @traceable
    def point_count(self) -> int:
        return len(self.points)

    def _sorted_points(self):
        """Sort points by tenor — helper to avoid lambda in @computed."""
        return sorted(self.points, key=_point_tenor_key)

    def set_rates_numerical(self, rates: list[float] | np.ndarray):
        """Update pillar rates without triggering reactive effects."""
        pts = self._sorted_points()
        for i, r in enumerate(rates):
            pts[i].fitted_rate = float(r)
        
        # Invalidate internal caches
        if hasattr(self, '_pillar_rates_cache'):
            object.__setattr__(self, '_pillar_rates_cache', None)
        if hasattr(self, '_interp_cache'):
            object.__setattr__(self, '_interp_cache', {})
        if hasattr(self, '_df_cache'):
            object.__setattr__(self, '_df_cache', {})

    @traceable
    def pillar_tenors(self) -> list:
        """Sorted pillar tenors — recomputes when any point's rate changes."""
        pts = self._sorted_points()
        return [p.tenor_years for p in pts]

    @traceable
    def pillar_rates(self) -> list:
        """Sorted pillar rates — triggers reactive dependency on all point rates."""
        pts = self._sorted_points()
        return [p.rate for p in pts]

    @traceable
    def pillar_names(self) -> list:
        """Sorted pillar names."""
        pts = self._sorted_points()
        return [p.name for p in pts]

    def fwd_at(self, tenor: float, period: float = 1.0) -> float:
        """Forward rate for the period starting at tenor.

        Derived from discount factors:
            rate = (df(tenor) / df(tenor + period) - 1) / period

        This is the implied rate for borrowing from `tenor` to
        `tenor + period`.  SQL-translatable: just /, -, * on DFs.
        """
        dfs = self.df_array([tenor, tenor + period])
        if dfs[1] == 0.0:
            return 0.0
        return (dfs[0] / dfs[1] - 1.0) / period

    def fwd_array(self, tenors: list[float], period: float = 1.0) -> list[float]:
        """Batch forward rates — one df_array call for all tenors.
        For each tenor t, computes (df(t) / df(t+period) - 1) / period.
        """
        if not tenors:
            return []
        # Build pairs: [t0, t0+p, t1, t1+p, ...]
        all_tenors = []
        for t in tenors:
            all_tenors.append(t)
            all_tenors.append(t + period)
        all_dfs = self.df_array(all_tenors)

        rates = []
        for i in range(0, len(all_dfs), 2):
            df_start = all_dfs[i]
            df_end = all_dfs[i + 1]
            if df_end == 0.0:
                rates.append(0.0)
            else:
                rates.append((df_start / df_end - 1.0) / period)
        return rates

    def df_at(self, tenor: float) -> float:
        """Discount factor at any tenor: (1 + rate)^(-tenor).

        Uses _interp on pillar zero rates (not fwd_at, which is a forward rate).
        """
        r = _interp(self.pillar_tenors, self.pillar_rates, tenor)
        return (1.0 + r) ** (-tenor)

    def interp(self, t: float) -> "Expr":
        """Build an Expr tree for the interpolated rate at tenor t.

        The interpolation weights are constants (baked in at build time).
        The pillar rates are Variable leaf nodes (symbolic variables).

        Cached: calling interp(3.0) twice returns the SAME Expr object,
        enabling cross-swap sub-expression sharing.

        Returns an Expr that can:
          .eval(ctx)  → float (same as _interp)
          .to_sql()   → SQL expression
          diff(expr, "USD_OIS_5Y") → derivative Expr
        """
        cache = getattr(self, '_interp_cache', None)
        if cache is None:
            cache = {}
            object.__setattr__(self, '_interp_cache', cache)
        if t in cache:
            return cache[t]

        from reactive.expr import Const, Variable as PRExpr

        pts = self._sorted_points()
        n = len(pts)
        if n == 0:
            expr = Const(0.0)
            cache[t] = expr
            return expr

        tenors = [p.tenor_years for p in pts]

        # Use the point objects' names to build Variable leaves.
        # Variable inherits VariableMixin, matching the YieldCurvePoint
        # names, so diff() treats them identically.
        def _leaf(point):
            return PRExpr(point.name)

        # Flat extrapolation at boundaries
        if t <= tenors[0]:
            expr = _leaf(pts[0])
        elif t >= tenors[-1]:
            expr = _leaf(pts[-1])
        else:
            expr = _leaf(pts[-1])  # fallback
            for i in range(n - 1):
                t1, t2 = tenors[i], tenors[i + 1]
                if t1 <= t <= t2:
                    r1 = _leaf(pts[i])
                    r2 = _leaf(pts[i + 1])
                    w = Const((t - t1) / (t2 - t1))
                    expr = r1 + (r2 - r1) * w
                    break

        cache[t] = expr
        return expr

    def _df_expr(self, t: float) -> "Expr":
        """Build a cached Expr tree for DF(t) = (1 + R(t))^(-t)."""
        cache = getattr(self, '_df_cache', None)
        if cache is None:
            cache = {}
            object.__setattr__(self, '_df_cache', cache)
        if t in cache:
            return cache[t]

        from reactive.expr import Const
        rate_expr = self.interp(t)
        expr = (Const(1.0) + rate_expr) ** Const(-t)
        cache[t] = expr
        return expr

    def df(self, t: float):
        """Discount factor at tenor *t*.

        Returns:
            TracedFloat — when tracing is active (@traceable trace mode)
            Expr        — when building is active (@traceable)
            float       — otherwise (default debug mode)
        """
        from reactive.traced import _is_tracing, _is_building
        if _is_tracing():
            from reactive.traced import TracedFloat
            return TracedFloat(self.df_at(t), self._df_expr(t))
        if _is_building():
            return self._df_expr(t)
        return self.df_at(t)

    def fwd(self, start: float, end: float) -> "Expr":
        """Build an Expr tree for the forward rate between start and end.
        
        fwd(start, end) = (df(start) / df(end) - 1.0) / (end - start)
        """
        dt = end - start
        if dt <= 0:
            return 0.0
            
        df_start = self.df(start)
        df_end = self.df(end)
        
        return (df_start / df_end - 1.0) / dt



    def df_array(self, tenors: list[float]) -> list[float]:
        """Batch discount factors — single sort, linear sweep.

        For N tenors and M pillars, this is O(N log N + N + M) instead of
        O(N * M) from calling df_at() in a loop.
        """
        if not tenors:
            return []
        p_tenors = self.pillar_tenors
        p_rates = self.pillar_rates
        n = len(p_tenors)
        if n == 0:
            return [1.0] * len(tenors)

        # Sort query tenors but remember original order
        indexed = sorted(enumerate(tenors), key=lambda x: x[1])
        result = [0.0] * len(tenors)
        j = 0  # pillar index

        for orig_idx, t in indexed:
            # Advance pillar pointer
            while j < n - 1 and p_tenors[j + 1] < t:
                j += 1

            # Interpolate
            if t <= p_tenors[0]:
                r = p_rates[0]
            elif t >= p_tenors[n - 1]:
                r = p_rates[n - 1]
            elif p_tenors[j] == p_tenors[j + 1] if j < n - 1 else True:
                r = p_rates[j]
            else:
                t1, t2 = p_tenors[j], p_tenors[j + 1]
                w = (t - t1) / (t2 - t1)
                r = p_rates[j] + w * (p_rates[j + 1] - p_rates[j])

            result[orig_idx] = (1.0 + r) ** (-t)

        return result


    # risk_quote() and benchmark_dv01s() are inherited from CurveBase

    @effect("pillar_rates")
    def on_rates_change(self, value):
        self.tick()
