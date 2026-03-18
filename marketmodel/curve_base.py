"""
curve_base — Abstract base class for all yield curve implementations.

Any curve type (linear zero-rate, integrated short-rate, log-linear, etc.)
must implement CurveBase.  Instruments and the fitter only depend on this
interface, making the interpolation method a pluggable implementation detail
that can be chosen per curve instance.

SQL compilation traces through the Expr-returning methods — implementations
must ensure all math is SQL-translatable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from store import Storable
from reactive.computed import computed, effect
from streaming import ticking


def _point_tenor_key(p):
    """Sort key for curve points — avoids lambda in @computed."""
    return p.tenor_years


class CurveBase(ABC):
    """Abstract interface for yield curve implementations.

    Instruments call:
        curve.df(t)            → Expr tree for discount factor  (symbolic)
        curve.fwd(start, end)  → Expr tree for forward rate     (symbolic)
        curve.df_at(t)         → float  (numerical)
        curve.fwd_at(t, p)     → float  (numerical)
        curve.df_array(tenors) → list[float]  (batch numerical)

    The fitter calls:
        curve._sorted_points() → sorted list of point objects
        curve.pillar_names     → list of variable names
        curve.pillar_tenors    → list of tenors
        curve.pillar_rates     → list of current parameter values
        curve.risk_quote(...)  → jacobian multiplication

    Subclasses implement the interpolation strategy independently of
    currency — a single curve class can be used for any currency or
    curve purpose (discount, projection, etc.).
    """

    # -- Pillar access (shared across all curve types) -----------------------

    @abstractmethod
    def _sorted_points(self) -> list:
        """Return points sorted by tenor."""

    @property
    @abstractmethod
    def pillar_names(self) -> list[str]:
        """Sorted pillar variable names."""

    @property
    @abstractmethod
    def pillar_tenors(self) -> list[float]:
        """Sorted pillar tenors."""

    @property
    @abstractmethod
    def pillar_rates(self) -> list[float]:
        """Sorted pillar parameter values (interpretation depends on curve type)."""

    # -- Symbolic Expr builders (the core of each curve type) ----------------

    @abstractmethod
    def df(self, t: float) -> "Expr":
        """Build an Expr tree for the discount factor at tenor t.

        Must be cached: same tenor → same Expr object (enables
        sub-expression sharing across multiple instruments).
        """

    @abstractmethod
    def fwd(self, start: float, end: float) -> "Expr":
        """Build an Expr tree for the forward rate between start and end."""

    @abstractmethod
    def interp(self, t: float) -> "Expr":
        """Build an Expr tree for the interpolated parameter at tenor t.

        What 'interpolated parameter' means depends on the curve type:
          - LinearZeroRateCurve: interpolated zero rate
          - IntegratedShortRateCurve: interpolated integrated rate R(t)
        """

    # -- Numerical evaluation (default implementations) ----------------------

    def df_at(self, tenor: float) -> float:
        """Numerical discount factor at any tenor.

        Default: evaluate the symbolic Expr tree.  Subclasses may override
        with a more efficient direct computation.
        """
        from reactive.expr import eval_cached
        expr = self.df(tenor)
        pts = self._sorted_points()
        ctx = {p.name: p.rate for p in pts}
        return eval_cached(expr, ctx)

    def fwd_at(self, tenor: float, period: float = 1.0) -> float:
        """Numerical forward rate for period starting at tenor.

        Default: uses df_at for backward compatibility.
        """
        df_start = self.df_at(tenor)
        df_end = self.df_at(tenor + period)
        if df_end == 0.0:
            return 0.0
        return (df_start / df_end - 1.0) / period

    def df_array(self, tenors: list[float]) -> list[float]:
        """Batch numerical discount factors.

        Default: loop over df_at.  Subclasses may override for efficiency.
        """
        return [self.df_at(t) for t in tenors]

    def fwd_array(self, tenors: list[float], period: float = 1.0) -> list[float]:
        """Batch forward rates."""
        return [self.fwd_at(t, period) for t in tenors]

    # -- Risk helpers (shared across all curve types) -----------------------

    def risk_quote(self, pillar_risks: dict[str, float]) -> dict[str, float]:
        """Map ∂npv/∂pillar_rate → ∂npv/∂quote_rate using the fitter's jacobian.

        Applies the jacobian matrix:
            ∂npv/∂quote[j] = Σ_i  ∂npv/∂pillar[i]  ×  ∂pillar[i]/∂quote[j]

        For a pass-through fitter, the jacobian is identity.
        """
        jacobian = getattr(self, 'jacobian', [])
        if not jacobian:
            return dict(pillar_risks)

        quote_risks: dict[str, float] = {}
        for entry in jacobian:
            pillar_name = None
            for pt in self._sorted_points():
                if pt.tenor_years == entry.output_tenor:
                    pillar_name = pt.name
                    break
            if pillar_name is None or pillar_name not in pillar_risks:
                continue

            pillar_risk = pillar_risks[pillar_name]
            quote_key = entry.quote_symbol if entry.quote_symbol else entry.symbol
            quote_risks[quote_key] = quote_risks.get(quote_key, 0.0) + \
                pillar_risk * entry.value

        return quote_risks

    def benchmark_dv01s(self) -> dict[str, float]:
        """Return the DV01 (per 1M notional) for each benchmark quote."""
        results = {}
        jacobian = getattr(self, 'jacobian', [])
        quotes = {}
        if not jacobian:
            for pt in getattr(self, 'points', []):
                quotes[pt.symbol] = pt.tenor_years
        else:
            for entry in jacobian:
                quotes[entry.quote_symbol] = entry.input_tenor

        for q_sym, tenor in quotes.items():
            annuity = 0.0
            t = 1.0
            while t <= tenor:
                annuity += self.df_at(t)
                t += 1.0
            if t - 1.0 < tenor:
                annuity += (tenor - (t - 1.0)) * self.df_at(tenor)
            results[q_sym] = 100.0 * annuity

        return results
