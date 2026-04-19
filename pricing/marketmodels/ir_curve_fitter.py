"""
CurveFitter — Simultaneous solve for yield curve pillars.

This is the bridge between market data (SwapQuote) and the curve (YieldCurvePoints).
It solves for all pillar rates simultaneously by minimizing the squared NPVs
of all target swaps.

Uses Portfolio to build the pricing formula and its symbolic Jacobian
from the same Expr tree — the derivative can't drift out of sync with the
price.  The Jacobian Expr trees are built once at construction and evaluated
cheaply each solver iteration via eval_cached().
"""

from __future__ import annotations

import logging
import sys
import numpy as np
import copy
from dataclasses import dataclass, field, replace
from scipy.optimize import least_squares

from reactive.traceable import traceable
from reactive.computed import effect
from streaming import ticking
from store import Storable

from reaktiv import batch

from pricing.marketmodels.curve_base import CurveBase
from pricing.marketmodels.ir_curve_yield import CurveJacobianEntry
from pricing.marketmodels.symbols import fit_symbol, jacobian_symbol

logger = logging.getLogger(__name__)

# Sentinel constant used by the Jacobian callable for pillars that have no
# symbolic derivative in the portfolio (cross-curve dependencies in xccy fits).
from reactive.expr import Const as _Const
_ZERO_CONST = _Const(0.0)


@ticking(exclude={"target_swaps", "curve", "points", "quotes"})
@dataclass
class CurveFitter(Storable):
    """Fitter that solves for all curve points at once.

    Triggers a re-solve whenever any target swap's rate changes.
    The solver minimizes sum(swap.npv ** 2).
    """
    __key__ = "name"

    name: str = ""
    currency: str = "USD"
    
    # Inputs: target swaps whose NPVs we want to solve for zero
    target_swaps: list = field(default_factory=list)

    # Market quotes whose rate changes trigger a re-solve
    quotes: list = field(default_factory=list)
    
    # Curve we are fitting (any CurveBase implementation)
    curve: object = None
    points: list = field(default_factory=list)

    _is_solving: bool = False

    @traceable
    def quote_trigger(self) -> float:
        """Dummy signal to watch all input quote rates."""
        # Using a sum as a simple trigger
        return sum(getattr(q, "rate", 0.0) for q in self.quotes)

    # @effect("quote_trigger")
    def on_market_change(self, value):
        """Re-solve the curve when any input quote rate changes."""
        self.solve()

    # ══════════════════════════════════════════════════════════════════
    # Solver (Portfolio + symbolic Jacobian)
    # ══════════════════════════════════════════════════════════════════

    def solve(self):
        """Solve using Portfolio — symbolic Jacobian from the same Expr tree."""
        self._is_solving = True
        try:
            from pricing.instruments.portfolio import Portfolio
            from pricing.engines.python_expr import PythonEngineExpr
            from pricing.risk.firstorder_analytic import FirstOrderAnalyticRisk
            
            engine = PythonEngineExpr()
            
            # Build a unit-notional 'fit_basket' for the solver.
            # This ensures residuals and jacobians are unscaled (NPV_per_1_notional).
            fit_basket = Portfolio()
            for swap in self.target_swaps:
                # Create a shallow copy with notional=1.0
                unit_swap = replace(swap, notional=1.0)
                # Important: link the unit_swap to the same curves as the original
                for attr in ['curve', 'discount_curve', 'projection_curve']:
                    if hasattr(swap, attr):
                        setattr(unit_swap, attr, getattr(swap, attr))
                
                name = getattr(swap, "symbol", "UNKNOWN")
                fit_basket.add_instrument(name, unit_swap)

            # Ordered lists for numpy conversion
            swap_names = fit_basket.names
            
            # Pre-compute the Jacobian Expr trees (done once, reused every iteration)
            risk_calc = FirstOrderAnalyticRisk(fit_basket)
            jac_exprs = risk_calc.jacobian() # {name: {label: Expr}}

            # Initial guess: bootstrap pillar-by-pillar for a tighter starting point.
            x0 = self._bootstrap_initial_guess(fit_basket, swap_names)

            from reactive.evaluation import PillarContext
            def _build_ctx(x) -> PillarContext:
                """Build a pillar rate context dict from the solver's x vector."""
                ctx = PillarContext()
                ctx.is_solving = True
                # Use one of the swaps to get the base context (curves, etc)
                if self.target_swaps:
                    ctx.update(self.target_swaps[0].pillar_context())
                
                ctx.update({
                    pt.name: float(rate_val)
                    for pt, rate_val in zip(self.points, x)
                })
                return ctx

            def objective(x):
                """Residual vector: NPV_i / notional_i for each target swap."""
                # 1. Update pillar rates (for reactive model sync)
                for pt, rate_val in zip(self.points, x):
                    pt.fitted_rate = float(rate_val)

                # 2. Evaluate residuals via Engine (using unit-notional basket)
                ctx = _build_ctx(x)
                residuals = engine.npvs(fit_basket, ctx)
                result = np.array([residuals[name] for name in swap_names])

                print(f"    [Fitter] Iteration x={x} -> NormNPVs={result}")
                sys.stdout.flush()
                return result

            def jacobian(x):
                """∂residual_i / ∂pillar_j — for THIS fitter's own pillars only.

                scipy expects a matrix of shape (m_swaps, len(x0)) = (m, n_own).
                ``portfolio.pillar_names`` is the union of ALL variables referenced
                by the swaps, which in a multi-curve / xccy setup is wider than our
                own pillar count.  We differentiate only w.r.t. ``self.points``
                (which is what x0 encodes) to match scipy's expected shape.
                """
                from reactive.expr import eval_cached
                ctx = _build_ctx(x)
                # Only our own pillar names — same order as x0
                own_pillar_names = [pt.name for pt in self.points]
                matrix = []
                for name in swap_names:
                    row = [
                        eval_cached(jac_exprs[name].get(pn, _ZERO_CONST), ctx)
                        for pn in own_pillar_names
                    ]
                    matrix.append(row)
                return np.array(matrix)

            # Solve!
            # Use 'trf' (Trust Region Reflective) instead of 'lm' to support bounds.
            # LM is slightly faster but 'trf' is more robust to bad market data.
            bounds = (-0.1, 0.5) # Hard physiological bounds (-10% to +50%)
            print(f"  [Fitter] Starting solve ({len(self.points)} pillars, method=trf)...")
            res = least_squares(objective, x0, jac=jacobian, method='trf', bounds=bounds)
            print(f"  [Fitter] Solve complete. Status: {res.status}")
            sys.stdout.flush()

            if not res.success:
                logger.warning(f"Fitter solver failed: {res.message}")
            
            # Map results back to YieldCurvePoints with a batch update
            with batch():
                for pt, final_rate in zip(self.points, res.x):
                    pt.set_fitted_rate(float(final_rate))

            # Publish the Jacobian after batch closes and values propagate
            if res.success:
                print(f"  [Fitter] Starting Jacobian calculation...")
                sys.stdout.flush()
                self._publish_jacobian(res)
                print(f"  [Fitter] Jacobian published.")
                sys.stdout.flush()
        finally:
            self._is_solving = False

    # ══════════════════════════════════════════════════════════════════
    # Bootstrap initial guess
    # ══════════════════════════════════════════════════════════════════

    def _bootstrap_initial_guess(self, portfolio, swap_names) -> np.ndarray:
        """Bootstrap pillar-by-pillar to get a tight initial guess.

        For each pillar (sorted by tenor), solves a single-variable problem:
        with shorter pillars fixed, find the rate that zeros the corresponding
        target swap NPV. Uses secant steps — cheap and robust.

        Returns the x0 vector for the full LM solver.
        """
        from reactive.expr import eval_cached

        n = len(self.points)
        x0 = np.array([
            p.initial_guess() if hasattr(p, 'initial_guess')
            else p.quote_ref.rate
            for p in self.points
        ])

        # Match each point to its target swap by tenor
        swap_by_tenor = {}
        for swap, sname in zip(self.target_swaps, swap_names):
            tenor = getattr(swap, 'tenor_years', 0.0)
            swap_by_tenor[tenor] = (swap, sname)

        # Bootstrap short → long
        order = sorted(range(n), key=lambda i: self.points[i].tenor_years)
        print(f"    [Fitter] Bootstrapping {self.name} ({n} pillars)...")
        sys.stdout.flush()

        for pi in order:
            pt = self.points[pi]
            match = swap_by_tenor.get(pt.tenor_years)
            if match is None:
                continue
            swap, swap_name = match

            npv_expr = portfolio.npv_exprs.get(swap_name)
            if npv_expr is None:
                continue

            # --- Ultra-Fast Path: Direct Numerical Solve ---
            # Pre-calculate schedule and parameters once per pillar
            notional = getattr(swap, 'notional', 10e6)
            fixed_rate = getattr(swap, 'fixed_rate', 0.0)
            target_tenor = float(pt.tenor_years)
            
            # Get payment schedule (tenors and year fractions)
            # Both Approx and OIS swaps support these legacy helpers or properties
            if hasattr(swap, 'target_dates'):
                pay_tenors = [float(t) for t in swap.target_dates()]
                resets = [0.0] + pay_tenors[:-1]
                if hasattr(swap, 'reset_dates'):
                    resets = [float(t) for t in swap.reset_dates()]
                taus = [pay_tenors[i] - resets[i] for i in range(len(pay_tenors))]
            else:
                # Fallback for bullet instruments
                pay_tenors = [target_tenor]
                taus = [target_tenor]

            def _eval_npv(r_val):
                x0[pi] = r_val
                # High-speed numerical update (bypasses reactive system completely)
                if hasattr(self.curve, "set_rates_numerical"):
                    self.curve.set_rates_numerical(x0)
                
                # Direct vectorized NPV calculation: payoff = float - fixed
                # (Standard simplification for initial guess - assumes unit notional)
                dfs = self.curve.df_array(pay_tenors)
                annuity = sum(t * df for t, df in zip(taus, dfs))
                # For shortcut float leg: PV = (1 - DF_maturity)
                float_leg_pv = (1.0 - dfs[-1])
                return (float_leg_pv - fixed_rate * annuity)

            # Secant method
            try:
                r_a = x0[pi]
                f_a = _eval_npv(r_a)
                bump = max(abs(r_a) * 0.01, 1e-6)
                r_b = r_a + bump
                f_b = _eval_npv(r_b)

                for _ in range(8):
                    if abs(f_b) < 1e-12: break
                    denom = f_b - f_a
                    if abs(denom) < 1e-25: break
                    r_new = r_b - f_b * (r_b - r_a) / denom
                    r_new = max(-0.1, min(0.5, r_new))
                    r_a, f_a = r_b, f_b
                    r_b, f_b = r_new, _eval_npv(r_new)

                x0[pi] = r_b
            except Exception:
                pass  # keep initial_guess on failure

        # Final pass: Apply bootstrapped values to reactive points once
        for i, p in enumerate(self.points):
            p.fitted_rate = float(x0[i])
        if hasattr(self.curve, 'invalidate_caches'):
            self.curve.invalidate_caches()

        return x0

    # ══════════════════════════════════════════════════════════════════
    # Shared: Jacobian publication
    # ══════════════════════════════════════════════════════════════════

    def _publish_jacobian(self, solver_res):
        """Publish the fitter sensitivity: ∂pillar_rate / ∂quote_rate.

        ``solver_res.jac`` from scipy least_squares has shape (m, n) where
        m = number of residuals (swaps) and n = number of free variables
        that were differentiated.  In a multi-curve xccy setup, the residuals
        of one fitter's swaps depend on pillars from another curve, so scipy
        may return a wider Jacobian than the number of pillars we're fitting
        here.  We only care about the first ``len(self.points)`` columns,
        which correspond to the x0 we passed in (our own pillar rates).
        """
        if self.curve is None:
            return

        n_points = len(self.points)
        j_full = solver_res.jac          # shape (m_swaps, n_all_vars)
        # Slice to our own pillars — columns 0..n_points-1 in x0 ordering.
        j_pillars = j_full[:, :n_points] if j_full.shape[1] > n_points else j_full

        try:
            j_inv = np.linalg.pinv(j_pillars)   # shape (n_points, m_swaps)

            new_jacobian_entries = []
            for i, p_out in enumerate(self.points):
                for j, s_in in enumerate(self.target_swaps):
                    dv01_val = getattr(s_in, "dv01", 0.0)
                    if dv01_val is None:
                        dv01_val = 0.0
                    notional = getattr(s_in, "notional",
                               getattr(s_in, "leg1_notional", 1.0))
                    dv01_scaled = (float(dv01_val) * 10000.0) / notional

                    val = j_inv[i, j] * dv01_scaled

                    q_sym = (self.quotes[j].symbol
                             if (self.quotes and j < len(self.quotes))
                             else s_in.symbol)
                    symbol = jacobian_symbol(self.currency, p_out.name, q_sym)
                    entry = CurveJacobianEntry(
                        symbol=symbol,
                        output_tenor=p_out.tenor_years,
                        input_tenor=s_in.tenor_years,
                        value=float(val),
                        quote_symbol=q_sym,
                    )
                    new_jacobian_entries.append(entry)

            self.curve.jacobian = new_jacobian_entries
            self.curve.tick()

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Jacobian inversion failed: {e}")

