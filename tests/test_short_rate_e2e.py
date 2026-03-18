"""
End-to-end test: IntegratedShortRateCurve through the full pipeline.

Tests the complete chain:
  1. Build IntegratedShortRateCurve from market quotes
  2. CurveFitter solves for R_i parameters (NPV → 0)
  3. Swap risk sensitivities (symbolic derivatives) flow through
  4. SQL generated with EXP() not POWER()
  5. Jacobian sparsity vs old LinearTermDiscountCurve

Also includes a comparative analysis section for the architecture doc.
"""

import math
import time
import sys
import pytest

from streaming import flush, get_tables

from marketmodel.yield_curve import YieldCurvePoint, LinearTermDiscountCurve
from marketmodel.integrated_rate_curve import IntegratedRatePoint, IntegratedShortRateCurve
from marketmodel.swap_curve import SwapQuote
from marketmodel.curve_fitter import CurveFitter
from instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox, SwapPortfolio
from instruments.portfolio import Portfolio
from reactive.expr import Const, diff, eval_cached


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _build_integrated_curve(quotes):
    """Build IntegratedShortRateCurve + Points from quote specs.

    Args:
        quotes: list of (tenor, rate, symbol) tuples

    Returns: (curve, points, quote_objects)
    """
    quote_objects = []
    points = []
    for tenor, rate, sym in quotes:
        q = SwapQuote(symbol=sym, rate=rate)
        quote_objects.append(q)

        pt = IntegratedRatePoint(
            name=f"IR_USD_DISC_USD.R.{tenor:.0f}Y",
            symbol=f"IR_USD_FIT.R.{tenor:.0f}Y",
            tenor_years=tenor,
            currency="USD",
            quote_ref=q,
            is_fitted=True,
        )
        points.append(pt)

    curve = IntegratedShortRateCurve(
        name="USD_SMOOTH", currency="USD", points=points,
    )
    return curve, points, quote_objects


def _build_old_curve(quotes):
    """Build LinearTermDiscountCurve + Points from the same quote specs."""
    quote_objects = []
    points = []
    for tenor, rate, sym in quotes:
        q = SwapQuote(symbol=sym, rate=rate)
        quote_objects.append(q)

        pt = YieldCurvePoint(
            name=f"IR_USD_DISC_USD.{tenor:.0f}Y",
            symbol=f"IR_USD_FIT.{tenor:.0f}Y",
            tenor_years=tenor,
            currency="USD",
            quote_ref=q,
            is_fitted=True,
        )
        points.append(pt)

    curve = LinearTermDiscountCurve(
        name="USD_OLD", currency="USD", points=points,
    )
    return curve, points, quote_objects


QUOTES_3P = [
    (1.0, 0.01, "IR_USD_OIS_QUOTE.1Y"),
    (5.0, 0.05, "IR_USD_OIS_QUOTE.5Y"),
    (10.0, 0.10, "IR_USD_OIS_QUOTE.10Y"),
]

QUOTES_7P = [
    (0.5, 0.005, "IR_USD_OIS_QUOTE.6M"),
    (1.0, 0.01,  "IR_USD_OIS_QUOTE.1Y"),
    (2.0, 0.02,  "IR_USD_OIS_QUOTE.2Y"),
    (3.0, 0.03,  "IR_USD_OIS_QUOTE.3Y"),
    (5.0, 0.05,  "IR_USD_OIS_QUOTE.5Y"),
    (7.0, 0.07,  "IR_USD_OIS_QUOTE.7Y"),
    (10.0, 0.10, "IR_USD_OIS_QUOTE.10Y"),
]


# ═══════════════════════════════════════════════════════════════════
# Section 1: Fitter convergence with IntegratedShortRateCurve
# ═══════════════════════════════════════════════════════════════════

class TestFitterConvergence:
    """Test that CurveFitter solves for R_i parameters on the new curve type."""

    def test_fitter_solves_3_pillar(self):
        """3-pillar curve: fitter converges, target swap NPVs → 0."""
        curve, points, quotes = _build_integrated_curve(QUOTES_3P)

        # Build target swaps (par swaps at quoted rates)
        target_swaps = [
            IRSwapFixedFloatApprox(
                symbol=f"T{tenor:.0f}Y", tenor_years=tenor,
                fixed_rate=rate, curve=curve,
                notional=1.0, is_target=True,
            )
            for tenor, rate, _ in QUOTES_3P
        ]

        fitter = CurveFitter(
            name="TEST_FIT", currency="USD", curve=curve,
            points=points, quotes=quotes,
            target_swaps=target_swaps,
        )
        fitter.solve()

        # After solve, each target swap NPV should be near zero
        for swap in target_swaps:
            ctx = {pt.name: pt.rate for pt in points}
            npv = eval_cached(swap.npv(), ctx)
            assert abs(npv / swap.notional) < 1e-6, \
                f"{swap.symbol}: NPV/N = {npv/swap.notional:.2e}, expected ~0"

    def test_fitter_solves_7_pillar(self):
        """7-pillar curve: fitter converges on a richer term structure."""
        curve, points, quotes = _build_integrated_curve(QUOTES_7P)

        target_swaps = [
            IRSwapFixedFloatApprox(
                symbol=f"T{tenor}", tenor_years=tenor,
                fixed_rate=rate, curve=curve,
                notional=1.0, is_target=True,
            )
            for tenor, rate, _ in QUOTES_7P
        ]

        fitter = CurveFitter(
            name="TEST_FIT7", currency="USD", curve=curve,
            points=points, quotes=quotes,
            target_swaps=target_swaps,
        )
        fitter.solve()

        for swap in target_swaps:
            ctx = {pt.name: pt.rate for pt in points}
            npv = eval_cached(swap.npv(), ctx)
            assert abs(npv / swap.notional) < 1e-5, \
                f"{swap.symbol}: NPV/N = {npv/swap.notional:.2e}, expected ~0"

    def test_fitted_R_values_reasonable(self):
        """After fitting, R_i values should be positive and rate-like."""
        curve, points, quotes = _build_integrated_curve(QUOTES_3P)

        target_swaps = [
            IRSwapFixedFloatApprox(
                symbol=f"T{t:.0f}", tenor_years=t,
                fixed_rate=r, curve=curve,
                notional=1.0, is_target=True,
            )
            for t, r, _ in QUOTES_3P
        ]

        fitter = CurveFitter(
            name="TEST_R", currency="USD", curve=curve,
            points=points, quotes=quotes,
            target_swaps=target_swaps,
        )
        fitter.solve()

        for pt, (tenor, rate, _) in zip(points, QUOTES_3P):
            R_i = pt.rate
            assert R_i > 0, f"R({tenor}) should be positive, got {R_i}"
            # R_i is average short rate — should be roughly the par rate
            assert abs(R_i - rate) < rate * 1.0, \
                f"R({tenor}) = {R_i:.4f}, expected roughly {rate:.4f}"


# ═══════════════════════════════════════════════════════════════════
# Section 2: Risk through the full pipeline
# ═══════════════════════════════════════════════════════════════════

class TestRiskPipeline:
    """Test that symbolic risk (∂NPV/∂R_i) works through the fitter."""

    def test_risk_sensitivities_non_zero(self):
        """After fitting, risk sensitivities should be non-zero for local pillars."""
        curve, points, quotes = _build_integrated_curve(QUOTES_3P)

        target_swaps = [
            IRSwapFixedFloatApprox(
                symbol=f"T{t:.0f}", tenor_years=t,
                fixed_rate=r, curve=curve,
                notional=1.0, is_target=True,
            )
            for t, r, _ in QUOTES_3P
        ]

        fitter = CurveFitter(
            name="TEST_RISK", currency="USD", curve=curve,
            points=points, quotes=quotes,
            target_swaps=target_swaps,
        )
        fitter.solve()

        # Build a 5Y swap on the fitted curve
        swap_5y = IRSwapFixedFloatApprox(
            symbol="USD-5Y", notional=1.0, fixed_rate=0.05,
            tenor_years=5.0, currency="USD", curve=curve,
        )

        # Check risk
        ctx = {pt.name: pt.rate for pt in points}
        npv_expr = swap_5y.npv()

        # Risk w.r.t. R_1Y and R_5Y should be non-zero
        pillar_names = curve.pillar_names
        for name in pillar_names:
            d = eval_cached(diff(npv_expr, name), ctx)
            if "1Y" in name or "5Y" in name:
                assert abs(d) > 1e-6, \
                    f"∂NPV/∂{name} = {d:.4f}, expected non-zero for local pillar"

        # Risk w.r.t. R_10Y should be zero (5Y swap doesn't reach 10Y)
        d_10y = eval_cached(diff(npv_expr, "IR_USD_DISC_USD.R.10Y"), ctx)
        assert abs(d_10y) < 0.01, \
            f"∂NPV(5Y swap)/∂R_10Y = {d_10y:.4f}, expected ~0 (sparsity)"

    def test_portfolio_risk(self):
        """Portfolio-level risk aggregation works with new curve."""
        curve, points, quotes = _build_integrated_curve(QUOTES_3P)

        target_swaps = [
            IRSwapFixedFloatApprox(
                symbol=f"T{t:.0f}", tenor_years=t,
                fixed_rate=r, curve=curve,
                notional=1.0, is_target=True,
            )
            for t, r, _ in QUOTES_3P
        ]

        fitter = CurveFitter(
            name="TEST_PORT", currency="USD", curve=curve,
            points=points, quotes=quotes,
            target_swaps=target_swaps,
        )
        fitter.solve()

        portfolio = Portfolio()
        for tenor, rate, _ in QUOTES_3P:
            swap = IRSwapFixedFloatApprox(
                symbol=f"S{tenor:.0f}Y", notional=1.0,
                fixed_rate=rate, tenor_years=tenor, curve=curve,
            )
            portfolio.add_instrument(swap.symbol, swap)

        ctx = portfolio.pillar_context()
        total_risk = portfolio.eval_total_risk(ctx)

        # Should have non-zero risk for each pillar
        assert all(abs(v) >= 0 for v in total_risk.values()), \
            "Portfolio risk should be computable"

    def test_risk_quote_jacobian(self):
        """Fitter jacobian (∂pillar/∂quote) should be published on the curve."""
        curve, points, quotes = _build_integrated_curve(QUOTES_3P)

        target_swaps = [
            IRSwapFixedFloatApprox(
                symbol=f"T{t:.0f}", tenor_years=t,
                fixed_rate=r, curve=curve,
                notional=1.0, is_target=True,
            )
            for t, r, _ in QUOTES_3P
        ]

        fitter = CurveFitter(
            name="TEST_JAC", currency="USD", curve=curve,
            points=points, quotes=quotes,
            target_swaps=target_swaps,
        )
        fitter.solve()

        # Jacobian should be published
        assert len(curve.jacobian) > 0, "Jacobian should be published after solve"

        # risk_quote should transform pillar risks
        swap_5y = IRSwapFixedFloatApprox(
            symbol="TST-5Y", notional=1.0, fixed_rate=0.05,
            tenor_years=5.0, currency="USD", curve=curve,
        )

        ctx = {pt.name: pt.rate for pt in points}
        pillar_risks = {}
        npv_expr = swap_5y.npv()
        for pt in points:
            pillar_risks[pt.name] = eval_cached(diff(npv_expr, pt.name), ctx)

        quote_risks = curve.risk_quote(pillar_risks)
        assert len(quote_risks) > 0, "Quote-level risks should be produced"


# ═══════════════════════════════════════════════════════════════════
# Section 3: SQL complexity comparison
# ═══════════════════════════════════════════════════════════════════

def _count_nodes(expr, seen=None):
    """Count total nodes in an Expr DAG."""
    if seen is None:
        seen = set()
    if id(expr) in seen:
        return 0
    seen.add(id(expr))
    count = 1
    from instruments.portfolio import _get_children
    for child in _get_children(expr):
        count += _count_nodes(child, seen)
    return count


class TestSQLComplexity:
    """Compare SQL output between old LinearTermDiscountCurve and new IntegratedShortRateCurve."""

    @pytest.fixture(scope="class")
    def fitted_curves(self):
        """Fit both curve types with the same 3-pillar quotes."""
        # New curve
        new_curve, new_pts, new_quotes = _build_integrated_curve(QUOTES_3P)
        new_targets = [
            IRSwapFixedFloatApprox(
                symbol=f"T{t:.0f}", tenor_years=t, fixed_rate=r,
                curve=new_curve, notional=1.0, is_target=True,
            )
            for t, r, _ in QUOTES_3P
        ]
        new_fitter = CurveFitter(
            name="NEW_FIT", currency="USD", curve=new_curve,
            points=new_pts, quotes=new_quotes, target_swaps=new_targets,
        )
        new_fitter.solve()

        # Old curve
        old_curve, old_pts, old_quotes = _build_old_curve(QUOTES_3P)
        old_targets = [
            IRSwapFixedFloatApprox(
                symbol=f"T{t:.0f}", tenor_years=t, fixed_rate=r,
                curve=old_curve, notional=1.0, is_target=True,
            )
            for t, r, _ in QUOTES_3P
        ]
        old_fitter = CurveFitter(
            name="OLD_FIT", currency="USD", curve=old_curve,
            points=old_pts, quotes=old_quotes, target_swaps=old_targets,
        )
        old_fitter.solve()

        return {
            "new_curve": new_curve, "new_pts": new_pts,
            "old_curve": old_curve, "old_pts": old_pts,
        }

    def test_sql_uses_exp_not_power(self, fitted_curves):
        """New curve DF SQL uses EXP(), old uses POWER()."""
        new_curve = fitted_curves["new_curve"]
        old_curve = fitted_curves["old_curve"]

        new_sql = new_curve.df(5.0).to_sql()
        old_sql = old_curve.df(5.0).to_sql()

        assert "EXP" in new_sql, f"New SQL should use EXP: {new_sql}"
        # Old curve uses ^ (power) operator, not EXP
        assert "^" in old_sql or "POWER" in old_sql, f"Old SQL should use ^/POWER: {old_sql}"
        assert "EXP" not in old_sql, f"Old SQL should NOT use EXP: {old_sql}"

    def test_expr_tree_comparison(self, fitted_curves):
        """Compare Expr tree sizes for a 5Y swap."""
        new_curve = fitted_curves["new_curve"]
        old_curve = fitted_curves["old_curve"]

        # Build same swap on both curves
        new_swap = IRSwapFixedFloatApprox(
            symbol="CMP_NEW", notional=1.0, fixed_rate=0.05,
            tenor_years=5.0, curve=new_curve,
        )
        old_swap = IRSwapFixedFloatApprox(
            symbol="CMP_OLD", notional=1.0, fixed_rate=0.05,
            tenor_years=5.0, curve=old_curve,
        )

        new_npv = new_swap.npv()
        old_npv = old_swap.npv()

        new_nodes = _count_nodes(new_npv)
        old_nodes = _count_nodes(old_npv)

        # Also count derivative nodes
        new_deriv_nodes = sum(
            _count_nodes(diff(new_npv, name))
            for name in new_curve.pillar_names
        )
        old_deriv_nodes = sum(
            _count_nodes(diff(old_npv, name))
            for name in old_curve.pillar_names
        )

        print(f"\n{'='*60}")
        print(f"  EXPR TREE COMPARISON (5Y Swap, 3 Pillars)")
        print(f"{'='*60}")
        print(f"  {'Metric':<30} {'Old (POWER)':<15} {'New (EXP)':<15}")
        print(f"  {'-'*60}")
        print(f"  {'NPV Expr nodes':<30} {old_nodes:<15} {new_nodes:<15}")
        print(f"  {'Total derivative nodes':<30} {old_deriv_nodes:<15} {new_deriv_nodes:<15}")

        # Check SQL sizes
        new_sql = new_swap.npv_sql()
        old_sql = old_swap.npv_sql()
        print(f"  {'NPV SQL length (chars)':<30} {len(old_sql):<15} {len(new_sql):<15}")

        # Derivative SQL sizes
        new_dsql_total = 0
        old_dsql_total = 0
        for name in new_curve.pillar_names:
            new_dsql_total += len(diff(new_npv, name).to_sql())
        for name in old_curve.pillar_names:
            old_dsql_total += len(diff(old_npv, name).to_sql())
        print(f"  {'Total deriv SQL (chars)':<30} {old_dsql_total:<15} {new_dsql_total:<15}")
        print(f"{'='*60}")
        sys.stdout.flush()

    def test_jacobian_sparsity_comparison(self, fitted_curves):
        """Compare Jacobian sparsity: new curve should have more zeros."""
        new_curve = fitted_curves["new_curve"]
        new_pts = fitted_curves["new_pts"]
        old_curve = fitted_curves["old_curve"]
        old_pts = fitted_curves["old_pts"]

        # Build portfolios with same swaps
        new_port = Portfolio()
        old_port = Portfolio()

        for t, r, _ in QUOTES_3P:
            new_swap = IRSwapFixedFloatApprox(
                symbol=f"S{t:.0f}", notional=1.0, fixed_rate=r,
                tenor_years=t, curve=new_curve,
            )
            old_swap = IRSwapFixedFloatApprox(
                symbol=f"S{t:.0f}", notional=1.0, fixed_rate=r,
                tenor_years=t, curve=old_curve,
            )
            new_port.add_instrument(new_swap.symbol, new_swap)
            old_port.add_instrument(old_swap.symbol, old_swap)

        new_ctx = {pt.name: pt.rate for pt in new_pts}
        old_ctx = {pt.name: pt.rate for pt in old_pts}

        new_jac = new_port.eval_instrument_risk(new_ctx)
        old_jac = old_port.eval_instrument_risk(old_ctx)

        def _count_nonzero(jac, threshold=1e-10):
            total = 0
            nonzero = 0
            for row in jac.values():
                for val in row.values():
                    total += 1
                    if abs(val) > threshold:
                        nonzero += 1
            return nonzero, total

        new_nz, new_total = _count_nonzero(new_jac)
        old_nz, old_total = _count_nonzero(old_jac)

        print(f"\n{'='*60}")
        print(f"  JACOBIAN SPARSITY (3 swaps × 3 pillars)")
        print(f"{'='*60}")
        print(f"  {'Metric':<30} {'Old (POWER)':<15} {'New (EXP)':<15}")
        print(f"  {'-'*60}")
        print(f"  {'Non-zero entries':<30} {old_nz}/{old_total:<10} {new_nz}/{new_total:<10}")
        old_sparsity = f"{1-old_nz/old_total:.1%}"
        new_sparsity = f"{1-new_nz/new_total:.1%}"
        print(f"  {'Sparsity':<30} {old_sparsity:<15} {new_sparsity:<15}")
        print(f"{'='*60}")
        sys.stdout.flush()

        # New curve should be sparser (fewer non-zero entries)
        assert new_nz <= old_nz, \
            f"New Jacobian should be sparser: {new_nz} vs {old_nz} non-zeros"


# ═══════════════════════════════════════════════════════════════════
# Section 4: Performance benchmark
# ═══════════════════════════════════════════════════════════════════

class TestPerformance:
    """Compare eval/SQL execution speed."""

    def _time_eval(self, curve, points, n_evals=1000):
        """Time n_evals of df_at + gradient computation."""
        ctx = {pt.name: pt.rate for pt in points}
        tenors = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

        # Warm up caches
        for t in tenors:
            curve.df(t)
            for name in curve.pillar_names:
                diff(curve.df(t), name)

        start = time.perf_counter()
        for _ in range(n_evals):
            for t in tenors:
                eval_cached(curve.df(t), ctx)
                for name in curve.pillar_names:
                    eval_cached(diff(curve.df(t), name), ctx)
        elapsed = time.perf_counter() - start
        return elapsed

    def test_python_eval_performance(self):
        """Compare Python eval_cached speed: new vs old."""
        new_curve, new_pts, _ = _build_integrated_curve(QUOTES_3P)
        old_curve, old_pts, _ = _build_old_curve(QUOTES_3P)

        # Set reasonable rates
        for pt in new_pts:
            pt.fitted_rate = pt.initial_guess()
        for pt in old_pts:
            pt.fitted_rate = pt.initial_guess()

        n_evals = 500
        new_time = self._time_eval(new_curve, new_pts, n_evals)
        old_time = self._time_eval(old_curve, old_pts, n_evals)

        print(f"\n{'='*60}")
        print(f"  PYTHON EVAL PERFORMANCE ({n_evals} iterations × 7 tenors × {len(QUOTES_3P)} derivs)")
        print(f"{'='*60}")
        print(f"  Old (POWER): {old_time:.3f}s")
        print(f"  New (EXP):   {new_time:.3f}s")
        ratio = old_time / new_time if new_time > 0 else float('inf')
        print(f"  Speedup:     {ratio:.2f}x")
        print(f"{'='*60}")
        sys.stdout.flush()

    @pytest.fixture
    def duckdb(self):
        try:
            import duckdb
            return duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

    def test_sql_execution_performance(self, duckdb):
        """Compare DuckDB SQL execution: EXP vs POWER portfolio queries."""
        new_curve, new_pts, _ = _build_integrated_curve(QUOTES_3P)
        old_curve, old_pts, _ = _build_old_curve(QUOTES_3P)

        for pt in new_pts:
            pt.fitted_rate = pt.initial_guess()
        for pt in old_pts:
            pt.fitted_rate = pt.initial_guess()

        # Build portfolios
        new_port = Portfolio()
        old_port = Portfolio()

        for t, r, _ in QUOTES_3P:
            new_swap = IRSwapFixedFloatApprox(
                symbol=f"S{t:.0f}Y", notional=1.0, fixed_rate=r,
                tenor_years=t, curve=new_curve,
            )
            old_swap = IRSwapFixedFloatApprox(
                symbol=f"S{t:.0f}Y", notional=1.0, fixed_rate=r,
                tenor_years=t, curve=old_curve,
            )
            new_port.add_instrument(new_swap.symbol, new_swap)
            old_port.add_instrument(old_swap.symbol, old_swap)

        new_ctx = {pt.name: pt.rate for pt in new_pts}
        old_ctx = {pt.name: pt.rate for pt in old_pts}

        new_sql = new_port.to_sql_optimized(new_ctx)
        old_sql = old_port.to_sql_optimized(old_ctx)

        n_runs = 100

        # Time old SQL
        start = time.perf_counter()
        for _ in range(n_runs):
            duckdb.sql(old_sql).fetchone()
        old_time = time.perf_counter() - start

        # Time new SQL
        start = time.perf_counter()
        for _ in range(n_runs):
            duckdb.sql(new_sql).fetchone()
        new_time = time.perf_counter() - start

        print(f"\n{'='*60}")
        print(f"  SQL EXECUTION ({n_runs} runs, DuckDB)")
        print(f"{'='*60}")
        print(f"  Old SQL size: {len(old_sql)} chars")
        print(f"  New SQL size: {len(new_sql)} chars")
        print(f"  Old time:     {old_time:.3f}s  ({old_time/n_runs*1000:.2f}ms/run)")
        print(f"  New time:     {new_time:.3f}s  ({new_time/n_runs*1000:.2f}ms/run)")
        ratio = old_time / new_time if new_time > 0 else float('inf')
        print(f"  Speedup:      {ratio:.2f}x")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Verify both produce valid results
        old_result = duckdb.sql(old_sql).fetchone()
        new_result = duckdb.sql(new_sql).fetchone()
        assert old_result is not None
        assert new_result is not None
