"""
Tests for the integrated short rate curve and supporting expr changes.

Covers:
  1. Exp/Log differentiation in expr.py
  2. IntegratedShortRateCurve numerical correctness
  3. Curve continuity and smoothness
  4. Jacobian sparsity
  5. SQL round-trip via DuckDB
  6. CurveBase interface compliance
  7. Equivalence with LinearTermDiscountCurve for flat curves
"""

import math
import pytest

from reactive.expr import (
    Const, Variable, Exp, Log, Func, diff, eval_cached, BinOp,
)
from marketmodel.yield_curve import LinearTermDiscountCurve, YieldCurvePoint
from marketmodel.integrated_rate_curve import (
    IntegratedShortRateCurve, IntegratedRatePoint,
)
from marketmodel.curve_base import CurveBase


# ═══════════════════════════════════════════════════════════════════
# Section 1: Exp / Log differentiation
# ═══════════════════════════════════════════════════════════════════

class TestExpDiff:
    """Test symbolic differentiation of exp() and log()."""

    def test_exp_eval(self):
        """Exp(x) evaluates to math.exp(x)."""
        x = Variable("x")
        expr = Exp(x)
        assert math.isclose(expr.eval({"x": 1.0}), math.e, rel_tol=1e-12)
        assert math.isclose(expr.eval({"x": 0.0}), 1.0, rel_tol=1e-12)
        assert math.isclose(expr.eval({"x": -1.0}), 1.0 / math.e, rel_tol=1e-12)

    def test_exp_diff(self):
        """d/dx exp(x) = exp(x)."""
        x = Variable("x")
        expr = Exp(x)
        deriv = diff(expr, "x")
        # At x=1: d/dx exp(x) = exp(1) = e
        ctx = {"x": 1.0}
        assert math.isclose(eval_cached(deriv, ctx), math.e, rel_tol=1e-10)

    def test_exp_chain_rule(self):
        """d/dx exp(2x) = 2 * exp(2x)."""
        x = Variable("x")
        expr = Exp(Const(2.0) * x)
        deriv = diff(expr, "x")
        ctx = {"x": 1.0}
        expected = 2.0 * math.exp(2.0)
        assert math.isclose(eval_cached(deriv, ctx), expected, rel_tol=1e-10)

    def test_exp_neg_x(self):
        """d/dx exp(-x) = -exp(-x)."""
        x = Variable("x")
        expr = Exp(-x)
        deriv = diff(expr, "x")
        ctx = {"x": 0.5}
        expected = -math.exp(-0.5)
        assert math.isclose(eval_cached(deriv, ctx), expected, rel_tol=1e-10)

    def test_exp_of_polynomial(self):
        """d/dx exp(x² + x) = (2x + 1) * exp(x² + x)."""
        x = Variable("x")
        expr = Exp(x * x + x)
        deriv = diff(expr, "x")
        ctx = {"x": 0.5}
        x_val = 0.5
        expected = (2 * x_val + 1) * math.exp(x_val ** 2 + x_val)
        assert math.isclose(eval_cached(deriv, ctx), expected, rel_tol=1e-10)

    def test_log_diff(self):
        """d/dx log(x) = 1/x."""
        x = Variable("x")
        expr = Log(x)
        deriv = diff(expr, "x")
        ctx = {"x": 2.0}
        assert math.isclose(eval_cached(deriv, ctx), 0.5, rel_tol=1e-10)

    def test_log_chain_rule(self):
        """d/dx log(2x + 1) = 2 / (2x + 1)."""
        x = Variable("x")
        expr = Log(Const(2.0) * x + Const(1.0))
        deriv = diff(expr, "x")
        ctx = {"x": 1.0}
        expected = 2.0 / 3.0
        assert math.isclose(eval_cached(deriv, ctx), expected, rel_tol=1e-10)

    def test_exp_reuses_node(self):
        """The derivative of exp(f) should reuse the same exp(f) Expr object."""
        x = Variable("x")
        exp_expr = Exp(x)
        deriv = diff(exp_expr, "x")
        # d/dx exp(x) = exp(x) * 1.0
        # The *1 simplification means the result IS exp(x) directly,
        # or a BinOp wrapping it. Either way, it should reference the node.
        assert deriv is exp_expr or (
            isinstance(deriv, BinOp) and deriv.op == "*" and
            (deriv.left is exp_expr or deriv.right is exp_expr)
        )

    def test_exp_eval_cached(self):
        """eval_cached handles Func('exp') natively."""
        x = Variable("x")
        expr = Exp(x)
        deriv = diff(expr, "x")
        ctx = {"x": 2.0}
        # Two evaluations should give same result
        v1 = eval_cached(deriv, ctx)
        v2 = eval_cached(deriv, ctx)
        assert math.isclose(v1, math.exp(2.0), rel_tol=1e-10)
        assert v1 == v2

    def test_exp_to_sql(self):
        """Exp(x) compiles to EXP() SQL."""
        x = Variable("x")
        expr = Exp(x)
        sql = expr.to_sql()
        assert "EXP" in sql


# ═══════════════════════════════════════════════════════════════════
# Section 2: IntegratedShortRateCurve — numerical correctness
# ═══════════════════════════════════════════════════════════════════

def _make_flat_curve(rate=0.05, tenors=(1.0, 5.0, 10.0)):
    """Helper: build an IntegratedShortRateCurve with a flat short rate.

    R_i = average short rate = rate (constant for flat curve).
    I_i = R_i × t_i = rate × t.
    DF(t) = exp(-rate × t).
    """
    points = []
    for t in tenors:
        pt = IntegratedRatePoint(
            name=f"R_{t:.0f}Y",
            symbol=f"IR_USD_FIT.R.{t:.0f}Y",
            tenor_years=t,
            fitted_rate=rate,  # R_i = average rate (same for all tenors when flat)
            is_fitted=True,
        )
        points.append(pt)
    return IntegratedShortRateCurve(name="USD_TEST", points=points)


def _make_steep_curve():
    """Helper: curve with increasing short rate (1% @ 1Y, 3% @ 5Y, 5% @ 10Y).

    We specify cumulative integrals I_i, then convert to average rates R_i = I_i/t_i.
    """
    # (tenor, cumulative_integral I_i)
    data = [
        (1.0, 0.01 * 1.0),                # I(1) = 0.01
        (5.0, 0.01 * 1.0 + 0.02 * 4.0),   # I(5) = 0.09
        (10.0, 0.09 + 0.04 * 5.0),         # I(10) = 0.29
    ]
    points = []
    for t, I_val in data:
        R_val = I_val / t  # average rate = cumulative integral / tenor
        pt = IntegratedRatePoint(
            name=f"R_{t:.0f}Y",
            symbol=f"IR_USD_FIT.R.{t:.0f}Y",
            tenor_years=t,
            fitted_rate=R_val,
            is_fitted=True,
        )
        points.append(pt)
    return IntegratedShortRateCurve(name="USD_STEEP", points=points)


class TestFlatCurve:
    """Flat short rate: r(t) = r₀ constant → R(T) = r₀·T, DF(T) = exp(-r₀·T)."""

    def test_df_at_knots(self):
        """DF at knot points should match exp(-r*T)."""
        curve = _make_flat_curve(rate=0.05)
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(-0.05 * t)
            actual = curve.df_at(t)
            assert math.isclose(actual, expected, rel_tol=1e-10), \
                f"DF({t}): {actual} != {expected}"

    def test_df_between_knots(self):
        """DF between knots should also match (flat curve is trivial)."""
        curve = _make_flat_curve(rate=0.05)
        for t in [0.5, 2.5, 7.5]:
            expected = math.exp(-0.05 * t)
            actual = curve.df_at(t)
            assert math.isclose(actual, expected, rel_tol=1e-6), \
                f"DF({t}): {actual} != {expected}"

    def test_I_at_knots(self):
        """I(t) at knots should equal R_i × t_i = rate × t for flat curve."""
        curve = _make_flat_curve(rate=0.03)
        for t in [1.0, 5.0, 10.0]:
            assert math.isclose(curve._I_at(t), 0.03 * t, rel_tol=1e-10)

    def test_forward_rate_flat(self):
        """Forward rate on a flat curve should be approximately the flat rate."""
        curve = _make_flat_curve(rate=0.04)
        for t in [1.0, 3.0, 7.0]:
            fwd = curve.fwd_at(t, period=1.0)
            # For truly flat short rate, fwd = R(t+1) - R(t) = r*1 = r
            assert math.isclose(fwd, 0.04, rel_tol=1e-4), \
                f"fwd({t}, 1.0) = {fwd}, expected ~0.04"


class TestSteepCurve:
    """Non-trivial curve — test interpolation correctness."""

    def test_I_continuity_at_knots(self):
        """I(t) should be continuous at knot boundaries."""
        curve = _make_steep_curve()
        pts = curve._sorted_points()
        for pt in pts:
            t = pt.tenor_years
            I_below = curve._I_at(t - 1e-10)
            I_exact = curve._I_at(t)
            I_above = curve._I_at(t + 1e-10)
            assert math.isclose(I_below, I_exact, rel_tol=1e-6), \
                f"I discontinuous from below at t={t}"
            assert math.isclose(I_exact, I_above, rel_tol=1e-6), \
                f"I discontinuous from above at t={t}"

    def test_area_preservation(self):
        """∫_{t_i}^{t_{i+1}} r(s)ds should equal I_{i+1} - I_i."""
        curve = _make_steep_curve()
        pts = curve._sorted_points()
        R_vals = [pt.rate for pt in pts]
        T_vals = [pt.tenor_years for pt in pts]
        I_vals = [R_vals[i] * T_vals[i] for i in range(len(pts))]

        for i in range(len(pts) - 1):
            # Numerical integration of r(t) from T[i] to T[i+1]
            n_steps = 10000
            h = (T_vals[i + 1] - T_vals[i]) / n_steps
            integral = 0.0
            for k in range(n_steps):
                t = T_vals[i] + (k + 0.5) * h
                I_left = curve._I_at(t - h / 2.0)
                I_right = curve._I_at(t + h / 2.0)
                # r(t) ≈ (I(t+ε) - I(t-ε)) / (2ε)
                r_approx = (I_right - I_left) / h
                integral += r_approx * h

            expected = I_vals[i + 1] - I_vals[i]
            assert math.isclose(integral, expected, rel_tol=1e-4), \
                f"Area not preserved on [{T_vals[i]}, {T_vals[i+1]}]: " \
                f"integral={integral}, ΔI={expected}"

    def test_df_positive(self):
        """Discount factors should always be positive."""
        curve = _make_steep_curve()
        for t_x10 in range(0, 120):
            t = t_x10 / 10.0
            assert curve.df_at(t) > 0, f"DF({t}) should be positive"

    def test_df_monotone_decreasing(self):
        """For positive rates, DF(t) should decrease with t."""
        curve = _make_steep_curve()
        prev_df = 1.0
        for t_x10 in range(1, 120):
            t = t_x10 / 10.0
            df = curve.df_at(t)
            assert df <= prev_df + 1e-12, \
                f"DF({t}) = {df} > DF({(t_x10-1)/10.0}) = {prev_df}"
            prev_df = df


# ═══════════════════════════════════════════════════════════════════
# Section 3: Symbolic Expr — interp, df, fwd
# ═══════════════════════════════════════════════════════════════════

class TestSymbolicExpr:
    """Test the Expr tree construction for the integrated rate curve."""

    def test_df_expr_eval_matches_numerical(self):
        """Symbolic DF expression should match numerical df_at."""
        curve = _make_steep_curve()
        pts = curve._sorted_points()
        ctx = {pt.name: pt.rate for pt in pts}

        for t_x10 in [5, 15, 30, 50, 75, 95]:
            t = t_x10 / 10.0
            expr_val = eval_cached(curve.df(t), ctx)
            num_val = curve.df_at(t)
            assert math.isclose(expr_val, num_val, rel_tol=1e-10), \
                f"Expr DF({t}) = {expr_val} != numerical {num_val}"

    def test_fwd_expr_eval_matches_numerical(self):
        """Symbolic forward expression should match numerical fwd_at."""
        curve = _make_steep_curve()
        pts = curve._sorted_points()
        ctx = {pt.name: pt.rate for pt in pts}

        for start in [1.0, 3.0, 5.0, 7.0]:
            expr_val = eval_cached(curve.fwd(start, start + 1.0), ctx)
            num_val = curve.fwd_at(start, 1.0)
            assert math.isclose(expr_val, num_val, rel_tol=1e-10), \
                f"Expr fwd({start}, {start+1}) = {expr_val} != numerical {num_val}"

    def test_df_expr_caching(self):
        """df(t) called twice returns the SAME Expr object."""
        curve = _make_steep_curve()
        expr1 = curve.df(3.0)
        expr2 = curve.df(3.0)
        assert expr1 is expr2, "df(t) should return cached Expr"

    def test_interp_expr_caching(self):
        """interp(t) called twice returns the SAME Expr object."""
        curve = _make_steep_curve()
        expr1 = curve.interp(3.0)
        expr2 = curve.interp(3.0)
        assert expr1 is expr2, "interp(t) should return cached Expr"

    def test_df_uses_exp(self):
        """df(t) Expr should use Func('exp'), not BinOp('**')."""
        curve = _make_steep_curve()
        expr = curve.df(3.0)
        assert isinstance(expr, Func) and expr.name == "exp", \
            f"df(t) should be Func('exp'), got {type(expr).__name__}"

    def test_df_sql_uses_exp(self):
        """df(t).to_sql() should use EXP(), not POWER()."""
        curve = _make_steep_curve()
        sql = curve.df(3.0).to_sql()
        assert "EXP" in sql, f"SQL should use EXP: {sql}"
        assert "POWER" not in sql, f"SQL should NOT use POWER: {sql}"


# ═══════════════════════════════════════════════════════════════════
# Section 4: Jacobian sparsity
# ═══════════════════════════════════════════════════════════════════

class TestJacobianSparsity:
    """Verify that diff(df(t), R_j) produces sparse derivatives."""

    def test_df_depends_on_bracketing_knots_only(self):
        """diff(df(t), R_j) should be zero for knots far from t."""
        curve = _make_flat_curve(rate=0.05)
        pts = curve._sorted_points()
        ctx = {pt.name: pt.rate for pt in pts}

        # DF at t=3.0 — bracket is [R_1Y, R_5Y], not R_10Y
        df_expr = curve.df(3.0)

        # Should have non-zero derivative w.r.t. R_1Y and R_5Y
        d_R1 = eval_cached(diff(df_expr, "R_1Y"), ctx)
        d_R5 = eval_cached(diff(df_expr, "R_5Y"), ctx)

        # Should have ZERO derivative w.r.t. R_10Y (not in the bracket)
        d_R10 = eval_cached(diff(df_expr, "R_10Y"), ctx)

        assert abs(d_R1) > 1e-10, f"∂DF/∂R_1Y should be non-zero, got {d_R1}"
        assert abs(d_R5) > 1e-10, f"∂DF/∂R_5Y should be non-zero, got {d_R5}"
        assert abs(d_R10) < 1e-15, f"∂DF/∂R_10Y should be zero, got {d_R10}"

    def test_fwd_depends_on_local_knots_only(self):
        """diff(fwd(2, 3), R_j) should be zero for distant knots."""
        curve = _make_flat_curve(rate=0.05)
        pts = curve._sorted_points()
        ctx = {pt.name: pt.rate for pt in pts}

        fwd_expr = curve.fwd(2.0, 3.0)

        # fwd(2,3) bracket at [R_1Y, R_5Y] — should NOT depend on R_10Y
        d_R10 = eval_cached(diff(fwd_expr, "R_10Y"), ctx)
        assert abs(d_R10) < 1e-15, f"∂fwd/∂R_10Y should be zero, got {d_R10}"

    def test_derivative_numerical_check(self):
        """Symbolic derivatives should match finite differences."""
        curve = _make_flat_curve(rate=0.05)
        pts = curve._sorted_points()
        ctx = {pt.name: pt.rate for pt in pts}

        df_expr = curve.df(3.0)
        bump = 1e-7

        for pt in pts:
            # Symbolic
            d_sym = eval_cached(diff(df_expr, pt.name), ctx)

            # Finite difference
            ctx_up = dict(ctx)
            ctx_up[pt.name] = ctx[pt.name] + bump
            ctx_dn = dict(ctx)
            ctx_dn[pt.name] = ctx[pt.name] - bump
            d_fd = (eval_cached(df_expr, ctx_up) - eval_cached(df_expr, ctx_dn)) / (2 * bump)

            if abs(d_sym) < 1e-15 and abs(d_fd) < 1e-10:
                continue  # Both essentially zero
            assert math.isclose(d_sym, d_fd, rel_tol=1e-4), \
                f"∂DF/∂{pt.name}: sym={d_sym}, fd={d_fd}"


# ═══════════════════════════════════════════════════════════════════
# Section 5: SQL round-trip
# ═══════════════════════════════════════════════════════════════════

class TestSqlRoundTrip:
    """Test SQL generation and DuckDB execution."""

    @pytest.fixture
    def duckdb(self):
        try:
            import duckdb
            return duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

    def test_df_sql_executes(self, duckdb):
        """df(t).to_sql() should produce valid DuckDB SQL."""
        curve = _make_flat_curve(rate=0.05)
        pts = curve._sorted_points()

        df_expr = curve.df(3.0)
        sql_body = df_expr.to_sql()

        # Build a SELECT with the pillar values as column aliases
        col_defs = ", ".join(f"{pt.rate} AS \"{pt.name}\"" for pt in pts)
        query = f"SELECT {sql_body} AS df FROM (SELECT {col_defs})"

        result = duckdb.sql(query).fetchone()[0]
        expected = math.exp(-0.05 * 3.0)
        assert math.isclose(result, expected, rel_tol=1e-6), \
            f"SQL DF(3.0) = {result}, expected {expected}"

    def test_derivative_sql_executes(self, duckdb):
        """diff(df(t), R_j).to_sql() should produce valid SQL."""
        curve = _make_flat_curve(rate=0.05)
        pts = curve._sorted_points()

        df_expr = curve.df(3.0)
        deriv = diff(df_expr, "R_5Y")
        sql_body = deriv.to_sql()

        col_defs = ", ".join(f"{pt.rate} AS \"{pt.name}\"" for pt in pts)
        query = f"SELECT {sql_body} AS ddf FROM (SELECT {col_defs})"

        result = duckdb.sql(query).fetchone()[0]
        # Should be a finite, reasonable number (derivative includes t_i factor)
        assert abs(result) < 10.0, f"Derivative seems too large: {result}"


# ═══════════════════════════════════════════════════════════════════
# Section 6: CurveBase interface compliance
# ═══════════════════════════════════════════════════════════════════

class TestCurveBaseInterface:
    """Both curve types should implement CurveBase identically."""

    def test_yield_curve_is_curvebase(self):
        """LinearTermDiscountCurve should be a CurveBase instance."""
        yc = LinearTermDiscountCurve(name="test")
        assert isinstance(yc, CurveBase)

    def test_integrated_curve_is_curvebase(self):
        """IntegratedShortRateCurve should be a CurveBase instance."""
        ic = IntegratedShortRateCurve(name="test")
        assert isinstance(ic, CurveBase)

    def test_interface_methods_present(self):
        """Both types should have all CurveBase methods."""
        for CurveType in [LinearTermDiscountCurve, IntegratedShortRateCurve]:
            assert hasattr(CurveType, 'df')
            assert hasattr(CurveType, 'fwd')
            assert hasattr(CurveType, 'interp')
            assert hasattr(CurveType, 'df_at')
            assert hasattr(CurveType, 'fwd_at')
            assert hasattr(CurveType, 'df_array')
            assert hasattr(CurveType, 'risk_quote')
            assert hasattr(CurveType, 'benchmark_dv01s')


# ═══════════════════════════════════════════════════════════════════
# Section 7: Cross-curve equivalence
# ═══════════════════════════════════════════════════════════════════

class TestCrossCurveEquivalence:
    """On a flat curve, both LinearTermDiscountCurve and IntegratedShortRateCurve
    should produce numerically similar discount factors."""

    def test_flat_curve_df_equivalence(self):
        """Compare DF values from both curve types for a flat rate."""
        rate = 0.05
        tenors = [1.0, 5.0, 10.0]

        # Old curve: zero rates
        old_points = [
            YieldCurvePoint(
                name=f"ZR_{t:.0f}Y",
                tenor_years=t,
                fitted_rate=rate,
                is_fitted=True,
            )
            for t in tenors
        ]
        old_curve = LinearTermDiscountCurve(name="OLD", points=old_points)

        # New curve: integrated rates
        new_curve = _make_flat_curve(rate=rate, tenors=tuple(tenors))

        # Compare DFs at various tenors
        for t in [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]:
            df_old = old_curve.df_at(t)
            df_new = new_curve.df_at(t)

            # Note: exp(-r*t) ≈ (1+r)^(-t) for small r, with ln(1+r) ≈ r
            # The continuous vs discrete compounding difference is expected
            # and grows with tenor. At 10Y and r=5%, the difference is ~1.2%.
            assert math.isclose(df_old, df_new, rel_tol=0.02), \
                f"DF({t}): old={df_old:.6f}, new={df_new:.6f} differ by >2%"

    def test_continuous_compounding_relationship(self):
        """The two curves should agree when r_zero = ln(1 + r_discrete)."""
        r_discrete = 0.05
        r_continuous = math.log(1.0 + r_discrete)  # Exact mapping
        tenors = [1.0, 5.0, 10.0]

        # Old: discrete compounding at r_discrete
        old_pts = [
            YieldCurvePoint(name=f"Z_{t:.0f}", tenor_years=t,
                           fitted_rate=r_discrete, is_fitted=True)
            for t in tenors
        ]
        old_curve = LinearTermDiscountCurve(name="OLD", points=old_pts)

        # New: continuous compounding at r_continuous (R_i = average rate)
        new_pts = [
            IntegratedRatePoint(name=f"R_{t:.0f}", tenor_years=t,
                                fitted_rate=r_continuous, is_fitted=True)
            for t in tenors
        ]
        new_curve = IntegratedShortRateCurve(name="NEW", points=new_pts)

        # Now they should match exactly
        for t in [1.0, 2.0, 5.0, 7.5, 10.0]:
            df_old = old_curve.df_at(t)
            df_new = new_curve.df_at(t)
            assert math.isclose(df_old, df_new, rel_tol=1e-6), \
                f"DF({t}): old={df_old:.8f}, new={df_new:.8f} (should match)"
