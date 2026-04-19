"""
test_expr_ergonomics.py — Expr tree ergonomic behaviour tests.

Covers two properties that allow instrument code to look like ordinary Python
float arithmetic while transparently building a correct, flat Expr DAG:

  1. Implicit scalar wrapping  — plain ints/floats on either side of an Expr
     operator are auto-wrapped via _wrap(), so ``expr * 0.0001`` is identical
     to ``expr * Const(0.0001)``.

  2. Sum flattening / accumulator collapse  — starting with ``pv = 0.0`` and
     accumulating ``pv += expr_term`` produces a *flat* Sum node of depth 1,
     not a left-recursive BinOp chain of depth N.  The number of coupon periods
     does not increase tree depth.

  3. Zero-identity stripping  — ``Const(0) + x`` returns ``x`` directly; the
     seed zero is dropped on the first accumulation step.

  4. Scalar early-return from @traceable  — returning a bare 0.0 from a
     @traceable body is legal; get_expr() auto-wraps it via _wrap().

See docs/architecture/reactive_expr_dag.md §Natural Code Ergonomics.
"""

import pytest
from reactive.expr import Const, Sum, BinOp, Variable, Expr, eval_cached, _wrap


# ── helpers ──────────────────────────────────────────────────────────────────

def _depth(expr) -> int:
    """Measure the maximum BinOp nesting depth of an expression tree."""
    if isinstance(expr, (Const, Variable)):
        return 0
    if isinstance(expr, Sum):
        # Sum is always depth 1 by definition (flat node)
        return 1
    if isinstance(expr, BinOp):
        return 1 + max(_depth(expr.left), _depth(expr.right))
    return 0


def make_var(name: str, value: float) -> Variable:
    return Variable(name)


# ── 1. Implicit scalar wrapping ───────────────────────────────────────────────

class TestImplicitScalarWrapping:
    """Plain Python scalars on either side of an Expr operator are auto-wrapped.

    Code like ``expr * 0.0001`` or ``0.5 * expr`` should be exactly equivalent
    to ``expr * Const(0.0001)`` — without any explicit Const() call.
    """

    def setup_method(self):
        self.x = Variable("x")
        self.ctx = {"x": 3.0}

    def test_right_operand_mul(self):
        """expr * scalar  — __mul__ calls _wrap(other)"""
        e1 = self.x * 2.0
        e2 = self.x * Const(2.0)
        assert eval_cached(e1, self.ctx) == pytest.approx(eval_cached(e2, self.ctx))

    def test_left_operand_mul(self):
        """scalar * expr  — __rmul__ is triggered"""
        e1 = 2.0 * self.x
        e2 = Const(2.0) * self.x
        assert eval_cached(e1, self.ctx) == pytest.approx(eval_cached(e2, self.ctx))

    def test_right_operand_add(self):
        """expr + scalar  — __add__ calls _wrap(other)"""
        e1 = self.x + 1.0
        e2 = self.x + Const(1.0)
        assert eval_cached(e1, self.ctx) == pytest.approx(eval_cached(e2, self.ctx))

    def test_left_operand_add(self):
        """scalar + expr  — __radd__ is triggered"""
        e1 = 1.0 + self.x
        e2 = Const(1.0) + self.x
        assert eval_cached(e1, self.ctx) == pytest.approx(eval_cached(e2, self.ctx))

    def test_right_operand_sub(self):
        e1 = self.x - 0.5
        e2 = self.x - Const(0.5)
        assert eval_cached(e1, self.ctx) == pytest.approx(eval_cached(e2, self.ctx))

    def test_left_operand_sub(self):
        e1 = 10.0 - self.x
        e2 = Const(10.0) - self.x
        assert eval_cached(e1, self.ctx) == pytest.approx(eval_cached(e2, self.ctx))

    def test_right_operand_div(self):
        e1 = self.x / 4.0
        e2 = self.x / Const(4.0)
        assert eval_cached(e1, self.ctx) == pytest.approx(eval_cached(e2, self.ctx))

    def test_chained_scalars_same_value(self):
        """Chained scalars: expr * a * b  equals  expr * (a * b)  — constant folding."""
        e1 = self.x * 2.0 * 3.0
        assert eval_cached(e1, self.ctx) == pytest.approx(3.0 * 2.0 * self.ctx["x"])

    def test_comparison_with_scalar(self):
        """expr > scalar  — __gt__ calls _wrap(other); no Const needed."""
        e1 = self.x > 0
        e2 = self.x > Const(0)
        assert eval_cached(e1, self.ctx) == eval_cached(e2, self.ctx)

    def test_integer_scalar_wrapped(self):
        """Integers (not just floats) are also auto-wrapped."""
        e = self.x * 10
        assert eval_cached(e, self.ctx) == pytest.approx(30.0)

    def test_wrap_function_directly(self):
        """_wrap() returns Expr unchanged, wraps scalars as Const."""
        v = Variable("y")
        assert _wrap(v) is v
        wrapped = _wrap(5.0)
        assert isinstance(wrapped, Const)
        assert wrapped.value == 5.0


# ── 2. Sum flattening / accumulator collapse ──────────────────────────────────

class TestSumFlattening:
    """The accumulator pattern ``pv = 0.0; pv += term`` stays flat.

    This is the key ergonomic property: code that looks like a simple float
    loop transparently produces a depth-1 Sum node, matching the explicit
    Sum([t1, t2, ...]) construction exactly.
    """

    def _accumulate(self, n: int):
        """Build a sum of n Variable terms using the accumulator pattern."""
        ctx = {}
        pv = 0.0
        for i in range(n):
            name = f"v{i}"
            ctx[name] = float(i + 1)
            pv += Variable(name)
        return pv, ctx

    def test_single_term_no_sum_node(self):
        """First += strips the 0.0 seed; result is just the term itself."""
        x = Variable("x")
        pv = 0.0
        pv += x
        # Const(0) + x → x (zero-identity, not wrapped in Sum)
        assert isinstance(pv, Variable)

    def test_two_terms_is_flat_sum(self):
        """Second += produces Sum([t1, t2]) — depth 1."""
        x, y = Variable("x"), Variable("y")
        pv = 0.0
        pv += x
        pv += y
        assert isinstance(pv, Sum)
        assert len(pv.terms) == 2
        assert _depth(pv) == 1

    def test_five_terms_still_depth_one(self):
        """Five terms still produce a flat Sum — depth does not grow."""
        pv, ctx = self._accumulate(5)
        assert isinstance(pv, Sum)
        assert _depth(pv) == 1

    def test_twenty_terms_still_depth_one(self):
        """20-period coupon loop: tree stays flat regardless of period count."""
        pv, ctx = self._accumulate(20)
        assert isinstance(pv, Sum)
        assert _depth(pv) == 1

    def test_accumulator_value_equals_list_sum(self):
        """Accumulator result equals Sum([...]) explicit construction."""
        vars_ = [Variable(f"v{i}") for i in range(6)]
        ctx = {f"v{i}": float(i) for i in range(6)}

        # Accumulator style
        pv_acc = 0.0
        for v in vars_:
            pv_acc += v

        # Explicit list style
        pv_list = Sum(vars_)

        assert eval_cached(pv_acc, ctx) == pytest.approx(eval_cached(pv_list, ctx))

    def test_sum_plus_sum_flattens(self):
        """Sum + Sum merges into a single flat Sum (no nesting)."""
        a, b, c, d = [Variable(f"v{i}") for i in range(4)]
        s1 = Sum([a, b])
        s2 = Sum([c, d])
        merged = s1 + s2
        assert isinstance(merged, Sum)
        assert len(merged.terms) == 4
        assert _depth(merged) == 1

    def test_accumulator_with_scaled_terms(self):
        """Each term is ``df * dt`` (BinOp): result is Sum of BinOps, depth 1."""
        ctx = {}
        pv = 0.0
        n = 4
        for i in range(n):
            df = Variable(f"df{i}")
            ctx[f"df{i}"] = 0.95 ** (i + 1)
            dt = 0.25  # scalar, auto-wrapped on multiply
            pv += df * dt
        assert isinstance(pv, Sum)
        assert _depth(pv) == 1
        expected = sum(0.95 ** (i + 1) * 0.25 for i in range(n))
        assert eval_cached(pv, ctx) == pytest.approx(expected)


# ── 3. Zero-identity stripping ────────────────────────────────────────────────

class TestZeroIdentityStripping:
    """Const(0) + x returns x, not Sum([Const(0), x])."""

    def test_zero_plus_expr(self):
        x = Variable("x")
        result = Const(0) + x
        assert result is x  # identity: 0 + x = x

    def test_zero_plus_zero_folds(self):
        result = Const(0) + Const(0)
        assert isinstance(result, Const)
        assert result.value == 0

    def test_expr_plus_zero(self):
        x = Variable("x")
        result = x + Const(0)
        assert result is x

    def test_float_zero_accumulator_seed(self):
        """float 0.0 used as seed: _wrap(0.0) = Const(0.0), then identity fires."""
        x = Variable("x")
        pv = 0.0        # plain Python float — not yet an Expr
        pv += x         # __radd__: _wrap(0.0) + x → Const(0)+x → x
        assert isinstance(pv, Variable)


# ── 4. Scalar early-return from @traceable ────────────────────────────────

class _StubCurve:
    """Minimal stand-in for a discount curve — no @ticking, no DH connection."""
    def df(self, t: float):
        from reactive.expr import Const
        return Const(1.0)  # flat 0% curve for simplicity

    def fwd(self, t1: float, t2: float):
        from reactive.expr import Const
        return Const(0.0)


class TestComputedExprScalarReturn:
    """@traceable bodies may return plain scalars; get_expr() wraps them."""

    def _make_dummy(self, return_value):
        """Build a class with a @traceable method returning return_value."""
        from reactive.traceable import traceable
        from reactive.expr import Expr
        stub = _StubCurve()

        rv = return_value  # capture

        class _Dummy:
            discount_curve = stub

            def __init__(self):
                # mock a reactive registry where 'prop' returns its traced value
                from unittest.mock import MagicMock
                node = MagicMock()
                node.read.side_effect = lambda: _Dummy.prop._compute(self)
                self._reactive = {"prop": node}

            @traceable
            def prop(self) -> Expr:
                return rv

        return _Dummy

    def test_get_expr_wraps_float(self):
        """Accessing a @traceable property as a function should wrap a plain float return as Const."""
        from reactive.expr import Const as C
        _Dummy = self._make_dummy(0.0)
        inst = _Dummy()
        result = inst.prop()  # Calling the @traceable property returns the Expr
        assert isinstance(result, C)
        assert result.value == 0.0

    def test_get_expr_wraps_nonzero_float(self):
        """Non-zero scalar returns are also wrapped correctly."""
        from reactive.expr import Const as C
        _Dummy = self._make_dummy(1.0)
        inst = _Dummy()
        result = inst.prop()
        assert isinstance(result, C)
        assert result.value == 1.0
