"""
reactive.traced — TracedFloat and tracing context.

A ``TracedFloat`` is a ``float`` subclass that silently carries an ``Expr``
tree alongside its concrete value.  In a debugger watch window it looks
like a plain float, but ``._expr`` reveals its symbolic provenance.

Tracing context
~~~~~~~~~~~~~~~
``_start_tracing() / _stop_tracing()`` set a thread-local flag that tells
curve methods (``df()``, ``interp()``, …) to return ``TracedFloat`` instead
of plain ``float``.  The ``@traceable`` decorator manages this flag.

For backward compatibility with ``@computed_expr``, a separate
``_start_building() / _stop_building()`` flag tells curves to return raw
``Expr`` (the old behaviour).

Usage::

    from reactive.traced import TracedFloat, _is_tracing

    x = TracedFloat(3.14, Variable("pi"))
    y = x * 2          # → TracedFloat(6.28, Variable("pi") * Const(2))
    float(y)            # → 6.28
    y._expr             # → BinOp('*', Variable("pi"), Const(2))
"""

from __future__ import annotations

import math as _math
import contextvars

from reactive.expr import Const, Expr, Func

# ---------------------------------------------------------------------------
# Context-local tracing / building context
# ---------------------------------------------------------------------------

_tracing_depth: contextvars.ContextVar[int] = contextvars.ContextVar("tracing_depth", default=0)
_building_depth: contextvars.ContextVar[int] = contextvars.ContextVar("building_depth", default=0)


def _start_tracing() -> None:
    """Increment the tracing depth (reentrant)."""
    _tracing_depth.set(_tracing_depth.get() + 1)


def _stop_tracing() -> None:
    """Decrement the tracing depth."""
    _tracing_depth.set(max(0, _tracing_depth.get() - 1))


def _is_tracing() -> bool:
    """Return True if any caller has activated tracing."""
    return _tracing_depth.get() > 0


def _start_building() -> None:
    """Activate Expr-building mode (for @computed_expr backward compat)."""
    _building_depth.set(_building_depth.get() + 1)


def _stop_building() -> None:
    """Deactivate Expr-building mode."""
    _building_depth.set(max(0, _building_depth.get() - 1))


def _is_building() -> bool:
    """True if Expr-building mode is active (used by curve.df())."""
    return _building_depth.get() > 0


# ---------------------------------------------------------------------------
# TracedFloat
# ---------------------------------------------------------------------------

def _other_expr(other: object) -> Expr | None:
    """Extract the Expr from *other*, or None if not possible."""
    # __expr__ protocol (TracedFloat, _TracedCallable, etc.)
    _get = getattr(other, "__expr__", None)
    if _get is not None:
        return _get()
    return None


class TracedFloat(float):
    """A float that carries its Expr provenance.

    Behaves exactly like a ``float`` in all contexts::

        isinstance(x, float)   → True
        sum([x, y, z])          → works (returns float — no tree)
        f"{x:.4f}"              → works
        x > 0.5                 → bool (standard comparison)

    Access the symbolic tree via ``._expr``::

        x._expr   → Expr node (Variable, BinOp, Func, …)
    """

    __slots__ = ("_expr",)

    def __new__(cls, value: float, expr: Expr | None = None) -> TracedFloat:
        obj = super().__new__(cls, float(value))
        obj._expr = expr if expr is not None else Const(value)
        return obj

    # -- __expr__ protocol ------------------------------------------------

    def __expr__(self) -> Expr:
        """Return the Expr tree (used by _wrap and other TracedFloat ops)."""
        return self._expr

    # -- helpers ----------------------------------------------------------

    def _binop(self, other: object, py_op) -> TracedFloat | float:
        """Binary operation: compute float result + compose Expr."""
        o_expr = _other_expr(other)
        if o_expr is not None:
            try:
                o_val = float(other)
                val = py_op(float(self), o_val)
            except (TypeError, ValueError):
                # other is pure Expr (no float value) — return Expr
                return py_op(self._expr, o_expr)
            except OverflowError:
                val = 1e100 # Graceful limit
            return TracedFloat(val, py_op(self._expr, o_expr))
        try:
            o_val = float(other)
            val = py_op(float(self), o_val)
        except (TypeError, ValueError):
            return NotImplemented
        except OverflowError:
            val = 1e100
        return TracedFloat(val,
                           py_op(self._expr, Const(o_val)))

    def _rbinop(self, other: object, py_op) -> TracedFloat | float:
        """Reverse binary operation."""
        o_expr = _other_expr(other)
        if o_expr is not None:
            try:
                o_val = float(other)
                val = py_op(o_val, float(self))
            except (TypeError, ValueError):
                return py_op(o_expr, self._expr)
            except OverflowError:
                val = 1e100
            return TracedFloat(val, py_op(o_expr, self._expr))
        try:
            o_val = float(other)
            val = py_op(o_val, float(self))
        except (TypeError, ValueError):
            return NotImplemented
        except OverflowError:
            val = 1e100
        return TracedFloat(val,
                           py_op(Const(o_val), self._expr))

    # -- Arithmetic operators ---------------------------------------------

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._rbinop(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._rbinop(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._rbinop(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._rbinop(other, lambda a, b: a / b)

    def __pow__(self, other):
        return self._binop(other, lambda a, b: a ** b)

    def __rpow__(self, other):
        return self._rbinop(other, lambda a, b: a ** b)

    def __neg__(self):
        return TracedFloat(-float(self), -self._expr)

    def __abs__(self):
        from reactive.expr import UnaryOp
        return TracedFloat(abs(float(self)), UnaryOp("abs", self._expr))

    # -- repr: looks like a plain float -----------------------------------

    def __repr__(self):
        return float.__repr__(self)

    def __str__(self):
        return float.__str__(self)


# ---------------------------------------------------------------------------
# Traced math functions — replacements for math.exp, math.log, etc.
# ---------------------------------------------------------------------------

def exp(x: float | TracedFloat) -> float | TracedFloat:
    """exp(x) — TracedFloat-aware replacement for math.exp."""
    if isinstance(x, TracedFloat):
        from reactive.expr import Exp
        val = float(x)
        if val > 700:
            return TracedFloat(1e100, Exp(x._expr))
        if val < -700:
            return TracedFloat(0.0, Exp(x._expr))
        return TracedFloat(_math.exp(val), Exp(x._expr))
    return _math.exp(x)


def log(x: float | TracedFloat) -> float | TracedFloat:
    """log(x) — TracedFloat-aware replacement for math.log."""
    if isinstance(x, TracedFloat):
        from reactive.expr import Log
        return TracedFloat(_math.log(float(x)), Log(x._expr))
    return _math.log(x)


def sqrt(x: float | TracedFloat) -> float | TracedFloat:
    """sqrt(x) — TracedFloat-aware replacement for math.sqrt."""
    if isinstance(x, TracedFloat):
        return TracedFloat(_math.sqrt(float(x)), Func("sqrt", [x._expr]))
    return _math.sqrt(x)


def fabs(x: float | TracedFloat) -> float | TracedFloat:
    """fabs(x) — TracedFloat-aware replacement for math.fabs."""
    if isinstance(x, TracedFloat):
        from reactive.expr import UnaryOp
        return TracedFloat(_math.fabs(float(x)), UnaryOp("abs", x._expr))
    return _math.fabs(x)
