"""
Expression tree for reactive computations.

Each node compiles to three targets:
- eval(ctx)    → Python value (powers reaktiv Computed)
- to_sql(col)  → PostgreSQL JSONB expression (DB push-down)
- to_pure(var) → Legend Pure expression (Legend integration)

Operator overloading builds the tree — no computation happens at definition time.
"""

import json
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Expr(ABC):
    """Abstract expression node. All concrete nodes subclass this."""

    @abstractmethod
    def eval(self, ctx: dict) -> Any:
        """Evaluate this expression against a context dict."""

    @abstractmethod
    def to_sql(self, col: str = "data") -> str:
        """Compile to a PostgreSQL JSONB expression fragment."""

    @abstractmethod
    def to_pure(self, var: str = "$row") -> str:
        """Compile to a Legend Pure expression string."""

    @abstractmethod
    def to_deephaven(self) -> str:
        """Compile to a Deephaven Java/Groovy formula string."""

    @abstractmethod
    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""

    # -- Arithmetic operators ------------------------------------------------

    def __add__(self, other: object) -> "Expr":
        r = _wrap(other)
        if isinstance(self, Const) and self.value == 0: return r
        if isinstance(r, Const) and r.value == 0: return self
        
        # Constant folding
        if isinstance(self, Const) and isinstance(r, Const):
            if isinstance(self.value, (int, float)) and isinstance(r.value, (int, float)):
                return Const(self.value + r.value)

        # Optimization: Flatten Sums
        if isinstance(self, Sum) or isinstance(r, Sum):
            terms = []
            if isinstance(self, Sum): terms.extend(self.terms)
            else: terms.append(self)
            if isinstance(r, Sum): terms.extend(r.terms)
            else: terms.append(r)
            return Sum(terms)

        # Default: upgrade to Sum for depth-1 optimization
        return Sum([self, r])

    def __radd__(self, other: object) -> "Expr":
        return _wrap(other) + self

    def __sub__(self, other: object) -> "Expr":
        r = _wrap(other)
        if isinstance(r, Const) and r.value == 0: return self
        
        # Constant folding
        if isinstance(self, Const) and isinstance(r, Const):
            if isinstance(self.value, (int, float)) and isinstance(r.value, (int, float)):
                return Const(self.value - r.value)
                
        # 0 - x -> -x
        if isinstance(self, Const) and self.value == 0:
            return -r
            
        return BinOp("-", self, r)

    def __rsub__(self, other: object) -> "Expr":
        return _wrap(other) - self

    def __mul__(self, other: object) -> "Expr":
        r = _wrap(other)
        if isinstance(self, Const):
            if self.value == 0: return Const(0.0)
            if self.value == 1: return r
        if isinstance(r, Const):
            if r.value == 0: return Const(0.0)
            if r.value == 1: return self
            
        # Constant folding
        if isinstance(self, Const) and isinstance(r, Const):
            if isinstance(self.value, (int, float)) and isinstance(r.value, (int, float)):
                return Const(self.value * r.value)

        return BinOp("*", self, r)

    def __rmul__(self, other: object) -> "Expr":
        return _wrap(other) * self

    def __truediv__(self, other: object) -> "Expr":
        r = _wrap(other)
        if isinstance(r, Const):
            if r.value == 1: return self
            if r.value == 0: return Const(0.0) # avoid div by zero in trees
            
        # Constant folding
        if isinstance(self, Const) and isinstance(r, Const):
            if isinstance(self.value, (int, float)) and isinstance(r.value, (int, float)):
                if r.value != 0:
                    return Const(self.value / r.value)
                    
        if isinstance(self, Const) and self.value == 0:
            return Const(0.0)
        return BinOp("/", self, r)

    def __rtruediv__(self, other: object) -> "Expr":
        return BinOp("/", _wrap(other), self)

    def __mod__(self, other: object) -> "Expr":
        return BinOp("%", self, _wrap(other))

    def __rmod__(self, other: object) -> "Expr":
        return BinOp("%", _wrap(other), self)

    def __pow__(self, other: object) -> "Expr":
        return BinOp("**", self, _wrap(other))

    def __rpow__(self, other: object) -> "Expr":
        return BinOp("**", _wrap(other), self)

    def __neg__(self) -> "Expr":
        return UnaryOp("neg", self)

    def __abs__(self) -> "Expr":
        return UnaryOp("abs", self)

    # -- Comparison operators ------------------------------------------------

    def __gt__(self, other: object) -> "Expr":
        return BinOp(">", self, _wrap(other))

    def __lt__(self, other: object) -> "Expr":
        return BinOp("<", self, _wrap(other))

    def __ge__(self, other: object) -> "Expr":
        return BinOp(">=", self, _wrap(other))

    def __le__(self, other: object) -> "Expr":
        return BinOp("<=", self, _wrap(other))

    def __eq__(self, other: object) -> "Expr":  # type: ignore[override]  # DSL: builds Expr tree
        if not isinstance(other, Expr) and other is None:
            return NotImplemented
        return BinOp("==", self, _wrap(other))

    def __ne__(self, other: object) -> "Expr":  # type: ignore[override]  # DSL: builds Expr tree
        if not isinstance(other, Expr) and other is None:
            return NotImplemented
        return BinOp("!=", self, _wrap(other))

    # -- Logical operators (use & | ~ since and/or/not can't be overridden) --

    def __and__(self, other: object) -> "Expr":
        return BinOp("and", self, _wrap(other))

    def __rand__(self, other: object) -> "Expr":
        return BinOp("and", _wrap(other), self)

    def __or__(self, other: object) -> "Expr":
        return BinOp("or", self, _wrap(other))

    def __ror__(self, other: object) -> "Expr":
        return BinOp("or", _wrap(other), self)

    def __invert__(self) -> "Expr":
        return UnaryOp("not", self)

    # -- String methods (chainable) ------------------------------------------

    def length(self) -> "Expr":
        return StrOp("length", self)

    def upper(self) -> "Expr":
        return StrOp("upper", self)

    def lower(self) -> "Expr":
        return StrOp("lower", self)

    def contains(self, substring: object) -> "Expr":
        return StrOp("contains", self, _wrap(substring))

    def starts_with(self, prefix: object) -> "Expr":
        return StrOp("starts_with", self, _wrap(prefix))

    def concat(self, other: object) -> "Expr":
        return StrOp("concat", self, _wrap(other))

    # -- Null helpers --------------------------------------------------------

    def is_null(self) -> "Expr":
        return IsNull(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_json()})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(value: object) -> Expr:
    """Wrap a Python literal as a Const if it's not already an Expr."""
    if isinstance(value, Expr):
        return value
    return Const(value)


_SQL_OPS = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%", "**": "^",
    ">": ">", "<": "<", ">=": ">=", "<=": "<=", "==": "=", "!=": "!=",
    "and": "AND", "or": "OR",
}

_PURE_OPS = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%", "**": "^",
    ">": ">", "<": "<", ">=": ">=", "<=": "<=", "==": "==", "!=": "!=",
    "and": "&&", "or": "||",
}


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

class Const(Expr):
    """A constant literal value."""

    def __init__(self, value: object) -> None:
        self.value = value

    def eval(self, ctx: dict) -> Any:
        return self.value

    def to_sql(self, col: str = "data") -> str:
        if isinstance(self.value, str):
            escaped = self.value.replace("'", "''")
            return f"'{escaped}'"
        if isinstance(self.value, bool):
            return "TRUE" if self.value else "FALSE"
        if self.value is None:
            return "NULL"
        return str(self.value)

    def to_pure(self, var: str = "$row") -> str:
        if isinstance(self.value, str):
            escaped = self.value.replace("'", "\\'")
            return f"'{escaped}'"
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        if self.value is None:
            return "[]"
        return str(self.value)

    def to_deephaven(self) -> str:
        if isinstance(self.value, str):
            escaped = self.value.replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        if self.value is None:
            return "null"
        return str(self.value)

    def to_json(self) -> dict:
        return {"type": "Const", "value": self.value}


class Field(Expr):
    """A reference to a field on the current object."""

    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, ctx: dict) -> Any:
        return ctx[self.name]

    def to_sql(self, col: str = "data") -> str:
        return f"({col}->>'{self.name}')"

    def to_pure(self, var: str = "$row") -> str:
        return f"{var}.{self.name}"

    def to_deephaven(self) -> str:
        return self.name

    def to_json(self) -> dict:
        return {"type": "Field", "name": self.name}


class VariableMixin:
    """Mixin: any object with a `.name` attribute gains Expr-leaf behaviour.

    This is the bridge between domain objects and the Expr tree. A class
    that mixes this in can be used directly as a leaf node in expression
    trees, supporting eval/to_sql/to_pure/diff.

    The mixin does NOT inherit from Expr (to avoid operator-overloading
    conflicts with @dataclass).  Instead, diff() and eval_cached() check
    for isinstance(x, VariableMixin).
    """

    # Subclass must provide: self.name (str)

    def expr_eval(self, ctx: dict) -> Any:
        return ctx[self.name]

    def expr_to_sql(self, col: str = "data") -> str:
        return f'"{self.name}"'

    def expr_to_pure(self, var: str = "$row") -> str:
        return f"${self.name}"

    def expr_to_deephaven(self) -> str:
        return self.name

    def expr_to_json(self) -> dict:
        return {"type": "Variable", "name": self.name}


class Variable(Expr, VariableMixin):
    """Standalone Expr leaf for a named variable (e.g. a market quote or pillar).

    For domain objects (like YieldCurvePoint), prefer mixing in
    VariableMixin directly so the object IS the leaf node.
    This class exists for cases where you need a standalone leaf.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, ctx: dict) -> Any:
        return self.expr_eval(ctx)

    def to_sql(self, col: str = "data") -> str:
        return self.expr_to_sql(col)

    def to_pure(self, var: str = "$row") -> str:
        return self.expr_to_pure(var)

    def to_deephaven(self) -> str:
        return self.expr_to_deephaven()

    def to_json(self) -> dict:
        return self.expr_to_json()

    def __repr__(self) -> str:
        return f"Variable({self.name!r})"



# ---------------------------------------------------------------------------
# Composite nodes
# ---------------------------------------------------------------------------

class BinOp(Expr):
    """Binary operation: left op right."""

    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def eval(self, ctx: dict) -> Any:
        left_val = self.left.eval(ctx)
        right_val = self.right.eval(ctx)
        if self.op == "+":
            return left_val + right_val
        if self.op == "-":
            return left_val - right_val
        if self.op == "*":
            return left_val * right_val
        if self.op == "/":
            return left_val / right_val if right_val != 0 else 0
        if self.op == "%":
            return left_val % right_val
        if self.op == "**":
            return left_val ** right_val
        if self.op == ">":
            return left_val > right_val
        if self.op == "<":
            return left_val < right_val
        if self.op == ">=":
            return left_val >= right_val
        if self.op == "<=":
            return left_val <= right_val
        if self.op == "==":
            return left_val == right_val
        if self.op == "!=":
            return left_val != right_val
        if self.op == "and":
            return left_val and right_val
        if self.op == "or":
            return left_val or right_val
        raise ValueError(f"Unknown binary op: {self.op}")

    def to_sql(self, col: str = "data") -> str:
        l_sql = self.left.to_sql(col)
        r_sql = self.right.to_sql(col)
        sql_op = _SQL_OPS[self.op]
        # Numeric fields from JSONB are text — cast for arithmetic/comparison
        if self.op in ("+", "-", "*", "/", "%", "**", ">", "<", ">=", "<="):
            l_sql = _cast_numeric_sql(self.left, col)
            r_sql = _cast_numeric_sql(self.right, col)
        return f"({l_sql} {sql_op} {r_sql})"

    def to_pure(self, var: str = "$row") -> str:
        l_pure = self.left.to_pure(var)
        r_pure = self.right.to_pure(var)
        pure_op = _PURE_OPS[self.op]
        return f"({l_pure} {pure_op} {r_pure})"

    def to_deephaven(self) -> str:
        l_dh = self.left.to_deephaven()
        r_dh = self.right.to_deephaven()
        if self.op == "**":
            return f"Math.pow({l_dh}, {r_dh})"
        # For 'and'/'or', Deephaven uses Java && / || which maps nicely
        dh_op = _PURE_OPS[self.op]
        return f"({l_dh} {dh_op} {r_dh})"

    def to_json(self) -> dict:
        return {
            "type": "BinOp",
            "op": self.op,
            "left": self.left.to_json(),
            "right": self.right.to_json(),
        }


class Sum(Expr):
    """Flat summation of N terms — depth 1 regardless of term count.

    Replaces the deep left-recursive BinOp('+') chains that arise from
    accumulator loops like ``pv = Const(0); pv += term_i``.

    Usage:
        terms = [df1 * Const(c1), df2 * Const(c2), ...]
        total = Sum(terms)               # depth 1
        # vs: Const(0) + t1 + t2 + ...  # depth N
    """

    def __init__(self, terms: list[Expr]) -> None:
        # Light symbolic cleanup: aggregate constants
        numeric_total = 0.0
        other_terms = []
        for t in terms:
            if isinstance(t, Const) and isinstance(t.value, (int, float)):
                numeric_total += t.value
            elif isinstance(t, Sum):
                # Extra safety: nested sums shouldn't happen with __add__ logic but handled here
                for sub in t.terms:
                    if isinstance(sub, Const) and isinstance(sub.value, (int, float)):
                        numeric_total += sub.value
                    else:
                        other_terms.append(sub)
            else:
                other_terms.append(t)
        
        # New terms: lead with the aggregate constant if non-zero, or if it's the only term
        final_terms = []
        if numeric_total != 0.0 or not other_terms:
            final_terms.append(Const(numeric_total))
        final_terms.extend(other_terms)
        self.terms = final_terms

    def eval(self, ctx: dict) -> Any:
        return sum(t.eval(ctx) for t in self.terms)

    def to_sql(self, col: str = "data") -> str:
        if not self.terms:
            return "0"
        parts = [_cast_numeric_sql(t, col) for t in self.terms]
        return "(" + " + ".join(parts) + ")"

    def to_pure(self, var: str = "$row") -> str:
        if not self.terms:
            return "0"
        return "(" + " + ".join(t.to_pure(var) for t in self.terms) + ")"

    def to_deephaven(self) -> str:
        if not self.terms:
            return "0"
        return "(" + " + ".join(t.to_deephaven() for t in self.terms) + ")"

    def to_json(self) -> dict:
        return {
            "type": "Sum",
            "terms": [t.to_json() for t in self.terms],
        }

class UnaryOp(Expr):
    """Unary operation: neg, abs, not."""

    def __init__(self, op: str, operand: Expr) -> None:
        self.op = op
        self.operand = operand

    def eval(self, ctx: dict) -> Any:
        v = self.operand.eval(ctx)
        if self.op == "neg":
            return -v
        if self.op == "abs":
            return abs(v)
        if self.op == "not":
            return not v
        raise ValueError(f"Unknown unary op: {self.op}")

    def to_sql(self, col: str = "data") -> str:
        s = _cast_numeric_sql(self.operand, col)
        if self.op == "neg":
            return f"(-{s})"
        if self.op == "abs":
            return f"ABS({s})"
        if self.op == "not":
            return f"NOT ({self.operand.to_sql(col)})"
        raise ValueError(f"Unknown unary op: {self.op}")

    def to_pure(self, var: str = "$row") -> str:
        p = self.operand.to_pure(var)
        if self.op == "neg":
            return f"(-{p})"
        if self.op == "abs":
            return f"abs({p})"
        if self.op == "not":
            return f"!({p})"
        raise ValueError(f"Unknown unary op: {self.op}")

    def to_deephaven(self) -> str:
        p = self.operand.to_deephaven()
        if self.op == "neg":
            return f"(-{p})"
        if self.op == "abs":
            return f"Math.abs({p})"
        if self.op == "not":
            return f"!({p})"
        raise ValueError(f"Unknown unary op: {self.op}")

    def to_json(self) -> dict:
        return {
            "type": "UnaryOp",
            "op": self.op,
            "operand": self.operand.to_json(),
        }


class Func(Expr):
    """Named function call: sqrt, ceil, floor, round, min, max, log, exp."""

    _PYTHON_FUNCS: ClassVar[dict] = {
        "sqrt": math.sqrt,
        "ceil": math.ceil,
        "floor": math.floor,
        "round": round,
        "log": math.log,
        "exp": math.exp,
        "min": min,
        "max": max,
    }

    _SQL_FUNCS: ClassVar[dict] = {
        "sqrt": "SQRT", "ceil": "CEIL", "floor": "FLOOR", "round": "ROUND",
        "log": "LN", "exp": "EXP", "min": "LEAST", "max": "GREATEST",
    }

    _PURE_FUNCS: ClassVar[dict] = {
        "sqrt": "sqrt", "ceil": "ceiling", "floor": "floor", "round": "round",
        "log": "log", "exp": "exp", "min": "min", "max": "max",
    }

    _DH_FUNCS: ClassVar[dict] = {
        "sqrt": "Math.sqrt", "ceil": "Math.ceil", "floor": "Math.floor", "round": "Math.round",
        "log": "Math.log", "exp": "Math.exp", "min": "Math.min", "max": "Math.max",
    }

    def __init__(self, name: str, args: list) -> None:
        self.name = name
        self.args = [_wrap(a) for a in args]

    def eval(self, ctx: dict) -> Any:
        fn = self._PYTHON_FUNCS.get(self.name)
        if fn is None:
            raise ValueError(f"Unknown function: {self.name}")
        evaluated = [a.eval(ctx) for a in self.args]
        return fn(*evaluated)

    def to_sql(self, col: str = "data") -> str:
        sql_name = self._SQL_FUNCS.get(self.name, self.name.upper())
        args_sql = ", ".join(_cast_numeric_sql(a, col) for a in self.args)
        return f"{sql_name}({args_sql})"

    def to_pure(self, var: str = "$row") -> str:
        pure_name = self._PURE_FUNCS.get(self.name, self.name)
        args_pure = ", ".join(a.to_pure(var) for a in self.args)
        return f"{pure_name}({args_pure})"

    def to_deephaven(self) -> str:
        dh_name = self._DH_FUNCS.get(self.name, self.name)
        args_dh = ", ".join(a.to_deephaven() for a in self.args)
        return f"{dh_name}({args_dh})"

    def to_json(self) -> dict:
        return {
            "type": "Func",
            "name": self.name,
            "args": [a.to_json() for a in self.args],
        }


# -- Convenience builders for common functions --------------------------------

def Exp(x) -> Expr:
    """exp(x) — builds a Func('exp', [x]) node.

    Preferred over Const(math.e) ** x because:
      - diff() handles Func('exp') directly: d/dx exp(f) = exp(f) * f'
      - SQL compiles to EXP() (single instruction) instead of POWER()
      - The derivative reuses the same exp(f) node (DAG sharing)
    """
    return Func("exp", [x])


def Log(x) -> Expr:
    """log(x) — builds a Func('log', [x]) node (natural logarithm).

    diff() handles: d/dx log(f) = f' / f
    SQL compiles to LN().
    """
    return Func("log", [x])


class If(Expr):
    """Conditional: if condition then then_ else else_.

    Compiles to CASE WHEN in SQL, if()|) in Pure.
    """

    def __init__(self, condition: Expr, then_: Expr, else_: Expr) -> None:
        self.condition = _wrap(condition)
        self.then_ = _wrap(then_)
        self.else_ = _wrap(else_)

    def eval(self, ctx: dict) -> Any:
        if self.condition.eval(ctx):
            return self.then_.eval(ctx)
        return self.else_.eval(ctx)

    def to_sql(self, col: str = "data") -> str:
        cond_sql = self.condition.to_sql(col)
        then_sql = self.then_.to_sql(col)
        else_sql = self.else_.to_sql(col)
        return f"CASE WHEN {cond_sql} THEN {then_sql} ELSE {else_sql} END"

    def to_pure(self, var: str = "$row") -> str:
        cond_pure = self.condition.to_pure(var)
        then_pure = self.then_.to_pure(var)
        else_pure = self.else_.to_pure(var)
        return f"if({cond_pure}, |{then_pure}, |{else_pure})"

    def to_deephaven(self) -> str:
        cond_dh = self.condition.to_deephaven()
        then_dh = self.then_.to_deephaven()
        else_dh = self.else_.to_deephaven()
        return f"({cond_dh} ? {then_dh} : {else_dh})"

    def to_json(self) -> dict:
        return {
            "type": "If",
            "condition": self.condition.to_json(),
            "then": self.then_.to_json(),
            "else": self.else_.to_json(),
        }


class Coalesce(Expr):
    """Return the first non-None value from a list of expressions."""

    def __init__(self, exprs: list) -> None:
        self.exprs = [_wrap(e) for e in exprs]

    def eval(self, ctx: dict) -> Any:
        for e in self.exprs:
            v = e.eval(ctx)
            if v is not None:
                return v
        return None

    def to_sql(self, col: str = "data") -> str:
        parts = ", ".join(e.to_sql(col) for e in self.exprs)
        return f"COALESCE({parts})"

    def to_pure(self, var: str = "$row") -> str:
        # Pure doesn't have a direct coalesce; chain if/isEmpty
        if len(self.exprs) == 0:
            return "[]"
        if len(self.exprs) == 1:
            return self.exprs[0].to_pure(var)
        first = self.exprs[0].to_pure(var)
        rest = Coalesce(self.exprs[1:]).to_pure(var)
        return f"if(isEmpty({first}), |{rest}, |{first})"

    def to_deephaven(self) -> str:
        if len(self.exprs) == 0:
            return "null"
        if len(self.exprs) == 1:
            return self.exprs[0].to_deephaven()
        # Requires java inline ternary to handle
        first = self.exprs[0].to_deephaven()
        rest = Coalesce(self.exprs[1:]).to_deephaven()
        return f"({first} != null ? {first} : {rest})"

    def to_json(self) -> dict:
        return {
            "type": "Coalesce",
            "exprs": [e.to_json() for e in self.exprs],
        }


class IsNull(Expr):
    """Check if an expression evaluates to null/None."""

    def __init__(self, operand: Expr) -> None:
        self.operand = _wrap(operand)

    def eval(self, ctx: dict) -> bool:
        return self.operand.eval(ctx) is None

    def to_sql(self, col: str = "data") -> str:
        return f"({self.operand.to_sql(col)} IS NULL)"

    def to_pure(self, var: str = "$row") -> str:
        return f"isEmpty({self.operand.to_pure(var)})"

    def to_deephaven(self) -> str:
        return f"({self.operand.to_deephaven()} == null)"

    def to_json(self) -> dict:
        return {
            "type": "IsNull",
            "operand": self.operand.to_json(),
        }


class StrOp(Expr):
    """String operation: length, upper, lower, contains, starts_with, concat."""

    def __init__(self, op: str, operand: Expr, arg: Expr | None = None) -> None:
        self.op = op
        self.operand = operand
        self.arg = arg

    def eval(self, ctx: dict) -> Any:
        v = self.operand.eval(ctx)
        if self.op == "length":
            return len(v)
        if self.op == "upper":
            return v.upper()
        if self.op == "lower":
            return v.lower()
        if self.op == "contains":
            assert self.arg is not None
            return self.arg.eval(ctx) in v
        if self.op == "starts_with":
            assert self.arg is not None
            return v.startswith(self.arg.eval(ctx))
        if self.op == "concat":
            assert self.arg is not None
            return v + str(self.arg.eval(ctx))
        raise ValueError(f"Unknown string op: {self.op}")

    def to_sql(self, col: str = "data") -> str:
        s = self.operand.to_sql(col)
        if self.op == "length":
            return f"LENGTH({s})"
        if self.op == "upper":
            return f"UPPER({s})"
        if self.op == "lower":
            return f"LOWER({s})"
        if self.op == "contains":
            assert self.arg is not None
            return f"({s} LIKE '%%' || {self.arg.to_sql(col)} || '%%')"
        if self.op == "starts_with":
            assert self.arg is not None
            return f"({s} LIKE {self.arg.to_sql(col)} || '%%')"
        if self.op == "concat":
            assert self.arg is not None
            return f"({s} || {self.arg.to_sql(col)})"
        raise ValueError(f"Unknown string op: {self.op}")

    def to_pure(self, var: str = "$row") -> str:
        p = self.operand.to_pure(var)
        if self.op == "length":
            return f"length({p})"
        if self.op == "upper":
            return f"toUpper({p})"
        if self.op == "lower":
            return f"toLower({p})"
        if self.op == "contains":
            assert self.arg is not None
            return f"contains({p}, {self.arg.to_pure(var)})"
        if self.op == "starts_with":
            assert self.arg is not None
            return f"startsWith({p}, {self.arg.to_pure(var)})"
        if self.op == "concat":
            assert self.arg is not None
            return f"({p} + {self.arg.to_pure(var)})"
        raise ValueError(f"Unknown string op: {self.op}")

    def to_deephaven(self) -> str:
        s = self.operand.to_deephaven()
        if self.op == "length":
            return f"{s}.length()"
        if self.op == "upper":
            return f"{s}.toUpperCase()"
        if self.op == "lower":
            return f"{s}.toLowerCase()"
        if self.op == "contains":
            assert self.arg is not None
            return f"{s}.contains({self.arg.to_deephaven()})"
        if self.op == "starts_with":
            assert self.arg is not None
            return f"{s}.startsWith({self.arg.to_deephaven()})"
        if self.op == "concat":
            assert self.arg is not None
            return f"({s} + {self.arg.to_deephaven()})"
        raise ValueError(f"Unknown string op: {self.op}")

    def to_json(self) -> dict:
        d = {"type": "StrOp", "op": self.op, "operand": self.operand.to_json()}
        if self.arg is not None:
            d["arg"] = self.arg.to_json()
        return d


# ---------------------------------------------------------------------------
# SQL helper
# ---------------------------------------------------------------------------

def _cast_numeric_sql(expr, col: str) -> str:
    """If expr is a Field, cast the JSONB text extraction to float.
    For constants, ensure they are treated as DOUBLE to avoid precision overflows in SQL engines.
    """
    if isinstance(expr, Field):
        return f"({col}->>'{expr.name}')::float"
    if isinstance(expr, VariableMixin):
        return expr.expr_to_sql(col)
    if isinstance(expr, Const) and isinstance(expr.value, (int, float)):
        return f"CAST({expr.value} AS DOUBLE)"
    return expr.to_sql(col)


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

_NODE_REGISTRY = {
    "Const": Const,
    "Field": Field,
    "Variable": Variable,
    "BinOp": BinOp,
    "UnaryOp": UnaryOp,
    "Func": Func,
    "If": If,
    "Coalesce": Coalesce,
    "IsNull": IsNull,
    "StrOp": StrOp,
}


def from_json(data: dict) -> Expr:
    """Deserialize a JSON dict back to an Expr tree."""
    if isinstance(data, str):  # type: ignore[unreachable]
        data = json.loads(data)  # type: ignore[unreachable]

    node_type = data["type"]

    if node_type == "Const":
        return Const(data["value"])

    if node_type == "Field":
        return Field(data["name"])

    if node_type == "BinOp":
        return BinOp(data["op"], from_json(data["left"]), from_json(data["right"]))

    if node_type == "UnaryOp":
        return UnaryOp(data["op"], from_json(data["operand"]))

    if node_type == "Func":
        return Func(data["name"], [from_json(a) for a in data["args"]])

    if node_type == "If":
        return If(from_json(data["condition"]), from_json(data["then"]), from_json(data["else"]))

    if node_type == "Coalesce":
        return Coalesce([from_json(e) for e in data["exprs"]])

    if node_type == "IsNull":
        return IsNull(from_json(data["operand"]))

    if node_type == "StrOp":
        arg = from_json(data["arg"]) if "arg" in data else None
        op = data["op"]
        operand = from_json(data["operand"])
        node = StrOp(op, operand, arg)
        return node

    if node_type == "Sum":
        return Sum([from_json(t) for t in data["terms"]])

    raise ValueError(f"Unknown expression type: {node_type}")


# ---------------------------------------------------------------------------
# Symbolic Differentiation (iterative — no recursion depth limit)
# ---------------------------------------------------------------------------

def diff(expr: Expr, wrt: str, _memo: dict | None = None) -> Expr:
    """Symbolic differentiation: ∂expr/∂Variable(wrt).

    Returns a new Expr tree representing the derivative.
    This enables risk calculations that compile to any target:
        risk = diff(npv_expr, "USD_OIS_5Y")
        risk.eval(ctx)   → Python float
        risk.to_sql()    → SQL expression

    Memoized: the same sub-expression differentiated w.r.t. the same
    variable returns the same Expr object.  This is critical because
    product/power rules create new references to existing sub-trees,
    and without memoization the derivative tree grows exponentially.

    ITERATIVE implementation — uses an explicit stack instead of Python
    call stack, so there is no recursion depth limit.

    Supports: +, -, *, /, **, neg, abs, Const, Variable, Field, Sum.
    """
    if _memo is None:
        _memo = {}

    # Fast path: already computed
    key = (id(expr), wrt)
    if key in _memo:
        return _memo[key]

    # ── Iterative post-order differentiation ──
    # We use a work stack of "frames". Each frame is a tuple:
    #   (expr, phase, *partial_results)
    # Phase 0: first visit — push children
    # Phase 1+: children done, combine results

    _ZERO = Const(0.0)
    _ONE = Const(1.0)

    stack: list = [(expr, 0)]
    result_stack: list = []  # holds derivative results

    while stack:
        node, phase, *args = stack.pop()

        nkey = (id(node), wrt)
        if nkey in _memo:
            result_stack.append(_memo[nkey])
            continue

        # ── Leaf nodes (no children to push) ──
        if isinstance(node, Const):
            r = _ZERO
            _memo[nkey] = r
            result_stack.append(r)
            continue

        if isinstance(node, (Variable, VariableMixin)):
            r = _ONE if node.name == wrt else _ZERO
            _memo[nkey] = r
            result_stack.append(r)
            continue

        if isinstance(node, Field):
            _memo[nkey] = _ZERO
            result_stack.append(_ZERO)
            continue

        # ── Sum node ──
        if isinstance(node, Sum):
            if phase == 0:
                # Push phase-1 continuation, then all terms
                stack.append((node, 1))
                for term in reversed(node.terms):
                    tkey = (id(term), wrt)
                    if tkey not in _memo:
                        stack.append((term, 0))
                    # else: already in memo, will be picked up from result_stack
            else:
                # Phase 1: collect derivatives of all terms
                dterms = []
                for term in node.terms:
                    tkey = (id(term), wrt)
                    if tkey in _memo:
                        dterms.append(_memo[tkey])
                    else:
                        dterms.append(result_stack.pop())
                # Filter out zero terms
                nonzero = [dt for dt in dterms if not (isinstance(dt, Const) and dt.value == 0.0)]
                if not nonzero:
                    r = _ZERO
                elif len(nonzero) == 1:
                    r = nonzero[0]
                else:
                    r = Sum(nonzero)
                _memo[nkey] = r
                result_stack.append(r)
            continue

        # ── BinOp ──
        if isinstance(node, BinOp):
            if phase == 0:
                # Push phase-1 continuation, then right, then left
                stack.append((node, 1))
                rkey = (id(node.right), wrt)
                if rkey not in _memo:
                    stack.append((node.right, 0))
                lkey = (id(node.left), wrt)
                if lkey not in _memo:
                    stack.append((node.left, 0))
            else:
                # Phase 1: both children are done
                lkey = (id(node.left), wrt)
                rkey = (id(node.right), wrt)
                dl = _memo[lkey] if lkey in _memo else result_stack.pop()
                dr = _memo[rkey] if rkey in _memo else result_stack.pop()
                # Store them in memo if not yet (they were popped from result_stack)
                if lkey not in _memo:
                    _memo[lkey] = dl
                if rkey not in _memo:
                    _memo[rkey] = dr

                if node.op == "+":
                    r = dl + dr
                elif node.op == "-":
                    r = dl - dr
                elif node.op == "*":
                    r = dl * node.right + node.left * dr
                elif node.op == "/":
                    r = (dl * node.right - node.left * dr) / (node.right ** Const(2.0))
                elif node.op == "**":
                    n = node.right
                    f = node.left
                    r = n * (f ** (n - Const(1.0))) * dl
                else:
                    raise ValueError(f"diff: unsupported BinOp '{node.op}'")

                _memo[nkey] = r
                result_stack.append(r)
            continue

        # ── Func (exp, log, sqrt) ──
        if isinstance(node, Func):
            if len(node.args) != 1:
                raise ValueError(f"diff: unsupported Func '{node.name}' with {len(node.args)} args")
            f = node.args[0]
            if phase == 0:
                stack.append((node, 1))
                fkey = (id(f), wrt)
                if fkey not in _memo:
                    stack.append((f, 0))
            else:
                fkey = (id(f), wrt)
                df = _memo[fkey] if fkey in _memo else result_stack.pop()
                if fkey not in _memo:
                    _memo[fkey] = df

                if node.name == "exp":
                    r = node * df
                elif node.name == "log":
                    r = df / f
                elif node.name == "sqrt":
                    r = df / (Const(2.0) * node)
                else:
                    raise ValueError(f"diff: unsupported Func '{node.name}'")

                _memo[nkey] = r
                result_stack.append(r)
            continue

        # ── UnaryOp ──
        if isinstance(node, UnaryOp):
            if phase == 0:
                stack.append((node, 1))
                okey = (id(node.operand), wrt)
                if okey not in _memo:
                    stack.append((node.operand, 0))
            else:
                okey = (id(node.operand), wrt)
                df = _memo[okey] if okey in _memo else result_stack.pop()
                if okey not in _memo:
                    _memo[okey] = df

                if node.op == "neg":
                    r = -df
                elif node.op == "abs":
                    f = node.operand
                    r = If(f > Const(0.0), df, If(f < Const(0.0), -df, _ZERO))
                else:
                    raise ValueError(f"diff: unsupported UnaryOp '{node.op}'")

                _memo[nkey] = r
                result_stack.append(r)
            continue

        # ── If ──
        if isinstance(node, If):
            if phase == 0:
                stack.append((node, 1))
                ekey = (id(node.else_), wrt)
                if ekey not in _memo:
                    stack.append((node.else_, 0))
                tkey = (id(node.then_), wrt)
                if tkey not in _memo:
                    stack.append((node.then_, 0))
            else:
                tkey = (id(node.then_), wrt)
                ekey = (id(node.else_), wrt)
                dt = _memo[tkey] if tkey in _memo else result_stack.pop()
                de = _memo[ekey] if ekey in _memo else result_stack.pop()
                if tkey not in _memo:
                    _memo[tkey] = dt
                if ekey not in _memo:
                    _memo[ekey] = de
                r = If(node.condition, dt, de)
                _memo[nkey] = r
                result_stack.append(r)
            continue

        raise ValueError(f"diff: unsupported Expr type '{type(node).__name__}'")

    return result_stack[-1]


# ---------------------------------------------------------------------------
# Cached evaluation (for DAGs produced by memoized diff)
# ---------------------------------------------------------------------------

def eval_cached(expr: Expr, ctx: dict, _cache: dict | None = None) -> Any:
    """Evaluate an Expr DAG with sub-expression caching.

    After memoized diff(), the derivative is a DAG (not a tree).
    Naive expr.eval(ctx) would re-evaluate shared sub-nodes exponentially.
    This function caches by node id(), evaluating each unique node once.

    ITERATIVE implementation — uses an explicit stack to avoid
    hitting Python's recursion limit on deep expression trees.

    Usage:
        deriv = diff(npv_expr, "USD_OIS_5Y")
        val = eval_cached(deriv, ctx)  # fast
    """
    if _cache is None:
        _cache = {}

    key = id(expr)
    if key in _cache:
        return _cache[key]

    # ── Iterative post-order evaluation ──
    stack: list = [(expr, 0)]
    result_stack: list = []

    while stack:
        node, phase, *_ = stack.pop()

        nkey = id(node)
        if nkey in _cache:
            result_stack.append(_cache[nkey])
            continue

        # ── Leaf nodes ──
        if isinstance(node, Const):
            r = node.eval(ctx)
            _cache[nkey] = r
            result_stack.append(r)
            continue

        if isinstance(node, (Variable, VariableMixin)):
            r = node.expr_eval(ctx)
            _cache[nkey] = r
            result_stack.append(r)
            continue

        if isinstance(node, Field):
            r = node.eval(ctx)
            _cache[nkey] = r
            result_stack.append(r)
            continue

        # ── Sum ──
        if isinstance(node, Sum):
            if phase == 0:
                stack.append((node, 1))
                for term in reversed(node.terms):
                    if id(term) not in _cache:
                        stack.append((term, 0))
            else:
                total = 0.0
                for term in node.terms:
                    tk = id(term)
                    if tk in _cache:
                        total += _cache[tk]
                    else:
                        total += result_stack.pop()
                _cache[nkey] = total
                result_stack.append(total)
            continue

        # ── BinOp ──
        if isinstance(node, BinOp):
            if phase == 0:
                stack.append((node, 1))
                if id(node.right) not in _cache:
                    stack.append((node.right, 0))
                if id(node.left) not in _cache:
                    stack.append((node.left, 0))
            else:
                lk, rk = id(node.left), id(node.right)
                lv = _cache[lk] if lk in _cache else result_stack.pop()
                rv = _cache[rk] if rk in _cache else result_stack.pop()
                if lk not in _cache:
                    _cache[lk] = lv
                if rk not in _cache:
                    _cache[rk] = rv

                op = node.op
                if op == "+":
                    r = lv + rv
                elif op == "-":
                    r = lv - rv
                elif op == "*":
                    r = lv * rv
                elif op == "/":
                    r = lv / rv if rv != 0 else 0
                elif op == "**":
                    try:
                        # Defensive against extreme notionals/rates in intermediate DAG nodes
                        if lv > 1e10 and rv > 10: r = 1e100
                        elif lv < -1e10 and rv > 10: r = -1e100
                        else: r = lv ** rv
                    except (OverflowError, FloatingPointError, ZeroDivisionError):
                        r = 1e100 if lv > 0 else -1e100 if lv < 0 else 0.0
                elif op == ">":
                    r = lv > rv
                elif op == "<":
                    r = lv < rv
                elif op == ">=":
                    r = lv >= rv
                elif op == "<=":
                    r = lv <= rv
                elif op == "==":
                    r = lv == rv
                elif op == "!=":
                    r = lv != rv
                else:
                    raise ValueError(f"eval_cached: unsupported BinOp '{op}'")
                _cache[nkey] = r
                result_stack.append(r)
            continue

        # ── UnaryOp ──
        if isinstance(node, UnaryOp):
            if phase == 0:
                stack.append((node, 1))
                if id(node.operand) not in _cache:
                    stack.append((node.operand, 0))
            else:
                ok = id(node.operand)
                ov = _cache[ok] if ok in _cache else result_stack.pop()
                if ok not in _cache:
                    _cache[ok] = ov
                if node.op == "neg":
                    r = -ov
                elif node.op == "abs":
                    r = abs(ov)
                else:
                    raise ValueError(f"eval_cached: unsupported UnaryOp '{node.op}'")
                _cache[nkey] = r
                result_stack.append(r)
            continue

        # ── Func ──
        if isinstance(node, Func):
            if phase == 0:
                stack.append((node, 1))
                for a in reversed(node.args):
                    if id(a) not in _cache:
                        stack.append((a, 0))
            else:
                vals = []
                for a in node.args:
                    ak = id(a)
                    if ak in _cache:
                        vals.append(_cache[ak])
                    else:
                        vals.append(result_stack.pop())
                fn = Func._PYTHON_FUNCS.get(node.name)
                if fn is None:
                    raise ValueError(f"eval_cached: unknown Func '{node.name}'")
                r = fn(*vals)
                _cache[nkey] = r
                result_stack.append(r)
            continue

        # ── If ──
        if isinstance(node, If):
            if phase == 0:
                # Evaluate condition first
                stack.append((node, 1))
                if id(node.condition) not in _cache:
                    stack.append((node.condition, 0))
            elif phase == 1:
                # Condition evaluated, now pick the branch
                ck = id(node.condition)
                cond = _cache[ck] if ck in _cache else result_stack.pop()
                if ck not in _cache:
                    _cache[ck] = cond
                branch = node.then_ if cond else node.else_
                bk = id(branch)
                if bk in _cache:
                    _cache[nkey] = _cache[bk]
                    result_stack.append(_cache[bk])
                else:
                    stack.append((node, 2))
                    stack.append((branch, 0))
            else:
                # Phase 2: branch evaluated
                r = result_stack.pop()
                _cache[nkey] = r
                result_stack.append(r)
            continue

        # Fallback for other node types
        r = node.eval(ctx)
        _cache[nkey] = r
        result_stack.append(r)

    return result_stack[-1]

