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
    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""

    # -- Arithmetic operators ------------------------------------------------

    def __add__(self, other: object) -> "Expr":
        return BinOp("+", self, _wrap(other))

    def __radd__(self, other: object) -> "Expr":
        return BinOp("+", _wrap(other), self)

    def __sub__(self, other: object) -> "Expr":
        return BinOp("-", self, _wrap(other))

    def __rsub__(self, other: object) -> "Expr":
        return BinOp("-", _wrap(other), self)

    def __mul__(self, other: object) -> "Expr":
        return BinOp("*", self, _wrap(other))

    def __rmul__(self, other: object) -> "Expr":
        return BinOp("*", _wrap(other), self)

    def __truediv__(self, other: object) -> "Expr":
        return BinOp("/", self, _wrap(other))

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

    def to_json(self) -> dict:
        return {"type": "Field", "name": self.name}


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

    def to_json(self) -> dict:
        return {
            "type": "BinOp",
            "op": self.op,
            "left": self.left.to_json(),
            "right": self.right.to_json(),
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

    def to_json(self) -> dict:
        return {
            "type": "Func",
            "name": self.name,
            "args": [a.to_json() for a in self.args],
        }


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

    def to_json(self) -> dict:
        d = {"type": "StrOp", "op": self.op, "operand": self.operand.to_json()}
        if self.arg is not None:
            d["arg"] = self.arg.to_json()
        return d


# ---------------------------------------------------------------------------
# SQL helper
# ---------------------------------------------------------------------------

def _cast_numeric_sql(expr: Expr, col: str) -> str:
    """If expr is a Field, cast the JSONB text extraction to float."""
    if isinstance(expr, Field):
        return f"({col}->>'{expr.name}')::float"
    return expr.to_sql(col)


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

_NODE_REGISTRY = {
    "Const": Const,
    "Field": Field,
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

    raise ValueError(f"Unknown expression type: {node_type}")
