from typing import Any
from .expr import Expr, Const, _cast_numeric_sql

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
        from .expr import _wrap
        numeric_total = 0.0
        other_terms = []
        for t in terms:
            t = _wrap(t)
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

    def to_json(self) -> dict:
        return {
            "type": "Sum",
            "terms": [t.to_json() for t in self.terms],
        }

