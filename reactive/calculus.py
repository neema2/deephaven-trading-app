from typing import Any
from .expr import Expr, Const, BinOp, UnaryOp, Func, If, Field, Variable, VariableMixin
from .sum_expr import Sum

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
            
        if phase == 0 and wrt not in node.variables:
            _memo[nkey] = _ZERO
            result_stack.append(_ZERO)
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

