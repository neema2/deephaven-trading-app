"""
reactive.traceable — @traceable decorator for dual-mode pricing functions.

A ``@traceable`` property runs on plain ``float``s by default (debug mode),
but can lazily produce an ``Expr`` tree on demand (trace mode) by re-running
the function with ``TracedFloat`` inputs from the curve.

The quant writes standard procedural code::

    @traceable
    def dv01(self) -> float:
        pv = 0.0
        for t in self.coupon_payment_dates():
            df = self.curve.df(t)
            pv += df * dcf * self.notional * 0.0001
        return pv

    swap.dv01      # → 4523.17  (plain float, fast, debuggable)
    swap.dv01()    # → Expr tree (lazy-traced, cached)

Internally, ``@traceable`` is a ``ComputedProperty`` that plugs into
the reaktiv Signal/Computed reactive framework.
"""

from __future__ import annotations

from reactive.computed import ComputedProperty

_SENTINEL = object()


class _TracedCallable(float):
    """Float subclass returned by @traceable — callable to get Expr tree.

    Acts as a plain float for arithmetic and display.  Calling ``()``
    triggers lazy tracing to produce the Expr tree (cached on the instance).

    Arithmetic with another _TracedCallable or TracedFloat triggers
    lazy tracing on both sides and returns a TracedFloat with the
    composed Expr tree.
    """

    __slots__ = ("_instance_ref", "_descriptor", "_expr_tree")

    def __new__(
        cls,
        value: float,
        instance: object,
        descriptor: "traceable",
    ) -> _TracedCallable:
        obj = super().__new__(cls, value)
        obj._instance_ref = instance
        obj._descriptor = descriptor
        obj._expr_tree = None
        return obj

    # -- Expr access (lazy) -----------------------------------------------

    def _ensure_expr(self):
        """Build the Expr tree via tracing (cached)."""
        if self._expr_tree is None:
            self._expr_tree = self._descriptor._trace_for_expr(
                self._instance_ref
            )
        return self._expr_tree

    def __expr__(self):
        """The __expr__ protocol — compatible with TracedFloat + _wrap."""
        return self._ensure_expr()

    def __call__(self, ctx=None):
        """Call to get the Expr tree, or evaluate it against a context.

        Usage::

            swap.dv01()         # → Expr tree
            swap.dv01(ctx)      # → float (Expr evaluated against ctx)
        """
        expr = self._ensure_expr()
        if ctx is not None:
            from reactive.evaluation import eval_cached
            return eval_cached(expr, ctx)
        return expr

    # -- Arithmetic: delegate to TracedFloat via lazy trace ---------------

    def _to_traced(self):
        """Convert to TracedFloat by lazily building the Expr tree."""
        from reactive.traced import TracedFloat
        return TracedFloat(float(self), self._ensure_expr())

    def __add__(self, other):
        return self._to_traced().__add__(other)

    def __radd__(self, other):
        return self._to_traced().__radd__(other)

    def __sub__(self, other):
        return self._to_traced().__sub__(other)

    def __rsub__(self, other):
        return self._to_traced().__rsub__(other)

    def __mul__(self, other):
        return self._to_traced().__mul__(other)

    def __rmul__(self, other):
        return self._to_traced().__rmul__(other)

    def __truediv__(self, other):
        return self._to_traced().__truediv__(other)

    def __rtruediv__(self, other):
        return self._to_traced().__rtruediv__(other)

    def __pow__(self, other):
        return self._to_traced().__pow__(other)

    def __rpow__(self, other):
        return self._to_traced().__rpow__(other)

    def __neg__(self):
        return self._to_traced().__neg__()

    def __abs__(self):
        return self._to_traced().__abs__()

    def __reduce__(self):
        """If serialization is attempted, gracefully degrade to a basic float primitive natively to prevent graph bloat."""
        return (float, (float(self),))

    # -- repr: plain float ------------------------------------------------

    def __repr__(self):
        return float.__repr__(self)


class traceable(ComputedProperty):
    """Decorator: dual-mode computed property.

    Default (debug mode): function runs on plain floats — zero overhead,
    fully transparent in debugger watch windows.

    On-demand (trace mode): function re-runs with TracedFloat inputs
    to build an Expr tree.  The tree is cached on the instance.

    Usage::

        @traceable
        def dv01(self) -> float:
            pv = 0.0
            for t in dates:
                df = self.curve.df(t)
                pv += df * dcf * self.notional * 0.0001
            return pv

    Result behaves like a ``CallableFloat``::

        swap.dv01      # → 4523.17          (float value)
        swap.dv01()    # → Expr tree         (for differentiation/SQL)
        swap.dv01(ctx) # → float             (Expr evaluated against ctx)
    """

    def __new__(cls, fn=None, *, is_variable=False):
        if fn is None:
            def wrapper(func):
                return cls(func, is_variable=is_variable)
            return wrapper
        return super().__new__(cls)

    def __init__(self, fn=None, *, is_variable=False):
        if fn is None:
            return
        # fn = self._compute: called by the reactive framework
        # We pass None for expr (cross-entity / proxy-based)
        super().__init__(self._compute, None, fn.__name__)
        self._user_fn = fn
        # Important: streaming/decorator.py checks self.fn.__annotations__
        try:
            self.fn.__annotations__ = getattr(fn, "__annotations__", {})
        except AttributeError:
            pass
        self._is_variable = is_variable
        self._expr_cache_attr = f"_{fn.__name__}_traced_expr"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        
        from reactive.traced import _is_tracing
        if _is_tracing():
            reactive = object.__getattribute__(instance, '_reactive')
            node = reactive.get(self.name)
            val = float(node.read()) if node is not None else 0.0
            return _TracedCallable(val, instance, self)
            
        return super().__get__(instance, owner)

    @property
    def expr(self):
        """Builds and caches an exact equivalent analytic mapping implicitly identical to python AST parsing by feeding structural Field variables dynamically."""
        if not hasattr(self, "_cached_expr") or self._cached_expr is None:
            from reactive.expr import Field, Const, Expr
            class _StructProxy:
                def __getattr__(self, name):
                    return Field(name)
            
            from reactive.traced import _start_tracing, _stop_tracing
            _start_tracing()
            try:
                res = self._user_fn(_StructProxy())
            except Exception:
                res = None
            finally:
                _stop_tracing()
                
            if res is not None and not isinstance(res, Expr):
                res = Const(res)
                
            self._cached_expr = res
        return self._cached_expr

    @expr.setter
    def expr(self, value):
        self._cached_expr = value

    def _compute(self, instance):
        """Called by the reactive framework (on every recomputation).

        Runs the user function in debug mode (plain floats) and wraps
        the result in a _TracedCallable.
        """
        result = self._user_fn(instance)
        
        # Fast path graceful degradation: if it's a complex type (dict, str, list),
        # do not wrap it in _TracedCallable (which subclasses float). Just return it natively.
        # This prevents Arrow serialization issues and behaves identically to @computed.
        if isinstance(result, (dict, str, list, tuple, bool)) or hasattr(result, "date") or result is None:
            return result
            


        val = float(result) if result is not None else 0.0
        return _TracedCallable(val, instance, self)

    def _trace_for_expr(self, instance):
        """Re-run the function with tracing to build the Expr tree (lazy).

        Called by _TracedCallable when Expr is first requested.
        The result is cached on the instance.
        """
        cached = getattr(instance, self._expr_cache_attr, _SENTINEL)
        if cached is not _SENTINEL:
            return cached

        # If explicitly flagged as an analytic variable, terminate tracing
        # instantly and drop the structural Variable node mapping directly 
        # to the governed instance boundaries!
        if self._is_variable:
            from reactive.expr import Variable
            name = getattr(instance, "name", self.name)
            expr = Variable(name)
            object.__setattr__(instance, self._expr_cache_attr, expr)
            return expr

        from reactive.traced import TracedFloat, _start_tracing, _stop_tracing
        from reactive.expr import Expr, Const

        _start_tracing()
        try:
            result = self._user_fn(instance)
        finally:
            _stop_tracing()

        if isinstance(result, TracedFloat):
            expr = result._expr
        elif isinstance(result, Expr) or isinstance(result, (dict, str, list, tuple, bool)):
            expr = result
        else:
            expr = Const(float(result) if result is not None else 0.0)

        # Cache on the instance
        object.__setattr__(instance, self._expr_cache_attr, expr)
        return expr

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)

        # Inject `eval_<name>` method for API compat with computed_expr
        def _eval(instance, ctx, self=self):
            expr = self._trace_for_expr(instance)
            if expr is None:
                return 0.0
            from reactive.evaluation import eval_cached
            if isinstance(expr, dict):
                return {k: eval_cached(v, ctx) for k, v in expr.items()}
            return eval_cached(expr, ctx)
        setattr(owner, f"eval_{name}", _eval)


