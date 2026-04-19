#!/usr/bin/env python3
"""
Test suite for the @traceable dual-mode expression architecture.
Converted to pytest format.
"""

import math
import pytest
from reactive.expr import Const, Variable, Exp, eval_cached, diff, _wrap
from reactive.traced import (
    TracedFloat,
    _start_tracing, _stop_tracing, _is_tracing,
    _start_building, _stop_building, _is_building,
    exp as traced_exp,
    log as traced_log,
)

class TestTracedFloat:
    def test_arithmetic(self):
        x = TracedFloat(3.0, Variable("x"))
        y = TracedFloat(2.0, Variable("y"))

        # Basic ops
        assert float(x + y) == 5.0
        assert float(x - y) == 1.0
        assert float(x * y) == 6.0
        assert float(x / y) == 1.5
        assert float(-x) == -3.0
        assert float(x ** y) == 9.0

        # Verify Expr trees evaluate correctly
        ctx = {"x": 3.0, "y": 2.0}
        assert eval_cached((x + y)._expr, ctx) == 5.0
        assert eval_cached((x * y)._expr, ctx) == 6.0
        
        # With different context
        ctx2 = {"x": 10.0, "y": 2.0}
        assert eval_cached((x + y)._expr, ctx2) == 12.0

    def test_mixed_arithmetic(self):
        x = TracedFloat(3.0, Variable("x"))
        assert float(x + 5.0) == 8.0
        assert float(5.0 + x) == 8.0
        assert float(x * 0.5) == 1.5
        assert float(0.5 * x) == 1.5
        
        ctx = {"x": 3.0}
        assert eval_cached((x + 5.0)._expr, ctx) == 8.0
        assert eval_cached((0.5 * x)._expr, ctx) == 1.5

    def test_chained_accumulation(self):
        pv = 0.0
        for i in range(3):
            df = TracedFloat(0.98 - 0.01 * i, Variable(f"df_{i}"))
            dcf = 0.25
            pv += df * dcf * 100.0
            
        assert isinstance(pv, TracedFloat)
        expected_pv = (0.98 + 0.97 + 0.96) * 0.25 * 100.0
        assert abs(float(pv) - expected_pv) < 1e-10
        
        ctx_alt = {"df_0": 0.95, "df_1": 0.90, "df_2": 0.85}
        expected_alt = (0.95 + 0.90 + 0.85) * 0.25 * 100.0
        assert abs(eval_cached(pv._expr, ctx_alt) - expected_alt) < 1e-10

    def test_type_inheritance(self):
        x = TracedFloat(3.0, Variable("x"))
        assert isinstance(x, float)

class TestTracedMath:
    def test_exp(self):
        a = TracedFloat(1.0, Variable("a"))
        e_a = traced_exp(a)
        assert isinstance(e_a, TracedFloat)
        assert abs(float(e_a) - math.exp(1.0)) < 1e-10
        assert abs(eval_cached(e_a._expr, {"a": 2.0}) - math.exp(2.0)) < 1e-10

    def test_log(self):
        b = TracedFloat(2.718, Variable("b"))
        l_b = traced_log(b)
        assert isinstance(l_b, TracedFloat)
        assert abs(float(l_b) - math.log(2.718)) < 1e-10

    def test_plain_fallback(self):
        assert traced_exp(1.0) == math.exp(1.0)
        assert traced_log(2.718) == math.log(2.718)

class TestProtocolInterop:
    def test_wrap_protocol(self):
        tf = TracedFloat(42.0, Variable("answer"))
        wrapped = _wrap(tf)
        assert isinstance(wrapped, Variable)
        assert wrapped.name == "answer"

    def test_expr_interop(self):
        tf = TracedFloat(42.0, Variable("answer"))
        expr = Const(10.0) + tf
        assert eval_cached(expr, {"answer": 42.0}) == 52.0

class TestTracingContext:
    def test_reentrant_tracing(self):
        assert not _is_tracing()
        _start_tracing()
        assert _is_tracing()
        _start_tracing()
        assert _is_tracing()
        _stop_tracing()
        assert _is_tracing()
        _stop_tracing()
        assert not _is_tracing()

    def test_building_context(self):
        assert not _is_building()
        _start_building()
        assert _is_building()
        _stop_building()
        assert not _is_building()
