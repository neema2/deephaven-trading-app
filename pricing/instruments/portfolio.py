"""
instruments/portfolio — Named collection of instrument Expr trees.

Provides the Portfolio class, which aggregates multiple IRSwapFixedFloatApprox
Expr trees on a shared curve.  This enables maximum sub-expression sharing
and provides symbolic Jacobian matrices for the fitter.
"""

from __future__ import annotations
from typing import Any

from reactive.expr import (
    Const, Expr, diff, eval_cached, 
    Variable, VariableMixin, Field,
    _cast_numeric_sql
)
from pricing.instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox
from pricing.instruments.base import Instrument


class Portfolio(Instrument):
    """A collection of named swap Expr trees on a shared curve.
    ...
    """

    def __init__(self):
        # Instrument.__init__ is not called because we have no dataclass fields
        # but we need to initialize reaktiv manually if we want it to work as storable
        # For now, stay as a plain object inheriting the interface.
        self._instruments: dict[str, Instrument] = {}

    def add_instrument(self, name: str, instrument: Instrument):
        """Add any pre-constructed instrument (swap, etc.) to the portfolio.
        Must support .npv() and optionally .notional.
        """
        self._instruments[name] = instrument
        return instrument

    @property
    def names(self) -> list[str]:
        return list(self._instruments.keys())

    def npv(self) -> Expr:
        """Total NPV of the portfolio as a single expression tree."""
        return sum(inst.npv() for inst in self._instruments.values())

    # ── Named dictionaries of Expr trees ───────────────────────────────

    @property
    def npv_exprs(self) -> dict[str, Expr]:
        """Named NPV expressions: {name: npv_expr}."""
        return {name: inst.npv() for name, inst in self._instruments.items()}

    @property

    @property
    def total_npv_expr(self) -> Expr:
        """Sum of all NPVs — a single Expr tree."""
        return sum(inst.npv() for inst in self._instruments.values())

    # ── Convenience evaluators ─────────────────────────────────────────

    def pillar_points(self) -> dict[str, Any]:
        """Aggregate all pillar point objects from child pricing.instruments."""
        points = {}
        for inst in self._instruments.values():
            points.update(inst.pillar_points())
        return points

    @property
    def pillar_names(self) -> list[str]:
        """Unified list of all pillar names across the portfolio."""
        names = set()
        for inst in self._instruments.values():
            names.update(inst.pillar_names)
        return sorted(list(names))

    def pillar_context(self) -> dict[str, float]:
        """Current pillar rates aggregated from all instruments' curves."""
        return {name: p.rate for name, p in self.pillar_points().items()}

