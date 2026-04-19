from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from store import Storable
from reactive.computed import computed
from reactive.traceable import traceable

if TYPE_CHECKING:
    from reactive.expr import Expr

@dataclass
class Instrument(Storable, ABC):
    """Base class for all financial instruments."""
    
    @traceable
    @abstractmethod
    def npv(self) -> float:
        """The Net Present Value of the instrument."""
        pass

    @property
    def pillar_names(self) -> list[str]:
        """Auto-discovery of pillar dependencies via Expr tree analysis."""
        # Use npv() to get the expression tree
        expr = self.npv()
        if hasattr(expr, "variables"):
             return sorted(list(expr.variables))
        return []

    def pillar_context(self) -> dict[str, float]:
        """Auto-discover curves in fields and extract their pillar rates.
        
        This satisfies the 'automated context construction' improvement.
        """
        return {name: p.rate for name, p in self.pillar_points().items()}

    def pillar_points(self) -> dict[str, Any]:
        """Auto-discover curves in fields and extract their pillar point objects."""
        points = {}
        # Simple scraping: look for anything with _sorted_points (CurveBase)
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(self, attr_name)
                # CurveBase implementations have _sorted_points()
                if hasattr(val, "_sorted_points"):
                    for p in val._sorted_points():
                        points[p.name] = p
            except Exception:
                continue
        return points
