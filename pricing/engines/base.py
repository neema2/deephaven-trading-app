from __future__ import annotations
from typing import Any

class ExecutionEngine:
    """Base class for all pricing execution strategies."""
    
    def npvs(self, portfolio: Any, ctx: dict) -> dict[str, float]:
        """Evaluate all instrument NPVs."""
        raise NotImplementedError()

    def total_risk(self, portfolio: Any, ctx: dict, risk_method: Any = None, regex: str = None, **kwargs) -> dict[str, float]:
        """Aggregate portfolio risk."""
        raise NotImplementedError()

    def instrument_risk(self, portfolio: Any, ctx: dict, risk_method: Any = None, regex: str = None, **kwargs) -> dict[str, dict[str, float]]:
        """Per-instrument risk breakdown."""
        raise NotImplementedError()
