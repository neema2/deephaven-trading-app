from __future__ import annotations
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pricing.instruments.base import Instrument

class FirstOrderRiskBase:
    """Base class for first-order risk helpers (Analytic and Numerical)."""
    def __init__(self, instrument: Instrument, regex: str = None, name: str = "Risk"):
        self.instrument = instrument
        self.regex = regex
        self.name = name

    def _pillars(self) -> list[str]:
        """Discovers and filters relevant pillars."""
        all_pillars = self.instrument.pillar_names
        if not self.regex:
            return all_pillars
        pattern = re.compile(self.regex)
        return [p for p in all_pillars if pattern.search(p)]

    def _get_instruments(self) -> dict[str, Instrument]:
        """Returns a mapping of instrument name to Instrument object."""
        if hasattr(self.instrument, "_instruments"):
            return self.instrument._instruments # Portfolio
        return {"Main": self.instrument}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} regex={self.regex!r} instrument={self.instrument!r}>"
