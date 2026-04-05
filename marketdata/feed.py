"""
MarketDataFeed — Abstract Base Class
=====================================
All market data sources (simulator, Polygon, Alpaca, etc.) implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from marketdata.bus import TickBus


class MarketDataFeed(ABC):
    """Abstract base class for market data feeds.

    Implementations produce Tick objects and publish them to a TickBus.
    """

    @abstractmethod
    async def start(self, bus: TickBus) -> None:
        """Start generating ticks and publishing to the bus.

        This method should run until ``stop()`` is called.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Signal the feed to stop producing ticks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this feed (e.g. 'simulator', 'polygon')."""

    @abstractmethod
    def symbols(self) -> dict[str, list[str]]:
        """Return the symbol universe grouped by type (e.g. equity, fx)."""
