"""
Market Data Server
==================
Standalone real-time market data service with pluggable feeds,
async pub/sub bus, and REST + WebSocket API.
"""

from typing import Any

from marketdata.bus import TickBus
from marketdata.client import MarketDataClient
from marketdata.feed import MarketDataFeed
from marketdata.feeds.simulator import SimulatorFeed
from marketdata.models import (
    CurveTick,
    FXTick,
    MarketDataMessage,
    RiskTick,
    SnapshotResponse,
    Subscription,
    Tick,
    get_symbol_key,
)

__all__ = [
    "CoinbaseExchangeFeed",
    "CurveTick",
    "FXTick",
    "MarketDataClient",
    "MarketDataFeed",
    "MarketDataMessage",
    "RiskTick",
    "SimulatorFeed",
    "SnapshotResponse",
    "Subscription",
    "Tick",
    "TickBus",
    "get_symbol_key",
]


def __getattr__(name: str) -> Any:
    if name == "CoinbaseExchangeFeed":
        from marketdata.feeds.coinbase_exchange import CoinbaseExchangeFeed

        return CoinbaseExchangeFeed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
