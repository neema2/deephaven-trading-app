"""
Market Data Models
==================
Pydantic v2 models shared across the market data pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Tick(BaseModel):
    """A single equity price tick from a market data feed."""
    type: Literal["equity"] = "equity"
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    change: float
    change_pct: float
    timestamp: datetime


class RiskTick(BaseModel):
    """Per-symbol risk snapshot computed from a price tick."""
    symbol: str
    position: int
    market_value: float
    unrealized_pnl: float
    delta: float
    gamma: float
    theta: float
    vega: float
    timestamp: datetime


class FXTick(BaseModel):
    """A single FX spot tick."""
    type: Literal["fx"] = "fx"
    pair: str           # "USD/JPY"
    bid: float
    ask: float
    mid: float
    spread_pips: float
    currency: str       # quote currency
    timestamp: datetime


class CurveTick(BaseModel):
    """A single yield curve point tick."""
    type: Literal["curve"] = "curve"
    label: str          # "USD_5Y"
    tenor_years: float
    rate: float
    discount_factor: float
    currency: str       # "USD" / "JPY"
    timestamp: datetime


class SwapTick(BaseModel):
    """A single interest rate swap quote tick."""
    type: Literal["swap"] = "swap"
    symbol: str
    rate: float
    timestamp: datetime


# Discriminated union of all market data message types
MarketDataMessage = Annotated[
    Tick | FXTick | CurveTick | SwapTick,
    Field(discriminator="type"),
]


def get_symbol_key(msg: Tick | FXTick | CurveTick) -> str:
    """Extract the symbol/pair/label key from any market data message."""
    if isinstance(msg, Tick):
        return msg.symbol
    if isinstance(msg, FXTick):
        return msg.pair
    if isinstance(msg, CurveTick):
        return msg.label
    if isinstance(msg, SwapTick):
        return msg.symbol
    raise ValueError(f"Unknown message type: {type(msg)}")


class Subscription(BaseModel):
    """Client subscription request — which types/symbols to stream."""
    types: list[str] | None = None    # ["equity","fx","curve"] or None=all
    symbols: list[str] | None = None  # symbol/pair/label filter, None=all


class SnapshotResponse(BaseModel):
    """REST response for a single symbol snapshot."""
    symbol: str
    tick: Tick | None = None
