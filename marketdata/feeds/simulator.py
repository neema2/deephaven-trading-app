"""
SimulatorFeed — Async Market Data Simulator
============================================
Generates realistic ticking price data using geometric Brownian motion.
Async rewrite of the original server/market_data.py.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any

from marketdata.bus import TickBus
from marketdata.feed import MarketDataFeed
from marketdata.models import FXTick, Tick

logger = logging.getLogger(__name__)

SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

BASE_PRICES = {
    "AAPL": 228.0, "GOOGL": 192.0, "MSFT": 415.0, "AMZN": 225.0,
    "TSLA": 355.0, "NVDA": 138.0, "META": 700.0, "NFLX": 1020.0,
}

# Random starting positions (shares held, negative = short)
POSITIONS = {sym: random.randint(-500, 500) for sym in SYMBOLS}

# ── FX pairs ─────────────────────────────────────────────────────────────────

FX_PAIRS = ["USD/JPY", "EUR/USD", "GBP/USD"]

FX_BASE: dict[str, dict[str, Any]] = {
    "USD/JPY": {"mid": 149.55, "spread": 0.10, "currency": "JPY"},
    "EUR/USD": {"mid": 1.0852, "spread": 0.0005, "currency": "USD"},
    "GBP/USD": {"mid": 1.2705, "spread": 0.0010, "currency": "USD"},
}


class SimulatorFeed(MarketDataFeed):
    """Async market data simulator producing realistic ticking prices.

    Uses geometric Brownian motion with 0.2% std dev per tick.
    Publishes both price Ticks and RiskTicks to the TickBus.
    """

    def __init__(self, tick_interval: float = 0.2) -> None:
        self._tick_interval = tick_interval
        self._current_prices: dict[str, float] = dict(BASE_PRICES)
        self._current_fx: dict[str, float] = {
            pair: data["mid"] for pair, data in FX_BASE.items()
        }
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "simulator"

    def symbols(self) -> dict[str, list[str]]:
        return {"equity": list(SYMBOLS), "fx": list(FX_PAIRS)}

    async def start(self, bus: TickBus) -> None:
        """Start generating ticks and publishing to the bus."""
        self._stop_event.clear()
        logger.info(
            "SimulatorFeed starting: %d equities + %d FX pairs, %.0fms interval",
            len(SYMBOLS), len(FX_PAIRS), self._tick_interval * 1000,
        )

        while not self._stop_event.is_set():
            try:
                now = datetime.now(timezone.utc)

                # ── Equity ticks ──────────────────────────────────────────
                for sym in SYMBOLS:
                    old = self._current_prices[sym]
                    move = random.gauss(0, 0.002)  # 0.2% std dev per tick
                    new_price = old * (1 + move)
                    self._current_prices[sym] = new_price

                    spread = new_price * 0.0001
                    bid = new_price - spread / 2
                    ask = new_price + spread / 2
                    volume = random.randint(100, 10_000)
                    change = new_price - old
                    change_pct = (change / old) * 100

                    tick = Tick(
                        symbol=sym,
                        price=new_price,
                        bid=bid,
                        ask=ask,
                        volume=volume,
                        change=change,
                        change_pct=change_pct,
                        timestamp=now,
                    )
                    await bus.publish(tick)

                # ── FX ticks ─────────────────────────────────────────────
                for pair in FX_PAIRS:
                    base = FX_BASE[pair]
                    old_mid = self._current_fx[pair]
                    mid_move = random.gauss(0, 0.0003) * old_mid  # ~3bp σ
                    new_mid = old_mid + mid_move
                    self._current_fx[pair] = new_mid

                    half_spread = base["spread"] / 2
                    bid = new_mid - half_spread
                    ask = new_mid + half_spread
                    spread_pips = base["spread"] * 10_000

                    fx_tick = FXTick(
                        pair=pair,
                        bid=round(bid, 5),
                        ask=round(ask, 5),
                        mid=round(new_mid, 5),
                        spread_pips=round(spread_pips, 1),
                        currency=base["currency"],
                        timestamp=now,
                    )
                    await bus.publish(fx_tick)

                await asyncio.sleep(self._tick_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("SimulatorFeed error: %s", e)
                await asyncio.sleep(1)

        logger.info("SimulatorFeed stopped")

    async def stop(self) -> None:
        """Signal the feed to stop producing ticks."""
        self._stop_event.set()
