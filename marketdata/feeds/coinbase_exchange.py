"""
CoinbaseExchangeFeed — Live ticker stream from Coinbase Exchange (public WebSocket).

Docs: https://docs.cdp.coinbase.com/exchange/websocket-feed/overview

Market data terms: https://www.coinbase.com/legal/market_data

Enable via environment (before starting the market data server)::

    export MARKETDATA_FEED=coinbase
    export COINBASE_PRODUCT_IDS=BTC-USD,ETH-USD,SOL-USD   # optional

Ticker messages are mapped onto the existing :class:`~marketdata.models.Tick` model
(``type="equity"``) so REST/WebSocket clients and Deephaven demos work unchanged;
``symbol`` holds the Coinbase product id (e.g. ``BTC-USD``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import websockets

from marketdata.bus import TickBus
from marketdata.feed import MarketDataFeed
from marketdata.models import Tick

logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "wss://ws-feed.exchange.coinbase.com"
DEFAULT_PRODUCT_IDS = ("BTC-USD", "ETH-USD", "SOL-USD")


def _parse_product_ids_from_env() -> list[str]:
    raw = os.environ.get("COINBASE_PRODUCT_IDS", "")
    if not raw.strip():
        return list(DEFAULT_PRODUCT_IDS)
    return [p.strip() for p in raw.split(",") if p.strip()]


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        s = value.replace("Z", "+00:00") if value.endswith("Z") else value
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.now(timezone.utc)


def coinbase_ticker_to_tick(msg: dict[str, Any]) -> Tick | None:
    """Convert a Coinbase ``ticker`` JSON object to a :class:`Tick`, or ``None``."""
    if msg.get("type") != "ticker":
        return None
    product_id = msg.get("product_id")
    if not product_id:
        return None

    price = _f(msg.get("price"))
    open_24h = _f(msg.get("open_24h"), price)
    bid = _f(msg.get("best_bid"))
    ask = _f(msg.get("best_ask"))
    vol_f = _f(msg.get("volume_24h"))
    volume = int(min(max(vol_f, 0), 2_147_483_647))

    change = price - open_24h
    change_pct = (change / open_24h) * 100.0 if open_24h else 0.0
    ts = _parse_time(msg.get("time"))

    return Tick(
        symbol=str(product_id),
        price=price,
        bid=bid,
        ask=ask,
        volume=volume,
        change=change,
        change_pct=change_pct,
        timestamp=ts,
    )


class CoinbaseExchangeFeed(MarketDataFeed):
    """Async feed: Coinbase Exchange WebSocket ``ticker`` channel → :class:`Tick` on the bus."""

    def __init__(
        self,
        product_ids: list[str] | None = None,
        ws_url: str | None = None,
        reconnect_delay_s: float = 3.0,
    ) -> None:
        self._product_ids = list(product_ids) if product_ids is not None else _parse_product_ids_from_env()
        self._ws_url = (ws_url or os.environ.get("COINBASE_WS_URL") or DEFAULT_WS_URL).strip()
        self._reconnect_delay_s = reconnect_delay_s
        self._stop_event = asyncio.Event()

    @property
    def name(self) -> str:
        return "coinbase"

    @property
    def product_ids(self) -> list[str]:
        return list(self._product_ids)

    async def start(self, bus: TickBus) -> None:
        self._stop_event.clear()
        logger.info(
            "CoinbaseExchangeFeed starting: products=%s url=%s",
            ",".join(self._product_ids),
            self._ws_url,
        )

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    sub = {
                        "type": "subscribe",
                        "product_ids": self._product_ids,
                        "channels": ["ticker"],
                    }
                    await ws.send(json.dumps(sub))
                    logger.info("Subscribed to Coinbase ticker channel")

                    async for raw in ws:
                        if self._stop_event.is_set():
                            return
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        mtype = data.get("type")
                        if mtype == "error":
                            logger.error("Coinbase WebSocket error: %s", data)
                            break
                        if mtype in ("subscriptions", "heartbeat"):
                            continue

                        tick = coinbase_ticker_to_tick(data)
                        if tick is not None:
                            await bus.publish(tick)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "Coinbase WebSocket disconnected (%s). Reconnecting in %.1fs...",
                    exc,
                    self._reconnect_delay_s,
                )
                await asyncio.sleep(self._reconnect_delay_s)

        logger.info("CoinbaseExchangeFeed stopped")

    async def stop(self) -> None:
        self._stop_event.set()
