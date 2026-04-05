"""Unit tests for Coinbase Exchange feed message mapping (no network)."""

from __future__ import annotations

import asyncio
import json
from datetime import timezone
from unittest.mock import patch

import pytest
from marketdata.bus import TickBus
from marketdata.feeds.coinbase_exchange import (
    CoinbaseExchangeFeed,
    coinbase_ticker_to_tick,
)


def test_coinbase_ticker_to_tick_maps_fields() -> None:
    msg = {
        "type": "ticker",
        "sequence": 1,
        "product_id": "BTC-USD",
        "price": "50000.5",
        "open_24h": "49000",
        "volume_24h": "123.45",
        "best_bid": "50000",
        "best_ask": "50001",
        "time": "2026-04-03T12:00:00.000000Z",
    }
    tick = coinbase_ticker_to_tick(msg)
    assert tick is not None
    assert tick.symbol == "BTC-USD"
    assert tick.price == 50000.5
    assert tick.bid == 50000.0
    assert tick.ask == 50001.0
    assert tick.volume == 123
    assert abs(tick.change - 1000.5) < 0.001
    assert tick.change_pct > 0
    assert tick.timestamp.tzinfo is not None


def test_coinbase_ticker_to_tick_rejects_non_ticker() -> None:
    assert coinbase_ticker_to_tick({"type": "subscriptions"}) is None
    assert coinbase_ticker_to_tick({"type": "ticker"}) is None  # no product_id


def test_coinbase_feed_defaults() -> None:
    f = CoinbaseExchangeFeed(product_ids=["BTC-USD", "ETH-USD"])
    assert f.name == "coinbase"
    assert f.product_ids == ["BTC-USD", "ETH-USD"]


def test_coinbase_feed_symbols() -> None:
    f = CoinbaseExchangeFeed(product_ids=["BTC-USD", "ETH-USD"])
    assert f.symbols() == {"equity": ["BTC-USD", "ETH-USD"], "fx": []}


def test_parse_time_offset() -> None:
    msg = {
        "type": "ticker",
        "product_id": "ETH-USD",
        "price": "1",
        "open_24h": "1",
        "volume_24h": "0",
        "best_bid": "1",
        "best_ask": "1",
        "time": "2026-04-03T12:00:00+00:00",
    }
    tick = coinbase_ticker_to_tick(msg)
    assert tick is not None
    assert tick.timestamp.tzinfo == timezone.utc

@pytest.mark.asyncio
async def test_coinbase_feed_start_publishes_tick_via_mock_ws() -> None:
    """``CoinbaseExchangeFeed.start`` subscribes and maps ticker JSON to ``bus.publish``."""
    ticker_msg = {
        "type": "ticker",
        "product_id": "BTC-USD",
        "price": "100",
        "open_24h": "90",
        "volume_24h": "1",
        "best_bid": "99",
        "best_ask": "101",
        "time": "2026-04-03T12:00:00.000000Z",
    }
    raw = json.dumps(ticker_msg)

    feed = CoinbaseExchangeFeed(
        product_ids=["BTC-USD"],
        ws_url="wss://test.invalid",
        reconnect_delay_s=0.01,
    )
    stop_evt = feed._stop_event

    class FakeWS:
        def __init__(self) -> None:
            self._first = True

        async def send(self, _data: str) -> None:
            return None

        def __aiter__(self) -> FakeWS:
            return self

        async def __anext__(self) -> str:
            if self._first:
                self._first = False
                return raw
            await stop_evt.wait()
            raise StopAsyncIteration

    class FakeConn:
        async def __aenter__(self) -> FakeWS:
            return FakeWS()

        async def __aexit__(self, *_args: object) -> None:
            return None

    bus = TickBus()

    with patch(
        "marketdata.feeds.coinbase_exchange.websockets.connect",
        return_value=FakeConn(),
    ):
        task = asyncio.create_task(feed.start(bus))
        for _ in range(200):
            if ("equity", "BTC-USD") in bus.latest:
                break
            await asyncio.sleep(0.01)
        await feed.stop()
        await asyncio.wait_for(task, timeout=2.0)

    tick = bus.latest.get(("equity", "BTC-USD"))
    assert tick is not None
    assert tick.symbol == "BTC-USD"
    assert tick.price == 100.0
