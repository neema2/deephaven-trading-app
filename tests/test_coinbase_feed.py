"""Unit tests for Coinbase Exchange feed message mapping (no network)."""

from __future__ import annotations

from datetime import timezone

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
