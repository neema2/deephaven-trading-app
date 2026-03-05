"""
Tests for the Market Data Server
=================================
Covers: TickBus, SimulatorFeed, risk_engine, FastAPI REST + WebSocket endpoints.
No Deephaven dependency required.
"""

import asyncio
import math

import pytest
from marketdata.bus import TickBus
from marketdata.feeds.simulator import (
    BASE_PRICES,
    FX_BASE,
    FX_PAIRS,
    POSITIONS,
    SYMBOLS,
    SimulatorFeed,
)
from marketdata.models import (
    CurveTick,
    FXTick,
    MarketDataMessage,
    Tick,
    get_symbol_key,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tick(symbol="AAPL", price=150.0) -> Tick:
    from datetime import datetime, timezone
    return Tick(
        symbol=symbol, price=price,
        bid=price - 0.01, ask=price + 0.01,
        volume=1000, change=0.5, change_pct=0.33,
        timestamp=datetime.now(timezone.utc),
    )


def _make_fx_tick(pair="USD/JPY", mid=149.55) -> FXTick:
    from datetime import datetime, timezone
    return FXTick(
        pair=pair, bid=mid - 0.05, ask=mid + 0.05,
        mid=mid, spread_pips=1.0, currency="JPY",
        timestamp=datetime.now(timezone.utc),
    )


def _make_curve_tick(label="USD_5Y", rate=0.041) -> CurveTick:
    from datetime import datetime, timezone
    return CurveTick(
        label=label, tenor_years=5.0, rate=rate,
        discount_factor=1.0 / (1.0 + rate) ** 5.0,
        currency="USD",
        timestamp=datetime.now(timezone.utc),
    )


# ── TickBus Tests ────────────────────────────────────────────────────────────

class TestTickBus:
    @pytest.mark.asyncio
    async def test_publish_updates_latest(self):
        bus = TickBus()
        tick = _make_tick("AAPL", 150.0)
        await bus.publish(tick)
        assert bus.latest[("equity", "AAPL")] == tick

    @pytest.mark.asyncio
    async def test_subscribe_receives_tick(self):
        bus = TickBus()
        sub_id, tick_iter = await bus.subscribe()
        tick = _make_tick("AAPL")
        await bus.publish(tick)

        received = await asyncio.wait_for(tick_iter.__anext__(), timeout=2.0)
        assert received.symbol == "AAPL"  # type: ignore[union-attr]
        await bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_subscribe_with_symbol_filter(self):
        bus = TickBus()
        sub_id, tick_iter = await bus.subscribe(symbols={"MSFT"})

        await bus.publish(_make_tick("AAPL"))
        await bus.publish(_make_tick("MSFT", 415.0))

        received = await asyncio.wait_for(tick_iter.__anext__(), timeout=2.0)
        assert received.symbol == "MSFT"  # type: ignore[union-attr]
        await bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_subscribe_all_symbols(self):
        bus = TickBus()
        sub_id, tick_iter = await bus.subscribe(symbols=None)

        await bus.publish(_make_tick("AAPL"))
        await bus.publish(_make_tick("MSFT"))

        t1 = await asyncio.wait_for(tick_iter.__anext__(), timeout=2.0)
        t2 = await asyncio.wait_for(tick_iter.__anext__(), timeout=2.0)
        assert {t1.symbol, t2.symbol} == {"AAPL", "MSFT"}  # type: ignore[union-attr]
        await bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_backpressure_drops_oldest(self):
        bus = TickBus(maxsize=2)
        sub_id, tick_iter = await bus.subscribe()

        # Publish 3 ticks — queue holds 2, oldest should be dropped
        await bus.publish(_make_tick("A", 1.0))
        await bus.publish(_make_tick("B", 2.0))
        await bus.publish(_make_tick("C", 3.0))

        t1 = await asyncio.wait_for(tick_iter.__anext__(), timeout=2.0)
        t2 = await asyncio.wait_for(tick_iter.__anext__(), timeout=2.0)
        # Oldest (A) dropped, should get B and C
        assert {t1.symbol, t2.symbol} == {"B", "C"}  # type: ignore[union-attr]
        await bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_subscription(self):
        bus = TickBus()
        sub_id, _ = await bus.subscribe()
        assert bus.subscriber_count == 1
        await bus.unsubscribe(sub_id)
        assert bus.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        bus = TickBus()
        s1, _ = await bus.subscribe()
        s2, _ = await bus.subscribe()
        assert bus.subscriber_count == 2
        await bus.unsubscribe(s1)
        await bus.unsubscribe(s2)


# ── Multi-Asset TickBus Tests ─────────────────────────────────────────────

class TestTickBusMultiAsset:
    @pytest.mark.asyncio
    async def test_publish_fx_tick(self):
        bus = TickBus()
        fx = _make_fx_tick("USD/JPY", 149.55)
        await bus.publish(fx)
        assert bus.latest[("fx", "USD/JPY")] == fx

    @pytest.mark.asyncio
    async def test_publish_curve_tick(self):
        bus = TickBus()
        curve = _make_curve_tick("USD_5Y", 0.041)
        await bus.publish(curve)
        assert bus.latest[("curve", "USD_5Y")] == curve

    @pytest.mark.asyncio
    async def test_type_filter_equity_only(self):
        bus = TickBus()
        sub_id, msg_iter = await bus.subscribe(types={"equity"})

        await bus.publish(_make_tick("AAPL"))
        await bus.publish(_make_fx_tick("USD/JPY"))

        received = await asyncio.wait_for(msg_iter.__anext__(), timeout=2.0)
        assert received.type == "equity"
        assert received.symbol == "AAPL"
        await bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_type_filter_fx_only(self):
        bus = TickBus()
        sub_id, msg_iter = await bus.subscribe(types={"fx"})

        await bus.publish(_make_tick("AAPL"))
        await bus.publish(_make_fx_tick("EUR/USD", 1.085))

        received = await asyncio.wait_for(msg_iter.__anext__(), timeout=2.0)
        assert received.type == "fx"
        assert received.pair == "EUR/USD"
        await bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_type_and_symbol_filter(self):
        bus = TickBus()
        sub_id, msg_iter = await bus.subscribe(
            types={"equity"}, symbols={"MSFT"},
        )

        await bus.publish(_make_tick("AAPL"))
        await bus.publish(_make_fx_tick("USD/JPY"))
        await bus.publish(_make_tick("MSFT", 415.0))

        received = await asyncio.wait_for(msg_iter.__anext__(), timeout=2.0)
        assert received.symbol == "MSFT"  # type: ignore[union-attr]
        await bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_all_types_subscription(self):
        bus = TickBus()
        sub_id, msg_iter = await bus.subscribe()  # types=None → all

        await bus.publish(_make_tick("AAPL"))
        await bus.publish(_make_fx_tick("USD/JPY"))
        await bus.publish(_make_curve_tick("USD_5Y"))

        msgs = []
        for _ in range(3):
            m = await asyncio.wait_for(msg_iter.__anext__(), timeout=2.0)
            msgs.append(m)

        types_seen = {m.type for m in msgs}
        assert types_seen == {"equity", "fx", "curve"}
        await bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_latest_keyed_by_type_and_symbol(self):
        bus = TickBus()
        await bus.publish(_make_tick("AAPL", 150.0))
        await bus.publish(_make_fx_tick("USD/JPY", 149.55))
        await bus.publish(_make_curve_tick("USD_5Y", 0.041))

        assert ("equity", "AAPL") in bus.latest
        assert ("fx", "USD/JPY") in bus.latest
        assert ("curve", "USD_5Y") in bus.latest
        assert len(bus.latest) == 3


# ── Model Tests ───────────────────────────────────────────────────────────

class TestModels:
    def test_tick_has_type_equity(self):
        tick = _make_tick()
        assert tick.type == "equity"

    def test_fx_tick_has_type_fx(self):
        fx = _make_fx_tick()
        assert fx.type == "fx"

    def test_curve_tick_has_type_curve(self):
        curve = _make_curve_tick()
        assert curve.type == "curve"

    def test_get_symbol_key_tick(self):
        assert get_symbol_key(_make_tick("AAPL")) == "AAPL"

    def test_get_symbol_key_fx(self):
        assert get_symbol_key(_make_fx_tick("USD/JPY")) == "USD/JPY"

    def test_get_symbol_key_curve(self):
        assert get_symbol_key(_make_curve_tick("USD_5Y")) == "USD_5Y"

    def test_discriminated_union_parse_equity(self):
        from pydantic import TypeAdapter
        adapter = TypeAdapter(MarketDataMessage)  # type: ignore[var-annotated]
        data = _make_tick().model_dump()
        msg = adapter.validate_python(data)
        assert isinstance(msg, Tick)

    def test_discriminated_union_parse_fx(self):
        from pydantic import TypeAdapter
        adapter = TypeAdapter(MarketDataMessage)  # type: ignore[var-annotated]
        data = _make_fx_tick().model_dump()
        msg = adapter.validate_python(data)
        assert isinstance(msg, FXTick)

    def test_discriminated_union_parse_curve(self):
        from pydantic import TypeAdapter
        adapter = TypeAdapter(MarketDataMessage)  # type: ignore[var-annotated]
        data = _make_curve_tick().model_dump()
        msg = adapter.validate_python(data)
        assert isinstance(msg, CurveTick)


# ── SimulatorFeed Tests ──────────────────────────────────────────────────────

class TestSimulatorFeedConfig:
    def test_equity_config_consistent(self):
        assert len(SYMBOLS) == 8
        assert set(BASE_PRICES.keys()) == set(SYMBOLS)
        assert set(POSITIONS.keys()) == set(SYMBOLS)
        for sym in SYMBOLS:
            assert BASE_PRICES[sym] > 0, f"{sym} has non-positive base price"
            assert isinstance(POSITIONS[sym], int), f"{sym} position is not int"

    def test_fx_config_consistent(self):
        assert len(FX_PAIRS) == 3
        assert set(FX_BASE.keys()) == set(FX_PAIRS)
        for _pair, data in FX_BASE.items():
            assert "mid" in data and "spread" in data and "currency" in data
            assert data["mid"] > 0


class TestSimulatorFeed:
    @pytest.mark.asyncio
    async def test_feed_produces_ticks(self):
        bus = TickBus()
        feed = SimulatorFeed(tick_interval=0.05)
        sub_id, tick_iter = await bus.subscribe()

        task = asyncio.create_task(feed.start(bus))
        ticks = []
        for _ in range(8):
            tick = await asyncio.wait_for(tick_iter.__anext__(), timeout=3.0)
            ticks.append(tick)

        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await bus.unsubscribe(sub_id)

        assert len(ticks) == 8

    @pytest.mark.asyncio
    async def test_feed_covers_all_symbols(self):
        bus = TickBus()
        feed = SimulatorFeed(tick_interval=0.05)
        sub_id, tick_iter = await bus.subscribe()

        task = asyncio.create_task(feed.start(bus))
        ticks = []
        # Collect enough ticks to cover all symbols + FX pairs (one cycle = 8 + 3 = 11)
        for _ in range(22):
            tick = await asyncio.wait_for(tick_iter.__anext__(), timeout=3.0)
            ticks.append(tick)

        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await bus.unsubscribe(sub_id)

        equity_symbols = {t.symbol for t in ticks if t.type == "equity"}
        fx_pairs = {t.pair for t in ticks if t.type == "fx"}
        assert equity_symbols == set(SYMBOLS)
        assert fx_pairs == set(FX_PAIRS)

    @pytest.mark.asyncio
    async def test_feed_ticks_have_valid_data(self):
        bus = TickBus()
        feed = SimulatorFeed(tick_interval=0.05)
        sub_id, tick_iter = await bus.subscribe(types={"equity"})

        task = asyncio.create_task(feed.start(bus))
        tick = await asyncio.wait_for(tick_iter.__anext__(), timeout=3.0)

        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await bus.unsubscribe(sub_id)

        assert tick.price > 0  # type: ignore[union-attr]
        assert tick.bid < tick.ask  # type: ignore[union-attr]
        assert 100 <= tick.volume <= 10_000  # type: ignore[union-attr]
        assert math.isfinite(tick.change_pct)  # type: ignore[union-attr]
        assert tick.timestamp is not None

    @pytest.mark.asyncio
    async def test_feed_produces_fx_ticks(self):
        bus = TickBus()
        feed = SimulatorFeed(tick_interval=0.05)
        sub_id, tick_iter = await bus.subscribe(types={"fx"})

        task = asyncio.create_task(feed.start(bus))
        fx = await asyncio.wait_for(tick_iter.__anext__(), timeout=3.0)

        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await bus.unsubscribe(sub_id)

        assert fx.type == "fx"
        assert fx.pair in FX_PAIRS
        assert fx.bid < fx.ask
        assert fx.mid > 0
        assert fx.spread_pips > 0
        assert fx.timestamp is not None

    @pytest.mark.asyncio
    async def test_feed_stop(self):
        bus = TickBus()
        feed = SimulatorFeed(tick_interval=0.05)
        task = asyncio.create_task(feed.start(bus))

        await asyncio.sleep(0.2)
        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Should not raise



# ── FastAPI Server Tests ─────────────────────────────────────────────────────

class TestRESTEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from marketdata.server import app
        with TestClient(app) as c:
            yield c

    def test_health(self, client):
        import time
        time.sleep(0.5)
        resp = client.get("/md/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["feed"] == "simulator"
        assert "asset_types" in data

    def test_symbols(self, client):
        resp = client.get("/md/symbols")
        assert resp.status_code == 200
        data = resp.json()
        assert "equity" in data
        assert "fx" in data
        assert len(data["equity"]) == 8
        assert len(data["fx"]) == 3

    def test_snapshot_all(self, client):
        import time
        time.sleep(0.5)  # let simulator produce some ticks
        resp = client.get("/md/snapshot")
        assert resp.status_code == 200

    def test_snapshot_by_type(self, client):
        import time
        time.sleep(0.5)
        resp = client.get("/md/snapshot/equity")
        assert resp.status_code == 200
        data = resp.json()
        # Should have equity symbols as keys
        if data:  # may be empty if simulator hasn't ticked yet
            for key in data:
                assert data[key]["type"] == "equity"

    def test_snapshot_fx_type(self, client):
        import time
        time.sleep(0.5)
        resp = client.get("/md/snapshot/fx")
        assert resp.status_code == 200

    def test_snapshot_single_equity(self, client):
        import time
        time.sleep(0.5)
        resp = client.get("/md/snapshot/equity/AAPL")
        assert resp.status_code == 200

    def test_snapshot_unknown(self, client):
        resp = client.get("/md/snapshot/equity/ZZZZ")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"] is None


class TestWebSocket:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from marketdata.server import app
        with TestClient(app) as c:
            yield c

    def test_websocket_connect_and_receive(self, client):
        import time
        time.sleep(0.3)
        with client.websocket_connect("/md/subscribe") as ws:
            # Should receive messages (subscribed to all by default)
            data = ws.receive_json()
            assert "type" in data
            assert "timestamp" in data

    def test_websocket_type_filtered_subscription(self, client):
        import time
        time.sleep(0.3)
        with client.websocket_connect("/md/subscribe") as ws:
            # Update subscription to equity only
            ws.send_json({"types": ["equity"]})
            ticks = []
            for _ in range(5):
                try:
                    data = ws.receive_json()
                    ticks.append(data)
                except Exception:
                    break
            # All ticks should be equity type
            for t in ticks:
                assert t.get("type") == "equity"

    def test_websocket_publish_curve(self, client):
        """Test bidirectional WS: client publishes a CurveTick."""
        import time
        time.sleep(0.3)
        from datetime import datetime, timezone
        with client.websocket_connect("/md/subscribe") as ws:
            # Publish a CurveTick via the WS
            ws.send_json({
                "type": "curve",
                "label": "TEST_5Y",
                "tenor_years": 5.0,
                "rate": 0.05,
                "discount_factor": 0.78,
                "currency": "USD",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Should eventually receive it back (subscribed to all)
            found_curve = False
            for _ in range(20):
                try:
                    data = ws.receive_json()
                    if data.get("type") == "curve" and data.get("label") == "TEST_5Y":
                        found_curve = True
                        break
                except Exception:
                    break
            assert found_curve


class TestPublishEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from marketdata.server import app
        with TestClient(app) as c:
            yield c

    def test_publish_curve_tick(self, client):
        from datetime import datetime, timezone
        resp = client.post("/md/publish", json={
            "type": "curve",
            "label": "USD_10Y",
            "tenor_years": 10.0,
            "rate": 0.04,
            "discount_factor": 0.67,
            "currency": "USD",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "published"
        assert data["type"] == "curve"
        assert data["key"] == "USD_10Y"

    def test_publish_fx_tick(self, client):
        from datetime import datetime, timezone
        resp = client.post("/md/publish", json={
            "type": "fx",
            "pair": "USD/CHF",
            "bid": 0.8800,
            "ask": 0.8810,
            "mid": 0.8805,
            "spread_pips": 1.0,
            "currency": "CHF",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "fx"
        assert data["key"] == "USD/CHF"
