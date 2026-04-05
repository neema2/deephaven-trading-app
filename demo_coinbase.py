"""
Demo: Coinbase → Market Data Server → Deephaven
================================================
Same dashboard tables as ``demo_trading.py``, but the market data server uses
:class:`~marketdata.feeds.coinbase_exchange.CoinbaseExchangeFeed` when
``MARKETDATA_FEED=coinbase``.

Prerequisites:
  pip install -e ".[streaming,marketdata]"

Optional env (before launch)::
  export COINBASE_PRODUCT_IDS=BTC-USD,ETH-USD,SOL-USD

Standalone::
  python3 demo_coinbase.py

Open the Deephaven IDE at http://localhost:10000 (default streaming port).
"""

import asyncio
import os

# Must be set before MarketDataServer spawns uvicorn (subprocess inherits env).
os.environ.setdefault("MARKETDATA_FEED", "coinbase")

import json  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_log = logging.getLogger("demo_coinbase")

_feed_thread: threading.Thread | None = None
_feed_stop = threading.Event()


def publish_tables(md_ws_url: str = "ws://localhost:8000/md/subscribe"):
    """Create DH writers, derive tables, consume market-data WebSocket (same as demo_trading)."""
    global _feed_thread, _feed_stop

    from streaming import TickingTable, agg

    prices = TickingTable({
        "Symbol": str,
        "Price": float,
        "Bid": float,
        "Ask": float,
        "Volume": int,
        "Change": float,
        "ChangePct": float,
    })

    risk = TickingTable({
        "Symbol": str,
        "Price": float,
        "Position": int,
        "MarketValue": float,
        "UnrealizedPnL": float,
        "Delta": float,
        "Gamma": float,
        "Theta": float,
        "Vega": float,
    })

    prices_live = prices.last_by("Symbol")
    risk_live = risk.last_by("Symbol")

    portfolio_summary = risk_live.agg_by([
        agg.sum(["TotalMV=MarketValue", "TotalPnL=UnrealizedPnL", "TotalDelta=Delta"]),
        agg.avg(["AvgGamma=Gamma", "AvgTheta=Theta", "AvgVega=Vega"]),
        agg.count("NumPositions"),
    ])

    top_movers = prices_live.sort_descending("ChangePct")
    volume_leaders = prices_live.sort_descending("Volume")

    prices.publish("prices_raw")
    prices_live.publish("prices_live")
    risk.publish("risk_raw")
    risk_live.publish("risk_live")
    portfolio_summary.publish("portfolio_summary")
    top_movers.publish("top_movers")
    volume_leaders.publish("volume_leaders")

    import random

    _positions: dict[str, int] = {}

    def _on_tick(tick: dict):
        sym = tick["symbol"]
        prices.write_row(
            sym, tick["price"], tick["bid"], tick["ask"],
            tick["volume"], tick["change"], tick["change_pct"],
        )
        if sym not in _positions:
            _positions[sym] = random.randint(100, 1000)
        pos = _positions[sym]
        risk.write_row(
            sym, tick["price"], pos,
            tick["price"] * pos,
            tick["change"] * pos,
            0.5 + random.random() * 0.3,
            0.02 + random.random() * 0.04,
            -0.1 - random.random() * 0.15,
            0.2 + random.random() * 0.2,
        )

    async def _consume(url: str):
        import websockets

        while not _feed_stop.is_set():
            try:
                _log.info("Connecting to Market Data Server at %s ...", url)
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps({"types": ["equity"]}))
                    _log.info("Connected — streaming Coinbase-backed equity ticks")
                    async for msg in ws:
                        if _feed_stop.is_set():
                            return
                        tick = json.loads(msg)
                        if tick.get("type") == "equity":
                            _on_tick(tick)
            except Exception as e:
                if _feed_stop.is_set():
                    return
                _log.warning("Market Data connection lost (%s). Retrying in 2s...", e)
                await asyncio.sleep(2)

    _feed_stop.clear()
    _feed_thread = threading.Thread(
        target=lambda: asyncio.run(_consume(md_ws_url)),
        daemon=True,
        name="md-consumer-coinbase",
    )
    _feed_thread.start()

    _log.info("Coinbase dashboard tables published — 7 tables")

    return {
        "prices_raw": prices, "prices_live": prices_live,
        "risk_raw": risk, "risk_live": risk_live,
        "portfolio_summary": portfolio_summary,
        "top_movers": top_movers, "volume_leaders": volume_leaders,
    }


def stop_feed():
    global _feed_thread
    _feed_stop.set()
    if _feed_thread and _feed_thread.is_alive():
        _feed_thread.join(timeout=5)
    _feed_thread = None


if __name__ == "__main__":
    print("\n── Platform: Coinbase + MarketDataServer + Deephaven ──")
    print("  MARKETDATA_FEED=%s" % os.environ.get("MARKETDATA_FEED", ""))

    from streaming.admin import StreamingServer

    streaming = StreamingServer(port=10000)
    streaming.start()
    print(f"  Streaming server: http://localhost:{streaming.port}")

    from marketdata.admin import MarketDataServer

    md = MarketDataServer(port=8000)
    asyncio.run(md.start())
    print(f"  Market data server: port {md.port} (feed={os.environ.get('MARKETDATA_FEED', 'simulator')})")

    tables = publish_tables(f"ws://localhost:{md.port}/md/subscribe")

    print()
    print("=" * 60)
    print("  Coinbase dashboard RUNNING")
    for name in tables:
        print(f"    • {name}")
    print("=" * 60)
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_feed()
        asyncio.run(md.stop())
        streaming.stop()
