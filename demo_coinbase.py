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

Dashboard wiring is shared with ``demo_trading.py`` via :mod:`demo_dashboard_common`.
"""

import asyncio
import logging

from demo_dashboard_common import publish_trading_dashboard_tables, stop_trading_dashboard_feed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_log = logging.getLogger("demo_coinbase")


def publish_tables(md_ws_url: str = "ws://localhost:8000/md/subscribe"):
    """Create DH writers, derive tables, consume market-data WebSocket."""
    return publish_trading_dashboard_tables(
        md_ws_url,
        logger=_log,
        consumer_thread_name="md-consumer-coinbase",
        connected_log_message="Connected — streaming Coinbase-backed equity ticks",
    )


def stop_feed():
    stop_trading_dashboard_feed()


def _run() -> None:
    import os
    import time

    os.environ.setdefault("MARKETDATA_FEED", "coinbase")

    print("\n── Platform: Coinbase + MarketDataServer + Deephaven ──")
    print("  MARKETDATA_FEED={}".format(os.environ.get("MARKETDATA_FEED", "")))

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


if __name__ == "__main__":
    _run()
