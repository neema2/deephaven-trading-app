"""
Demo: Trading Server
====================
Publishes ticking price + risk tables to Deephaven, fed by the Market Data Server.

Platform tier:
    StreamingServer   — Deephaven JVM
    MarketDataServer  — REST + WS + QuestDB

Published tables (available to all DH clients):
    prices_raw        — append-only equity ticks
    prices_live       — latest price per symbol
    risk_raw          — per-tick risk metrics
    risk_live         — latest risk per symbol
    portfolio_summary — aggregate portfolio view
    top_movers        — sorted by % change
    volume_leaders    — sorted by volume

Dashboard wiring is shared with ``demo_coinbase.py`` via :mod:`demo_dashboard_common`.

Standalone:
    python3 demo_trading.py

Testable:
    from demo_trading import publish_tables, stop_feed
    publish_tables("ws://localhost:8000/md/subscribe")
"""

import asyncio
import logging
import time

from demo_dashboard_common import publish_trading_dashboard_tables, stop_trading_dashboard_feed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_log = logging.getLogger("demo_trading")


def publish_tables(md_ws_url: str = "ws://localhost:8000/md/subscribe"):
    """Create DH writers, derive all tables, start WS consumer."""
    return publish_trading_dashboard_tables(
        md_ws_url,
        logger=_log,
        consumer_thread_name="md-consumer",
        connected_log_message="Connected — streaming equity ticks",
    )


def stop_feed():
    """Stop the WS consumer thread."""
    stop_trading_dashboard_feed()


if __name__ == "__main__":
    print("\n── Platform: starting infrastructure ──")

    from streaming.admin import StreamingServer

    streaming = StreamingServer(port=10000)
    streaming.start()
    print(f"  Streaming server started on port {streaming.port}")
    print(f"  Web IDE: http://localhost:{streaming.port}")

    from marketdata.admin import MarketDataServer

    md = MarketDataServer(port=8000)
    asyncio.run(md.start())
    print(f"  Market data server started on port {md.port}")

    tables = publish_tables(f"ws://localhost:{md.port}/md/subscribe")

    print()
    print("=" * 60)
    print("  Trading Server is RUNNING")
    print()
    print("  Published tables:")
    for name in tables:
        print(f"    • {name}")
    print()
    print("  Tables populate as Market Data Server feeds ticks.")
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
