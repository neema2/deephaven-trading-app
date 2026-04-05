"""
Shared Deephaven + market-data WebSocket wiring for trading-style demos.

Used by ``demo_trading.py`` and ``demo_coinbase.py`` so the dashboard pipeline
(``TickingTable`` setup, derived tables, WS consumer) stays in one place.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
from typing import Any

_feed_thread: threading.Thread | None = None
_feed_stop = threading.Event()


def publish_trading_dashboard_tables(
    md_ws_url: str,
    *,
    logger: logging.Logger,
    consumer_thread_name: str = "md-consumer",
    connected_log_message: str = "Connected — streaming equity ticks",
) -> dict[str, Any]:
    """Create DH writers, derive tables, start WS consumer to ``md_ws_url``.

    Call after ``StreamingServer.start()`` (JVM must be up before importing ``streaming``).
    """
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

    _positions: dict[str, int] = {}

    def _on_tick(tick: dict) -> None:
        sym = tick["symbol"]
        prices.write_row(
            sym,
            tick["price"],
            tick["bid"],
            tick["ask"],
            tick["volume"],
            tick["change"],
            tick["change_pct"],
        )
        if sym not in _positions:
            _positions[sym] = random.randint(100, 1000)
        pos = _positions[sym]
        risk.write_row(
            sym,
            tick["price"],
            pos,
            tick["price"] * pos,
            tick["change"] * pos,
            0.5 + random.random() * 0.3,
            0.02 + random.random() * 0.04,
            -0.1 - random.random() * 0.15,
            0.2 + random.random() * 0.2,
        )

    async def _consume(url: str) -> None:
        import websockets

        while not _feed_stop.is_set():
            try:
                logger.info("Connecting to Market Data Server at %s ...", url)
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps({"types": ["equity"]}))
                    logger.info(connected_log_message)
                    async for msg in ws:
                        if _feed_stop.is_set():
                            return
                        tick = json.loads(msg)
                        if tick.get("type") == "equity":
                            _on_tick(tick)
            except Exception as e:
                if _feed_stop.is_set():
                    return
                logger.warning("Market Data connection lost (%s). Retrying in 2s...", e)
                await asyncio.sleep(2)

    _feed_stop.clear()
    _feed_thread = threading.Thread(
        target=lambda: asyncio.run(_consume(md_ws_url)),
        daemon=True,
        name=consumer_thread_name,
    )
    _feed_thread.start()

    logger.info("Trading dashboard tables published — 7 tables available to all clients")

    return {
        "prices_raw": prices,
        "prices_live": prices_live,
        "risk_raw": risk,
        "risk_live": risk_live,
        "portfolio_summary": portfolio_summary,
        "top_movers": top_movers,
        "volume_leaders": volume_leaders,
    }


def stop_trading_dashboard_feed() -> None:
    """Stop the WebSocket consumer thread started by :func:`publish_trading_dashboard_tables`."""
    global _feed_thread
    _feed_stop.set()
    if _feed_thread and _feed_thread.is_alive():
        _feed_thread.join(timeout=5)
    _feed_thread = None
