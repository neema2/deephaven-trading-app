"""
ir.consumer — Reusable WebSocket consumer for market data ticks.

Connects to the Market Data Server, consumes FX ticks, triggers the
reactive cascade, and calls an ``on_tick`` callback after each update.

Usage::

    from pricing.scenarios.consumer import MarketDataConsumer

    consumer = MarketDataConsumer(
        fx_spots=graph["fx_spots"],
        on_tick=lambda n: print(f"Tick {n}"),
    )
    consumer.start_background()
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from collections.abc import Callable

from streaming import flush

logger = logging.getLogger(__name__)

DEFAULT_MD_URL = "ws://localhost:8000/md/subscribe"
DEFAULT_RECONNECT_DELAY = 2


class MarketDataConsumer:
    """Reusable WS consumer for FX ticks with auto-reconnect.

    Parameters
    ----------
    fx_spots : dict[str, FXSpot]
        Map of pair → FXSpot instances to update.
    on_tick : callable(tick_count) or None
        Called after each FX update + flush.  Use for risk recompute,
        summary printing, etc.
    curve_publish_queue : deque or None
        If set, CurveTick dicts are drained and sent back to the hub.
    initial_curve_ticks : list or None
        CurveTicks to publish on first connect.
    md_url : str
        WebSocket URL for the market data server.
    reconnect_delay : int
        Seconds to wait before reconnecting after a failure.
    """

    def __init__(
        self,
        fx_spots: dict,
        on_tick: Callable[[int], None] | None = None,
        curve_publish_queue: deque | None = None,
        initial_curve_ticks: list | None = None,
        md_url: str = DEFAULT_MD_URL,
        reconnect_delay: int = DEFAULT_RECONNECT_DELAY,
    ) -> None:
        self.fx_spots = fx_spots
        self.on_tick = on_tick
        self.curve_publish_queue = curve_publish_queue or deque()
        self.initial_curve_ticks = initial_curve_ticks or []
        self.md_url = md_url
        self.reconnect_delay = reconnect_delay
        self._thread: threading.Thread | None = None

    async def _consume(self) -> None:
        """Connect to the Market Data Server and consume FX ticks."""
        import websockets

        tick_count = 0

        while True:
            try:
                logger.info("Connecting to Market Data Server at %s ...", self.md_url)
                async with websockets.connect(self.md_url) as ws:
                    await ws.send(json.dumps({"types": ["fx"]}))
                    logger.info("Connected — consuming FX ticks")

                    # Publish initial CurveTicks
                    for ct in self.initial_curve_ticks:
                        await ws.send(json.dumps(ct))

                    async for msg_str in ws:
                        tick = json.loads(msg_str)
                        if tick.get("type") != "fx":
                            continue

                        pair = tick["pair"]
                        fx = self.fx_spots.get(pair)
                        if fx is None:
                            continue

                        # Reactive cascade: FX → curve rates → swap NPVs
                        fx.batch_update(bid=tick["bid"], ask=tick["ask"])
                        flush()

                        tick_count += 1

                        # Callback for demo-specific logic
                        if self.on_tick is not None:
                            self.on_tick(tick_count)

                        # Drain and publish CurveTicks
                        while self.curve_publish_queue:
                            await ws.send(
                                json.dumps(self.curve_publish_queue.popleft())
                            )

            except Exception as e:
                logger.warning(
                    "Market Data connection lost (%s). Retrying in %ds...",
                    e, self.reconnect_delay,
                )
                await asyncio.sleep(self.reconnect_delay)

    def _run_loop(self) -> None:
        """Entry point for the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._consume())

    def start_background(self, name: str = "md-consumer") -> threading.Thread:
        """Start the consumer in a daemon thread.

        Returns the thread (already started).
        """
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=name,
        )
        self._thread.start()
        return self._thread
