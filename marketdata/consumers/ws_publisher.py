"""
WebSocket Publisher — Fan-Out Consumer
=======================================
Subscribes to the TickBus and fans out ticks to connected WebSocket clients,
filtered by each client's symbol subscription.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from marketdata.bus import TickBus
from marketdata.models import Subscription, get_symbol_key

logger = logging.getLogger(__name__)


@dataclass
class _ClientState:
    """Tracks a connected WebSocket client and its subscription."""
    websocket: WebSocket
    types: set[str] | None = None    # None = all types
    symbols: set[str] | None = None  # None = all symbols


class WebSocketPublisher:
    """Fans out TickBus ticks to connected WebSocket clients.

    Each client connects via WebSocket and sends a Subscription message
    to specify which symbols to receive. The publisher filters ticks
    per client and sends matching ones as JSON.
    """

    def __init__(self, bus: TickBus) -> None:
        self._bus = bus
        self._clients: dict[int, _ClientState] = {}  # id(ws) → state
        self._lock = asyncio.Lock()
        self._sub_id: str | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start consuming from the TickBus and dispatching to clients."""
        self._sub_id, tick_iter = await self._bus.subscribe()
        self._task = asyncio.create_task(self._dispatch_loop(tick_iter))
        logger.info("WebSocketPublisher started")

    async def stop(self) -> None:
        """Stop the dispatch loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._sub_id:
            await self._bus.unsubscribe(self._sub_id)
        logger.info("WebSocketPublisher stopped")

    async def register(self, ws: WebSocket, subscription: Subscription) -> None:
        """Register a new WebSocket client with its subscription."""
        types = set(subscription.types) if subscription.types else None
        symbols = set(subscription.symbols) if subscription.symbols else None
        async with self._lock:
            self._clients[id(ws)] = _ClientState(
                websocket=ws, types=types, symbols=symbols,
            )
        logger.info(
            "WS client registered: %s (types=%s, symbols=%s)",
            id(ws), types or "ALL", symbols or "ALL",
        )

    async def update_subscription(
        self, ws: WebSocket, subscription: Subscription
    ) -> None:
        """Update an existing client's type and symbol filter."""
        types = set(subscription.types) if subscription.types else None
        symbols = set(subscription.symbols) if subscription.symbols else None
        async with self._lock:
            client = self._clients.get(id(ws))
            if client:
                client.types = types
                client.symbols = symbols

    async def unregister(self, ws: WebSocket) -> None:
        """Remove a disconnected client."""
        async with self._lock:
            self._clients.pop(id(ws), None)
        logger.info("WS client unregistered: %s", id(ws))

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def _dispatch_loop(self, msg_iter: AsyncIterator[Any]) -> None:
        """Main loop: read messages from bus, fan out to matching clients."""
        try:
            async for msg in msg_iter:
                sym_key = get_symbol_key(msg)
                async with self._lock:
                    clients = list(self._clients.values())

                for client in clients:
                    if client.types is not None and msg.type not in client.types:
                        continue
                    if client.symbols is not None and sym_key not in client.symbols:
                        continue
                    try:
                        await client.websocket.send_text(
                            msg.model_dump_json()
                        )
                    except (WebSocketDisconnect, RuntimeError, Exception) as e:
                        logger.debug("WS send failed for %s: %s", id(client.websocket), e)
                        # Client will be cleaned up when their handler exits
        except asyncio.CancelledError:
            pass
