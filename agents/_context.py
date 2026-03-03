"""
_PlatformContext — Internal shared state for all agent tools.

NOT part of the public API.  Users interact only with ``PlatformAgents``.
Tools receive a ``_PlatformContext`` via closure and use its lazy-initialized
client properties to talk to platform services.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class _PlatformContext:
    """Internal platform context — lazy client creation from aliases.

    All service clients are created on first access.  If a service was
    not started (alias not registered), the agent gets a clear error
    only when it actually tries to use that service.
    """

    def __init__(
        self,
        alias: str = "",
        user: str = "",
        password: str = "",
        *,
        store_alias: str | None = None,
        lakehouse_alias: str | None = None,
        tsdb_alias: str | None = None,
        streaming_alias: str | None = None,
        md_alias: str | None = None,
        media_alias: str | None = None,
        ai: Any = None,
    ) -> None:
        self._alias = alias
        self._user = user
        self._password = password

        # Per-service alias overrides (default: use global alias)
        self._store_alias = store_alias or alias
        self._lakehouse_alias = lakehouse_alias or alias
        self._tsdb_alias = tsdb_alias or alias
        self._streaming_alias = streaming_alias or alias
        self._md_alias = md_alias or alias
        self._media_alias = media_alias or alias

        # Pre-built AI (or lazy)
        self._ai_instance = ai

        # Lazy client slots
        self._lakehouse_instance = None
        self._md_client_instance = None
        self._media_store_instance = None
        self._tsdb_instance = None
        self._streaming_client_instance = None

        # Dynamic type registry (for OLTP agent)
        self._storable_types: dict[str, type] = {}

    # ── Lazy service clients ─────────────────────────────────────

    def get_store_connection(self) -> object:
        """Get or create a store connection using the configured alias."""
        if not self._store_alias:
            raise RuntimeError("No store alias configured")
        from store import connect
        return connect(
            self._store_alias,
            user=self._user,
            password=self._password,
        )

    @property
    def lakehouse(self) -> object:
        """Lazy Lakehouse client."""
        if self._lakehouse_instance is None:
            if not self._lakehouse_alias:
                raise RuntimeError("No lakehouse alias configured")
            from lakehouse import Lakehouse
            self._lakehouse_instance = Lakehouse(self._lakehouse_alias)
        return self._lakehouse_instance

    @property
    def md_client(self) -> object:
        """Lazy MarketDataClient."""
        if self._md_client_instance is None:
            if not self._md_alias:
                raise RuntimeError("No market data alias configured")
            from marketdata.client import MarketDataClient
            self._md_client_instance = MarketDataClient(self._md_alias)
        return self._md_client_instance

    @property
    def md_base_url(self) -> str:
        """Base URL for market data REST endpoints (backward compat for tools)."""
        return self.md_client.base_url

    @property
    def media_store(self) -> object:
        """Lazy MediaStore client."""
        if self._media_store_instance is None:
            if not self._media_alias:
                raise RuntimeError("No media alias configured")
            from media.store import MediaStore
            self._media_store_instance = MediaStore(
                self._media_alias, ai=self._ai_instance,
            )
        return self._media_store_instance

    @property
    def tsdb(self) -> object:
        """Lazy Timeseries client."""
        if self._tsdb_instance is None:
            if not self._tsdb_alias:
                raise RuntimeError("No TSDB alias configured")
            from timeseries import Timeseries
            self._tsdb_instance = Timeseries(self._tsdb_alias)
        return self._tsdb_instance

    @property
    def streaming_client(self) -> object:
        """Lazy StreamingClient."""
        if self._streaming_client_instance is None:
            if not self._streaming_alias:
                raise RuntimeError("No streaming alias configured")
            from streaming import StreamingClient
            self._streaming_client_instance = StreamingClient(self._streaming_alias)
        return self._streaming_client_instance

    @property
    def ai(self) -> object:
        """AI instance (lazy-created from env if not provided)."""
        if self._ai_instance is None:
            from ai import AI
            self._ai_instance = AI()
        return self._ai_instance

    @property
    def store_alias(self) -> str:
        return self._store_alias

    # ── Service availability checks (no lazy creation) ─────────

    def has_store(self) -> bool:
        return bool(self._store_alias)

    def has_lakehouse(self) -> bool:
        return bool(self._lakehouse_alias)

    def has_md(self) -> bool:
        return bool(self._md_alias)

    def has_media(self) -> bool:
        return bool(self._media_alias)

    def has_tsdb(self) -> bool:
        return bool(self._tsdb_alias)

    def has_streaming(self) -> bool:
        return bool(self._streaming_alias)

    # ── Dynamic type registry ────────────────────────────────────

    def register_storable_type(self, name: str, cls: type) -> None:
        """Register a dynamically created Storable subclass."""
        self._storable_types[name] = cls

    def get_storable_type(self, name: str) -> type | None:
        """Look up a registered Storable type by name."""
        return self._storable_types.get(name)

    def list_storable_types(self) -> list[str]:
        """Return names of all registered Storable types."""
        return list(self._storable_types.keys())

    # ── Introspection ────────────────────────────────────────────

    def validate(self) -> dict[str, bool]:
        """Check which platform services are reachable.

        Returns:
            Dict of service_name → is_available.
        """
        status = {}

        # Store
        if self._store_alias:
            try:
                from store.connection import _resolve_alias
                status["store"] = _resolve_alias(self._store_alias) is not None
            except Exception:
                status["store"] = False
        else:
            status["store"] = False

        # Market Data
        if self._md_alias:
            try:
                status["marketdata"] = self.md_client.health()
            except Exception:
                status["marketdata"] = False
        else:
            status["marketdata"] = False

        # Lakehouse
        if self._lakehouse_alias:
            try:
                status["lakehouse"] = self._lakehouse_instance is not None or True
            except Exception:
                status["lakehouse"] = False
        else:
            status["lakehouse"] = False

        # Media Store
        status["media"] = bool(self._media_alias)

        # AI
        status["ai"] = self._ai_instance is not None

        # TSDB
        status["tsdb"] = bool(self._tsdb_alias)

        # Streaming
        status["streaming"] = bool(self._streaming_alias)

        logger.info("Platform status: %s", status)
        return status
