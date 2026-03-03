"""
streaming.client — Lightweight client for connecting to a streaming server.

Wraps the pydeephaven session — users never import pydeephaven directly.

Usage::

    from streaming import StreamingClient

    # Via alias (registered by StreamingServer)
    with StreamingClient("demo") as c:
        tables = c.list_tables()

    # Or explicit host/port (backward compat)
    with StreamingClient(host="localhost", port=10000) as c:
        df = c.open_table("prices_live").to_arrow().to_pandas()
        c.run_script('filtered = prices_live.where(["Symbol = `AAPL`"])')
"""

from __future__ import annotations

from typing import Any


class StreamingClient:
    """Lightweight client for querying a remote streaming server.

    Connects via alias or explicit host/port.
    Uses pydeephaven (no Java needed on the client machine).
    """

    def __init__(
        self,
        alias_or_host: str | None = None,
        port: int | None = None,
        *,
        host: str | None = None,
    ) -> None:
        from pydeephaven import Session

        resolved = self._resolve(alias_or_host, host, port)
        self.host = resolved.get("host", "localhost")
        self.port = resolved.get("port", 10000)
        self.session = Session(host=self.host, port=self.port)
        print(f"Connected to streaming server at {self.host}:{self.port}")

    @staticmethod
    def _resolve(
        alias_or_host: str | None,
        host: str | None,
        port: int | None,
    ) -> dict:
        """Resolve alias or explicit host/port."""
        # If first arg given with no port, try alias resolution
        if alias_or_host is not None and port is None and host is None:
            from streaming._registry import resolve_alias
            resolved = resolve_alias(alias_or_host)
            if resolved is not None:
                return {"host": "localhost", "port": resolved["port"]}
            # Not an alias — treat as host
            return {"host": alias_or_host, "port": 10000}
        # Explicit params
        return {
            "host": alias_or_host or host or "localhost",
            "port": port or 10000,
        }

    def list_tables(self) -> Any:
        """Return names of all tables in the server's global scope."""
        return self.session.tables

    def open_table(self, name: str) -> Any:
        """Open a server-side table by name."""
        return self.session.open_table(name)

    def run_script(self, script: str) -> None:
        """Execute a Python script on the server. Use this to create
        custom derived tables that live server-side."""
        self.session.run_script(script)

    def bind_table(self, name: str, table: Any) -> None:
        """Publish a client-created table to the server's global scope
        so it is visible in the web IDE and to other sessions."""
        self.session.bind_table(name=name, table=table)

    def close(self) -> None:
        """Close the session."""
        self.session.close()
        print("Session closed.")

    def __enter__(self) -> "StreamingClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
