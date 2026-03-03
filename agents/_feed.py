"""
Feed Agent — Real-Time Market Data
====================================
Configure real-time feeds, publish custom ticks, inspect market data health.

Tools:
    - list_md_symbols         — symbol universe by type
    - get_md_snapshot         — current prices from REST
    - get_feed_health         — tick counts, latency, uptime
    - publish_custom_tick     — publish to bus via POST /md/publish
    - describe_feed_setup     — explain how to create a new feed

Usage::

    from agents._feed import create_feed_agent

    agent = create_feed_agent(ctx)
    result = agent.run("What symbols are currently streaming?")
"""

from __future__ import annotations

import json
import logging

from ai import Agent, tool
from agents._context import _PlatformContext

logger = logging.getLogger(__name__)

FEED_SYSTEM_PROMPT = """\
You are the Market Data Feed Agent — a platform specialist that manages \
real-time market data feeds.

You can:
1. List available symbols by asset type (equity, fx, curve).
2. Get current market data snapshots (prices, volumes, spreads).
3. Check feed health (latency, tick counts, uptime).
4. Publish custom ticks to the market data bus.
5. Describe how to set up new feed implementations.

The market data server runs on a configurable port and provides:
- REST endpoints: /md/health, /md/symbols, /md/snapshot
- WebSocket: /md/subscribe for real-time streaming
- POST /md/publish for injecting custom data

When describing feeds:
- Explain the MarketDataFeed ABC and how SimulatorFeed implements it.
- Cover the TickBus pub/sub architecture.
- Mention supported tick types: Tick (equity), FXTick (fx), CurveTick (curve).
"""


def create_feed_tools(ctx: _PlatformContext) -> list:
    """Create Feed agent tools bound to a _PlatformContext."""

    def _md_url(path: str) -> str:
        return f"{ctx.md_base_url}{path}"

    @tool
    def list_md_symbols() -> str:
        """List all symbols currently available in the market data feed, grouped by type.

        Returns equity symbols, FX pairs, and curve labels.
        """
        try:
            import httpx
            resp = httpx.get(_md_url("/md/symbols"), timeout=5.0)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def get_md_snapshot(msg_type: str = "", symbol: str = "") -> str:
        """Get current market data snapshot.

        Args:
            msg_type: Optional type filter — "equity", "fx", or "curve". Empty = all types.
            symbol: Optional specific symbol (e.g. "AAPL", "USD/JPY"). Empty = all symbols.
        """
        try:
            import httpx
            if msg_type and symbol:
                url = _md_url(f"/md/snapshot/{msg_type}/{symbol}")
            elif msg_type:
                url = _md_url(f"/md/snapshot/{msg_type}")
            else:
                url = _md_url("/md/snapshot")
            resp = httpx.get(url, timeout=5.0)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def get_feed_health() -> str:
        """Check the health of the market data feed server.

        Returns server status, uptime info, and connectivity check.
        """
        try:
            import httpx
            resp = httpx.get(_md_url("/md/health"), timeout=5.0)
            health = resp.json() if resp.status_code == 200 else {"status": "unhealthy"}
            health["http_status"] = resp.status_code
            health["base_url"] = ctx.md_base_url
            return json.dumps(health)
        except Exception as e:
            return json.dumps({"error": str(e), "base_url": ctx.md_base_url})

    @tool
    def publish_custom_tick(tick_json: str) -> str:
        """Publish a custom tick to the market data bus.

        The tick is broadcast to all WebSocket subscribers. Use this to inject
        custom price data or test feeds.

        Args:
            tick_json: JSON tick object. Must include a "type" field.
                       Equity: {"type": "equity", "symbol": "TEST", "price": 100.0, "bid": 99.9, "ask": 100.1, "volume": 1000}
                       FX: {"type": "fx", "pair": "EUR/GBP", "bid": 0.85, "ask": 0.8505, "mid": 0.8503}
                       Curve: {"type": "curve", "label": "USD_5Y", "rate": 0.045, "tenor": "5Y"}
        """
        try:
            import httpx
            tick = json.loads(tick_json)
            resp = httpx.post(_md_url("/md/publish"), json=tick, timeout=5.0)
            resp.raise_for_status()
            return json.dumps({
                "status": "published",
                "tick": tick,
                "http_status": resp.status_code,
            })
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def describe_feed_setup(asset_class: str = "equity") -> str:
        """Describe how to set up a new market data feed for a given asset class.

        Explains the MarketDataFeed ABC, TickBus architecture, and integration points.

        Args:
            asset_class: Asset class — "equity", "fx", "curve", or "custom".
        """
        descriptions = {
            "equity": {
                "tick_type": "Tick",
                "required_fields": ["symbol", "price", "bid", "ask", "volume", "timestamp"],
                "feed_class": "MarketDataFeed (ABC in marketdata/feed.py)",
                "example_impl": "SimulatorFeed (marketdata/feeds/simulator.py)",
                "setup_steps": [
                    "1. Create a new class inheriting from MarketDataFeed",
                    "2. Implement start(bus) — connect to data source, publish Tick objects to bus",
                    "3. Implement stop() — clean up connections",
                    "4. Implement name property — return feed name",
                    "5. Register in MarketData server lifespan or use POST /md/publish for external feeds",
                ],
            },
            "fx": {
                "tick_type": "FXTick",
                "required_fields": ["pair", "bid", "ask", "mid", "spread_pips", "currency", "timestamp"],
                "feed_class": "MarketDataFeed (ABC)",
                "setup_steps": [
                    "1. Inherit MarketDataFeed, publish FXTick objects",
                    "2. FX pairs use 'pair' field (e.g. 'EUR/USD'), not 'symbol'",
                    "3. Include spread_pips and currency metadata",
                ],
            },
            "curve": {
                "tick_type": "CurveTick",
                "required_fields": ["label", "rate", "tenor", "currency", "timestamp"],
                "feed_class": "MarketDataFeed (ABC)",
                "setup_steps": [
                    "1. Curves are typically derived reactively from FX/equity data",
                    "2. Publish CurveTick via POST /md/publish from a reactive @effect",
                    "3. See demo_ir_swap.py for the reactive curve derivation pattern",
                ],
            },
            "custom": {
                "tick_type": "Any JSON with 'type' field",
                "setup_steps": [
                    "1. Use POST /md/publish to inject any JSON with a 'type' field",
                    "2. Subscribers filter by 'types' and 'symbols' in their subscription",
                    "3. For persistent custom feeds, implement MarketDataFeed ABC",
                    "4. The TickBus keys messages by (type, symbol_key) tuple",
                ],
            },
        }
        desc = descriptions.get(asset_class, descriptions["custom"])
        return json.dumps(desc, indent=2)

    return [list_md_symbols, get_md_snapshot, get_feed_health,
            publish_custom_tick, describe_feed_setup]


def create_feed_agent(ctx: _PlatformContext, **kwargs) -> Agent:
    """Create a Feed Agent bound to a _PlatformContext."""
    tools = create_feed_tools(ctx)
    return Agent(
        tools=tools,
        system_prompt=FEED_SYSTEM_PROMPT,
        name="feed",
        **kwargs,
    )
