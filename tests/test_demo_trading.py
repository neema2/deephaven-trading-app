"""
Mirror test for demo_trading.py
=================================
Verifies the full demo flow:

  1. Create TickingTable writers for prices + risk
  2. Derived tables: last_by, agg_by, sort_descending
  3. Publish 7 tables via publish_tables()
  4. WS consumer receives ticks from market_data_server
  5. Tables populate with live data — verified via snapshot()
"""

import time

import pytest

from demo_trading import publish_tables, stop_feed
from streaming import LiveTable, TickingTable, flush, snapshot


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tables(streaming_server, market_data_server):
    """Start the trading demo — same as demo's __main__ block.

    Publishes 7 tables and starts the WS consumer thread.
    """
    md_url = f"ws://localhost:{market_data_server.port}/md/subscribe"
    t = publish_tables(md_url)
    # Let ticks flow from market data server → WS consumer → writers
    from streaming.admin import _needs_docker
    time.sleep(8 if _needs_docker() else 5)
    flush()
    yield t
    stop_feed()


# ── Tests ────────────────────────────────────────────────────────────────

class TestDemoTrading:
    """Mirrors demo_trading.py — TickingTable + derived table creation."""

    def test_seven_tables_published(self, tables) -> None:
        """publish_tables returns 7 named tables."""
        assert len(tables) == 7
        for name in ("prices_raw", "prices_live", "risk_raw", "risk_live",
                      "portfolio_summary", "top_movers", "volume_leaders"):
            assert name in tables, f"Missing table: {name}"

    def test_prices_raw_is_ticking_table(self, tables) -> None:
        """prices_raw is a TickingTable (the raw writer)."""
        assert isinstance(tables["prices_raw"], TickingTable)

    def test_risk_raw_is_ticking_table(self, tables) -> None:
        """risk_raw is a TickingTable (the raw writer)."""
        assert isinstance(tables["risk_raw"], TickingTable)

    def test_prices_live_is_live_table(self, tables) -> None:
        """prices_live = prices_raw.last_by('Symbol') — a LiveTable."""
        assert isinstance(tables["prices_live"], LiveTable)

    def test_risk_live_is_live_table(self, tables) -> None:
        """risk_live = risk_raw.last_by('Symbol') — a LiveTable."""
        assert isinstance(tables["risk_live"], LiveTable)

    def test_portfolio_summary_is_live_table(self, tables) -> None:
        """portfolio_summary = risk_live.agg_by — a LiveTable."""
        assert isinstance(tables["portfolio_summary"], LiveTable)

    def test_top_movers_is_live_table(self, tables) -> None:
        """top_movers = prices_live.sort_descending — a LiveTable."""
        assert isinstance(tables["top_movers"], LiveTable)

    def test_volume_leaders_is_live_table(self, tables) -> None:
        """volume_leaders = prices_live.sort_descending — a LiveTable."""
        assert isinstance(tables["volume_leaders"], LiveTable)

    def test_prices_raw_has_ticked_rows(self, tables) -> None:
        """After WS consumer runs, prices_raw has rows."""
        snap = snapshot(tables["prices_raw"])
        assert len(snap) > 0

    def test_risk_raw_has_ticked_rows(self, tables) -> None:
        """Risk writer ticks alongside prices."""
        snap = snapshot(tables["risk_raw"])
        assert len(snap) > 0

    def test_prices_raw_columns(self, tables) -> None:
        """prices_raw has the expected columns from the demo."""
        snap = snapshot(tables["prices_raw"])
        for col in ("Symbol", "Price", "Bid", "Ask", "Volume", "Change", "ChangePct"):
            assert col in snap.columns, f"Missing column: {col}"

    def test_risk_raw_columns(self, tables) -> None:
        """risk_raw has the expected columns from the demo."""
        snap = snapshot(tables["risk_raw"])
        for col in ("Symbol", "Price", "Position", "MarketValue",
                     "UnrealizedPnL", "Delta", "Gamma", "Theta", "Vega"):
            assert col in snap.columns, f"Missing column: {col}"

    def test_stop_feed(self, tables) -> None:
        """stop_feed() terminates the WS consumer thread."""
        from demo_trading import _feed_thread
        stop_feed()
        assert _feed_thread is None or not _feed_thread.is_alive()
