"""
RLS Unit Tests — SQL Rewriting, Hybrid Routing, Flight Round-Trip
===================================================================
In-process tests requiring no external infrastructure.
Tests the three core components:
  1. RLSRewriter — sqlglot-based SQL rewriting with ACL joins
  2. Hybrid routing — _is_flight_query correctly classifies queries
  3. Flight round-trip — in-process Flight server with token auth and row isolation
"""

from __future__ import annotations

import json
import threading
import time

import duckdb
import pyarrow as pa
import pyarrow.flight as flight
import pytest

from lakehouse.rls_server import RLSFlightServer, RLSPolicy, RLSRewriter


# ── Test Data ────────────────────────────────────────────────────────────────

POLICIES = [
    RLSPolicy(
        table_name="sales_data",
        acl_table="sales_acl",
        join_column="row_id",
        user_column="user_token",
    ),
]

USERS = {
    "alice-token": "alice",
    "bob-token": "bob",
}


# ── TestRLSRewriter ──────────────────────────────────────────────────────────


class TestRLSRewriter:
    """Test sqlglot-based SQL rewriting for RLS."""

    def setup_method(self) -> None:
        self.rewriter = RLSRewriter(POLICIES)

    def test_simple_select_gets_acl_join(self) -> None:
        """SELECT * FROM sales_data → gets ACL join injected."""
        sql = "SELECT * FROM sales_data"
        result = self.rewriter.rewrite(sql, "alice-token")
        # Should contain JOIN to sales_acl
        assert "sales_acl" in result
        assert "alice-token" in result
        assert "JOIN" in result.upper()

    def test_qualified_table_gets_acl_join(self) -> None:
        """Fully qualified table name should also be rewritten."""
        sql = "SELECT * FROM lakehouse.default.sales_data"
        result = self.rewriter.rewrite(sql, "bob-token")
        assert "sales_acl" in result
        assert "bob-token" in result

    def test_unprotected_table_passes_through(self) -> None:
        """Tables without RLS policies should not be rewritten."""
        sql = "SELECT * FROM trades"
        result = self.rewriter.rewrite(sql, "alice-token")
        # Should not contain any ACL join
        assert "sales_acl" not in result
        assert "alice-token" not in result

    def test_where_clause_preserved(self) -> None:
        """Existing WHERE clauses should be preserved after rewriting."""
        sql = "SELECT * FROM sales_data WHERE region = 'US'"
        result = self.rewriter.rewrite(sql, "alice-token")
        assert "sales_acl" in result
        assert "alice-token" in result
        # Original WHERE condition should still be present
        assert "region" in result

    def test_aliased_table(self) -> None:
        """Table aliases should be handled correctly."""
        sql = "SELECT t.* FROM sales_data t WHERE t.amount > 100"
        result = self.rewriter.rewrite(sql, "alice-token")
        assert "sales_acl" in result
        assert "alice-token" in result

    def test_multi_table_only_protected_rewritten(self) -> None:
        """Only protected tables get ACL joins in multi-table queries."""
        sql = "SELECT * FROM sales_data s, trades t WHERE s.id = t.id"
        result = self.rewriter.rewrite(sql, "alice-token")
        assert "sales_acl" in result
        assert "alice-token" in result
        # trades should not get an ACL join
        assert result.count("JOIN") == 1 or result.upper().count("JOIN") == 1

    def test_extract_tables(self) -> None:
        """extract_tables should return all table names."""
        sql = "SELECT * FROM sales_data s JOIN trades t ON s.id = t.id"
        tables = RLSRewriter.extract_tables(sql)
        assert "sales_data" in tables
        assert "trades" in tables

    def test_needs_rewrite(self) -> None:
        """needs_rewrite should identify protected table references."""
        assert self.rewriter.needs_rewrite("SELECT * FROM sales_data")
        assert not self.rewriter.needs_rewrite("SELECT * FROM trades")

    def test_protected_tables_property(self) -> None:
        """protected_tables should return the set of table names."""
        assert self.rewriter.protected_tables == {"sales_data"}

    def test_invalid_sql_passes_through(self) -> None:
        """Invalid SQL should pass through without error."""
        sql = "THIS IS NOT SQL AT ALL {{{}}}"
        result = self.rewriter.rewrite(sql, "alice-token")
        assert result == sql  # passed through unchanged


# ── TestHybridRouting ───────────────────────────────────────────────────────


class TestHybridRouting:
    """Test _is_flight_query routing logic."""

    def test_no_token_always_direct(self) -> None:
        """Without token, all queries go direct."""
        from lakehouse.query import Lakehouse
        lh = Lakehouse.__new__(Lakehouse)
        lh._token = None
        lh._protected_tables = {"sales_data"}
        assert not lh._is_flight_query("SELECT * FROM sales_data")

    def test_no_protected_tables_always_direct(self) -> None:
        """With token but no protected tables, all queries go direct."""
        from lakehouse.query import Lakehouse
        lh = Lakehouse.__new__(Lakehouse)
        lh._token = "alice-token"
        lh._protected_tables = set()
        assert not lh._is_flight_query("SELECT * FROM sales_data")

    def test_protected_table_routes_to_flight(self) -> None:
        """Protected table query should route to Flight."""
        from lakehouse.query import Lakehouse
        lh = Lakehouse.__new__(Lakehouse)
        lh._token = "alice-token"
        lh._protected_tables = {"sales_data"}
        assert lh._is_flight_query("SELECT * FROM sales_data")

    def test_open_table_stays_direct(self) -> None:
        """Open table query should stay direct."""
        from lakehouse.query import Lakehouse
        lh = Lakehouse.__new__(Lakehouse)
        lh._token = "alice-token"
        lh._protected_tables = {"sales_data"}
        assert not lh._is_flight_query("SELECT * FROM trades")

    def test_mixed_query_routes_to_flight(self) -> None:
        """Query mixing open + protected tables should route to Flight."""
        from lakehouse.query import Lakehouse
        lh = Lakehouse.__new__(Lakehouse)
        lh._token = "alice-token"
        lh._protected_tables = {"sales_data"}
        assert lh._is_flight_query(
            "SELECT * FROM sales_data s JOIN trades t ON s.id = t.id"
        )

    def test_qualified_name_routes_to_flight(self) -> None:
        """Fully qualified table name should still be detected."""
        from lakehouse.query import Lakehouse
        lh = Lakehouse.__new__(Lakehouse)
        lh._token = "alice-token"
        lh._protected_tables = {"sales_data"}
        assert lh._is_flight_query(
            "SELECT * FROM lakehouse.default.sales_data"
        )


# ── TestRLSFlightRoundTrip ──────────────────────────────────────────────────


def _find_free_port() -> int:
    """Find a free TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def flight_server():
    """Start an in-process RLS Flight server with sample data."""
    # Create server-side DuckDB with sample data
    conn = duckdb.connect()

    conn.execute("""
        CREATE TABLE sales_data (
            row_id INTEGER,
            product VARCHAR,
            region VARCHAR,
            amount DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO sales_data VALUES
            (1, 'Widget A', 'US', 100.0),
            (2, 'Widget B', 'EU', 200.0),
            (3, 'Widget C', 'US', 150.0),
            (4, 'Widget D', 'EU', 300.0),
            (5, 'Widget E', 'APAC', 250.0)
    """)

    conn.execute("""
        CREATE TABLE sales_acl (
            row_id INTEGER,
            user_token VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO sales_acl VALUES
            (1, 'alice-token'),
            (2, 'alice-token'),
            (3, 'bob-token'),
            (4, 'bob-token'),
            (5, 'bob-token')
    """)

    conn.execute("""
        CREATE TABLE trades (
            trade_id INTEGER,
            symbol VARCHAR,
            quantity INTEGER
        )
    """)
    conn.execute("""
        INSERT INTO trades VALUES
            (1, 'AAPL', 100),
            (2, 'GOOGL', 50)
    """)

    port = _find_free_port()
    server = RLSFlightServer(
        duckdb_conn=conn,
        policies=POLICIES,
        users=USERS,
        host="localhost",
        port=port,
    )

    # Start server in background thread
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    time.sleep(0.5)  # Wait for server to start

    yield server, port

    server.shutdown()


class _TestAuthHandler(flight.ClientAuthHandler):
    """Client auth handler for tests."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token.encode("utf-8")
        self._session_token: bytes = b""

    def authenticate(self, outgoing, incoming) -> None:
        outgoing.write(self._token)
        self._session_token = incoming.read()

    def get_token(self) -> bytes:
        return self._session_token


class TestRLSFlightRoundTrip:
    """Test end-to-end Flight SQL with RLS."""

    def test_alice_sees_own_rows(self, flight_server) -> None:
        """Alice should only see rows 1, 2 (her ACL entries)."""
        server, port = flight_server
        client = flight.FlightClient(f"grpc://localhost:{port}")
        client.authenticate(_TestAuthHandler("alice-token"))

        ticket = flight.Ticket(b"SELECT * FROM sales_data")
        reader = client.do_get(ticket)
        table = reader.read_all()

        assert table.num_rows == 2
        row_ids = sorted(table.column("row_id").to_pylist())
        assert row_ids == [1, 2]
        client.close()

    def test_bob_sees_own_rows(self, flight_server) -> None:
        """Bob should only see rows 3, 4, 5 (his ACL entries)."""
        server, port = flight_server
        client = flight.FlightClient(f"grpc://localhost:{port}")
        client.authenticate(_TestAuthHandler("bob-token"))

        ticket = flight.Ticket(b"SELECT * FROM sales_data")
        reader = client.do_get(ticket)
        table = reader.read_all()

        assert table.num_rows == 3
        row_ids = sorted(table.column("row_id").to_pylist())
        assert row_ids == [3, 4, 5]
        client.close()

    def test_alice_with_where_clause(self, flight_server) -> None:
        """Alice's WHERE clause should be preserved alongside RLS filter."""
        server, port = flight_server
        client = flight.FlightClient(f"grpc://localhost:{port}")
        client.authenticate(_TestAuthHandler("alice-token"))

        ticket = flight.Ticket(b"SELECT * FROM sales_data WHERE region = 'US'")
        reader = client.do_get(ticket)
        table = reader.read_all()

        # Alice has rows 1 (US) and 2 (EU); only row 1 is US
        assert table.num_rows == 1
        assert table.column("row_id").to_pylist() == [1]
        assert table.column("region").to_pylist() == ["US"]
        client.close()

    def test_open_table_no_rls(self, flight_server) -> None:
        """Queries to unprotected tables should return all rows."""
        server, port = flight_server
        client = flight.FlightClient(f"grpc://localhost:{port}")
        client.authenticate(_TestAuthHandler("alice-token"))

        ticket = flight.Ticket(b"SELECT * FROM trades")
        reader = client.do_get(ticket)
        table = reader.read_all()

        assert table.num_rows == 2  # All trades visible
        client.close()

    def test_unauthenticated_rejected(self, flight_server) -> None:
        """Requests with invalid tokens should be rejected."""
        server, port = flight_server
        client = flight.FlightClient(f"grpc://localhost:{port}")

        with pytest.raises(flight.FlightUnauthenticatedError):
            client.authenticate(_TestAuthHandler("invalid-token"))

        client.close()

    def test_get_protected_tables_action(self, flight_server) -> None:
        """get_protected_tables action should return the policy table set."""
        server, port = flight_server
        client = flight.FlightClient(f"grpc://localhost:{port}")
        client.authenticate(_TestAuthHandler("alice-token"))

        action = flight.Action("get_protected_tables", b"")
        results = list(client.do_action(action))

        assert len(results) == 1
        tables = json.loads(results[0].body.to_pybytes())
        assert tables == ["sales_data"]
        client.close()

    def test_list_flights_returns_tables(self, flight_server) -> None:
        """list_flights should return available tables."""
        server, port = flight_server
        client = flight.FlightClient(f"grpc://localhost:{port}")
        client.authenticate(_TestAuthHandler("alice-token"))

        flights = list(client.list_flights())
        # Should have at least sales_data, sales_acl, trades
        assert len(flights) >= 3
        client.close()
