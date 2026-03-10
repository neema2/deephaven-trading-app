#!/usr/bin/env python3
"""
RLS Demo — Row-Level Security via Arrow Flight SQL Gateway
=============================================================
Demonstrates the full RLS workflow using the real lakehouse stack.

Usage::

    python demo_rls.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-5s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Sample Data ───────────────────────────────────────────────────────────

TRADES = [
    {"trade_id": 1, "symbol": "AAPL", "quantity": 100, "price": 185.50, "side": "BUY"},
    {"trade_id": 2, "symbol": "GOOGL", "quantity": 50, "price": 140.25, "side": "BUY"},
    {"trade_id": 3, "symbol": "MSFT", "quantity": 75, "price": 420.00, "side": "SELL"},
    {"trade_id": 4, "symbol": "AMZN", "quantity": 30, "price": 178.90, "side": "BUY"},
    {"trade_id": 5, "symbol": "TSLA", "quantity": 60, "price": 195.00, "side": "SELL"},
]

SALES_DATA = [
    {"row_id": 1, "product": "Enterprise License", "region": "US", "amount": 50000.0, "sales_rep": "alice"},
    {"row_id": 2, "product": "Cloud Subscription", "region": "EU", "amount": 35000.0, "sales_rep": "alice"},
    {"row_id": 3, "product": "Support Package", "region": "US", "amount": 12000.0, "sales_rep": "bob"},
    {"row_id": 4, "product": "Enterprise License", "region": "APAC", "amount": 45000.0, "sales_rep": "bob"},
    {"row_id": 5, "product": "Cloud Subscription", "region": "US", "amount": 28000.0, "sales_rep": "bob"},
    {"row_id": 6, "product": "Training", "region": "EU", "amount": 8000.0, "sales_rep": "charlie"},
]

SALES_ACL = [
    {"row_id": 1, "user_token": "alice-token"},
    {"row_id": 2, "user_token": "alice-token"},
    {"row_id": 3, "user_token": "bob-token"},
    {"row_id": 4, "user_token": "bob-token"},
    {"row_id": 5, "user_token": "bob-token"},
]


# ── Helpers ───────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def _print_rows(rows: list[dict], max_rows: int = 20) -> None:
    if not rows:
        print("  (no rows)")
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows[:max_rows])) for k in keys}
    header = " | ".join(k.ljust(widths[k]) for k in keys)
    print(f"  {header}")
    print(f"  {'-+-'.join('-' * widths[k] for k in keys)}")
    for row in rows[:max_rows]:
        line = " | ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys)
        print(f"  {line}")
    print(f"  ({len(rows)} rows)")


# ── Main Demo ─────────────────────────────────────────────────────────────

async def run_demo(args: argparse.Namespace) -> None:
    from lakehouse.admin import LakehouseServer, RLSPolicy
    from lakehouse import Lakehouse

    # ── Step 1: Start the Lakehouse Stack with RLS ──
    _banner("Step 1: Starting Lakehouse Stack with RLS")

    server = LakehouseServer(
        data_dir="data/lakehouse-rls",
        rls_policies=[
            RLSPolicy(
                table_name="sales_data",
                acl_table="sales_acl",
                join_column="row_id",
                user_column="user_token",
            ),
        ],
        rls_users={
            "alice-token": "alice",
            "bob-token": "bob",
        },
    )
    await server.start()
    server.register_alias("rls-demo")

    print("  ✅ Lakehouse stack started (PG + Lakekeeper + MinIO + RLS Flight)")
    print(f"     Catalog:      {server.catalog_url}")
    print(f"     S3:           {server.s3_endpoint}")
    print(f"     Flight port:  {server.flight_port}")

    try:
        # ── Step 2: Ingest Data ──
        _banner("Step 2: Ingesting Data into Lakehouse")

        lh = Lakehouse("rls-demo")
        n = lh.ingest("trades", TRADES)
        print(f"  ✅ Ingested {n} rows into 'trades' (open table)")

        n = lh.ingest("sales_data", SALES_DATA)
        print(f"  ✅ Ingested {n} rows into 'sales_data' (RLS-protected)")

        n = lh.ingest("sales_acl", SALES_ACL)
        print(f"  ✅ Ingested {n} rows into 'sales_acl' (ACL table)")
        lh.close()

        # ── Step 3: Query as Alice ──
        _banner("Step 3: Query as Alice (token='alice-token')")

        alice = Lakehouse("rls-demo", token="alice-token")

        print("\n  📊 Alice queries 'trades' (open table → direct DuckDB):")
        _print_rows(alice.query("SELECT * FROM lakehouse.default.trades"))

        print("\n  🔒 Alice queries 'sales_data' (protected → Flight SQL + RLS):")
        alice_sales = alice.query("SELECT * FROM lakehouse.default.sales_data")
        _print_rows(alice_sales)

        print("\n  🔒 Alice queries 'sales_data' with WHERE (RLS + user filter):")
        _print_rows(alice.query(
            "SELECT * FROM lakehouse.default.sales_data WHERE region = 'US'"
        ))
        alice.close()

        # ── Step 4: Query as Bob ──
        _banner("Step 4: Query as Bob (token='bob-token')")

        bob = Lakehouse("rls-demo", token="bob-token")

        print("\n  📊 Bob queries 'trades' (open table → direct DuckDB):")
        _print_rows(bob.query("SELECT * FROM lakehouse.default.trades"))

        print("\n  🔒 Bob queries 'sales_data' (protected → Flight SQL + RLS):")
        bob_sales = bob.query("SELECT * FROM lakehouse.default.sales_data")
        _print_rows(bob_sales)
        bob.close()

        # ── Summary ──
        _banner("Summary")
        print(f"  • trades (open):      5 rows visible to all users")
        print(f"  • sales_data (RLS):   Alice sees {len(alice_sales)} rows, Bob sees {len(bob_sales)} rows")
        print(f"  • Charlie's row (6):  invisible to both (no ACL entry)")
        print(f"  • Open tables:        zero Flight overhead (direct DuckDB)")
        print(f"  • Protected tables:   routed through Flight SQL + ACL joins")
        print()

    finally:
        await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLS Demo — Row-Level Security via Arrow Flight SQL")
    args = parser.parse_args()

    try:
        asyncio.run(run_demo(args))
    except KeyboardInterrupt:
        print("\nDemo stopped.")
