"""
Risk Client — Exposure Monitoring & Alerts
===========================================
Example: A risk analyst connects to the Deephaven server, monitors
large exposures, creates filtered risk views, and periodically
prints risk summaries.

Usage:  python3 risk_client.py [--host localhost] [--port 10000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming import StreamingClient


def main(host="localhost", port=10000):
    with StreamingClient(host=host, port=port) as client:
        # ── 1. List available tables ─────────────────────────────────
        print("\nAvailable server tables:")
        for name in client.list_tables():
            print(f"  • {name}")

        # ── 2. Create server-side risk views via run_script ──────────
        client.run_script("""
from deephaven import agg

# Large exposures: positions where |MarketValue| > 50,000
large_exposures = risk_live.where(
    ["Math.abs(MarketValue) > 50000"]
)

# Risk summary by sign of position (long vs short)
risk_by_direction = risk_live.update(
    ["Direction = Position > 0 ? `LONG` : `SHORT`"]
).agg_by(
    [
        agg.sum_(["TotalMV=MarketValue", "TotalPnL=UnrealizedPnL", "TotalDelta=Delta"]),
        agg.count_("Count"),
    ],
    by=["Direction"],
)

# Greeks heatmap data: all risk metrics side by side
greeks_monitor = risk_live.update([
    "AbsDelta = Math.abs(Delta)",
    "AbsGamma = Math.abs(Gamma)",
    "RiskScore = Math.abs(Delta) + Math.abs(Gamma) * 100 + Math.abs(Vega) * 10",
]).sort_descending("RiskScore")
""")
        print("Created server-side risk views:")
        print("  • large_exposures — positions with |MV| > $50k")
        print("  • risk_by_direction — aggregated long vs short")
        print("  • greeks_monitor — ranked by composite risk score")

        # ── 3. Check for tables published by other clients ──────────
        all_tables = client.list_tables()
        print("\nAll tables on server (including from other clients):")
        for name in all_tables:
            print(f"  • {name}")

        # If the quant client published a watchlist, we can read it
        if "quant_watchlist" in all_tables:
            df = client.open_table("quant_watchlist").to_arrow().to_pandas()
            print("\n  Found quant_watchlist (published by quant client):")
            print(df[["Symbol", "Price", "ChangePct"]].to_string(index=False))

        # ── 4. Periodic risk report ──────────────────────────────────
        print("\n✓ Risk views published. Visible to ALL other clients + web IDE.")
        print("Printing risk summary every 10 seconds. Press Ctrl+C to stop.\n")

        try:
            while True:
                # Snapshot portfolio summary
                summary = client.open_table("portfolio_summary").to_arrow().to_pandas()
                print(f"[{time.strftime('%H:%M:%S')}] Portfolio Summary:")
                print(summary.to_string(index=False))

                # Snapshot large exposures
                exposures = client.open_table("large_exposures").to_arrow().to_pandas()
                if len(exposures) > 0:
                    print(f"\n  Large Exposures ({len(exposures)} positions):")
                    print(exposures[["Symbol", "Position", "MarketValue", "Delta"]].to_string(index=False))
                else:
                    print("  No large exposures.")

                # Direction breakdown
                direction = client.open_table("risk_by_direction").to_arrow().to_pandas()
                print("\n  Long vs Short:")
                print(direction.to_string(index=False))

                print("-" * 60)
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nDisconnecting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk client for Deephaven Trading Server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
