"""
Portfolio Manager Client — Summary & P&L Snapshots
====================================================
Example: A PM connects to the Deephaven server, views portfolio
summary, takes periodic P&L snapshots, and exports to pandas
for local analysis.

Usage:  python3 pm_client.py [--host localhost] [--port 10000]
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

        # ── 2. Create server-side P&L tracking table ─────────────────
        client.run_script("""
from deephaven import agg

# P&L ranked view: best and worst performers
pnl_ranked = risk_live.sort_descending("UnrealizedPnL")

# Position sizing: market value as percentage of total portfolio
pm_positions = risk_live.update([
    "AbsMV = Math.abs(MarketValue)",
]).sort_descending("AbsMV")
""")
        print("Created server-side PM views:")
        print("  • pnl_ranked — positions ranked by P&L")
        print("  • pm_positions — positions by absolute market value")

        # ── 3. Show initial portfolio state ──────────────────────────
        summary = client.open_table("portfolio_summary").to_arrow().to_pandas()
        print("\nPortfolio Summary:")
        print(summary.to_string(index=False))

        risk = client.open_table("risk_live").to_arrow().to_pandas()
        print(f"\nAll Positions ({len(risk)} symbols):")
        print(risk[["Symbol", "Position", "MarketValue", "UnrealizedPnL"]].to_string(index=False))

        # ── 4. Periodic P&L snapshots for local analysis ─────────────
        print("\n✓ PM views are live in the Deephaven web IDE.")
        print("  Open http://localhost:10000 and look for:")
        print("    • pnl_ranked")
        print("    • pm_positions")
        print("    • portfolio_summary")
        print("\nTaking P&L snapshots every 15 seconds. Press Ctrl+C to stop.\n")

        snapshots = []
        try:
            while True:
                ts = time.strftime('%H:%M:%S')
                summary = client.open_table("portfolio_summary").to_arrow().to_pandas()
                summary["Timestamp"] = ts
                snapshots.append(summary)

                total_pnl = summary["TotalPnL"].iloc[0] if len(summary) > 0 else 0
                total_mv = summary["TotalMV"].iloc[0] if len(summary) > 0 else 0
                print(f"[{ts}]  Total MV: ${total_mv:>12,.2f}  |  Total P&L: ${total_pnl:>10,.2f}  |  Snapshots: {len(snapshots)}")

                time.sleep(15)
        except KeyboardInterrupt:
            print(f"\n\nCollected {len(snapshots)} snapshots.")
            if snapshots:
                import pandas as pd
                history = pd.concat(snapshots, ignore_index=True)
                print("\nP&L History:")
                print(history.to_string(index=False))
                # Could save: history.to_csv("pnl_history.csv")
            print("Disconnecting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PM client for Deephaven Trading Server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
