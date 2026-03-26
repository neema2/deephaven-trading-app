#!/usr/bin/env bash
# ── Parallel test runner ────────────────────────────────────────────
# Runs main tests and demo tests IN PARALLEL on isolated port ranges.
#
# Port isolation:
#   run_tests.sh      → base ports (10000, 8765, 9200, ...)
#   run_demo_tests.sh → offset ports (10100, 8865, 9300, ...) via PORT_OFFSET=100
#
# Usage:  ./run_all_tests.sh
set -uo pipefail

cd "$(dirname "$0")"

# ── Global cleanup (once, before either suite starts) ───────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PARALLEL TEST RUNNER — main + demo on isolated ports       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Cleaning up ALL stale services..."

# Kill base ports + offset ports (PORT_OFFSET=200)
for port in 10000 8765 9200 9209 8922 5490 8183 9004 9005 9102 9103 8050 \
            10200 8965 9400 9409 9122 5690 8383 9204 9205 9302 9303; do
    lsof -ti :"$port" 2>/dev/null | xargs kill -9 2>/dev/null || true
done
pkill -9 -f postgres 2>/dev/null || true
pkill -9 -f lakekeeper 2>/dev/null || true
pkill -9 -f minio 2>/dev/null || true

# Clean up stale SysV shared memory segments
for seg in $(ipcs -m 2>/dev/null | awk '/^m / || /^0x/ {print $2}' | grep -E '^[0-9]+$'); do
    ipcrm -m "$seg" 2>/dev/null || true
done

# Clean up stale Deephaven Docker containers
if command -v docker &>/dev/null; then
    docker ps -aq --filter "name=dh-streaming" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
fi
sleep 1

# ── Launch both suites (SKIP_CLEANUP=1 so they don't kill each other) ─
export SKIP_CLEANUP=1

echo ""
echo "Starting main tests (base ports) and demo tests (ports +200)..."

./run_tests.sh &
PID_MAIN=$!

./run_demo_tests.sh &
PID_DEMO=$!

# ── Wait for both and report ────────────────────────────────────────
MAIN_RC=0
DEMO_RC=0

wait $PID_MAIN || MAIN_RC=$?
wait $PID_DEMO || DEMO_RC=$?

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RESULTS                                                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
if [ $MAIN_RC -eq 0 ]; then
    echo "║  ✅ Main tests       PASSED                                 ║"
else
    echo "║  ❌ Main tests       FAILED  (exit $MAIN_RC)                       ║"
fi
if [ $DEMO_RC -eq 0 ]; then
    echo "║  ✅ Demo tests       PASSED                                 ║"
else
    echo "║  ❌ Demo tests       FAILED  (exit $DEMO_RC)                       ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"

# Exit non-zero if either failed
[ $MAIN_RC -eq 0 ] && [ $DEMO_RC -eq 0 ]
