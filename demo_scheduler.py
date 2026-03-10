#!/usr/bin/env python3
"""
Demo: Scheduler — Cron + Pipelines + Parallel Execution
=========================================================

Shows the full scheduler lifecycle:

  1. Platform setup (embedded PG + WorkflowEngine + SchedulerServer)
  2. @schedule decorator — simple function + pipeline with depends_on
  3. Programmatic API — Schedule + Task registration
  4. Fire schedules — single-task and multi-task pipelines
  5. Diamond DAG — parallel branches (a → b,c → d)
  6. Failure propagation — failed task → dependent skipped
  7. Tick loop — cron-based automatic firing
  8. Pause/resume — management API
  9. Duration tracking — per-task timing
 10. History — query past runs

No external infrastructure required — everything starts embedded.

Usage::

    python3 demo_scheduler.py
"""

import logging
import tempfile
import time

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ── Importable task functions (must be at module level for resolve_fn) ────


def extract_data():
    """Simulates data extraction."""
    time.sleep(0.02)
    print("    ✓ extract_data: pulled 1,000 rows from source")
    return "1000 rows"


def transform_data():
    """Simulates data transformation."""
    time.sleep(0.03)
    print("    ✓ transform_data: cleaned and normalized")
    return "transformed"


def load_data():
    """Simulates data loading."""
    time.sleep(0.02)
    print("    ✓ load_data: wrote to warehouse")
    return "loaded"


def validate_data():
    """Simulates data validation (runs parallel with transform)."""
    time.sleep(0.02)
    print("    ✓ validate_data: schema checks passed")
    return "valid"


def publish_report():
    """Simulates report publishing (depends on transform + validate)."""
    time.sleep(0.01)
    print("    ✓ publish_report: report generated")
    return "published"


def always_fails():
    """Simulates a task that always fails."""
    raise RuntimeError("Connection refused: database unreachable")


def send_alert():
    """Simulates sending an alert (depends on always_fails)."""
    print("    ✓ send_alert: notification sent")
    return "alerted"


def slow_task():
    """Simulates a slow task for duration tracking."""
    time.sleep(0.1)
    print("    ✓ slow_task: heavy computation done")
    return "computed"


def fast_task():
    """Simulates a fast task for duration tracking."""
    print("    ✓ fast_task: cache hit")
    return "cached"


def heartbeat():
    """Simple heartbeat task."""
    print("    ✓ heartbeat: alive")
    return "ok"


# ── @schedule decorated functions (module level for importability) ────────

from scheduler import schedule as _schedule


@_schedule("*/5 * * * *")
def ingest_events():
    print("    ✓ ingest_events: pulled latest events")
    return "ingested"


@_schedule("0 2 * * *", name="etl_pipeline")
def etl_extract():
    print("    ✓ etl_extract: reading source data")
    return "extracted"


@_schedule("0 2 * * *", name="etl_pipeline", depends_on=["etl_extract"])
def etl_transform():
    print("    ✓ etl_transform: cleaning data")
    return "transformed"


@_schedule("0 2 * * *", name="etl_pipeline", depends_on=["etl_transform"])
def etl_load():
    print("    ✓ etl_load: writing to warehouse")
    return "loaded"


# ── Demo ──────────────────────────────────────────────────────────────────


def main():
    from scheduler import Schedule, Scheduler, Task
    from scheduler.admin import SchedulerServer

    MODULE = "demo_scheduler"

    # ── 1. Platform setup ─────────────────────────────────────────────

    print("=" * 70)
    print("  SCHEDULER DEMO")
    print("=" * 70)
    print()
    print("Starting platform services...")

    tmp = tempfile.mkdtemp(prefix="demo_scheduler_")
    server = SchedulerServer(data_dir=tmp)
    server.start(poll_interval=0)
    server.register_alias("demo")

    scheduler = Scheduler("demo")

    print("  ✓ SchedulerServer (embedded PG + WorkflowEngine)")
    print("  ✓ Scheduler client via alias 'demo'")
    print()

    # ── 2. @schedule decorator ────────────────────────────────────────

    print("-" * 70)
    print("1. @schedule DECORATOR")
    print("-" * 70)
    print()

    count = server.collect_schedules()
    print(f"  Registered {count} schedules via @schedule decorator:")
    print("    - ingest_events (*/5 * * * *) — single task")
    print("    - etl_pipeline (0 2 * * *) — 3-task pipeline with depends_on")
    print()

    # ── 3. Programmatic API ───────────────────────────────────────────

    print("-" * 70)
    print("2. PROGRAMMATIC API")
    print("-" * 70)
    print()

    # Simple single-task schedule
    scheduler.register(Schedule(
        name="heartbeat",
        cron_expr="*/1 * * * *",
        tasks=[Task("heartbeat", fn=f"{MODULE}:heartbeat")],
    ))
    print("  Registered: heartbeat (*/1 * * * *, 1 task)")

    # Linear pipeline
    scheduler.register(Schedule(
        name="data_pipeline",
        cron_expr="0 * * * *",
        description="Extract → Transform → Load",
        tasks=[
            Task("extract", fn=f"{MODULE}:extract_data"),
            Task("transform", fn=f"{MODULE}:transform_data", depends_on=["extract"]),
            Task("load", fn=f"{MODULE}:load_data", depends_on=["transform"]),
        ],
    ))
    print("  Registered: data_pipeline (0 * * * *, 3 tasks: extract → transform → load)")

    # Diamond DAG (parallel branches)
    scheduler.register(Schedule(
        name="diamond",
        cron_expr="0 2 * * *",
        description="Extract → (Transform || Validate) → Publish",
        tasks=[
            Task("extract", fn=f"{MODULE}:extract_data"),
            Task("transform", fn=f"{MODULE}:transform_data", depends_on=["extract"]),
            Task("validate", fn=f"{MODULE}:validate_data", depends_on=["extract"]),
            Task("publish", fn=f"{MODULE}:publish_report", depends_on=["transform", "validate"]),
        ],
    ))
    print("  Registered: diamond (0 2 * * *, 4 tasks: extract → transform,validate → publish)")

    # Failure test
    scheduler.register(Schedule(
        name="fragile",
        cron_expr="0 3 * * *",
        tasks=[
            Task("always_fails", fn=f"{MODULE}:always_fails"),
            Task("send_alert", fn=f"{MODULE}:send_alert", depends_on=["always_fails"]),
        ],
    ))
    print("  Registered: fragile (0 3 * * *, 2 tasks: always_fails → send_alert)")

    # Duration tracking
    scheduler.register(Schedule(
        name="timing_test",
        cron_expr="0 4 * * *",
        tasks=[
            Task("slow", fn=f"{MODULE}:slow_task"),
            Task("fast", fn=f"{MODULE}:fast_task"),
        ],
    ))
    print("  Registered: timing_test (0 4 * * *, 2 tasks: slow + fast)")
    print()

    # ── 4. Fire single-task schedule ──────────────────────────────────

    print("-" * 70)
    print("3. FIRE: Single-Task Schedule")
    print("-" * 70)
    print()

    run = scheduler.fire("heartbeat")
    print(f"  Run: {run.run_id[:8]}...")
    print(f"  State: {run._store_state}")
    print(f"  Tasks: {list(run.task_results.keys())}")
    print()

    # ── 5. Fire linear pipeline ───────────────────────────────────────

    print("-" * 70)
    print("4. FIRE: Linear Pipeline (extract → transform → load)")
    print("-" * 70)
    print()

    run = scheduler.fire("data_pipeline")
    print()
    print(f"  Run: {run.run_id[:8]}...")
    print(f"  State: {run._store_state}")
    for name, tr in run.task_results.items():
        status = tr.status if hasattr(tr, 'status') else tr.get('status', '?')
        dur = tr.duration_ms if hasattr(tr, 'duration_ms') else tr.get('duration_ms', 0)
        print(f"    {name}: {status} ({dur:.1f}ms)")
    print()

    # ── 6. Fire diamond DAG ───────────────────────────────────────────

    print("-" * 70)
    print("5. FIRE: Diamond DAG (parallel branches)")
    print("-" * 70)
    print("  Graph: extract → (transform || validate) → publish")
    print()

    run = scheduler.fire("diamond")
    print()
    print(f"  Run: {run.run_id[:8]}...")
    print(f"  State: {run._store_state}")
    for name, tr in run.task_results.items():
        status = tr.status if hasattr(tr, 'status') else tr.get('status', '?')
        dur = tr.duration_ms if hasattr(tr, 'duration_ms') else tr.get('duration_ms', 0)
        print(f"    {name}: {status} ({dur:.1f}ms)")
    print()

    # ── 7. Failure propagation ────────────────────────────────────────

    print("-" * 70)
    print("6. FAILURE PROPAGATION")
    print("-" * 70)
    print("  always_fails will ERROR → send_alert will be SKIPPED")
    print()

    run = scheduler.fire("fragile")
    print()
    print(f"  Run: {run.run_id[:8]}...")
    print(f"  State: {run._store_state}")
    for name, tr in run.task_results.items():
        status = tr.status if hasattr(tr, 'status') else tr.get('status', '?')
        error = ""
        if hasattr(tr, 'error') and tr.error:
            error = f" — {tr.error[:50]}"
        elif isinstance(tr, dict) and tr.get('error'):
            error = f" — {tr['error'][:50]}"
        print(f"    {name}: {status}{error}")
    print()

    # ── 8. Duration tracking ──────────────────────────────────────────

    print("-" * 70)
    print("7. DURATION TRACKING")
    print("-" * 70)
    print()

    run = scheduler.fire("timing_test")
    print()
    slow_dur = run.task_results["slow"].duration_ms if hasattr(run.task_results.get("slow"), 'duration_ms') else 0
    fast_dur = run.task_results["fast"].duration_ms if hasattr(run.task_results.get("fast"), 'duration_ms') else 0
    print(f"  slow: {slow_dur:.1f}ms")
    print(f"  fast: {fast_dur:.1f}ms")
    print(f"  slow is {slow_dur / max(fast_dur, 0.01):.0f}x slower than fast")
    print()

    # ── 9. Tick loop ──────────────────────────────────────────────────

    print("-" * 70)
    print("8. TICK — Cron-Based Automatic Firing")
    print("-" * 70)
    print()

    runs = scheduler.tick()
    print(f"  Tick fired {len(runs)} schedule(s):")
    for r in runs:
        print(f"    - {r.schedule_name}: {r._store_state}")
    print()

    # ── 10. Pause / Resume ────────────────────────────────────────────

    print("-" * 70)
    print("9. PAUSE / RESUME")
    print("-" * 70)
    print()

    scheduler.pause("heartbeat")
    print("  Paused: heartbeat")

    runs_after_pause = scheduler.tick()
    heartbeat_fired = any(r.schedule_name == "heartbeat" for r in runs_after_pause)
    print(f"  Tick after pause: heartbeat fired? {heartbeat_fired}")

    scheduler.resume("heartbeat")
    print("  Resumed: heartbeat")
    print()

    # ── 11. List & History ────────────────────────────────────────────

    print("-" * 70)
    print("10. LIST & HISTORY")
    print("-" * 70)
    print()

    all_schedules = scheduler.list_schedules()
    print(f"  {len(all_schedules)} registered schedules:")
    for s in all_schedules:
        task_count = len(s.tasks) if hasattr(s.tasks, '__len__') else '?'
        print(f"    - {s.name} ({s.cron_expr}, {task_count} tasks)")

    print()
    runs = scheduler.history("data_pipeline")
    print(f"  data_pipeline history: {len(runs)} run(s)")
    print()

    # ── Cleanup ───────────────────────────────────────────────────────

    server.stop()

    print("=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
