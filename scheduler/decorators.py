"""
@schedule decorator — syntactic sugar for registering scheduled functions.

Usage::

    @schedule("*/5 * * * *")
    def ingest_events():
        Lakehouse("demo").sync_events()

    @schedule("0 2 * * *", name="etl")
    def extract():
        return Lakehouse("demo").extract()

    @schedule("0 2 * * *", name="etl", depends_on=["extract"])
    def transform():
        return Lakehouse("demo").transform()

The decorator auto-derives the importable path (module:qualname) for Task.fn.
Functions sharing the same name are grouped into a single Schedule with
multiple Tasks. A lone function becomes a Schedule with one Task.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from scheduler.models import Schedule, Task

if TYPE_CHECKING:
    from scheduler.client import Scheduler as SchedulerClient

# Pending tasks grouped by schedule name — collected at import time
_pending_tasks: list[dict] = []


def schedule(cron_expr: str, name: str | None = None,
             depends_on: list[str] | None = None, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory: register a function as a scheduled task.

    Args:
        cron_expr: Cron expression (e.g. "*/5 * * * *").
        name: Schedule name. Defaults to fn.__name__. Functions sharing
              the same name are grouped into a single Schedule.
        depends_on: Task names this function depends on (for pipelines).
        **kwargs: Additional Schedule fields (max_retries, timeout_s, etc.).

    The decorator auto-derives Task.fn as 'module:qualname' from the
    decorated function, making it importlib-resolvable.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sched_name = name or fn.__name__
        task_name = fn.__name__
        task_fn = f"{fn.__module__}:{fn.__qualname__}"
        _pending_tasks.append({
            "schedule_name": sched_name,
            "cron_expr": cron_expr,
            "task_name": task_name,
            "task_fn": task_fn,
            "depends_on": depends_on or [],
            "kwargs": kwargs,
        })
        return fn
    return decorator


def collect_schedules(scheduler: SchedulerClient) -> int:
    """Flush all @schedule-decorated functions to PG via a Scheduler client.

    Groups tasks by schedule name — multiple tasks with the same name
    become a single Schedule with multiple Tasks.

    Idempotent — safe to call on every startup.
    Returns the number of schedules registered.
    """
    # Group by schedule name
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in _pending_tasks:
        groups[entry["schedule_name"]].append(entry)

    count = 0
    for sched_name, entries in groups.items():
        # Build tasks from all entries in this group
        tasks = [
            Task(
                name=e["task_name"],
                fn=e["task_fn"],
                depends_on=e["depends_on"],
            )
            for e in entries
        ]
        # Use cron/kwargs from first entry (all should match for same group)
        first = entries[0]
        sched = Schedule(
            name=sched_name,
            cron_expr=first["cron_expr"],
            tasks=tasks,
            **first["kwargs"],
        )
        scheduler.register(sched)
        count += 1

    return count
