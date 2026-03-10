"""
Scheduler — cron-based scheduling with durable task execution.

User API::

    from scheduler import Scheduler, Schedule, Task, Run, TaskResult
    from scheduler import schedule

    @schedule("*/5 * * * *")
    def ingest_events():
        Lakehouse("demo").sync_events()

    @schedule("0 2 * * *", name="etl")
    def extract(): ...

    @schedule("0 2 * * *", name="etl", depends_on=["extract"])
    def transform(): ...

Platform API lives in ``scheduler.admin``.
"""

from scheduler.client import Scheduler
from scheduler.dag import CycleError
from scheduler.decorators import schedule
from scheduler.models import Run, Schedule, Task, TaskResult

__all__ = [
    "CycleError",
    "Run",
    "Schedule",
    "Scheduler",
    "Task",
    "TaskResult",
    "schedule",
]
