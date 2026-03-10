"""
Pre-built Schedule definitions for common platform operations.

Usage::

    sched = make_sync_schedule()
    scheduler.register(sched)
"""

from __future__ import annotations

from scheduler.models import Schedule, Task


def make_sync_schedule(cron_expr: str = "*/5 * * * *") -> Schedule:
    """ETL: PG events + QuestDB ticks → Iceberg.

    Task functions must be importable at the configured paths.
    """
    return Schedule(
        name="sync",
        cron_expr=cron_expr,
        description="Incremental ETL from PG + QuestDB to Iceberg",
        tasks=[
            Task(name="sync_events", fn="lakehouse.sync:SyncEngine.sync_events"),
            Task(name="sync_ticks", fn="lakehouse.sync:SyncEngine.sync_ticks"),
        ],
    )


def make_media_embed_schedule(cron_expr: str = "0 * * * *") -> Schedule:
    """Re-embed documents that are missing embeddings.

    Task function must be importable at the configured path.
    """
    return Schedule(
        name="media_embed",
        cron_expr=cron_expr,
        description="Re-embed documents missing vector embeddings",
        tasks=[
            Task(name="find_and_embed", fn="media.store:MediaStore.reembed_missing"),
        ],
    )
