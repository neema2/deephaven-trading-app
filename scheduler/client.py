"""
Scheduler — user-facing client (alias-based).

Public import via ``scheduler``::

    from scheduler import Scheduler

    scheduler = Scheduler("demo")
    scheduler.register(Schedule(...))
    scheduler.fire("etl")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from scheduler.models import Run, Schedule

if TYPE_CHECKING:
    from scheduler.server import SchedulerServer

logger = logging.getLogger(__name__)


class Scheduler:
    """User-facing scheduler API.

    Connects to a SchedulerServer via alias. All operations delegate
    to the server — the user never touches infrastructure directly.
    """

    def __init__(self, alias: str) -> None:
        """
        Args:
            alias: Registered alias name (e.g. ``"demo"``).
        """
        from scheduler._registry import resolve_alias
        resolved = resolve_alias(alias)
        if resolved is None:
            raise ValueError(
                f"No scheduler registered under alias {alias!r}. "
                f"Call SchedulerServer.register_alias({alias!r}) first."
            )
        self._server: SchedulerServer = resolved["server"]

    # ── Registration ──────────────────────────────────────────────────

    def register(self, schedule: Schedule) -> Schedule:
        """Register a schedule (persists to PG)."""
        return self._server.register(schedule)

    # ── Trigger ───────────────────────────────────────────────────────

    def fire(self, name: str) -> Run:
        """Manually trigger a schedule by name."""
        return self._server.fire(name)

    def tick(self, now: datetime | None = None) -> list[Run]:
        """Check due schedules, fire any that are due."""
        return self._server.tick(now)

    # ── Management ────────────────────────────────────────────────────

    def pause(self, name: str) -> Schedule:
        """Pause a schedule (ACTIVE → PAUSED)."""
        return self._server.pause(name)

    def resume(self, name: str) -> Schedule:
        """Resume a schedule (PAUSED → ACTIVE)."""
        return self._server.resume(name)

    def delete(self, name: str) -> Schedule:
        """Soft-delete a schedule (→ DELETED)."""
        return self._server.delete(name)

    # ── Query ─────────────────────────────────────────────────────────

    def list_schedules(self) -> list[Schedule]:
        """Return all non-deleted schedules."""
        return self._server.list_schedules()

    def history(self, name: str, limit: int = 20) -> list[Run]:
        """Return past runs for a schedule, most recent first."""
        return self._server.history(name, limit)
