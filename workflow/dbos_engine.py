"""
DBOS-backed implementation of WorkflowEngine.

This module is an internal implementation detail — application code should
never import from here.  Use ``workflow.WorkflowEngine`` instead.
"""

import functools
from collections.abc import Callable
from typing import Any

from dbos import DBOS
from dbos import Queue as DBOSQueue

from workflow.engine import WorkflowEngine, WorkflowHandle, WorkflowStatus

# Map DBOS status strings → our WorkflowStatus enum
_STATUS_MAP = {
    "PENDING": WorkflowStatus.PENDING,
    "ENQUEUED": WorkflowStatus.PENDING,
    "SUCCESS": WorkflowStatus.SUCCESS,
    "ERROR": WorkflowStatus.ERROR,
    "CANCELLED": WorkflowStatus.CANCELLED,
    "RETRIES_EXCEEDED": WorkflowStatus.ERROR,
}


class DBOSEngine(WorkflowEngine):
    """WorkflowEngine backed by DBOS Transact.

    Parameters
    ----------
    pg_url : str
        SQLAlchemy connection string to the PostgreSQL database where DBOS
        stores its system tables (``dbos`` schema).
    name : str
        Application name registered with DBOS.
    """

    def __init__(self, pg_url: str, *, name: str = "workflow-app") -> None:
        self._dbos = DBOS(config={
            "name": name,
            "system_database_url": pg_url,
        })
        self._queues: dict[str, DBOSQueue] = {}
        self._launched = False

    # ── Lifecycle ────────────────────────────────────────────────────

    def launch(self) -> "DBOSEngine":
        """Create system tables and start background threads.

        Must be called after all workflow/step functions have been
        registered (via :meth:`workflow` / :meth:`step`) but before
        any of them are invoked.
        """
        if not self._launched:
            DBOS.launch()
            self._launched = True
        return self

    def destroy(self) -> None:
        """Shut down DBOS cleanly."""
        if self._launched:
            DBOS.destroy()
            self._launched = False

    def __enter__(self) -> "DBOSEngine":
        self.launch()
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()

    # ── Running workflows / steps ────────────────────────────────────

    def workflow(self, fn: Callable, *args: Any, **kwargs: Any) -> WorkflowHandle:
        """Run *fn* as a durable workflow.

        All arguments must be serialisable (pickle-safe) because they are
        persisted for crash recovery.  Do **not** pass DB connections,
        engines, or other non-serialisable objects — access those through
        module-level references instead.
        """
        wrapped = self._ensure_workflow(fn)
        handle = DBOS.start_workflow(wrapped, *args, **kwargs)
        return WorkflowHandle(
            workflow_id=handle.get_workflow_id(),
            _engine=self,
        )

    def run(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run *fn* as a durable workflow **synchronously** in-process.

        Convenient for workflows whose arguments include non-serialisable
        objects (e.g. an engine reference).  The workflow still benefits
        from step-level checkpointing but cannot be recovered across
        process restarts.
        """
        wrapped = self._ensure_workflow(fn)
        return wrapped(*args, **kwargs)

    def step(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        wrapped = self._ensure_step(fn)
        return wrapped(*args, **kwargs)

    # ── Queues ───────────────────────────────────────────────────────

    def create_queue(self, name: str, concurrency: int = 10) -> None:
        """Pre-create a named queue with the given concurrency limit."""
        if name not in self._queues:
            self._queues[name] = DBOSQueue(name, concurrency)

    def queue(
        self,
        queue_name: str,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> WorkflowHandle:
        if queue_name not in self._queues:
            self._queues[queue_name] = DBOSQueue(queue_name)
        q = self._queues[queue_name]
        wrapped = self._ensure_workflow(fn)
        handle = q.enqueue(wrapped, *args, **kwargs)
        return WorkflowHandle(
            workflow_id=handle.get_workflow_id(),
            _engine=self,
        )

    # ── Durable timers ───────────────────────────────────────────────

    def sleep(self, seconds: float) -> None:
        DBOS.sleep(seconds)

    # ── Inter-workflow messaging ─────────────────────────────────────

    def send(self, workflow_id: str, topic: str, value: Any) -> None:
        DBOS.send(workflow_id, value, topic)

    def recv(self, topic: str, timeout: float | None = None) -> Any:
        timeout_s = timeout if timeout is not None else 60
        return DBOS.recv(topic, timeout_seconds=timeout_s)

    # ── Workflow management ──────────────────────────────────────────

    def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
        status = DBOS.get_workflow_status(workflow_id)
        if status is None:
            raise ValueError(f"No workflow found with id {workflow_id}")
        return _STATUS_MAP.get(status.status, WorkflowStatus.RUNNING)

    def get_workflow_result(
        self, workflow_id: str, *, timeout: float | None = None
    ) -> Any:
        handle = DBOS.retrieve_workflow(workflow_id)
        result = handle.get_result()
        return result

    # ── Internal helpers ─────────────────────────────────────────────

    _workflow_registry: dict[Callable, Callable] = {}
    _step_registry: dict[Callable, Callable] = {}

    def _ensure_workflow(self, fn: Callable) -> Callable:
        """Wrap *fn* with @DBOS.workflow() exactly once."""
        if fn not in self._workflow_registry:
            @DBOS.workflow()
            @functools.wraps(fn)
            def wrapper(*a: Any, **kw: Any) -> Any:
                return fn(*a, **kw)
            self._workflow_registry[fn] = wrapper
        return self._workflow_registry[fn]

    def _ensure_step(self, fn: Callable) -> Callable:
        """Wrap *fn* with @DBOS.step() exactly once."""
        if fn not in self._step_registry:
            @DBOS.step()
            @functools.wraps(fn)
            def wrapper(*a: Any, **kw: Any) -> Any:
                return fn(*a, **kw)
            self._step_registry[fn] = wrapper
        return self._step_registry[fn]
