"""
WorkflowDispatcher — helper for durable multi-step state progressions.

Use inside workflows to make state transitions checkpointed (exactly-once
on crash recovery):

    dispatcher = WorkflowDispatcher(engine, client)

    def settlement_workflow(entity_id):
        order = engine.step(lambda: client.read(Order, entity_id))
        engine.step(lambda: call_clearing_house(order))
        dispatcher.durable_transition(order, "SETTLED")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from store import UserConnection

if TYPE_CHECKING:
    from workflow.engine import WorkflowEngine


class WorkflowDispatcher:
    """Helper for durable state transitions inside workflows.

    Wraps client.transition() in engine.step() so each transition is
    checkpointed — on crash recovery it replays from the checkpoint
    rather than re-executing the transition.
    """

    def __init__(self, engine: WorkflowEngine, client: UserConnection) -> None:
        self._engine = engine
        self._client = client

    def durable_transition(self, obj: Any, new_state: str, **kwargs: Any) -> Any:
        """Execute a state transition as a checkpointed workflow step.

        Exactly-once semantics: if the workflow crashes after this step
        completes, the transition is not re-executed on recovery.
        """
        return self._engine.step(
            lambda: self._client.transition(obj, new_state, **kwargs)
        )
