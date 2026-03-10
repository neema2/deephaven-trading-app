"""
Zero-trust Python object store backed by PostgreSQL JSONB + Row-Level Security.
"""

from store.base import Embedded, Storable
from store.client import VersionConflict
from store.connection import connect
from store.state_machine import (
    GuardFailure,
    InvalidTransition,
    StateMachine,
    Transition,
    TransitionNotPermitted,
)
from store.subscriptions import ChangeEvent, EventListener

__all__ = [
    "ChangeEvent",
    "Embedded",
    "EventListener",
    "GuardFailure",
    "InvalidTransition",
    "StateMachine",
    "Storable",
    "Transition",
    "TransitionNotPermitted",
    "VersionConflict",
    "connect",
]
