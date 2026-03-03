"""
agents._eval — Multi-Dimensional Agent Evaluation Framework
============================================================
Goes far beyond tool selection: evaluates data models, curation quality,
cross-dataset linking, metadata capture, query correctness, and analysis quality.

Public surface::

    from agents._eval import AgentEval, AgentEvalCase, AgentEvalResult
    from agents._eval import EvalDimension, EvalPhase
    from agents._eval import scorers, judges
"""

from agents._eval.framework import (
    AgentEval,
    AgentEvalCase,
    AgentEvalResult,
    EvalDimension,
    EvalPhase,
)

__all__ = [
    "AgentEval",
    "AgentEvalCase",
    "AgentEvalResult",
    "EvalDimension",
    "EvalPhase",
]
