"""
ir — Interest Rate domain models and utilities.

Shared reactive domain models, graph construction, and market data
consumption used by both ``demo_ir_swap.py`` and ``demo_ir_risk.py``.
"""

from ir.risk import Risk_IR_DV01

__all__ = ["Risk_IR_DV01"]
