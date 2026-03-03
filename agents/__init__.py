"""
agents — Platform Agent Team
==============================
Eight specialist agents that use real platform APIs to onboard, curate,
query, and analyze data.

Usage::

    from agents import PlatformAgents

    team = PlatformAgents(alias="demo", user="alice", password="pw")
    result = team.run("Create a trades dataset and build a star schema")

    # Direct agent access
    result = team.oltp.run("Create a trades table with symbol, price, quantity")
    result = team.quant.run("Compute realized vol for AAPL")
"""

from agents._team import PlatformAgents

__all__ = ["PlatformAgents"]
