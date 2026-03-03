"""
PlatformAgents — The only public class in the agents package.
===============================================================
Eight specialist agents orchestrated by an LLM router.

Usage::

    from agents import PlatformAgents

    team = PlatformAgents(alias="demo", user="alice", password="pw")
    result = team.run("Create a trades dataset and build a star schema")

    # Or use a specific agent directly
    result = team.oltp.run("Create a trades table with symbol, price, quantity")
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from ai import Agent, AgentTeam
from agents._context import _PlatformContext

logger = logging.getLogger(__name__)

# Agent descriptions for the router
_AGENT_DESCRIPTIONS = {
    "oltp": (
        "OLTP Dataset Agent — creates and manages operational datasets in PostgreSQL. "
        "Use for: defining new data schemas, creating tables, inserting records, "
        "bulk-loading from CSV/Parquet, querying OLTP data."
    ),
    "lakehouse": (
        "Lakehouse Curation Agent — transforms operational data into analytical Iceberg tables. "
        "Use for: designing star schemas (fact/dimension tables), ETL transforms, "
        "building datacube reports, querying the lakehouse via SQL."
    ),
    "feed": (
        "Market Data Feed Agent — manages real-time market data feeds. "
        "Use for: listing available symbols, checking feed health, "
        "publishing custom ticks, explaining feed architecture."
    ),
    "timeseries": (
        "Time Series Agent — manages historical market data. "
        "Use for: OHLCV bars, tick history, realized volatility, "
        "cross-exchange comparisons, historical data ingestion."
    ),
    "document": (
        "Document Agent — manages unstructured documents. "
        "Use for: uploading PDFs/reports, searching documents, "
        "extracting structured data from text, bulk uploads, tagging."
    ),
    "dashboard": (
        "Dashboard Agent — builds real-time streaming dashboards. "
        "Use for: creating ticking tables, streaming joins/aggregations, "
        "setting up StoreBridge for event streaming, reactive models."
    ),
    "query": (
        "Query Agent — universal data access. Knows every data store and picks the right one. "
        "Use for: answering natural language questions about data, "
        "discovering what data is available, cross-store queries."
    ),
    "quant": (
        "Quant Agent — analytical and quantitative workflows. "
        "Use for: statistics, correlations, regressions, anomaly detection, "
        "time series decomposition, visualization recommendations."
    ),
}


class PlatformAgents:
    """Multi-agent team for data engineering tasks.

    The only public class in the ``agents`` package.  Wraps 8 specialist
    agents in an ``AgentTeam`` with LLM-based routing.

    Usage::

        from agents import PlatformAgents

        team = PlatformAgents(alias="demo", user="alice", password="pw")

        # Full team — LLM routes automatically
        result = team.run("Create a trades dataset and build a star schema")

        # Direct agent access via typed properties
        result = team.oltp.run("Create a trades table")
        result = team.lakehouse.run("Design a star schema for trades")
        result = team.quant.run("Compute realized vol for AAPL")

    Args:
        alias: Default alias for all platform services.
        user: Default user for authenticated services (e.g., Store).
        password: Default password.
        store_alias: Override alias for the object store.
        lakehouse_alias: Override alias for the lakehouse.
        tsdb_alias: Override alias for the time-series DB.
        streaming_alias: Override alias for the streaming server.
        md_alias: Override alias for the market data server.
        media_alias: Override alias for the media store.
        ai: Pre-built AI instance (created from env if not provided).
        agents: Subset of agents to activate (default: all 8).
            Valid names: oltp, lakehouse, feed, timeseries, document,
            dashboard, query, quant.
        temperature: Router LLM temperature (default: 0.5).
        max_delegations: Max delegation rounds per run (default: 8).
    """

    def __init__(
        self,
        alias: str = "",
        user: str = "",
        password: str = "",
        *,
        store_alias: str | None = None,
        lakehouse_alias: str | None = None,
        tsdb_alias: str | None = None,
        streaming_alias: str | None = None,
        md_alias: str | None = None,
        media_alias: str | None = None,
        ai: Any = None,
        agents: list[str] | None = None,
        temperature: float = 0.5,
        max_delegations: int = 8,
    ):
        # Build internal context
        self._ctx = _PlatformContext(
            alias=alias,
            user=user,
            password=password,
            store_alias=store_alias,
            lakehouse_alias=lakehouse_alias,
            tsdb_alias=tsdb_alias,
            streaming_alias=streaming_alias,
            md_alias=md_alias,
            media_alias=media_alias,
            ai=ai,
        )

        # Lazy-import agent factories to avoid circular imports
        from agents._oltp import create_oltp_agent
        from agents._lakehouse import create_lakehouse_agent
        from agents._feed import create_feed_agent
        from agents._timeseries import create_timeseries_agent
        from agents._document import create_document_agent
        from agents._dashboard import create_dashboard_agent
        from agents._query import create_query_agent
        from agents._datascience import create_datascience_agent

        _factories = {
            "oltp": create_oltp_agent,
            "lakehouse": create_lakehouse_agent,
            "feed": create_feed_agent,
            "timeseries": create_timeseries_agent,
            "document": create_document_agent,
            "dashboard": create_dashboard_agent,
            "query": create_query_agent,
            "quant": create_datascience_agent,
        }

        # Build requested agents (or all)
        agent_names = agents or list(_factories.keys())
        self._agents: dict[str, Agent] = {}
        for name in agent_names:
            factory = _factories.get(name)
            if factory is None:
                logger.warning("Unknown agent: %s (skipping)", name)
                continue
            agent = factory(self._ctx)
            agent._description = _AGENT_DESCRIPTIONS.get(name, "")
            self._agents[name] = agent

        # Build the team
        self._team = AgentTeam(
            agents=self._agents,
            ai=ai,
            max_delegations=max_delegations,
            temperature=temperature,
        )

        logger.info(
            "PlatformAgents initialized with %d agents: %s",
            len(self._agents), list(self._agents.keys()),
        )

    # ── Team-level operations ────────────────────────────────────

    def run(self, prompt: str):
        """Run a data engineering task.

        The router LLM decomposes the task and delegates to specialist agents.

        Args:
            prompt: Natural language request.

        Returns:
            TeamResult with the synthesized response and delegation log.
        """
        logger.info("PlatformAgents.run: %s", prompt[:100])
        return self._team.run(prompt)

    # ── Typed agent properties ───────────────────────────────────

    @property
    def oltp(self) -> Agent:
        """OLTP Dataset Agent — creates and manages operational datasets."""
        return self._agents["oltp"]

    @property
    def lakehouse(self) -> Agent:
        """Lakehouse Curation Agent — star schemas, ETL, datacubes."""
        return self._agents["lakehouse"]

    @property
    def feed(self) -> Agent:
        """Market Data Feed Agent — real-time market data feeds."""
        return self._agents["feed"]

    @property
    def timeseries(self) -> Agent:
        """Time Series Agent — historical market data."""
        return self._agents["timeseries"]

    @property
    def document(self) -> Agent:
        """Document Agent — unstructured document management."""
        return self._agents["document"]

    @property
    def dashboard(self) -> Agent:
        """Dashboard Agent — real-time streaming dashboards."""
        return self._agents["dashboard"]

    @property
    def query(self) -> Agent:
        """Query Agent — universal data access across all stores."""
        return self._agents["query"]

    @property
    def quant(self) -> Agent:
        """Quant Agent — statistics, regressions, anomaly detection."""
        return self._agents["quant"]

    # ── Collection protocol ──────────────────────────────────────

    def __iter__(self) -> Iterator[tuple[str, Agent]]:
        return iter(self._agents.items())

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents
