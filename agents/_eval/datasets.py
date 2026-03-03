"""
Built-in Eval Datasets
=======================
Curated eval cases for each agent at varying difficulty levels.

Usage::

    from agents._eval.datasets import OLTP_EVAL_CASES, LAKEHOUSE_EVAL_CASES

    evaluator = AgentEval(agents={"oltp": oltp_agent})
    results = evaluator.run(OLTP_EVAL_CASES)
"""

from agents._eval.framework import AgentEvalCase


# ── OLTP Agent Eval Cases ─────────────────────────────────────────────

OLTP_EVAL_CASES = [
    # Basic: simple table creation
    AgentEvalCase(
        input="Create a trades table with fields: symbol (string), quantity (int), price (float), side (string)",
        agent="oltp",
        expected_tools=["create_dataset"],
        expected_schema={"fields": ["symbol", "quantity", "price", "side"]},
        tags=["oltp", "create"],
        difficulty="basic",
        description="Simple 4-field trade table",
    ),
    AgentEvalCase(
        input="What dataset types are currently registered?",
        agent="oltp",
        expected_tools=["list_storable_types"],
        tags=["oltp", "list"],
        difficulty="basic",
        description="List existing types",
    ),
    AgentEvalCase(
        input="Describe the schema of the Trade dataset",
        agent="oltp",
        expected_tools=["describe_type"],
        expected_output_contains=["symbol", "price"],
        tags=["oltp", "describe"],
        difficulty="basic",
        description="Describe a known type",
    ),

    # Intermediate: creation + insert
    AgentEvalCase(
        input=(
            "Create an Instrument dataset with: ticker (string), name (string), "
            "sector (string), exchange (string), market_cap (float), is_active (bool). "
            "Then insert 3 sample instruments: AAPL/Apple/Tech/NASDAQ/3000000000000, "
            "GOOGL/Alphabet/Tech/NASDAQ/2000000000000, JPM/JPMorgan/Finance/NYSE/500000000000"
        ),
        agent="oltp",
        expected_tools=["create_dataset", "insert_records"],
        expected_schema={"fields": ["ticker", "name", "sector", "exchange", "market_cap", "is_active"]},
        tags=["oltp", "create", "insert"],
        difficulty="intermediate",
        description="Create table + seed data",
    ),
    AgentEvalCase(
        input="Query all records from the Instrument dataset, limit 10",
        agent="oltp",
        expected_tools=["query_dataset"],
        tags=["oltp", "query"],
        difficulty="intermediate",
        description="Query with limit",
    ),

    # Advanced: complex schema design
    AgentEvalCase(
        input=(
            "Design and create a complete order management schema. I need: "
            "an Order type (order_id string, symbol string, side string, order_type string, "
            "quantity int, price float, status string, trader string, desk string, "
            "submitted_at string), and an Execution type (exec_id string, order_id string, "
            "symbol string, fill_quantity int, fill_price float, executed_at string). "
            "Create both types."
        ),
        agent="oltp",
        expected_tools=["create_dataset"],
        expected_schema={"fields": ["order_id", "symbol", "side", "order_type", "quantity", "price"]},
        tags=["oltp", "create", "multi-entity"],
        difficulty="advanced",
        description="Multi-entity order management schema",
    ),
]


# ── Lakehouse Agent Eval Cases ────────────────────────────────────────

LAKEHOUSE_EVAL_CASES = [
    # Basic: list and describe
    AgentEvalCase(
        input="What tables are in the lakehouse?",
        agent="lakehouse",
        expected_tools=["list_lakehouse_tables"],
        tags=["lakehouse", "list"],
        difficulty="basic",
        description="List lakehouse tables",
    ),
    AgentEvalCase(
        input="Describe the schema of the trades table in the lakehouse",
        agent="lakehouse",
        expected_tools=["describe_lakehouse_table"],
        tags=["lakehouse", "describe"],
        difficulty="basic",
        description="Describe a table schema",
    ),

    # Intermediate: query and ingest
    AgentEvalCase(
        input="Run a SQL query to count the total number of trades per symbol in the lakehouse",
        agent="lakehouse",
        expected_tools=["query_lakehouse"],
        expected_output_contains=["SELECT", "GROUP BY"],
        tags=["lakehouse", "query"],
        difficulty="intermediate",
        description="Aggregation query",
    ),
    AgentEvalCase(
        input=(
            "Ingest the NYC taxi parquet data from "
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet "
            "into a lakehouse table called 'taxi_rides'"
        ),
        agent="lakehouse",
        expected_tools=["ingest_to_lakehouse"],
        tags=["lakehouse", "ingest"],
        difficulty="intermediate",
        description="Parquet URL ingestion",
    ),

    # Advanced: star schema design + datacube
    AgentEvalCase(
        input=(
            "I have a trades dataset with fields: trade_id, symbol, sector, side, quantity, "
            "price, notional, trader, desk, trade_date. Design a proper star schema with "
            "fact and dimension tables for analytical reporting."
        ),
        agent="lakehouse",
        expected_tools=["design_star_schema"],
        expected_output_contains=["fact_", "dim_"],
        tags=["lakehouse", "star_schema", "design"],
        difficulty="advanced",
        description="Star schema design from trade data",
    ),
    AgentEvalCase(
        input=(
            "Build a datacube report on the fact_trades table, grouped by sector and symbol, "
            "with side as the pivot column"
        ),
        agent="lakehouse",
        expected_tools=["build_datacube"],
        tags=["lakehouse", "datacube"],
        difficulty="advanced",
        description="Datacube with group_by + pivot_by",
    ),
]


# ── Query Agent Eval Cases ────────────────────────────────────────────

QUERY_EVAL_CASES = [
    AgentEvalCase(
        input="What is AAPL trading at right now?",
        agent="query",
        expected_tools=["get_md_snapshot"],
        tags=["query", "marketdata"],
        difficulty="basic",
        description="Simple live price query",
    ),
    AgentEvalCase(
        input="What datasets are available on the platform?",
        agent="query",
        expected_tools=["list_all_datasets"],
        tags=["query", "catalog"],
        difficulty="basic",
        description="Platform catalog query",
    ),
    AgentEvalCase(
        input="Show me all trades where the notional value exceeded $1 million",
        agent="query",
        expected_tools=["query_lakehouse"],
        tags=["query", "lakehouse"],
        difficulty="intermediate",
        description="Filtered lakehouse query",
    ),
    AgentEvalCase(
        input="Find research documents about interest rate risk",
        agent="query",
        expected_tools=["search_documents"],
        tags=["query", "media"],
        difficulty="intermediate",
        description="Document search",
    ),
    AgentEvalCase(
        input=(
            "What was AAPL's average daily volume over the last week, "
            "and how does it compare to its 30-day average?"
        ),
        agent="query",
        expected_tools=["query_tsdb"],
        tags=["query", "tsdb", "comparison"],
        difficulty="advanced",
        description="Cross-period TSDB comparison",
    ),
]


# ── Data Science Agent Eval Cases ─────────────────────────────────────

DATASCIENCE_EVAL_CASES = [
    AgentEvalCase(
        input="Compute descriptive statistics for the price column in the trades dataset",
        agent="datascience",
        expected_tools=["compute_statistics"],
        expected_output_contains=["mean", "std"],
        tags=["datascience", "stats"],
        difficulty="basic",
        description="Basic descriptive statistics",
    ),
    AgentEvalCase(
        input="What is the correlation between AAPL and GOOGL prices?",
        agent="datascience",
        expected_tools=["compute_correlation"],
        tags=["datascience", "correlation"],
        difficulty="intermediate",
        description="Pairwise correlation",
    ),
    AgentEvalCase(
        input=(
            "Run a regression to predict trade PnL from quantity, price, and volatility. "
            "Report R², coefficients, and residual analysis."
        ),
        agent="datascience",
        expected_tools=["run_regression"],
        expected_output_contains=["R²", "coefficient"],
        tags=["datascience", "regression"],
        difficulty="advanced",
        description="Multi-factor regression with diagnostics",
    ),
]


# ── Feed Agent Eval Cases ─────────────────────────────────────────────

FEED_EVAL_CASES = [
    AgentEvalCase(
        input="What symbols are currently available in the market data feed?",
        agent="feed",
        expected_tools=["list_md_symbols"],
        tags=["feed", "list"],
        difficulty="basic",
        description="List feed symbols",
    ),
    AgentEvalCase(
        input="Get the current market data snapshot for all equity symbols",
        agent="feed",
        expected_tools=["get_md_snapshot"],
        tags=["feed", "snapshot"],
        difficulty="basic",
        description="Equity snapshot",
    ),
]


# ── Document Agent Eval Cases ─────────────────────────────────────────

DOCUMENT_EVAL_CASES = [
    AgentEvalCase(
        input="What documents are stored in the platform?",
        agent="document",
        expected_tools=["list_documents"],
        tags=["document", "list"],
        difficulty="basic",
        description="List documents",
    ),
    AgentEvalCase(
        input="Search for documents about credit risk management",
        agent="document",
        expected_tools=["search_documents"],
        tags=["document", "search"],
        difficulty="basic",
        description="Document search",
    ),
]


# ── Timeseries Agent Eval Cases ───────────────────────────────────────

TIMESERIES_EVAL_CASES = [
    AgentEvalCase(
        input="Show me the OHLCV bars for AAPL at 1-minute intervals",
        agent="timeseries",
        expected_tools=["get_bars"],
        tags=["timeseries", "bars"],
        difficulty="basic",
        description="Basic OHLCV bars",
    ),
    AgentEvalCase(
        input="Compute the 20-day realized volatility for TSLA",
        agent="timeseries",
        expected_tools=["compute_realized_vol"],
        tags=["timeseries", "vol"],
        difficulty="intermediate",
        description="Realized volatility",
    ),
]


# ── All cases combined ────────────────────────────────────────────────

ALL_EVAL_CASES = (
    OLTP_EVAL_CASES
    + LAKEHOUSE_EVAL_CASES
    + QUERY_EVAL_CASES
    + DATASCIENCE_EVAL_CASES
    + FEED_EVAL_CASES
    + DOCUMENT_EVAL_CASES
    + TIMESERIES_EVAL_CASES
)
