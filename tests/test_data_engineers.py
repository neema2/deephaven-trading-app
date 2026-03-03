"""
Tests for the agents/ package — PlatformAgents.

Pure-Python tests (no services required) for:
- _PlatformContext
- Dynamic Storable creation
- Eval framework (scoring dimensions, artifact collection)
- Tool creation (each agent's tool list)
- PlatformAgents construction

Integration tests (require GEMINI_API_KEY + services) are marked with
@requires_gemini and @requires_services.
"""

import dataclasses
import json
import os
import pytest

from agents._context import _PlatformContext
from agents._oltp import create_oltp_tools, create_oltp_agent, _build_storable_class, _TYPE_MAP
from agents._lakehouse import create_lakehouse_tools, create_lakehouse_agent
from agents._feed import create_feed_tools, create_feed_agent
from agents._timeseries import create_timeseries_tools, create_timeseries_agent
from agents._document import create_document_tools, create_document_agent
from agents._dashboard import create_dashboard_tools, create_dashboard_agent
from agents._query import create_query_tools, create_query_agent
from agents._datascience import create_datascience_tools, create_datascience_agent
from agents._team import PlatformAgents, _AGENT_DESCRIPTIONS
from agents._eval.framework import (
    AgentEval, AgentEvalCase, AgentEvalResult, EvalDimension, EvalPhase,
    _score_tool_selection, _score_output_contains, _score_schema_quality,
    _score_table_creation, _score_metadata_completeness, _score_link_quality,
    _score_query_correctness, DEFAULT_DIMENSIONS,
)
from agents._eval.scorers import (
    score_naming_conventions, score_type_appropriateness,
    score_schema_completeness, score_row_count_preservation,
    score_star_schema_design, score_sql_validity,
)
from agents._eval.judges import (
    DATA_MODEL_RUBRIC, CURATION_QUALITY_RUBRIC, STAR_SCHEMA_RUBRIC,
    METADATA_QUALITY_RUBRIC, ANALYSIS_QUALITY_RUBRIC,
)
from agents._eval.datasets import (
    OLTP_EVAL_CASES, LAKEHOUSE_EVAL_CASES, QUERY_EVAL_CASES,
    DATASCIENCE_EVAL_CASES, ALL_EVAL_CASES,
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
requires_gemini = pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")


# ── PlatformContext ────────────────────────────────────────────────────


class TestPlatformContext:

    def test_create_context(self):
        ctx = _PlatformContext(alias="test")
        assert ctx._store_alias == "test"
        assert ctx._md_alias == "test"
        assert ctx._lakehouse_instance is None
        assert ctx._media_store_instance is None

    def test_storable_type_registry(self):
        ctx = _PlatformContext()
        assert ctx.list_storable_types() == []

        cls = _build_storable_class("Trade", [
            {"name": "symbol", "type": "str"},
            {"name": "price", "type": "float"},
        ])
        ctx.register_storable_type("Trade", cls)
        assert ctx.list_storable_types() == ["Trade"]
        assert ctx.get_storable_type("Trade") is cls
        assert ctx.get_storable_type("Unknown") is None

    def test_validate_no_services(self):
        ctx = _PlatformContext()
        status = ctx.validate()
        assert status["store"] is False
        assert status["lakehouse"] is False
        assert status["media"] is False
        assert status["ai"] is False


# ── Dynamic Storable Creation ──────────────────────────────────────────


class TestDynamicStorable:

    def test_build_basic_class(self):
        cls = _build_storable_class("Order", [
            {"name": "symbol", "type": "str"},
            {"name": "quantity", "type": "int"},
            {"name": "price", "type": "float"},
            {"name": "is_active", "type": "bool"},
        ])
        assert cls.__name__ == "Order"
        assert dataclasses.is_dataclass(cls)
        fields = {f.name: f.type for f in dataclasses.fields(cls) if not f.name.startswith("_")}
        assert fields == {"symbol": str, "quantity": int, "price": float, "is_active": bool}

    def test_build_with_defaults(self):
        cls = _build_storable_class("Instrument", [
            {"name": "ticker", "type": "str", "default": "AAPL"},
            {"name": "market_cap", "type": "float", "default": 0.0},
        ])
        obj = cls()
        assert obj.ticker == "AAPL"
        assert obj.market_cap == 0.0

    def test_build_with_auto_defaults(self):
        cls = _build_storable_class("Simple", [
            {"name": "name", "type": "str"},
            {"name": "count", "type": "int"},
        ])
        obj = cls()
        assert obj.name == ""
        assert obj.count == 0

    def test_type_map_coverage(self):
        assert _TYPE_MAP["str"] is str
        assert _TYPE_MAP["string"] is str
        assert _TYPE_MAP["int"] is int
        assert _TYPE_MAP["integer"] is int
        assert _TYPE_MAP["float"] is float
        assert _TYPE_MAP["double"] is float
        assert _TYPE_MAP["bool"] is bool
        assert _TYPE_MAP["boolean"] is bool

    def test_inherits_storable(self):
        from store.base import Storable
        cls = _build_storable_class("Test", [{"name": "x", "type": "int"}])
        assert issubclass(cls, Storable)


# ── Tool Creation ──────────────────────────────────────────────────────


class TestToolCreation:

    def test_oltp_tools(self):
        ctx = _PlatformContext()
        tools = create_oltp_tools(ctx)
        names = [t.__name__ for t in tools]
        assert "list_storable_types" in names
        assert "describe_type" in names
        assert "create_dataset" in names
        assert "insert_records" in names
        assert "query_dataset" in names
        assert "ingest_from_file" in names
        assert len(tools) == 6

    def test_lakehouse_tools(self):
        ctx = _PlatformContext()
        tools = create_lakehouse_tools(ctx)
        names = [t.__name__ for t in tools]
        assert "list_lakehouse_tables" in names
        assert "describe_lakehouse_table" in names
        assert "design_star_schema" in names
        assert "build_datacube" in names
        assert "query_lakehouse" in names
        assert len(tools) == 7

    def test_feed_tools(self):
        ctx = _PlatformContext()
        tools = create_feed_tools(ctx)
        names = [t.__name__ for t in tools]
        assert "list_md_symbols" in names
        assert "get_md_snapshot" in names
        assert "get_feed_health" in names
        assert "publish_custom_tick" in names
        assert "describe_feed_setup" in names
        assert len(tools) == 5

    def test_timeseries_tools(self):
        ctx = _PlatformContext()
        tools = create_timeseries_tools(ctx)
        names = [t.__name__ for t in tools]
        assert "list_tsdb_series" in names
        assert "get_bars" in names
        assert "get_tick_history" in names
        assert "compute_realized_vol" in names
        assert len(tools) == 6

    def test_document_tools(self):
        ctx = _PlatformContext()
        tools = create_document_tools(ctx)
        names = [t.__name__ for t in tools]
        assert "upload_document" in names
        assert "list_documents" in names
        assert "search_documents" in names
        assert "extract_structured_data" in names
        assert "bulk_upload" in names
        assert "tag_documents" in names
        assert len(tools) == 6

    def test_dashboard_tools(self):
        ctx = _PlatformContext()
        tools = create_dashboard_tools(ctx)
        names = [t.__name__ for t in tools]
        assert "list_ticking_tables" in names
        assert "create_ticking_table" in names
        assert "create_derived_table" in names
        assert "setup_store_bridge" in names
        assert "create_reactive_model" in names
        assert "publish_table" in names
        assert len(tools) == 6

    def test_query_tools(self):
        ctx = _PlatformContext()
        tools = create_query_tools(ctx)
        names = [t.__name__ for t in tools]
        assert "query_store" in names
        assert "query_lakehouse" in names
        assert "query_tsdb" in names
        assert "search_documents" in names
        assert "get_md_snapshot" in names
        assert "list_all_datasets" in names
        assert "describe_dataset" in names
        assert len(tools) == 7

    def test_datascience_tools(self):
        ctx = _PlatformContext()
        tools = create_datascience_tools(ctx)
        names = [t.__name__ for t in tools]
        assert "run_sql_analysis" in names
        assert "compute_statistics" in names
        assert "compute_correlation" in names
        assert "detect_anomalies" in names
        assert "run_regression" in names
        assert "time_series_decompose" in names
        assert "suggest_visualization" in names
        assert len(tools) == 7


# ── Tool Execution (pure Python, no services) ─────────────────────────


class TestToolExecution:

    def test_oltp_list_types_empty(self):
        ctx = _PlatformContext()
        tools = create_oltp_tools(ctx)
        list_fn = [t for t in tools if t.__name__ == "list_storable_types"][0]
        result = json.loads(list_fn())
        assert result == []

    def test_oltp_create_dataset(self):
        ctx = _PlatformContext()
        tools = create_oltp_tools(ctx)
        create_fn = [t for t in tools if t.__name__ == "create_dataset"][0]

        fields = json.dumps([
            {"name": "symbol", "type": "str"},
            {"name": "price", "type": "float"},
            {"name": "quantity", "type": "int"},
        ])
        result = json.loads(create_fn(name="Trade", fields_json=fields))
        assert result["status"] == "created"
        assert result["type_name"] == "Trade"
        assert len(result["fields"]) == 3

        # Verify it's registered
        assert ctx.get_storable_type("Trade") is not None

    def test_oltp_create_duplicate(self):
        ctx = _PlatformContext()
        tools = create_oltp_tools(ctx)
        create_fn = [t for t in tools if t.__name__ == "create_dataset"][0]

        fields = json.dumps([{"name": "x", "type": "int"}])
        create_fn(name="Dup", fields_json=fields)
        result = json.loads(create_fn(name="Dup", fields_json=fields))
        assert "error" in result

    def test_oltp_create_invalid_type(self):
        ctx = _PlatformContext()
        tools = create_oltp_tools(ctx)
        create_fn = [t for t in tools if t.__name__ == "create_dataset"][0]

        fields = json.dumps([{"name": "x", "type": "complex_number"}])
        result = json.loads(create_fn(name="Bad", fields_json=fields))
        assert "error" in result

    def test_oltp_describe_unknown_type(self):
        ctx = _PlatformContext()
        tools = create_oltp_tools(ctx)
        desc_fn = [t for t in tools if t.__name__ == "describe_type"][0]
        result = json.loads(desc_fn(type_name="NonExistent"))
        assert "error" in result

    def test_feed_describe_setup(self):
        ctx = _PlatformContext()
        tools = create_feed_tools(ctx)
        desc_fn = [t for t in tools if t.__name__ == "describe_feed_setup"][0]

        for asset_class in ["equity", "fx", "curve", "custom"]:
            result = json.loads(desc_fn(asset_class=asset_class))
            assert "setup_steps" in result

    def test_dashboard_create_reactive_model(self):
        ctx = _PlatformContext()
        tools = create_dashboard_tools(ctx)
        model_fn = [t for t in tools if t.__name__ == "create_reactive_model"][0]

        fields = json.dumps([
            {"name": "symbol", "type": "str"},
            {"name": "quantity", "type": "int"},
            {"name": "price", "type": "float"},
        ])
        computeds = json.dumps([
            {"name": "market_value", "formula": "self.quantity * self.price", "description": "Position value"},
        ])
        result = json.loads(model_fn(name="Position", fields_json=fields, computeds_json=computeds))
        assert "generated_code" in result
        assert "class Position" in result["generated_code"]
        assert "@computed" in result["generated_code"]

    def test_query_list_all_datasets_empty(self):
        ctx = _PlatformContext()
        tools = create_query_tools(ctx)
        list_fn = [t for t in tools if t.__name__ == "list_all_datasets"][0]
        result = json.loads(list_fn())
        # No services configured, so catalog should be mostly empty
        assert isinstance(result, dict)


# ── Eval Framework ─────────────────────────────────────────────────────


class TestEvalFramework:

    def test_eval_phases(self):
        assert EvalPhase.TOOL_SELECTION.value == 1
        assert EvalPhase.OUTPUT_QUALITY.value == 2
        assert EvalPhase.INTEGRATION.value == 3
        assert EvalPhase.END_TO_END.value == 4

    def test_default_dimensions(self):
        assert len(DEFAULT_DIMENSIONS) >= 7
        names = [d.name for d in DEFAULT_DIMENSIONS]
        assert "tool_selection" in names
        assert "schema_quality" in names
        assert "query_correctness" in names

    def test_eval_case_creation(self):
        case = AgentEvalCase(
            input="Create a trades table",
            agent="oltp",
            expected_tools=["create_dataset"],
            difficulty="basic",
        )
        assert case.agent == "oltp"
        assert case.difficulty == "basic"

    def test_score_tool_selection_perfect(self):
        case = AgentEvalCase(expected_tools=["create_dataset"])
        artifacts = {"actual_tools": ["create_dataset", "insert_records"]}
        assert _score_tool_selection(case, artifacts) == 1.0

    def test_score_tool_selection_miss(self):
        case = AgentEvalCase(expected_tools=["create_dataset", "insert_records"])
        artifacts = {"actual_tools": ["list_storable_types"]}
        assert _score_tool_selection(case, artifacts) == 0.0

    def test_score_tool_selection_partial(self):
        case = AgentEvalCase(expected_tools=["create_dataset", "insert_records"])
        artifacts = {"actual_tools": ["create_dataset"]}
        assert _score_tool_selection(case, artifacts) == 0.5

    def test_score_tool_selection_empty_expected(self):
        case = AgentEvalCase(expected_tools=[])
        artifacts = {"actual_tools": ["anything"]}
        assert _score_tool_selection(case, artifacts) == 1.0

    def test_score_output_contains(self):
        case = AgentEvalCase(expected_output_contains=["symbol", "price"])
        artifacts = {"actual_output": "Created table with symbol and price fields"}
        assert _score_output_contains(case, artifacts) == 1.0

    def test_score_output_contains_partial(self):
        case = AgentEvalCase(expected_output_contains=["symbol", "price", "volume"])
        artifacts = {"actual_output": "Created symbol and price"}
        score = _score_output_contains(case, artifacts)
        assert abs(score - 2/3) < 0.01

    def test_score_schema_quality_match(self):
        case = AgentEvalCase(expected_schema={"fields": ["symbol", "price"]})
        artifacts = {"created_schema": {"fields": [{"name": "symbol"}, {"name": "price"}]}}
        assert _score_schema_quality(case, artifacts) == 1.0

    def test_score_schema_quality_partial(self):
        case = AgentEvalCase(expected_schema={"fields": ["a", "b", "c"]})
        artifacts = {"created_schema": {"fields": [{"name": "a"}, {"name": "b"}]}}
        score = _score_schema_quality(case, artifacts)
        assert abs(score - 2/3) < 0.01

    def test_score_table_creation(self):
        case = AgentEvalCase(expected_tables=["fact_trades", "dim_instrument"])
        artifacts = {"created_tables": ["fact_trades", "dim_instrument"]}
        assert _score_table_creation(case, artifacts) == 1.0

    def test_score_metadata_completeness(self):
        case = AgentEvalCase(expected_metadata={"has_description": True, "tags": ["trades"]})
        artifacts = {"metadata": {"has_description": True, "tags": ["trades", "finance"]}}
        assert _score_metadata_completeness(case, artifacts) == 1.0

    def test_score_query_correctness_exact(self):
        case = AgentEvalCase(expected_result=42)
        artifacts = {"query_result": 42}
        assert _score_query_correctness(case, artifacts) == 1.0

    def test_score_query_correctness_approx(self):
        case = AgentEvalCase(expected_result=100.0)
        artifacts = {"query_result": 95.0}
        score = _score_query_correctness(case, artifacts)
        assert score == 0.95

    def test_active_dimensions_by_phase(self):
        evaluator = AgentEval(agents={}, max_phase=EvalPhase.TOOL_SELECTION)
        active = evaluator.active_dimensions
        assert all(d.phase == EvalPhase.TOOL_SELECTION for d in active)

        evaluator2 = AgentEval(agents={}, max_phase=EvalPhase.END_TO_END)
        active2 = evaluator2.active_dimensions
        assert len(active2) > len(active)


# ── Eval Scorers ───────────────────────────────────────────────────────


class TestEvalScorers:

    def test_naming_conventions_snake_case(self):
        case = AgentEvalCase()
        artifacts = {"created_schema": {
            "type_name": "Trade",
            "fields": [{"name": "order_id"}, {"name": "trade_price"}, {"name": "is_active"}],
        }}
        score = score_naming_conventions(case, artifacts)
        assert score == 1.0

    def test_naming_conventions_bad(self):
        case = AgentEvalCase()
        artifacts = {"created_schema": {
            "type_name": "trade",  # should be PascalCase
            "fields": [{"name": "OrderId"}, {"name": "PRICE"}],
        }}
        score = score_naming_conventions(case, artifacts)
        assert score < 0.5

    def test_type_appropriateness(self):
        case = AgentEvalCase()
        artifacts = {"created_schema": {
            "fields": [
                {"name": "order_id", "type": "int"},
                {"name": "trade_price", "type": "float"},
                {"name": "is_active", "type": "bool"},
                {"name": "symbol_name", "type": "str"},
            ]
        }}
        score = score_type_appropriateness(case, artifacts)
        assert score == 1.0

    def test_schema_completeness(self):
        case = AgentEvalCase()
        assert score_schema_completeness(case, {"created_schema": {"fields": []}}) == 0.0
        assert score_schema_completeness(case, {"created_schema": {"fields": ["a"]}}) == 0.3
        assert score_schema_completeness(case, {"created_schema": {"fields": ["a", "b", "c", "d", "e"]}}) == 1.0

    def test_sql_validity(self):
        case = AgentEvalCase()
        assert score_sql_validity(case, {"generated_sql": "SELECT * FROM trades"}) == 1.0
        assert score_sql_validity(case, {"generated_sql": "DELETE FROM trades"}) == 0.3
        assert score_sql_validity(case, {"generated_sql": "SELECT symbol ("}) == 0.5  # unbalanced parens

    def test_star_schema_design_good(self):
        case = AgentEvalCase()
        artifacts = {"star_schema_design": {
            "fact_tables": [{"name": "fact_trades", "columns": [{"role": "measure", "type": "float"}]}],
            "dimension_tables": [{"name": "dim_instrument", "columns": [{"role": "attribute", "type": "str"}]}],
            "relationships": [{"fact": "fact_trades", "dimension": "dim_instrument", "join_key": "symbol"}],
        }}
        score = score_star_schema_design(case, artifacts)
        assert score >= 0.8

    def test_star_schema_design_empty(self):
        case = AgentEvalCase()
        artifacts = {"star_schema_design": {"fact_tables": [], "dimension_tables": [], "relationships": []}}
        score = score_star_schema_design(case, artifacts)
        assert score == 0.0


# ── Eval Datasets ──────────────────────────────────────────────────────


class TestEvalDatasets:

    def test_oltp_cases(self):
        assert len(OLTP_EVAL_CASES) >= 5
        for case in OLTP_EVAL_CASES:
            assert case.agent == "oltp"
            assert case.input

    def test_lakehouse_cases(self):
        assert len(LAKEHOUSE_EVAL_CASES) >= 4
        for case in LAKEHOUSE_EVAL_CASES:
            assert case.agent == "lakehouse"

    def test_query_cases(self):
        assert len(QUERY_EVAL_CASES) >= 4
        for case in QUERY_EVAL_CASES:
            assert case.agent == "query"

    def test_datascience_cases(self):
        assert len(DATASCIENCE_EVAL_CASES) >= 3
        for case in DATASCIENCE_EVAL_CASES:
            assert case.agent == "datascience"

    def test_all_cases_have_tags(self):
        for case in ALL_EVAL_CASES:
            assert case.tags, f"Case missing tags: {case.input[:40]}"

    def test_all_cases_have_difficulty(self):
        for case in ALL_EVAL_CASES:
            assert case.difficulty in ("basic", "intermediate", "advanced")

    def test_total_cases(self):
        assert len(ALL_EVAL_CASES) >= 20


# ── Eval Judges ────────────────────────────────────────────────────────


class TestEvalJudges:

    def test_rubrics_are_strings(self):
        for rubric in [DATA_MODEL_RUBRIC, CURATION_QUALITY_RUBRIC,
                       STAR_SCHEMA_RUBRIC, METADATA_QUALITY_RUBRIC,
                       ANALYSIS_QUALITY_RUBRIC]:
            assert isinstance(rubric, str)
            assert len(rubric) > 50


# ── DataEngineerTeam ───────────────────────────────────────────────────


class TestPlatformAgents:

    def test_agent_descriptions(self):
        assert len(_AGENT_DESCRIPTIONS) == 8
        for name in ["oltp", "lakehouse", "feed", "timeseries",
                      "document", "dashboard", "query", "quant"]:
            assert name in _AGENT_DESCRIPTIONS
            assert len(_AGENT_DESCRIPTIONS[name]) > 20

    @requires_gemini
    def test_team_construction(self):
        team = PlatformAgents()
        assert len(team) == 8
        assert "oltp" in team
        assert "query" in team
        assert "quant" in team

    @requires_gemini
    def test_team_subset(self):
        team = PlatformAgents(agents=["oltp", "lakehouse"])
        assert len(team) == 2
        assert "oltp" in team
        assert "lakehouse" in team
        assert "feed" not in team

    @requires_gemini
    def test_typed_properties(self):
        team = PlatformAgents(agents=["oltp", "quant"])
        assert team.oltp is not None
        assert team.quant is not None
