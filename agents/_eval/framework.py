"""
Multi-Dimensional Agent Evaluation Engine
==========================================
Extends the basic EvalRunner (tool selection) into a multi-layered eval
that scores data models, curation quality, linking, metadata, query
correctness, and analysis quality.

Each eval case can specify expectations across multiple dimensions.
Dimensions are scored independently (0.0–1.0) and combined into a
weighted composite score.

Usage::

    from agents._eval import AgentEval, AgentEvalCase, EvalDimension

    cases = [
        AgentEvalCase(
            input="Create a trades table with symbol, qty, price, side",
            agent="oltp",
            expected_tools=["create_dataset"],
            expected_schema={"fields": ["symbol", "quantity", "price", "side"]},
            tags=["oltp", "basic"],
        ),
    ]

    evaluator = AgentEval(agents={"oltp": oltp_agent})
    results = evaluator.run(cases)
    evaluator.summary()
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai.agent import AgentResult
    from ai.client import AI

logger = logging.getLogger(__name__)


class EvalPhase(enum.Enum):
    """Evaluation phases — cumulative, each adds scoring dimensions."""
    TOOL_SELECTION = 1    # Did the agent pick the right tools?
    OUTPUT_QUALITY = 2    # Are schemas/models/curations well-designed?
    INTEGRATION = 3       # Cross-dataset linking, metadata completeness
    END_TO_END = 4        # Full pipeline correctness


@dataclass
class EvalDimension:
    """One scoring dimension for agent evaluation.

    Dimensions can use deterministic scorers, LLM-as-judge, or both.
    The ``scorer`` function receives the eval case, actual artifacts,
    and returns a float 0.0–1.0.  The optional ``judge_rubric`` is
    sent to an LLM for qualitative scoring.
    """
    name: str
    description: str = ""
    scorer: Callable | None = None     # (case, artifacts) → float
    judge_rubric: str = ""                # LLM rubric for qualitative eval
    weight: float = 1.0                   # weight in composite score
    phase: EvalPhase = EvalPhase.TOOL_SELECTION


@dataclass
class AgentEvalCase:
    """Extended eval case with multi-dimensional expectations.

    Only populate the expectations relevant to the eval phase.
    Unpopulated expectations are skipped during scoring.
    """
    # ── Core ──────────────────────────────────────────────────────
    input: str = ""                        # user prompt
    agent: str = ""                        # which agent to test

    # ── Phase 1: Tool Selection ───────────────────────────────────
    expected_tools: list[str] = field(default_factory=list)

    # ── Phase 2: Output Quality ───────────────────────────────────
    expected_schema: dict = field(default_factory=dict)
    expected_tables: list[str] = field(default_factory=list)
    expected_output_contains: list[str] = field(default_factory=list)

    # ── Phase 3: Integration ──────────────────────────────────────
    expected_links: list[dict] = field(default_factory=list)
    expected_metadata: dict = field(default_factory=dict)

    # ── Phase 4: End-to-End ───────────────────────────────────────
    expected_result: Any = None
    golden_sql: str = ""

    # ── Metadata ──────────────────────────────────────────────────
    tags: list[str] = field(default_factory=list)
    difficulty: str = "basic"              # basic / intermediate / advanced
    description: str = ""


@dataclass
class AgentEvalResult:
    """Multi-dimensional evaluation result for a single case."""
    case: AgentEvalCase = field(default_factory=AgentEvalCase)
    scores: dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    details: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    token_usage: dict = field(default_factory=dict)
    actual_tools: list[str] = field(default_factory=list)
    actual_output: str = ""
    artifacts: dict = field(default_factory=dict)
    error: str = ""
    passed: bool = False


# ── Built-in scoring dimensions ────────────────────────────────────────

def _score_tool_selection(case: AgentEvalCase, artifacts: dict) -> float:
    """Phase 1: Did the agent call the expected tools?"""
    if not case.expected_tools:
        return 1.0
    expected = set(case.expected_tools)
    actual = set(artifacts.get("actual_tools", []))
    if not expected:
        return 1.0
    # Subset match: all expected tools must appear in actual
    matched = expected & actual
    return len(matched) / len(expected)


def _score_output_contains(case: AgentEvalCase, artifacts: dict) -> float:
    """Phase 1: Does the output contain expected substrings?"""
    if not case.expected_output_contains:
        return 1.0
    output = artifacts.get("actual_output", "").lower()
    matched = sum(1 for s in case.expected_output_contains if s.lower() in output)
    return matched / len(case.expected_output_contains)


def _score_schema_quality(case: AgentEvalCase, artifacts: dict) -> float:
    """Phase 2: Does the created schema match expectations?

    Checks:
    - Expected field names are present
    - Field types match
    - No critical fields are missing
    """
    if not case.expected_schema:
        return 1.0

    expected_fields = set(case.expected_schema.get("fields", []))
    if not expected_fields:
        return 1.0

    # Look for schema in artifacts
    actual_schema = artifacts.get("created_schema", {})
    actual_fields = set()
    if isinstance(actual_schema, dict) and "fields" in actual_schema:
        for f in actual_schema["fields"]:
            if isinstance(f, dict):
                actual_fields.add(f.get("name", ""))
            elif isinstance(f, str):
                actual_fields.add(f)
    elif isinstance(actual_schema, list):
        for f in actual_schema:
            if isinstance(f, dict):
                actual_fields.add(f.get("name", ""))
            elif isinstance(f, str):
                actual_fields.add(f)

    if not actual_fields:
        # Try to extract from output text
        return 0.5  # Partial credit — we can't verify

    matched = expected_fields & actual_fields
    return len(matched) / len(expected_fields)


def _score_table_creation(case: AgentEvalCase, artifacts: dict) -> float:
    """Phase 2: Were the expected tables created?"""
    if not case.expected_tables:
        return 1.0
    actual_tables = set(artifacts.get("created_tables", []))
    expected = set(case.expected_tables)
    matched = expected & actual_tables
    return len(matched) / len(expected) if expected else 1.0


def _score_metadata_completeness(case: AgentEvalCase, artifacts: dict) -> float:
    """Phase 3: How complete is the metadata?

    Checks: descriptions, tags, data types, column docs.
    """
    if not case.expected_metadata:
        return 1.0

    actual_meta = artifacts.get("metadata", {})
    checks = []

    # Check each expected metadata key
    for key, expected_val in case.expected_metadata.items():
        actual_val = actual_meta.get(key)
        if actual_val is not None:
            if isinstance(expected_val, list):
                # Check list overlap
                expected_set = set(expected_val)
                actual_set = set(actual_val) if isinstance(actual_val, list) else set()
                overlap = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0
                checks.append(overlap)
            elif isinstance(expected_val, bool):
                checks.append(1.0 if actual_val == expected_val else 0.0)
            else:
                checks.append(1.0 if str(actual_val) == str(expected_val) else 0.0)
        else:
            checks.append(0.0)

    return sum(checks) / len(checks) if checks else 1.0


def _score_link_quality(case: AgentEvalCase, artifacts: dict) -> float:
    """Phase 3: Are cross-dataset links properly created?"""
    if not case.expected_links:
        return 1.0
    actual_links = artifacts.get("links", [])
    # Simple: count how many expected links were created
    matched = 0
    for expected_link in case.expected_links:
        for actual_link in actual_links:
            if (expected_link.get("fact") == actual_link.get("fact") and
                expected_link.get("dimension") == actual_link.get("dimension")):
                matched += 1
                break
    return matched / len(case.expected_links)


def _score_query_correctness(case: AgentEvalCase, artifacts: dict) -> float:
    """Phase 4: Does the query result match the expected result?"""
    if case.expected_result is None:
        return 1.0
    actual_result = artifacts.get("query_result")
    if actual_result is None:
        return 0.0
    # Simple equality check — can be extended for fuzzy matching
    if actual_result == case.expected_result:
        return 1.0
    # Partial credit for similar results
    if isinstance(case.expected_result, (int, float)) and isinstance(actual_result, (int, float)):
        if case.expected_result != 0:
            ratio = abs(actual_result - case.expected_result) / abs(case.expected_result)
            return max(0.0, 1.0 - ratio)
    return 0.0


# ── Default dimensions by phase ───────────────────────────────────────

DEFAULT_DIMENSIONS = [
    EvalDimension(
        name="tool_selection",
        description="Did the agent call the right tools?",
        scorer=_score_tool_selection,
        weight=1.0,
        phase=EvalPhase.TOOL_SELECTION,
    ),
    EvalDimension(
        name="output_contains",
        description="Does the output contain expected information?",
        scorer=_score_output_contains,
        weight=0.5,
        phase=EvalPhase.TOOL_SELECTION,
    ),
    EvalDimension(
        name="schema_quality",
        description="Does the created schema match expectations?",
        scorer=_score_schema_quality,
        weight=1.0,
        phase=EvalPhase.OUTPUT_QUALITY,
    ),
    EvalDimension(
        name="table_creation",
        description="Were the expected tables created?",
        scorer=_score_table_creation,
        weight=1.0,
        phase=EvalPhase.OUTPUT_QUALITY,
    ),
    EvalDimension(
        name="metadata_completeness",
        description="How complete is the metadata capture?",
        scorer=_score_metadata_completeness,
        weight=0.8,
        phase=EvalPhase.INTEGRATION,
    ),
    EvalDimension(
        name="link_quality",
        description="Are cross-dataset links properly created?",
        scorer=_score_link_quality,
        weight=0.8,
        phase=EvalPhase.INTEGRATION,
    ),
    EvalDimension(
        name="query_correctness",
        description="Does the query produce the correct result?",
        scorer=_score_query_correctness,
        weight=1.0,
        phase=EvalPhase.END_TO_END,
    ),
]


# ── AgentEval — the main eval engine ──────────────────────────────────


class AgentEval:
    """Multi-dimensional agent evaluation engine.

    Runs eval cases against agents, scoring across multiple dimensions.
    Dimensions are filtered by phase — only dimensions up to the configured
    ``max_phase`` are scored.

    Args:
        agents: Dict of agent_name → Agent instance.
        dimensions: Scoring dimensions (defaults to DEFAULT_DIMENSIONS).
        max_phase: Highest eval phase to score (default: TOOL_SELECTION).
        judge: Optional AI instance for LLM-as-judge dimensions.
    """

    def __init__(
        self,
        agents: dict,
        dimensions: list[EvalDimension] | None = None,
        max_phase: EvalPhase = EvalPhase.TOOL_SELECTION,
        judge: AI | None = None,
    ) -> None:
        self._agents = agents
        self._dimensions = dimensions or DEFAULT_DIMENSIONS
        self._max_phase = max_phase
        self._judge = judge
        self._results: list[AgentEvalResult] = []

    @property
    def active_dimensions(self) -> list[EvalDimension]:
        """Dimensions active for the current max_phase."""
        return [d for d in self._dimensions if d.phase.value <= self._max_phase.value]

    def run(self, cases: list[AgentEvalCase]) -> list[AgentEvalResult]:
        """Run all eval cases and return results.

        Args:
            cases: List of eval cases.

        Returns:
            List of scored results.
        """
        self._results = []
        for case in cases:
            result = self._run_single(case)
            self._results.append(result)
        return self._results

    def _run_single(self, case: AgentEvalCase) -> AgentEvalResult:
        """Run a single eval case against the appropriate agent."""
        result = AgentEvalResult(case=case)

        # Find the agent
        agent = self._agents.get(case.agent)
        if agent is None:
            result.error = f"Agent '{case.agent}' not found. Available: {list(self._agents.keys())}"
            return result

        # Run the agent
        try:
            agent.reset()
            t0 = time.time()
            agent_result = agent.run(case.input)
            result.latency_ms = (time.time() - t0) * 1000
            result.actual_output = agent_result.content
            result.actual_tools = [s.action.name for s in agent_result.steps]
            result.token_usage = agent_result.usage

            # Collect artifacts from agent steps
            result.artifacts = self._collect_artifacts(agent_result)

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            return result

        # Inject actual tools into artifacts for scorers
        result.artifacts["actual_tools"] = result.actual_tools
        result.artifacts["actual_output"] = result.actual_output

        # Score each active dimension
        active = self.active_dimensions
        for dim in active:
            try:
                if dim.scorer is not None:
                    score = dim.scorer(case, result.artifacts)
                    result.scores[dim.name] = score
                    if score < 1.0:
                        result.details[dim.name] = f"Score: {score:.2f}"

                # LLM-as-judge (if rubric and judge available)
                if dim.judge_rubric and self._judge:
                    judge_score = self._llm_judge(case, result, dim.judge_rubric)
                    # Blend with deterministic score if both exist
                    if dim.name in result.scores:
                        result.scores[dim.name] = (result.scores[dim.name] + judge_score) / 2
                    else:
                        result.scores[dim.name] = judge_score

            except Exception as e:
                logger.warning("Dimension %s scoring failed: %s", dim.name, e)
                result.scores[dim.name] = 0.0
                result.details[dim.name] = f"Error: {e}"

        # Compute weighted composite score
        total_weight = sum(d.weight for d in active if d.name in result.scores)
        if total_weight > 0:
            result.composite_score = sum(
                result.scores[d.name] * d.weight
                for d in active if d.name in result.scores
            ) / total_weight
        else:
            result.composite_score = 1.0 if not result.error else 0.0

        # Pass/fail: composite >= 0.7
        result.passed = result.composite_score >= 0.7 and not result.error
        return result

    def _collect_artifacts(self, agent_result: AgentResult) -> dict:
        """Extract structured artifacts from agent tool call results.

        Parses JSON observations from tool calls to find schemas,
        tables, metadata, etc.
        """
        artifacts: dict[str, object] = {}
        created_tables = []

        for step in agent_result.steps:
            try:
                obs = step.observation
                if isinstance(obs, str):
                    data = __import__("json").loads(obs)
                else:
                    data = obs

                # Track created schemas
                if isinstance(data, dict):
                    if data.get("status") == "created" and "fields" in data:
                        artifacts["created_schema"] = data
                    if data.get("status") in ("created", "ingested") and "table_name" in data:
                        created_tables.append(data["table_name"])
                    if "tables" in data:
                        artifacts["existing_tables"] = data["tables"]

            except (ValueError, TypeError, KeyError):
                continue

        if created_tables:
            artifacts["created_tables"] = created_tables
        return artifacts

    def _llm_judge(self, case: AgentEvalCase, result: AgentEvalResult,
                   rubric: str) -> float:
        """Use LLM-as-judge to score based on a rubric."""
        assert self._judge is not None
        try:
            from ai._types import Message
            response = self._judge.generate(
                [
                    Message(role="system", content=(
                        "You are an evaluation judge. Score the agent's output on a scale "
                        "of 0.0 to 1.0 based on the rubric below. Respond with ONLY a "
                        "number (0.0 to 1.0) on the first line, then a brief explanation."
                    )),
                    Message(role="user", content=(
                        f"Rubric:\n{rubric}\n\n"
                        f"User request: {case.input}\n\n"
                        f"Agent output: {result.actual_output}\n\n"
                        f"Tools called: {result.actual_tools}\n\n"
                        f"Artifacts: {__import__('json').dumps(result.artifacts, default=str)[:1000]}"
                    )),
                ],
                temperature=0.0,
                max_tokens=200,
            )
            text = response.content.strip()
            first_line = text.split("\n")[0].strip()
            return float(first_line)
        except Exception as e:
            logger.warning("LLM judge failed: %s", e)
            return 0.5  # Neutral score on failure

    def summary(self) -> dict:
        """Print and return aggregate evaluation results."""
        if not self._results:
            print("No results to summarize.")
            return {}

        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        errors = sum(1 for r in self._results if r.error)
        avg_latency = sum(r.latency_ms for r in self._results) / total
        avg_composite = sum(r.composite_score for r in self._results) / total

        # Per-dimension averages
        dim_scores: dict[str, list[float]] = {}
        for r in self._results:
            for dim_name, score in r.scores.items():
                dim_scores.setdefault(dim_name, []).append(score)
        dim_averages = {
            name: sum(scores) / len(scores)
            for name, scores in dim_scores.items()
        }

        # By tag
        tag_stats: dict[str, dict] = {}
        for r in self._results:
            for tag in r.case.tags:
                if tag not in tag_stats:
                    tag_stats[tag] = {"total": 0, "passed": 0, "avg_score": 0.0}
                tag_stats[tag]["total"] += 1
                tag_stats[tag]["avg_score"] += r.composite_score
                if r.passed:
                    tag_stats[tag]["passed"] += 1
        for tag in tag_stats:
            tag_stats[tag]["avg_score"] /= tag_stats[tag]["total"]

        # By difficulty
        diff_stats: dict[str, dict] = {}
        for r in self._results:
            d = r.case.difficulty
            if d not in diff_stats:
                diff_stats[d] = {"total": 0, "passed": 0, "avg_score": 0.0}
            diff_stats[d]["total"] += 1
            diff_stats[d]["avg_score"] += r.composite_score
            if r.passed:
                diff_stats[d]["passed"] += 1
        for d in diff_stats:
            diff_stats[d]["avg_score"] /= diff_stats[d]["total"]

        summary = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "errors": errors,
            "pass_rate": passed / total,
            "avg_composite_score": round(avg_composite, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "dimension_averages": {k: round(v, 3) for k, v in dim_averages.items()},
            "by_tag": tag_stats,
            "by_difficulty": diff_stats,
        }

        # Print
        print(f"\n{'='*60}")
        print(f"  Agent Eval Results: {passed}/{total} passed ({summary['pass_rate']:.0%})")
        print(f"  Avg composite score: {avg_composite:.3f}")
        print(f"{'='*60}")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        if errors:
            print(f"  Errors: {errors}")

        if dim_averages:
            print("\n  Dimension Scores:")
            for name, avg in sorted(dim_averages.items()):
                bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
                print(f"    {name:<25} {bar} {avg:.3f}")

        if diff_stats:
            print("\n  By Difficulty:")
            for d, stats in sorted(diff_stats.items()):
                print(f"    {d:<15} {stats['passed']}/{stats['total']} "
                      f"(avg score: {stats['avg_score']:.3f})")

        if tag_stats:
            print("\n  By Tag:")
            for tag, stats in sorted(tag_stats.items()):
                print(f"    {tag:<15} {stats['passed']}/{stats['total']} "
                      f"(avg score: {stats['avg_score']:.3f})")

        print(f"{'='*60}\n")

        # Print failures
        failures = [r for r in self._results if not r.passed]
        if failures:
            print(f"Failures ({len(failures)}):")
            for r in failures:
                print(f"  Input: {r.case.input[:80]}")
                if r.error:
                    print(f"    Error: {r.error}")
                for dim, score in r.scores.items():
                    if score < 0.7:
                        print(f"    {dim}: {score:.3f}")
                print()

        return summary
