"""
LLM-as-Judge Rubrics
=====================
Pre-defined rubrics for qualitative evaluation of agent outputs.
Each rubric is a string template that gets sent to an LLM judge along
with the agent's output and artifacts.

Usage::

    from agents._eval.judges import DATA_MODEL_RUBRIC
    dim = EvalDimension(name="data_model", judge_rubric=DATA_MODEL_RUBRIC)
"""

# ── Phase 2: Output Quality ───────────────────────────────────────────

DATA_MODEL_RUBRIC = """\
Score the quality of this data model (schema design) from 0.0 to 1.0.

Criteria:
- Field names are clear, descriptive, and follow snake_case convention (0.2)
- Data types are appropriate for each field (0.2)
- The schema has reasonable completeness — no obvious missing fields (0.2)
- Normalization is appropriate — not too denormalized, not over-normalized (0.2)
- Primary key / identity fields are present where needed (0.2)

A score of 1.0 means the schema is production-ready. A score of 0.5 means
it's functional but needs improvement. A score below 0.3 means it has
significant design issues.
"""

CURATION_QUALITY_RUBRIC = """\
Score the quality of this data curation (OLTP → analytical tables) from 0.0 to 1.0.

Criteria:
- Fact tables have the correct grain (one row per event/measurement) (0.2)
- Dimensions are properly separated from facts (0.2)
- Measures are numeric and aggregatable (0.15)
- No data loss during transformation (row counts preserved where appropriate) (0.15)
- Foreign keys properly link facts to dimensions (0.15)
- Naming follows star-schema conventions (fact_*, dim_*) (0.15)

A score of 1.0 means the curation follows Kimball star-schema best practices.
"""

STAR_SCHEMA_RUBRIC = """\
Score this star schema design from 0.0 to 1.0.

Criteria:
- Clear separation of facts (events with measures) and dimensions (descriptive) (0.25)
- Proper grain defined for each fact table (0.2)
- Surrogate keys used where appropriate (0.1)
- Relationships (foreign keys) correctly link facts to dimensions (0.2)
- Naming conventions followed (fact_*, dim_*) (0.1)
- No unnecessary denormalization (0.15)
"""

# ── Phase 3: Integration Quality ──────────────────────────────────────

METADATA_QUALITY_RUBRIC = """\
Score the metadata capture quality from 0.0 to 1.0.

Criteria:
- Table/dataset descriptions are present and informative (0.25)
- Column descriptions explain the meaning of each field (0.25)
- Tags are applied for discoverability (0.15)
- Data types are documented accurately (0.15)
- Lineage information captured (source → target mapping) (0.2)
"""

CROSS_DATASET_LINKING_RUBRIC = """\
Score the quality of cross-dataset linking from 0.0 to 1.0.

Criteria:
- Foreign keys are correctly identified between related datasets (0.3)
- Join paths are valid and produce correct results (0.3)
- Referential integrity is maintained (no orphan keys) (0.2)
- Link documentation explains the business relationship (0.2)
"""

# ── Phase 4: End-to-End ───────────────────────────────────────────────

QUERY_CORRECTNESS_RUBRIC = """\
Score the correctness of this query result from 0.0 to 1.0.

Criteria:
- The query answers the user's question accurately (0.4)
- The SQL is syntactically correct and efficient (0.2)
- The result set has the expected columns and row count (0.2)
- Edge cases are handled (nulls, empty results, etc.) (0.2)
"""

ANALYSIS_QUALITY_RUBRIC = """\
Score the quality of this data analysis from 0.0 to 1.0.

Criteria:
- Statistical methodology is appropriate for the question (0.25)
- Results are correctly computed (0.25)
- Interpretation is accurate and well-explained (0.25)
- Limitations and caveats are noted where appropriate (0.15)
- Visualization recommendations (if any) are appropriate (0.1)
"""

PIPELINE_RUBRIC = """\
Score the end-to-end data pipeline from 0.0 to 1.0.

Criteria:
- Data flows correctly from source to destination (0.3)
- No data loss or corruption during the pipeline (0.25)
- Pipeline handles errors gracefully (0.15)
- Final output is usable and answers the original question (0.3)
"""
