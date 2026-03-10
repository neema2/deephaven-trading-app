"""
Integration tests for structured extraction — real Gemini API calls.

Requires GEMINI_API_KEY env var. Tests skip if not set.
"""

import os
from dataclasses import dataclass

import pytest
from ai._extraction import extract
from ai._types import ExtractionResult

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
requires_gemini = pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")


@pytest.fixture(scope="module")
def llm():
    if not GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not set")
    from ai._llm import GeminiLLM
    return GeminiLLM(api_key=GEMINI_API_KEY)


# ── Tests ─────────────────────────────────────────────────────────────────


@requires_gemini
class TestStructuredExtraction:

    def test_extract_basic(self, llm):
        """Extract name and age from text."""
        result = extract(
            llm,
            text="John Smith is 35 years old and lives in New York.",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person's full name"},
                    "age": {"type": "integer", "description": "Person's age"},
                    "city": {"type": "string", "description": "City of residence"},
                },
                "required": ["name", "age"],
            },
        )

        assert isinstance(result, ExtractionResult)
        assert isinstance(result.data, dict)
        assert result.data["name"] == "John Smith"
        assert result.data["age"] == 35
        assert result.data["city"] == "New York"

    def test_extract_financial(self, llm):
        """Extract financial data from text."""
        result = extract(
            llm,
            text="Apple Inc. (AAPL) closed at $178.50 on March 15, 2024, up 2.3% from the previous session.",
            schema={
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "ticker": {"type": "string"},
                    "price": {"type": "number"},
                    "date": {"type": "string"},
                    "change_pct": {"type": "number"},
                },
                "required": ["ticker", "price"],
            },
        )

        assert result.data["ticker"] == "AAPL"
        assert result.data["price"] == 178.50
        assert result.data["change_pct"] == 2.3

    def test_extract_with_dataclass(self, llm):
        """Extract into a Python dataclass."""

        @dataclass
        class Person:
            name: str
            age: int
            occupation: str = ""

        result = extract(
            llm,
            text="Dr. Sarah Chen, 42, is a quantum physicist at MIT.",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "occupation": {"type": "string"},
                },
                "required": ["name", "age"],
            },
            model_class=Person,
        )

        assert isinstance(result.data, Person)
        assert "Sarah Chen" in result.data.name
        assert result.data.age == 42
        assert "physicist" in result.data.occupation.lower()

    def test_extract_list(self, llm):
        """Extract a list of items from text."""
        result = extract(
            llm,
            text="The portfolio contains Apple (AAPL), Microsoft (MSFT), and Google (GOOGL).",
            schema={
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "company": {"type": "string"},
                                "ticker": {"type": "string"},
                            },
                        },
                    },
                },
            },
        )

        assert "holdings" in result.data
        holdings = result.data["holdings"]
        assert len(holdings) == 3
        tickers = [h["ticker"] for h in holdings]
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

    def test_extract_null_fields(self, llm):
        """Fields not found in text should be null."""
        result = extract(
            llm,
            text="Alice works at a tech company.",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
            },
        )

        assert result.data["name"] == "Alice"
        assert result.data.get("age") is None
        assert result.data.get("email") is None
