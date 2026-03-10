"""
Integration tests for embedding providers — real Gemini API calls.

Requires GEMINI_API_KEY env var. Tests skip if not set.
"""

import math
import os

import pytest
from ai._embeddings import EmbeddingProvider, GeminiEmbeddings

# ── Skip if no API key ───────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
requires_gemini = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="GEMINI_API_KEY not set",
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gemini():
    """Create a GeminiEmbeddings instance (768-dim)."""
    if not GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not set")
    return GeminiEmbeddings(api_key=GEMINI_API_KEY, dimension=768)


# ── Helper ────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Tests ─────────────────────────────────────────────────────────────────


@requires_gemini
class TestGeminiEmbeddings:

    def test_provider_abc(self, gemini):
        """GeminiEmbeddings should be an instance of EmbeddingProvider."""
        assert isinstance(gemini, EmbeddingProvider)

    def test_dimension(self, gemini):
        """dimension property should return configured value."""
        assert gemini.dimension == 768

    def test_embed_single(self, gemini):
        """Embed one text, verify returns 768-dim float vector."""
        vectors = gemini.embed(["The quick brown fox jumps over the lazy dog."])
        assert len(vectors) == 1
        assert len(vectors[0]) == 768
        assert all(isinstance(v, float) for v in vectors[0])

    def test_embed_batch(self, gemini):
        """Embed multiple texts in one call, verify correct count and dims."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "The stock market closed higher today.",
            "Python is a popular programming language.",
        ]
        vectors = gemini.embed(texts)
        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 768

    def test_embed_query(self, gemini):
        """Embed a query, verify dimension."""
        vec = gemini.embed_query("What is machine learning?")
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)

    def test_embed_empty_list(self, gemini):
        """Embedding empty list should return empty list."""
        vectors = gemini.embed([])
        assert vectors == []

    def test_similarity_meaningful(self, gemini):
        """Similar texts should have higher cosine similarity than dissimilar ones."""
        texts = [
            "The cat sat on the mat.",           # 0: about cats
            "A kitten was resting on the rug.",   # 1: also about cats (similar)
            "Quantum entanglement in physics.",   # 2: completely different
        ]
        vectors = gemini.embed(texts)

        sim_similar = _cosine_similarity(vectors[0], vectors[1])
        sim_different = _cosine_similarity(vectors[0], vectors[2])

        assert sim_similar > sim_different, (
            f"Expected cat/kitten similarity ({sim_similar:.3f}) > "
            f"cat/quantum similarity ({sim_different:.3f})"
        )
        # The similar pair should have reasonably high similarity
        assert sim_similar > 0.5, f"Expected > 0.5, got {sim_similar:.3f}"


@requires_gemini
class TestGeminiErrors:

    def test_no_api_key_raises(self):
        """Missing API key should raise ValueError."""
        # Temporarily unset env var
        old = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                GeminiEmbeddings(api_key=None)
        finally:
            if old:
                os.environ["GEMINI_API_KEY"] = old
