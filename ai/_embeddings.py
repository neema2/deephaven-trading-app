"""
Embedding Providers — Provider-agnostic embedding API.

Supports:
  - GeminiEmbeddings: Google Gemini text-embedding via google-genai SDK

Usage::

    from ai import GeminiEmbeddings

    emb = GeminiEmbeddings(api_key="...", dimension=768)
    vectors = emb.embed(["hello world", "how are you"])
    query_vec = emb.embed_query("search query")
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts for document indexing.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats), one per input text.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single search query.

        Some providers use different task types for queries vs documents
        to improve retrieval quality.

        Args:
            text: The search query text.

        Returns:
            Embedding vector (list of floats).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding dimension."""
        ...


class GeminiEmbeddings(EmbeddingProvider):
    """
    Google Gemini embedding provider via the google-genai SDK.

    Uses task types for optimal retrieval:
      - RETRIEVAL_DOCUMENT for indexing documents (embed)
      - RETRIEVAL_QUERY for search queries (embed_query)

    Args:
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        model: Embedding model name (default: gemini-embedding-001).
        dimension: Output embedding dimension (default: 768).
            gemini-embedding-001 natively outputs 3072; truncated to this value.
        max_retries: Number of retry attempts on transient errors (default: 3).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var."
            )
        self._model = model
        self._dimension = dimension
        self._max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the genai client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts for document indexing using RETRIEVAL_DOCUMENT task type.

        Supports batch embedding — multiple texts in one API call.
        """
        if not texts:
            return []

        from google.genai import types

        client = self._get_client()
        result = self._call_with_retry(
            lambda: client.models.embed_content(
                model=self._model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._dimension,
                ),
            )
        )

        vectors = [list(e.values) for e in result.embeddings]
        logger.debug("Embedded %d texts → %d-dim vectors", len(texts), self._dimension)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a search query using RETRIEVAL_QUERY task type.

        Uses a different task type than embed() for better retrieval quality.
        """
        from google.genai import types

        client = self._get_client()
        result = self._call_with_retry(
            lambda: client.models.embed_content(
                model=self._model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self._dimension,
                ),
            )
        )

        vector = list(result.embeddings[0].values)
        logger.debug("Embedded query → %d-dim vector", self._dimension)
        return vector

    def _call_with_retry(self, fn: Callable[..., Any], retries: int | None = None) -> Any:
        """Call fn with exponential backoff retry on transient errors."""
        max_retries = retries if retries is not None else self._max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Retry on rate limits and transient server errors
                if "429" in error_str or "500" in error_str or "503" in error_str or "resource_exhausted" in error_str:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini API error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries, wait, e,
                    )
                    time.sleep(wait)
                else:
                    # Non-transient error — don't retry
                    raise

        assert last_error is not None
        raise last_error
