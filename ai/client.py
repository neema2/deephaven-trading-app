"""
AI — Single entry point for all AI capabilities.

Usage::

    from ai import AI, Message
    from media import MediaStore

    ai = AI(api_key="...")
    ms = MediaStore(s3_endpoint="...", ai=ai)

    # RAG
    result = ai.ask("What are credit default swaps?", documents=ms)

    # Direct generation
    response = ai.generate("Explain quantum computing")
    response = ai.generate([
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ])

    # Streaming
    for chunk in ai.stream("Tell me about neural networks"):
        print(chunk, end="")

    # Structured extraction
    data = ai.extract("John Smith, 35, engineer", schema={...})

    # Tool calling
    response = ai.run_tool_loop(
        [Message(role="user", content="Search for docs")],
        tools=ai.search_tools(ms),
    )
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

from ai._types import (
    ExtractionResult,
    LLMResponse,
    Message,
    RAGResult,
)

if TYPE_CHECKING:
    from media.store import MediaStore

    from ai._embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class AI:
    """
    Single entry point for all AI capabilities.

    Wraps embedding, LLM, RAG, extraction, and tool calling behind
    one object. Provider details are internal — users never see them.

    Args:
        api_key: API key for the AI provider. Falls back to GEMINI_API_KEY env var.
        provider: Provider name (default: "gemini"). Currently only "gemini" is supported.
        embedding_dim: Embedding dimension (default: 768).
        model: LLM model name. Defaults to provider's best model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "gemini",
        embedding_dim: int = 768,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key required. Pass api_key= or set GEMINI_API_KEY env var."
            )
        self._provider = provider
        self._embedding_dim = embedding_dim

        if provider == "gemini":
            from ai._embeddings import GeminiEmbeddings
            from ai._llm import GeminiLLM

            self._embedder = GeminiEmbeddings(
                api_key=self._api_key,
                dimension=embedding_dim,
            )
            self._llm = GeminiLLM(
                api_key=self._api_key,
                model=model or "gemini-3-flash-preview",
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ── Generation ───────────────────────────────────────────────────────

    def generate(
        self,
        messages: str | list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: A string (shorthand for single user message) or list of Messages.
            tools: Optional tool declarations.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Returns:
            LLMResponse with content and/or tool_calls.
        """
        msgs = self._normalize_messages(messages)
        return self._llm.generate(msgs, tools=tools, temperature=temperature, max_tokens=max_tokens)

    def stream(
        self,
        messages: str | list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """
        Stream response chunks from the LLM.

        Args:
            messages: A string or list of Messages.

        Yields:
            Partial content strings as they arrive.
        """
        msgs = self._normalize_messages(messages)
        return self._llm.stream(msgs, tools=tools, temperature=temperature, max_tokens=max_tokens)

    # ── RAG ──────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        documents: MediaStore | None = None,
        system_prompt: str | None = None,
        search_mode: str = "hybrid",
        limit: int = 5,
        temperature: float = 0.3,
    ) -> RAGResult:
        """
        Ask a question with document-grounded answers (RAG).

        Args:
            question: The question to answer.
            documents: A MediaStore instance to retrieve context from.
            system_prompt: Optional custom system prompt.
            search_mode: "hybrid", "semantic", or "text" (default: "hybrid").
            limit: Number of chunks to retrieve.
            temperature: LLM temperature.

        Returns:
            RAGResult with answer, sources, and usage stats.
        """
        if documents is None:
            raise ValueError("documents= is required. Pass a MediaStore instance.")

        from ai._rag import RAGPipeline
        pipeline = RAGPipeline(
            llm=self._llm,
            media_store=documents,
            search_mode=search_mode,
        )
        return pipeline.ask(
            question,
            system_prompt=system_prompt,
            limit=limit,
            temperature=temperature,
        )

    # ── Extraction ───────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        schema: dict,
        model_class: type | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> ExtractionResult:
        """
        Extract structured data from text.

        Args:
            text: The unstructured text to extract from.
            schema: JSON Schema describing the expected output.
            model_class: Optional dataclass to instantiate with extracted data.
            system_prompt: Optional custom system prompt.
            temperature: LLM temperature.

        Returns:
            ExtractionResult with data (dict or model_class instance).
        """
        from ai._extraction import extract as _extract
        return _extract(
            self._llm,
            text=text,
            schema=schema,
            model_class=model_class,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    # ── Tool calling ─────────────────────────────────────────────────────

    def run_tool_loop(
        self,
        messages: str | list[Message],
        tools: list[dict] | None = None,
        execute_tool: Callable | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_iterations: int = 5,
    ) -> LLMResponse:
        """
        Run a generate → tool call → execute → respond loop.

        Args:
            messages: A string or list of Messages.
            tools: Tool declarations.
            execute_tool: Callable(name, arguments) → str.
            temperature: Sampling temperature.
            max_tokens: Max tokens per generation.
            max_iterations: Safety limit on rounds.

        Returns:
            Final LLMResponse.
        """
        msgs = self._normalize_messages(messages)
        return self._llm.run_tool_loop(
            msgs,
            tools=tools,
            execute_tool=execute_tool,
            temperature=temperature,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
        )

    def search_tools(self, media_store: MediaStore) -> list[dict]:
        """
        Get tool declarations for document search on a MediaStore.

        Pass the result to generate(tools=...) or run_tool_loop(tools=...).

        Args:
            media_store: A MediaStore instance.

        Returns:
            List of tool declaration dicts.
        """
        from ai._tools import ToolRegistry, create_search_tools
        registry = ToolRegistry()
        for tool in create_search_tools(media_store):
            registry.register(tool)
        return registry.list_declarations()

    # ── Internal ─────────────────────────────────────────────────────────

    @property
    def embedder(self) -> EmbeddingProvider:
        """Internal embedding provider. Used by MediaStore."""
        return self._embedder

    def _normalize_messages(self, messages: str | list[Message]) -> list[Message]:
        """Convert string shorthand to Message list."""
        if isinstance(messages, str):
            return [Message(role="user", content=messages)]
        return messages
