"""
RAG Pipeline — Retrieve-Augment-Generate for document-grounded answers.

Embeds the user's question, retrieves relevant document chunks,
builds a prompt with context, and generates an LLM answer with citations.

Usage::

    from ai import GeminiLLM, GeminiEmbeddings, RAGPipeline
    from media import MediaStore

    llm = GeminiLLM(api_key="...")
    embedder = GeminiEmbeddings(api_key="...")
    ms = MediaStore(..., ai=ai_instance)

    rag = RAGPipeline(llm=llm, media_store=ms)
    result = rag.ask("What are credit default swaps?")
    print(result.answer)
    print(result.sources)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai._llm import LLMClient
from ai._types import Message, RAGResult

if TYPE_CHECKING:
    from media.store import MediaStore

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided documents.

Rules:
- Use ONLY the information from the documents below to answer.
- If the answer is not in the documents, say "I don't have enough information in the available documents to answer that."
- Cite your sources by mentioning the document title or filename.
- Be concise and direct."""


CONTEXT_TEMPLATE = """## Retrieved Documents

{context}

## Question
{question}"""


class RAGPipeline:
    """
    Retrieve-Augment-Generate pipeline for document-grounded answers.

    Args:
        llm: An LLMClient instance for generation.
        media_store: A MediaStore instance with ai= for retrieval.
        search_mode: Search strategy — "hybrid", "semantic", or "text" (default: "hybrid").
    """

    def __init__(
        self,
        llm: LLMClient,
        media_store: MediaStore,
        search_mode: str = "hybrid",
    ) -> None:
        self._llm = llm
        self._media_store = media_store
        self._search_mode = search_mode

    def ask(
        self,
        question: str,
        system_prompt: str | None = None,
        limit: int = 5,
        temperature: float = 0.3,
    ) -> RAGResult:
        """
        Ask a question with document-grounded answers.

        Args:
            question: The user's question.
            system_prompt: Optional custom system prompt (default: built-in RAG prompt).
            limit: Number of chunks to retrieve (default: 5).
            temperature: LLM temperature (default: 0.3 for factual answers).

        Returns:
            RAGResult with answer, sources, and usage stats.
        """
        # 1. Retrieve relevant chunks
        sources = self._retrieve(question, limit=limit)

        # 2. Build context from retrieved chunks
        context = self._build_context(sources)

        # 3. Build messages
        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        user_content = CONTEXT_TEMPLATE.format(context=context, question=question)

        messages = [
            Message(role="system", content=sys_prompt),
            Message(role="user", content=user_content),
        ]

        # 4. Generate answer
        response = self._llm.generate(messages, temperature=temperature)

        logger.info("RAG: retrieved %d sources, generated %d chars",
                     len(sources), len(response.content))

        return RAGResult(
            answer=response.content,
            sources=sources,
            usage=response.usage,
        )

    def _retrieve(self, question: str, limit: int) -> list[dict]:
        """Retrieve relevant document chunks using configured search mode."""
        if self._search_mode == "hybrid":
            return self._media_store.hybrid_search(question, limit=limit)
        elif self._search_mode == "semantic":
            return self._media_store.semantic_search(question, limit=limit)
        elif self._search_mode == "text":
            return self._media_store.search(question, limit=limit)
        else:
            raise ValueError(f"Unknown search_mode: {self._search_mode}")

    def _build_context(self, sources: list[dict]) -> str:
        """Format retrieved chunks into a context string for the prompt."""
        if not sources:
            return "(No relevant documents found.)"

        parts = []
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Unknown")
            filename = src.get("filename", "")
            chunk_text = src.get("chunk_text", "")

            # For text-only search, there's no chunk_text — use title
            if not chunk_text:
                chunk_text = f"(Document: {title})"

            header = f"[Source {i}: {title}"
            if filename:
                header += f" ({filename})"
            header += "]"

            parts.append(f"{header}\n{chunk_text}")

        return "\n\n".join(parts)
