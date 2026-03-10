"""
LLM Client — Provider-agnostic LLM interface with tool/function calling.

Supports:
  - GeminiLLM: Google Gemini via google-genai SDK

Usage::

    from ai import GeminiLLM, Message

    llm = GeminiLLM(api_key="...", model="gemini-3-flash-preview")

    # Simple generation
    response = llm.generate([Message(role="user", content="Hello!")])
    print(response.content)

    # With tools
    tools = [{"name": "search", "description": "Search docs", "parameters": {...}}]
    response = llm.generate([Message(role="user", content="Find reports")], tools=tools)
    if response.tool_calls:
        # Execute tool, send result back
        ...
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any, cast

from ai._types import LLMResponse, Message, ToolCall

logger = logging.getLogger(__name__)


# ── Abstract base class ──────────────────────────────────────────────────


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a response from the model.

        Args:
            messages: Conversation history as list of Message objects.
            tools: Optional list of tool declarations (OpenAI-style format).
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with content and/or tool_calls.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """
        Stream response chunks from the model.

        Args:
            Same as generate().

        Yields:
            Partial content strings as they arrive.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier."""
        ...

    def run_tool_loop(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        execute_tool: Callable | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_iterations: int = 5,
    ) -> LLMResponse:
        """
        Run a generate → tool call → execute → respond loop until the model
        returns a text response (no more tool calls) or max_iterations is reached.

        This handles all the plumbing for tool-calling conversations, including
        preserving provider-specific metadata (thought signatures, etc).

        Args:
            messages: Initial conversation history.
            tools: Tool declarations.
            execute_tool: Callable(name, arguments) → str. Called for each tool call.
                If None, tool calls are returned without execution.
            temperature: Sampling temperature.
            max_tokens: Max tokens per generation.
            max_iterations: Safety limit on tool-call rounds (default: 5).

        Returns:
            Final LLMResponse (the one with text content, or last response if
            max_iterations reached).
        """
        if execute_tool is None:
            return self.generate(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)

        msgs = list(messages)  # Don't mutate caller's list

        for _ in range(max_iterations):
            response = self.generate(msgs, tools=tools, temperature=temperature, max_tokens=max_tokens)

            if not response.tool_calls:
                return response

            # Append assistant message (preserves thought signatures)
            msgs.append(response.to_message())

            # Execute each tool call and append results
            for tc in response.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                msgs.append(Message(
                    role="tool",
                    content=result,
                    name=tc.name,
                    tool_call_id=tc.id,
                ))

        return response


# ── Gemini implementation ────────────────────────────────────────────────


class GeminiLLM(LLMClient):
    """
    Google Gemini LLM provider via the google-genai SDK.

    Supports text generation, conversation history, streaming,
    and tool/function calling.

    Args:
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        model: Model name (default: gemini-3-flash-preview).
        max_retries: Retry attempts on transient errors (default: 3).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var."
            )
        self._model = model
        self._max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the genai client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response using Gemini."""

        client = self._get_client()
        contents = self._messages_to_contents(messages)
        config = self._build_config(tools, temperature, max_tokens)

        result = self._call_with_retry(
            lambda: client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        )

        return self._parse_response(result)

    def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """Stream response chunks from Gemini."""

        client = self._get_client()
        contents = self._messages_to_contents(messages)
        config = self._build_config(tools, temperature, max_tokens)

        response_stream = client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    # ── Internal helpers ─────────────────────────────────────────────────

    def _messages_to_contents(self, messages: list[Message]) -> list:
        """Convert our Message list to Gemini contents format."""
        from google.genai import types

        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue

            if msg.role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg.content)],
                ))

            elif msg.role == "assistant":
                # Use raw content if available (preserves thought_signature for Gemini 3)
                if hasattr(msg, '_raw_content') and msg._raw_content is not None:
                    # _raw_content is a Gemini Content object stored from prior response
                    raw = cast(types.Content, msg._raw_content)
                    contents.append(raw)
                else:
                    parts = []
                    if msg.content:
                        parts.append(types.Part.from_text(text=msg.content))
                    for tc in msg.tool_calls:
                        parts.append(types.Part.from_function_call(
                            name=tc.name,
                            args=tc.arguments,
                        ))
                    if parts:
                        contents.append(types.Content(role="model", parts=parts))

            elif msg.role == "tool":
                # Tool response — send as function_response
                try:
                    response_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": msg.content}
                # Gemini FunctionResponse requires a dict, not a list
                if not isinstance(response_data, dict):
                    response_data = {"result": response_data}

                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=msg.name,
                        response=response_data,
                    )],
                ))

        # Store system instruction for config
        self._last_system_instruction = system_instruction
        return contents

    def _build_config(self, tools: list[dict] | None, temperature: float, max_tokens: int) -> Any:
        """Build Gemini generation config."""
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # System instruction
        if hasattr(self, '_last_system_instruction') and self._last_system_instruction:
            config_kwargs["system_instruction"] = self._last_system_instruction

        # Tools
        if tools:
            gemini_tools = self._convert_tools(tools)
            config_kwargs["tools"] = gemini_tools

        return types.GenerateContentConfig(**config_kwargs)

    def _convert_tools(self, tools: list[dict]) -> list:
        """Convert our tool format to Gemini function declarations."""
        from google.genai import types

        declarations = []
        for tool in tools:
            decl = {
                "name": tool["name"],
                "description": tool.get("description", ""),
            }
            if "parameters" in tool:
                decl["parameters"] = tool["parameters"]
            declarations.append(decl)

        return [types.Tool(function_declarations=declarations)]  # type: ignore[arg-type]  # Gemini accepts dicts

    def _parse_response(self, result: Any) -> LLMResponse:
        """Parse Gemini response into our LLMResponse format."""
        content = ""
        tool_calls = []

        if result.candidates:
            candidate = result.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    content += part.text
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        id=fc.id or f"call_{fc.name}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))

        usage = {}
        if hasattr(result, 'usage_metadata') and result.usage_metadata:
            um = result.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, 'prompt_token_count', 0),
                "completion_tokens": getattr(um, 'candidates_token_count', 0),
                "total_tokens": getattr(um, 'total_token_count', 0),
            }

        # Preserve raw content for thought_signature support (Gemini 3)
        raw_content = None
        if result.candidates:
            raw_content = result.candidates[0].content

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=self._model,
            _raw_content=raw_content,
        )

    def _call_with_retry(self, fn: Callable[..., Any], retries: int | None = None) -> Any:
        """Call fn with exponential backoff on transient errors."""
        max_retries = retries if retries is not None else self._max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "429" in error_str or "500" in error_str or "503" in error_str or "resource_exhausted" in error_str:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini API error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries, wait, e,
                    )
                    time.sleep(wait)
                else:
                    raise

        raise last_error  # type: ignore[misc]
