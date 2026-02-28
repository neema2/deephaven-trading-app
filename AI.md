# AI — Embeddings, LLM, RAG, Extraction & Tool Calling

Single `AI` class wraps all AI capabilities. Provider details are internal — users never see them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  AI(api_key="...")                                               │
│                                                                  │
│  ai.generate("prompt")        → LLMResponse                     │
│  ai.stream("prompt")          → Generator[str]                   │
│  ai.ask(q, documents=ms)      → RAGResult (retrieve + generate) │
│  ai.extract(text, schema)     → ExtractionResult                 │
│  ai.run_tool_loop(msgs, ...)  → LLMResponse (with tool calls)   │
│  ai.search_tools(ms)          → tool declarations for LLM       │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────────┐    │
│  │ Embeddings│  │ LLM Client   │  │ RAG Pipeline            │    │
│  │ (768-dim) │  │ (Gemini 3)   │  │ retrieve → augment →    │    │
│  │           │  │              │  │ generate with citations  │    │
│  └──────────┘  └──────────────┘  └─────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  MediaStore(ai=ai)                                               │
│                                                                  │
│  ms.upload(file)          → auto-chunk + embed                   │
│  ms.search("keywords")   → full-text (tsvector)                 │
│  ms.semantic_search(q)    → vector (pgvector cosine)             │
│  ms.hybrid_search(q)     → RRF fusion (best of both)            │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | Gemini `gemini-embedding-001` (768-dim) | Document + query vectors |
| **LLM** | Gemini `gemini-3-flash-preview` | Generation, tool calling, extraction |
| **Vector search** | pgvector HNSW (cosine) | Semantic similarity |
| **Text search** | PG tsvector + GIN | Keyword matching |
| **Hybrid search** | RRF (k=60) | Fuses text + vector rankings |
| **Chunking** | Sentence-aware, 512-token, 50-token overlap | Split docs for embedding |

---

## Quick Start

```python
from ai import AI, Message
from media import MediaStore

ai = AI()                                           # reads GEMINI_API_KEY env var
ms = MediaStore(s3_endpoint="localhost:9002", ai=ai) # auto-embeds on upload
```

### Upload & Search

```python
doc = ms.upload("reports/q1.pdf", title="Q1 Report", tags=["finance"])

ms.search("interest rate swap")           # full-text (keywords)
ms.semantic_search("risk transfer")       # vector (meaning)
ms.hybrid_search("credit derivatives")    # RRF fusion (best)
```

### RAG — Document-Grounded Q&A

```python
result = ai.ask("What are credit default swaps?", documents=ms)
print(result.answer)    # Grounded answer with citations
print(result.sources)   # Retrieved chunks used as context
```

Options:
```python
result = ai.ask(
    "How does DV01 work?",
    documents=ms,
    search_mode="semantic",    # "hybrid" (default), "semantic", or "text"
    limit=5,                   # chunks to retrieve (default: 5)
    temperature=0.3,           # LLM temperature (default: 0.3)
    system_prompt="...",       # custom system prompt
)
```

### Structured Extraction

```python
result = ai.extract(
    text="Goldman Sachs Q3 2024: revenue $12.7B, EPS $8.40, ROE 10.4%.",
    schema={
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "revenue_billions": {"type": "number"},
            "eps": {"type": "number"},
            "roe_pct": {"type": "number"},
        },
        "required": ["company"],
    },
)
print(result.data)
# {"company": "Goldman Sachs", "revenue_billions": 12.7, "eps": 8.40, "roe_pct": 10.4}
```

With dataclass instantiation:
```python
from dataclasses import dataclass

@dataclass
class Earnings:
    company: str
    revenue_billions: float = 0.0
    eps: float = 0.0

result = ai.extract(text, schema=schema, model_class=Earnings)
print(result.data.company)  # "Goldman Sachs"
```

### Direct Generation

```python
# String shorthand (single user message)
response = ai.generate("Explain convexity in fixed income.")
print(response.content)

# Full conversation
response = ai.generate([
    Message(role="system", content="You are a quant analyst."),
    Message(role="user", content="Explain gamma hedging."),
])
```

### Streaming

```python
for chunk in ai.stream("Explain Black-Scholes assumptions"):
    print(chunk, end="", flush=True)
```

### Tool Calling

The LLM can search your documents autonomously:

```python
# Get tool declarations
tools = ai.search_tools(ms)

# LLM decides when to call tools, results fed back automatically
response = ai.run_tool_loop(
    "Find documents about Basel III and summarize the capital requirements.",
    tools=tools,
)
print(response.content)
```

Available search tools (auto-generated from MediaStore):
- `search_documents` — full-text keyword search
- `semantic_search` — vector similarity search
- `hybrid_search` — RRF fusion search
- `list_documents` — browse available documents

Custom tools:
```python
from ai import Tool

my_tool = Tool(
    name="get_price",
    description="Get the current price of a stock.",
    parameters={
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
    fn=lambda symbol: '{"price": 150.25}',
)

response = ai.run_tool_loop(
    "What is AAPL trading at?",
    tools=[{"name": my_tool.name, "description": my_tool.description, "parameters": my_tool.parameters}],
    execute_tool=lambda name, args: my_tool.fn(**args),
)
```

---

## Public API

7 symbols exported from `ai/`:

| Symbol | Kind | Description |
|--------|------|-------------|
| `AI` | class | Single entry point for all AI capabilities |
| `Message` | dataclass | Conversation message (role, content) |
| `LLMResponse` | dataclass | Response from generate/run_tool_loop |
| `ToolCall` | dataclass | Tool call in LLMResponse.tool_calls |
| `RAGResult` | dataclass | Response from ask() — answer + sources |
| `ExtractionResult` | dataclass | Response from extract() — structured data |
| `Tool` | dataclass | Custom tool definition (name, schema, function) |

### AI Constructor

```python
AI(
    api_key=None,          # Falls back to GEMINI_API_KEY env var
    provider="gemini",     # Currently only "gemini" supported
    embedding_dim=768,     # Embedding vector dimension
    model=None,            # LLM model (default: gemini-3-flash-preview)
)
```

### AI Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `.generate()` | `(messages, tools=, temperature=, max_tokens=)` | `LLMResponse` | Generate text. `messages` can be a string or list[Message]. |
| `.stream()` | `(messages, tools=, temperature=, max_tokens=)` | `Generator[str]` | Stream response chunks. |
| `.ask()` | `(question, documents=, search_mode=, limit=, temperature=, system_prompt=)` | `RAGResult` | RAG: retrieve + generate with citations. |
| `.extract()` | `(text, schema, model_class=, system_prompt=, temperature=)` | `ExtractionResult` | Extract structured data from text. |
| `.run_tool_loop()` | `(messages, tools=, execute_tool=, max_iterations=)` | `LLMResponse` | Generate → call tools → respond loop. |
| `.search_tools()` | `(media_store)` | `list[dict]` | Get tool declarations for a MediaStore. |

### MediaStore with AI

```python
MediaStore(
    s3_endpoint="localhost:9002",
    ai=ai,              # pass AI instance — auto-embeds on upload
)
```

When `ai=` is provided, `upload()` automatically chunks text and generates embeddings. `semantic_search()` and `hybrid_search()` become available.

---

## Search Modes

| Mode | Method | How it works | Best for |
|------|--------|-------------|----------|
| **Full-text** | `ms.search(q)` | PG tsvector weighted ranking | Exact keyword matches |
| **Semantic** | `ms.semantic_search(q)` | pgvector cosine similarity on chunks | Meaning-based queries |
| **Hybrid** | `ms.hybrid_search(q)` | RRF fusion of text + semantic | General queries (default for RAG) |

### Search Weights (full-text)

| Weight | Source |
|--------|--------|
| **A** (highest) | Title |
| **B** | Filename + tags |
| **C** | Extracted text |

---

## Chunking & Embeddings

Documents are automatically chunked and embedded on upload when `ai=` is set.

| Parameter | Value |
|-----------|-------|
| Chunk size | ~512 tokens |
| Overlap | ~50 tokens |
| Splitting | Sentence-aware boundaries |
| Embedding model | `gemini-embedding-001` |
| Dimension | 768 |
| Task types | `RETRIEVAL_DOCUMENT` (index), `RETRIEVAL_QUERY` (search) |
| Index | HNSW (cosine distance) |

---

## Demo

```bash
export GEMINI_API_KEY="your-key"
pip install -e ".[ai,media]"
python3 demo_rag.py
```

Exercises all features: upload → 3 search modes → RAG Q&A → extraction → streaming → tool calling.

---

## Test Coverage

| Test suite | Count | What |
|------------|-------|------|
| `test_ai_client.py` | 12 | Clean API: generate, stream, ask, extract, tools, import hygiene |
| `test_embeddings.py` | 8 | Gemini embeddings: batch, query, similarity |
| `test_llm.py` | 12 | LLM: generate, stream, tool calling, run_tool_loop |
| `test_extraction.py` | 5 | Structured extraction: basic, financial, dataclass, list, nulls |
| `test_rag.py` | 5 | RAG pipeline: ask, sources, semantic mode |
| `test_tools.py` | 7 | Tool registry, search tools, LLM+tool integration |
| `test_embed_upload.py` | 12 | Upload+embed, semantic search, hybrid search |
| **Total** | **61** | All real API calls, real PG, real MinIO |
