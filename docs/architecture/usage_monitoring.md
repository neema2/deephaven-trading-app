# AI Usage and Cost Monitoring

Development costs in an agentic project like `py-flow` are driven primarily by **Reasoning/Coding sessions** rather than runtime API usage. This document provides a guide for approximating model usage and spend using the platform's local telemetry.

## Telemetry Sources

The Antigravity platform stores conversation history in binary protobuf files:
`~/.gemini/antigravity/conversations/*.pb`

> [!NOTE]
> Because these files are binary and often use model switching (Gemini vs. Claude via Google Ultimate), simple plaintext searches may fail. Usage must be approximated by log volume and session intensity.

## Approximation Protocol

To view the approximate "Development Intensity" by day, use the following `ls` command to group logs by date and size:

```bash
# List logs sorted by date and size to identify peak development days
ls -lh --time-style=long-iso ~/.gemini/antigravity/conversations/*.pb | awk '{print $6, $5, $8}' | sort
```

### Conversion Heuristics (Approximate)
| Log Size | Model Weight | Est. Tokens | Rationale |
| :--- | :--- | :--- | :--- |
| **< 1MB** | Light | ~250k | Quick fixes, unit test debugging. |
| **1MB - 10MB** | Moderate | 250k - 2.5M | Standard feature development. |
| **10MB+** | Heavy | 2.5M - 10M+ | Major refactors (e.g., IRS Swap Pricing logic). |

## Estimating Spend (Weighting by Model Choice)

When calculating the "Market Value" of spend in the Google Ultimate package, weight the usage by the model switch used during the session:

*   **Claude 3 Opus**: High-density reasoning; most expensive market rate (~$15/1M in). Used for core architecture/bi-temporal logic.
*   **Claude 3.5 Sonnet / Gemini 1.5 Pro**: Standard development (~$3/1M in).
*   **Gemini 1.5 Flash**: Runtime tests and boilerplate (~$0.10/1M in).

## Automated Scanning (Metadata Analysis)

Scanning binary `.pb` logs is difficult due to compression. The most reliable way to identify model switches is to query the conversation metadata directly via the `py-flow` Python API.

### Option A: Python Monitoring Script
You can use the `AgentMemory` class to list conversations and extract the model associated with each session (assuming the UI sets this in the metadata).

```python
from ai.memory import AgentMemory
from store import connect

# Connect to the object store
conn = connect()
memory = AgentMemory(store_conn=conn)

# List recent conversations
convos = memory.list_conversations(limit=100)

print(f"{'Date':<12} {'Model':<20} {'Tokens':<10} {'Msgs':<5}")
print("-" * 50)

for c in convos:
    date = c.created_at[:10]
    # The UI typically stores the model name in the metadata field
    model = c.metadata.get("model_name", "Unknown")
    tokens = c.metadata.get("total_tokens", 0)
    print(f"{date:<12} {model:<20} {tokens:<10} {c.message_count:<5}")
```

### Option B: Binary Grep (Heuristic)
If you cannot connect to the database, you can use a heuristic `grep` on the log files. While the model names are often compressed, searching for specific "fingerprint" strings can sometimes reveal the active provider.

```bash
# Search for model fingerprints in binary logs
grep -aoE "claude-3|gemini-[0-9]+" ~/.gemini/antigravity/conversations/*.pb
```

---

## Why the Initial Scan Failed
The `.pb` files are protobuf-encoded and often compressed. A standard `grep` will return nothing for strings like "Claude" or "Gemini" if they are inside a compressed byte-stream. Always prioritize the **Metadata Analysis** via Python when possible.
