"""Prompt templates for LLM-based atomic fact mining."""

from __future__ import annotations

MINING_SYSTEM_PROMPT = (
    "You are an information extraction engine. "
    "Return a JSON array only, with no extra prose, describing atomic facts found in the provided window_text."
)

MINING_USER_PROMPT_TEMPLATE = """
Document metadata:
- doc_id: {doc_id}
- doc_name: {doc_name}
- pages: {page_start}–{page_end}
- window_length: {window_chars} characters

window_text:
```
{window_text}
```

JSON schema:
[
  {{
    "kind": "definition|numeric|kv|table_row|sentence",
    "text": "<verbatim span within window_text>",
    "char_start": <int>,
    "char_end": <int>,
    "labels": ["key: value", ...],
    "evidence": [{{"type": "sentence", "index": <int>}}, ...],
    "tags": ["tag", ...]
  }},
  ...
]

Rules:
- Output MUST be a single JSON array with no trailing commentary.
- window_text[char_start:char_end] == text.
- 0 ≤ char_start < char_end ≤ len(window_text).
- Each text value ≤ 300 characters and should capture an atomic, concrete fact.
- Prefer short definitions, numeric values with units, key–value snippets, table rows, or concise declarative sentences.
- Skip boilerplate, headers, footers, and vague fragments.
- Emit at most {max_items} items for this window.

Example (truncated to two items):
[
  {{
    "kind": "definition",
    "text": "Retention policy keeps data for 30 days.",
    "char_start": 128,
    "char_end": 166,
    "labels": ["term: Retention policy", "duration: 30 days"],
    "tags": ["policy"]
  }},
  {{
    "kind": "numeric",
    "text": "99.9% availability",
    "char_start": 302,
    "char_end": 320,
    "labels": ["metric: availability", "unit: percent"],
    "tags": ["numeric"]
  }}
]
"""
