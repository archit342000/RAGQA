"""Prompt templates for LLM-based atomic fact mining."""

from __future__ import annotations

MINING_SYSTEM_PROMPT = (
    "You are an expert factual span miner. "
    "Only respond with a JSON array that follows the schema and rules. "
    "If no valid facts exist, respond with []. No prose, no code fences."
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

JSON schema (array of objects):
[
  {{
    "kind": "definition|numeric|kv|table_row|sentence",
    "text": "<verbatim span within window_text>",
    "char_start": <int>,
    "char_end": <int>,
    "labels": ["key: value", ...],
    "evidence": [{{"type": "sentence", "index": <int>}}, ...],
    "tags": ["tag", ...]
  }}
]

Rules:
1. window_text[char_start:char_end] == text
2. 0 ≤ char_start < char_end ≤ len(window_text)
3. Prefer atomic, concrete facts (short definitions, numeric-with-unit, key–value pairs, table rows, concise declaratives)
4. Avoid boilerplate, headers, footers, and vague fragments
5. Limit each text field to ≤ 300 characters
6. Emit at most {max_items} items for this window
7. Return [] when no valid atoms exist

Example output (2 items maximum):
[
  {{
    "kind": "definition",
    "text": "The retention policy keeps backups for 30 days.",
    "char_start": 118,
    "char_end": 162,
    "labels": ["term: retention policy", "duration: 30 days"],
    "tags": ["policy"]
  }},
  {{
    "kind": "numeric",
    "text": "99.9% uptime",
    "char_start": 278,
    "char_end": 288,
    "labels": ["metric: availability", "unit: percent"],
    "tags": ["numeric"]
  }}
]
"""
