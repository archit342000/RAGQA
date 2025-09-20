"""Prompt templates for LLM-driven question synthesis and mining."""

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

SYNTH_SYSTEM = (
    "You craft diverse, unambiguous questions answerable solely from the given window. "
    "Output a JSON array only; no commentary or code fences. "
    "Each item must contain keys: question, wh, type, answer_text, evidence. "
    "Do not emit character indices. "
    "Avoid vague openings and tautologies. "
    "Include at least one concrete entity or metric from the window in every question. "
    "Limit the array to at most {max_q} items and keep answer_text ≤ 300 characters."
)

SYNTH_USER_TEMPLATE = """
Document metadata:
- doc_id: {doc_id}
- doc_name: {doc_name}
- pages: {page_start}–{page_end}
- window_len: {window_len}

Instructions:
- Generate focused questions that can be answered with evidence contained entirely in this window.
- Cover a mix of numeric, comparison, procedural, temporal, definitional, and multi-hop (within-window) queries.
- Diversify WH forms (what/which/who/when/where/why/how/how many/how much) where possible.
- Produce no more than {max_q} items; respond with [] if nothing fits the criteria.
- Return a JSON array only; omit trailing text.

Expected JSON structure (pseudo-code example):
[
  {{
    "question": "Which tier supports SSO and costs under $100?",
    "wh": "which",
    "type": "comparison",
    "answer_text": "Basic Plus",
    "evidence": [{{"type": "sentence", "index": 7}}]
  }},
  {{
    "question": "When did the retention policy change take effect?",
    "wh": "when",
    "type": "temporal",
    "answer_text": "March 1, 2024",
    "evidence": [{{"type": "sentence", "index": 3}}]
  }}
]

window_text:
```
{window_text}
```
"""
