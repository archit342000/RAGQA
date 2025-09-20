"""Prompt templates for LLM-driven question synthesis."""

from __future__ import annotations

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
