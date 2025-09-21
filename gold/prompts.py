"""Prompt templates for LLM-driven question synthesis."""

from __future__ import annotations

SYNTH_SYSTEM = (
    "You generate diverse, unambiguous questions answerable only from the provided window. "
    "Output strictly a JSON array — no commentary, code fences, or extra text. "
    "Each object must include: question, wh, type, answer_text, evidence. "
    "Do not output character indices. "
    "Avoid vague phrasing and tautologies. "
    "Every question must cite at least one concrete entity, fact, or metric from the window. "
    "Limit to {max_q} items; keep answer_text ≤ 300 characters."
)

SYNTH_USER_TEMPLATE = """
Document metadata:
- doc_id: {doc_id}
- doc_name: {doc_name}
- pages: {page_start}–{page_end}
- window_len: {window_len}

Instructions:
- Generate precise questions that can be answered solely from this window.
- Cover a variety: numeric, comparison, procedural, temporal, definitional, and multi-hop (within-window).
- Vary WH forms: what, which, who, when, where, why, how, how many, how much.
- Produce ≤ {max_q} items; if no valid question exists, return [].
- Output must be a JSON array only.

Expected JSON format (illustrative):
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

Window text:
{window_text}
"""
