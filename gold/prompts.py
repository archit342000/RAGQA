"""Prompt templates for LLM-driven question synthesis and mining."""

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

MINING_SYSTEM_PROMPT = (
    "You extract distinct, atomic factual statements from the provided document window. "
    "Return only a JSON array. Each item must contain keys: kind, text, char_start, char_end, labels, evidence, tags. "
    "char_start and char_end are 0-indexed character offsets into window_text (end exclusive). "
    "Copy the exact span from window_text into text so the offsets match perfectly. "
    "Skip speculative, redundant, or cross-window facts. "
    "Emit at most {max_items} items; output [] when no atomic facts qualify."
)

MINING_USER_PROMPT_TEMPLATE = """
Document metadata:
- doc_id: {doc_id}
- doc_name: {doc_name}
- pages: {page_start}–{page_end}
- window_chars: {window_chars}

Task:
1. Identify up to {max_items} atomic facts explicitly supported by this window.
2. Each fact must map to a contiguous span in window_text.
3. For every fact provide:
   - "kind": short category such as "definition", "numeric", "event", or "other".
   - "text": the exact characters from window_text that express the fact.
   - "char_start" / "char_end": integers pointing to the span (end exclusive).
   - "labels": short descriptive keywords (may be empty).
   - "evidence": optional structured hints (e.g., sentence references); use an empty list if unsure.
   - "tags": optional quality or content tags like ["numeric"] or ["table"].
4. Avoid overlapping spans and do not hallucinate information not present in the window.
5. Respond with a JSON array only—no prose or code fences.

Example:
[
  {{
    "kind": "definition",
    "text": "Retention Policy is defined as keeping data for 30 days.",
    "char_start": 128,
    "char_end": 186,
    "labels": ["Retention Policy"],
    "evidence": [{{"type": "sentence", "index": 2}}],
    "tags": ["definition", "numeric"]
  }}
]

window_text:
```
{window_text}
```
"""

__all__ = [
    "SYNTH_SYSTEM",
    "SYNTH_USER_TEMPLATE",
    "MINING_SYSTEM_PROMPT",
    "MINING_USER_PROMPT_TEMPLATE",
]
