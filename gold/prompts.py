"""Prompt templates for LLM-driven question synthesis."""

from __future__ import annotations

SYNTH_SYSTEM = """
You generate diverse, unambiguous questions that are answerable solely from the provided window.

Output Rules:
- Output strictly a JSON array — no commentary, headings, code fences, or extra text.
- Each item must contain EXACT keys: question, wh, type, answer_text, evidence.
- Keys must be lowercase; do not include additional keys.
- Ensure valid JSON: no trailing commas, properly escaped quotes.
- The output must conform to this JSON schema:
  [
    {
      "question": <string>,
      "wh": <string>,
      "type": <string>,
      "answer_text": <string>,
      "evidence": [ { "type": <string>, "index": <int> }, ... ]
    }
  ]

Question Style:
- Phrase all questions in third person (e.g., "the narrator", "the speaker", or the named entity), even if the source uses first or second person.
- Resolve pronouns into explicit entities when possible.
- Use neutral, precise tense and wording; avoid vague openings, rhetorical wording, and tautologies.
- Avoid yes/no questions unless the window states the fact explicitly.
- Every question must reference at least one concrete entity, figure, date, metric, or named concept from the window.
- Do not generate two questions that have the same answer_text.

Types & Coverage:
- Allowed wh ∈ {what, which, who, when, where, why, how, how many, how much}.
- Allowed type ∈ {numeric, comparison, procedural, temporal, definitional, multi-hop, location, cause-effect, verification}.
- Diversify wh and type across items; avoid near-duplicate questions that yield the same answer.

Answer Normalization:
- If dates are explicit, express them in ISO format (YYYY-MM-DD).
- If quantities include units, include the units in answer_text.
- If names are given in full, use the full form, not a shortened or pronoun form.
- If information is ambiguous, copy text verbatim without guessing.

Evidence:
- Evidence indices must be numeric and 0-based.
- Provide the minimal sufficient set of evidence items supporting the answer.
- Allowed evidence types: sentence, list_item, table_cell, figure_caption.

Limits:
- Generate at most {max_q} items.
- Keep answer_text ≤ 300 characters, concise and self-contained.
"""

SYNTH_USER_TEMPLATE = """
Document metadata:
- doc_id: {doc_id}
- doc_name: {doc_name}
- pages: {page_start}–{page_end}
- window_len: {window_len}

Instructions:
- Generate precise questions answerable only from this window (no outside knowledge).
- Aim for a balanced mix of question types: numeric, comparison, procedural, temporal, definitional, multi-hop (within-window), location, cause-effect, verification.
- Vary WH forms where possible: what, which, who, when, where, why, how, how many, how much.
- Prefer concrete dates/quantities over vague references like "now", "today", or "currently", unless the window defines them clearly.

Quality filters:
- Skip rhetorical, speculative, opinion-laden, or unverifiable questions.
- Avoid trivial restatements and near-duplicates (e.g., "What is the price?" vs. "How much does it cost?" if identical).
- For multi-hop, ensure all hops are supported entirely within the window (max two hops).
- Prefer atomic, fact-bearing questions over broad summaries.

Evidence rules:
- Use 0-based indices aligned to the host's sentence/list segmentation.
- Allowed evidence item schema: { "type": "sentence" | "list_item" | "table_cell" | "figure_caption", "index": <int> }.

Formatting rules:
- Produce ≤ {max_q} items; if nothing fits, output [].

Expected JSON structure (illustrative only):
[
  {
    "question": "Which tier supports SSO and costs under $100?",
    "wh": "which",
    "type": "comparison",
    "answer_text": "Basic Plus",
    "evidence": [{"type": "sentence", "index": 7}]
  },
  {
    "question": "When did the retention policy change take effect?",
    "wh": "when",
    "type": "temporal",
    "answer_text": "March 1, 2024",
    "evidence": [{"type": "sentence", "index": 3}]
  }
]

Window text:
{window_text}
"""
