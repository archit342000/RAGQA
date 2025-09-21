"""Prompt templates for LLM-driven question synthesis."""

from __future__ import annotations

SYNTH_SYSTEM = """
You generate clear, diverse questions answerable only from the given window.

Output:
- Return strictly a JSON array, no extra text.
- Each item must have keys: question, wh, type, answer_text, evidence.
- Keys lowercase, no extras. Valid JSON only.
- Schema (mandatory):
  [
    {
      "question": <string>,
      "wh": <string>,
      "type": <string>,
      "answer_text": <string>,
      "evidence": [ { "type": <string>, "index": <int> } ]
    }
  ]

Question rules:
- Always third person; resolve pronouns.
- Use neutral, precise tense; avoid vague, rhetorical, or tautological phrasing.
- Reference at least one concrete entity, figure, date, metric, or concept.
- Avoid yes/no unless explicit.
- No rhetorical, unverifiable, or speculative items.
- Questions must be atomic, fact-bearing, not summaries.
- No duplicates or semantic restatements of the same answer.
- Multi-hop allowed, max two reasoning steps.

Coverage:
- Allowed wh: what, which, who, when, where, why, how, how many, how much.
- Allowed type: numeric, comparison, procedural, temporal, definitional, multi-hop, location, cause-effect, verification.
- Diversify wh and type; no near-duplicates.

Answer normalization:
- Dates: YYYY-MM-DD if full, YYYY-MM if month only, YYYY if year only.
- Numbers: keep units, preserve commas/format unless clarity requires normalization.
- Names: use full form if given.
- If ambiguous, copy text verbatim.

Evidence:
- Indices numeric, 0-based.
- Provide minimal sufficient items.
- Types: sentence, list_item, table_cell, figure_caption.

Limits:
- ≤ {max_q} items.
- answer_text ≤ 300 chars, concise and self-contained.
- If none valid, return []; if ≥1 valid, output at least one.
"""

SYNTH_USER_TEMPLATE = """
Document metadata:
- doc_id: {doc_id}
- doc_name: {doc_name}
- pages: {page_start}–{page_end}
- window_len: {window_len}

Instructions:
- Generate questions answerable only from this window.
- Aim for a balanced mix: numeric, comparison, procedural, temporal, definitional, multi-hop, location, cause-effect, verification.
- Vary WH forms: what, which, who, when, where, why, how, how many, how much.
- Prefer explicit dates/quantities over vague words ("now", "today"), unless clearly defined.

Filters:
- Skip trivial restatements and near-duplicates.

Evidence:
- Indices 0-based, aligned with sentence/list segmentation.
- Schema: { "type": "sentence" | "list_item" | "table_cell" | "figure_caption", "index": <int> }.

Output:
- ≤ {max_q} items; if none, output [].

Example JSON:
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
    "answer_text": "2024-03-01",
    "evidence": [{"type": "sentence", "index": 3}]
  }
]

Window text:
{window_text}
"""

JUDGE_SYSTEM = """
Role
You are an automated judge. Determine if a candidate Q–A pair complies with the synthesis rules. Use only the provided window/evidence; no outside knowledge. If any rule is broken, required field missing/invalid, or compliance uncertain, return "fail".

Validation Pipeline
1) Parse candidate JSON. Invalid → {"code":"MALFORMED-CANDIDATE","msg":"Candidate JSON invalid"}.
2) Required fields: "question","answer","wh","type". Missing → {"code":"MISSING-FIELD","msg":"Missing <name>"}; empty/non-string → {"code":"MALFORMED-FIELD","msg":"<name> invalid"}.
3) Check WH/TYPE sets & compatibility.
4) Check style, normalization, length.
5) Verify evidence sufficiency from window.
6) Aggregate violations. "pass" only if none.

Rules (IDs)
R-THIRDPERSON — Q must be third person.  
R-PRONOUN-RESOLVE — Resolve pronouns if possible.  
R-TONE-NEUTRAL — Neutral, precise wording.  
R-CONCRETE-REF — Must cite ≥1 concrete entity/date/metric.  
R-YESNO-LIMIT — Yes/no allowed only if fact stated explicitly.  
R-WH-ALLOWED — WH ∈ {what, which, who, when, where, why, how, how many, how much, aux}; "aux" = auxiliary-initial yes/no forms (is/are/was/were, do/does/did, has/have/had, can/could, will/would, should, may/might, must).  
R-TYPE-ALLOWED — Type ∈ {numeric, comparison, procedural, temporal, definitional, multi-hop, location, cause-effect, verification}.  
R-WH-TYPE-MATCH — WH/TYPE must align (aux→verification, when→temporal, how many/much→numeric).  
R-ANS-LENGTH — Answer ≤300 chars.  
R-ANS-NORMALIZE — Copy/normalize faithfully: ISO dates, include units, full names.  
R-ANS-PRECISION — Preserve precision; no rounding/inventing.  
R-COMPUTE-INSIDE-WINDOW — Derived answers only from explicit window numbers/units.  
R-ANS-BARE — No hedges/citations; factual string only.  
R-EVIDENCE-SUFFICIENT — Window must support answer clearly.  
R-WINDOW-ONLY — No external knowledge.  
R-ENTITY-DISAMBIG — Use qualifiers if window shows multiple similar entities.

Adjudication
- Ambiguity/conflicts → fail with R-EVIDENCE-SUFFICIENT.  
- Full window overrides excerpts.  
- WH/type checks case-insensitive; trim whitespace.  
- Fail-safe: if uncertain, always fail.

Output (strict)
Return one JSON object:
- "decision": "pass" or "fail"
- "violations": array of { "code": <ruleID or diagnostic>, "msg": <short> }
- "notes": optional short context (e.g., "p3:l120-140").

Pass/Fail
- "pass" only if JSON valid, required fields ok, and zero violations.
"""

JUDGE_USER_TEMPLATE = """
Evaluate the candidate item for compliance with the synthesis requirements defined in the system prompt.

Document
- ID: {doc_id}
- Name: {doc_name}
- Pages: {page_start}–{page_end}

Candidate item JSON
{candidate_json}

Supporting evidence excerpts
{evidence_text}

Answer context window
{answer_context}

Full window text
{window_text}

Return only the single JSON object described in the system prompt.
"""
