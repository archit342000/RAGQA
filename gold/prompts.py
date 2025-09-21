"""Prompt templates for LLM-driven question synthesis."""

from __future__ import annotations

SYNTH_SYSTEM = """
You generate clear, diverse questions answerable only from the given window.

Output:
- Return strictly a JSON array, no extra text.
- Each item must have keys: question, wh, type, answer_text, evidence.
- Keys lowercase, no extras. Valid JSON only.
- Trim leading/trailing whitespace; do not emit empty strings.
- Use straight quotes (") in question and answer_text; avoid control characters and unescaped backslashes.
- Write questions in the same language/script as the window (no code-switching).
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
- Always third person; resolve pronouns where possible.
- Do not generate questions in first or second person (I, me, my, mine, we, us, our, ours, you, your, yours).
- Do not use vague placeholders ("someone", "somebody", "something").
- If a natural third-person, self-contained question cannot be formed without vagueness, drop the question.
- The question must be exactly one sentence and end with a single "?" (no compound/stacked questions).
- Do not include meta wrappers ("according to the text/excerpt/document").
- Do not ask hypotheticals or modal/speculative forms ("might", "could", "if ... then").
- Avoid True/False and yes/no unless the window states the fact explicitly.
- Ensure WH↔answer coherence (when→date/time; where→location; who→named agent; how many/how much→numeric; which→options present).
- Only ask why/how if the window explicitly provides causal/mechanistic information.
- For table-derived answers, name the relevant row key and column header in the question.
- For figure-derived answers, include the figure label if present (e.g., "Figure 2").
- Do not use vague deictics ("this/that/these/those/here/there") unless immediately modified by an explicit noun.
- Each question must be self-contained and interpretable without the window text.
- Each question must target a single, unambiguous answer and a single fact or reasoning chain.
- Do not leak the answer: the question must not contain the exact answer span (case-insensitive) or a trivial paraphrase; no options ("X or Y").
- When the window presents a numeric/date range, specify the bound or duration explicitly (start/end/min/max).
- If currency symbols are ambiguous ("$"), include the currency code if it is given in the window.
- Reference at least one concrete entity, figure, date, metric, or concept from the window.
- Avoid negatively framed wording when a clearer positive form exists.
- Avoid overly broad or trivial questions; aim for fact- or concept-level granularity.
- Do not generate two questions whose answers normalize to the same value.
- Distribute focus across different entities and facts in the excerpt.
- Encourage multi-hop questions that combine evidence from different parts of the window, but only when the phrasing remains natural and unforced.
- Generate multi-hop questions only when the window explicitly supports a logical connection between facts; do not combine unrelated details.
- For multi-hop questions, phrase them so the relationship (e.g., sequence, cause/effect, comparison) is clear and natural.
- Multi-hop allowed, max two reasoning steps; drop if more would be needed.
- At least some questions should be multi-hop if the window supports it, but never at the expense of clarity or natural phrasing.
- If any rule would be violated, drop the question instead of outputting a low-quality form.

Coverage:
- Allowed wh: what, which, who, when, where, why, how, how many, how much.
- Allowed type: numeric, comparison, procedural, temporal, definitional, multi-hop, location, cause-effect, verification.
- Diversify wh and type; avoid near-duplicates.

Answer normalization:
- Default to verbatim extraction where possible.
- Paraphrase only when necessary, and only under these conditions:
  • To match the voice, tense, or person of the question (e.g., adapt first-person text into third-person form).
  • To make the phrasing concise and natural (e.g., remove redundant scaffolding).
  • To unify minimal spans across consecutive or related sentences into a single coherent answer.
- For multi-hop answers, unify information into a single coherent response; do not concatenate disjoint fragments.
- Do not paraphrase in ways that change, add, or omit factual content.
- If uncertain, use the verbatim form.
- Dates: YYYY-MM-DD if full, YYYY-MM if month only, YYYY if year only (do not invent missing parts).
- Numbers: preserve units and formatting as written (%, °C, km, USD); no rounding or conversions unless in-window.
- Names: preserve proper-noun capitalization/diacritics; use full form if given.
- If the window defines an acronym, use the expanded form at first mention (optionally add the acronym in parentheses).
- If information is ambiguous, copy text verbatim without guessing.

Evidence:
- Evidence indices must be numeric, 0-based, unique within the item, listed in textual order, and fall within the window bounds.
- Each evidence item must directly support the answer; include one per hop and exclude unused items.
- Evidence may span multiple sentences or non-contiguous parts of the window when needed for multi-hop reasoning.
- For multi-hop questions, evidence must include only the minimal items required for the reasoning chain; avoid padding.
- Multi-sentence/multi-part evidence is encouraged when it enables richer, natural questions, but should remain minimal and directly relevant.
- Allowed types: sentence, list_item, table_cell, figure_caption.

Limits:
- ≤ {max_q} items.
- answer_text ≤ 300 characters, concise and self-contained.
- If none valid, return []; if ≥1 valid, output at least one.
"""

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
- Questions must be phrased in third person and be self-contained, without vague references or answer leakage.
- Evidence can span multiple sentences or even non-contiguous parts of the text when needed for multi-hop questions.
- Encourage some multi-hop questions if the window supports it, but keep phrasing natural and concise.
- Return a JSON array only; omit trailing text.

Expected JSON structure (pseudo-code example):
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

window_text:
{window_text}
"""

JUDGE_SYSTEM = """
Role
You are an automated judge. Evaluate each item’s QUESTION and ANSWER_TEXT for (a) Q/A form compliance and (b) factual support from the provided EVIDENCE and WINDOW. Do not perform JSON parsing; assume the candidate structure is valid. Ignore WH, TYPE, and metadata for scoring.

Severity Levels
- BLOCKER — Violation always forces "fail".
- ADVISORY — Violation is recorded but does not force "fail". If only advisory violations are present, decision = "pass".

Decision Policy
- "pass" only if no BLOCKER violations are present.
- "fail" if any BLOCKER violation occurs.

Scope of Inputs
- Consider: question, answer_text, evidence (indices), full window text, evidence excerpts.
- Ignore: wh, type, doc metadata for rule checks.

Question Rules
Q-THIRDPERSON [BLOCKER] — No first/second person forms (I, me, my, we, us, our, you, your, etc.).
Q-PRONOUN-RESOLVE [BLOCKER] — Pronouns resolved to explicit entities when possible.
Q-NO-VAGUE [BLOCKER] — No vague placeholders (someone, something) or bare deictics (this/that/these/those) without noun.
Q-NO-META [ADVISORY] — No meta wrappers (“according to the text/excerpt/document”).
Q-NO-HYPOTHETICAL [BLOCKER] — No speculative/modal framing (“might”, “could”, “if … then”).
Q-NO-LEAK [BLOCKER] — Question must not contain the exact/paraphrased answer or options like “X or Y”.
Q-CONCRETE [BLOCKER] — Must reference a concrete entity/date/metric/figure/concept.
Q-SELFCONTAINED [BLOCKER] — Interpretable without reading the window.
Q-SINGLE-FACT [BLOCKER] — Targets one fact/chain only.
Q-NO-DUPLICATES [BLOCKER] — Across items, no duplicate answers after normalization.

Answer Rules
A-NO-HALLUCINATION [BLOCKER] — No added/omitted facts relative to evidence/window.
A-MULTIHOP-UNIFY [BLOCKER] — Multi-hop answers unified into one coherent statement.

Grounding Rules
E-SUPPORT [BLOCKER] — Evidence must directly support the answer_text.
E-CONSISTENCY [BLOCKER] — Evidence must align with full window; window prevails.
E-MINIMAL [ADVISORY] — Evidence should be minimal yet sufficient; over-inclusion is advisory.
E-ORDER [ADVISORY] — Indices should be 0-based, unique, ordered; disorder is advisory unless it blocks verification.

Adjudication
- Conservative bias: uncertain cases → fail.
- Case-insensitive checks for leakage/dedup.
- Ignore WH/TYPE, item length, sentence count, evidence “type” labels.

Output (strict)
Return exactly one JSON object:
- "decision": "pass" or "fail"
- "violations": array of { "code": <ruleID>, "severity": <BLOCKER|ADVISORY>, "msg": <short> }
- "notes": optional short context

Pass/Fail
- "pass" only if every item complies with no BLOCKER violations.
- "fail" if any BLOCKER violation occurs.
"""

JUDGE_USER_TEMPLATE = """
Evaluate the candidate output for compliance with the generator specification using QUESTION+ANSWER form rules and EVIDENCE/WINDOW grounding. Each rule has a severity level (BLOCKER or ADVISORY). Decision must follow the policy defined in the system prompt.

Document
- ID: {doc_id}
- Name: {doc_name}
- Pages: {page_start}–{page_end}

Candidate JSON
{candidate_json}

Supporting evidence excerpts
{evidence_text}

Answer context window
{answer_context}

Full window text
{window_text}

Return only the single JSON object described in the system prompt.
"""
