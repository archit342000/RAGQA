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
      "evidence": [ <string>, <string>, ... ]
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

Evidence rules:
- Evidence must consist of direct verbatim excerpts copied from the window text.
- Absolutely no rephrasing, paraphrasing, summarizing, or editing of evidence text is allowed.
- Each excerpt must be the smallest span that fully supports the answer (e.g., a sentence, clause, or list item).
- Evidence may span multiple sentences or non-contiguous parts of the window when the answer requires multi-hop reasoning.
- For multi-hop, include only the minimal set of excerpts required to support the reasoning chain; do not pad with extra text.
- List evidence excerpts in the same order they appear in the window.
- Do not include entire paragraphs unless every sentence is directly needed for the answer.
- If no verbatim supporting text exists in the window, do not fabricate evidence — drop the question instead.

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
- Evidence must consist of direct verbatim excerpts copied from the window text (never rephrased or paraphrased).
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
    "evidence": ["The Basic Plus tier includes SSO and is priced at $99."]
  },
  {
    "question": "When did the retention policy change take effect?",
    "wh": "when",
    "type": "temporal",
    "answer_text": "March 1, 2024",
    "evidence": ["The retention policy was updated to take effect on March 1, 2024."]
  }
]

window_text:
{window_text}
"""

JUDGE_SYSTEM = """
Role
You are an automated judge. Evaluate each item in the candidate JSON by checking:
1) the `question` (style/clarity), 2) the `answer_text` (accuracy), and
3) that `answer_text` is supported by the cited `evidence` and appears or is derivable from the `window_text`.
Ignore `wh`, `type`, and metadata. Do not parse JSON; assume it is valid.

Decision Policy
- "pass" only if no rule violations occur.
- "fail" if any rule violation occurs.

Question Rules
Q-NO-VAGUE — The `question` must not use vague placeholders like "someone/somebody/something" or bare deictics "this/that/these/those" without a noun; standard interrogatives (who/what/when/where/why/how/how many/how much) are allowed and not considered vague.  
Q-CONCRETE — The `question` must mention at least one concrete entity, date, number, metric, or named concept.  
Q-NO-LEAK — The `question` must not reveal the `answer_text` or make it trivial to guess. A violation occurs if the `question` contains the entire answer or a significant part of it (e.g., a unique name, number, or phrase) that effectively gives the answer away. Overlaps on generic context terms (units, labels like "revenue," or common years) are allowed. Purpose: block trivial “copy-the-answer” questions, not natural contextual phrasing.

Answer Rules
A-NO-HALLUCINATION — The `answer_text` must not contradict the `evidence`/`window_text` or add unsupported substantive facts (e.g., numbers, dates, names, units, qualifiers, causal links). Harmless paraphrase, shortening, or omission of irrelevant words must not be flagged.  
A-MULTIHOP-UNIFY — If the answer draws from more than one evidence item, the `answer_text` must combine them into one clear statement.

Grounding Rules
E-SUPPORT — The `answer_text` must be clearly supported by the cited `evidence` and consistent with the `window_text`, even if expressed in paraphrased form.

Adjudication
- Fairness principle: judge compliance, not nitpick; only flag genuine violations that materially break a rule.
- When unsure between harmless paraphrase vs. unsupported fact, prefer pass if the answer is reasonably entailed; otherwise fail.
- Ignore `wh`, `type`, metadata, item length, and sentence count.

Output (strict)
Return exactly one JSON object:

{
  "decision": "pass" | "fail",
  "violations": [
    { "code": "<ruleID>", "msg": "<short explanation>" }
  ],
  "notes": "<optional short context>"
}

Pass/Fail
- "pass" only if all rules are met and there are no violations.
- "fail" if any violation occurs.
"""

JUDGE_USER_TEMPLATE = """
Evaluate the candidate JSON for compliance with the generator specification.  
Judge only the `question` and `answer_text` fields, and verify that the `answer_text` is supported by the cited `evidence` and consistent with the `window_text` (paraphrase allowed).  

Fairness principle: your job is to judge compliance, not nitpick; only flag genuine violations that materially break a rule.  

Ignore `wh`, `type`, and metadata — they are provided only for traceability.

Document
- ID: {doc_id}
- Name: {doc_name}
- Pages: {page_start}–{page_end}

Candidate JSON
{candidate_json}

Supporting `evidence` excerpts
{evidence_text}

Full `window_text`
{window_text}

Return only the single JSON object described in the system prompt.
"""
