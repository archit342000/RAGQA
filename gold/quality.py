"""Quality checks for synthesized question-answer pairs."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}

_PRONOUN_HEADS = {"it", "its", "they", "their", "this", "that", "these", "those"}
_AUX_START = {
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "is",
    "are",
    "was",
    "were",
    "will",
    "has",
    "have",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def detect_wh(q: str) -> str:
    text = q.strip().lower()
    if text.startswith("how many"):
        return "how_many"
    if text.startswith("how much"):
        return "how_much"
    if text.startswith("how"):
        return "how"
    if text.startswith("why"):
        return "why"
    if text.startswith("which"):
        return "which"
    if text.startswith("who"):
        return "who"
    if text.startswith("when"):
        return "when"
    if text.startswith("where"):
        return "where"
    if text.startswith("what"):
        return "what"
    first_tokens = _tokenize(text)
    if first_tokens and first_tokens[0] in _AUX_START:
        return "aux"
    return "what"


def is_entity_anchored(
    question: str,
    anchor: Mapping[str, str] | str,
    window_text: str | None = None,
) -> bool:
    """Determine whether a question references a known entity anchor."""

    q_tokens = {tok for tok in _tokenize(question) if tok not in _STOPWORDS}
    if not q_tokens:
        return False

    if isinstance(anchor, Mapping):
        anchor_tokens: Set[str] = set()
        preferred_keys = (
            "term",
            "topic",
            "acronym",
            "metric",
            "entity_type",
            "process",
            "decision",
            "event",
            "responsibility",
            "subject",
            "doc_name",
            "heading",
            "section",
            "scope",
        )
        for key in preferred_keys:
            value = anchor.get(key)
            if value:
                anchor_tokens.update(_tokenize(value))
        if not anchor_tokens:
            for value in anchor.values():
                anchor_tokens.update(_tokenize(value))
        anchor_tokens = {tok for tok in anchor_tokens if tok not in _STOPWORDS}
        if not anchor_tokens:
            return False
        return bool(q_tokens & anchor_tokens)

    answer_text = str(anchor or "")
    window = window_text or ""
    anchor_tokens = {
        tok for tok in _tokenize(answer_text + " " + window) if tok not in _STOPWORDS
    }
    if not anchor_tokens:
        return False
    return bool(q_tokens & anchor_tokens)


def has_banned_opening(question: str, banned: Sequence[str]) -> bool:
    prefix = question.strip().lower()
    return any(prefix.startswith(entry.lower()) for entry in banned)


def no_vague_pronoun(question: str) -> bool:
    tokens = _tokenize(question)
    if not tokens:
        return False
    return tokens[0] not in _PRONOUN_HEADS


def no_vague_pronouns(question: str) -> bool:
    """Backward compatible alias exposed for legacy callers."""

    return no_vague_pronoun(question)


def readability_bounds(question: str, min_len: int = 8, max_len: int = 160) -> bool:
    stripped = question.strip()
    if not stripped.endswith("?"):
        return False
    if len(stripped) < min_len or len(stripped) > max_len:
        return False
    words = stripped.split()
    return len(words) >= 3


def answerability_check(
    page_text: str,
    answerspan: Tuple[int, int],
    question: str,
    slots: Mapping[str, str] | None,
) -> bool:
    """Validate that the answer span and question are grounded in the source."""

    if not page_text or not question:
        return False
    if not isinstance(answerspan, tuple) or len(answerspan) != 2:
        return False
    try:
        start = int(answerspan[0])
        end = int(answerspan[1])
    except (TypeError, ValueError):
        return False
    if start < 0 or end <= start or end > len(page_text):
        return False

    answer_text = page_text[start:end].strip()
    if not answer_text:
        return False

    question_tokens = {tok for tok in _tokenize(question) if tok not in _STOPWORDS}
    answer_tokens = {tok for tok in _tokenize(answer_text) if tok not in _STOPWORDS}
    if not answer_tokens:
        return False

    if question_tokens & answer_tokens:
        return True

    slot_tokens: Set[str] = set()
    for value in (slots or {}).values():
        slot_tokens.update(_tokenize(value))
    slot_tokens = {tok for tok in slot_tokens if tok not in _STOPWORDS}
    if slot_tokens & answer_tokens:
        return True

    radius = 120
    window_start = max(0, start - radius)
    window_end = min(len(page_text), end + radius)
    context_tokens = {
        tok for tok in _tokenize(page_text[window_start:window_end]) if tok not in _STOPWORDS
    }
    if question_tokens & context_tokens:
        return True

    return False


def enforce_wh_distribution(
    questions: Sequence[str],
    targets: Mapping[str, float],
    seed: int,
) -> List[str]:
    """Subsample questions so WH categories respect requested shares."""

    if not questions:
        return []

    total = len(questions)
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, question in enumerate(questions):
        buckets[detect_wh(question)].append(idx)

    rng = random.Random(seed)
    selected: Set[int] = set()

    for wh, indices in buckets.items():
        share = targets.get(wh)
        if share is None:
            selected.update(indices)
            continue
        share = max(float(share), 0.0)
        if share == 0.0:
            continue
        limit = math.ceil(total * share)
        if limit <= 0:
            limit = 1
        limit = min(len(indices), limit)
        if limit <= 0:
            continue
        shuffled = indices[:]
        rng.shuffle(shuffled)
        selected.update(shuffled[:limit])

    for wh, indices in buckets.items():
        if wh in targets and targets[wh] > 0 and indices:
            if not any(idx in selected for idx in indices):
                selected.add(indices[0])

    if not selected:
        return list(questions)

    ordered = sorted(selected)
    return [questions[idx] for idx in ordered]


def mmr_select(
    questions: Sequence[str],
    context: str,
    *,
    k: int,
    lambda_: float = 0.7,
) -> List[str]:
    """Select a diverse subset of questions via Maximal Marginal Relevance."""

    if k <= 0 or not questions:
        return []

    context_tokens = {tok for tok in _tokenize(context) if tok not in _STOPWORDS}
    question_tokens: List[Set[str]] = [
        {tok for tok in _tokenize(q) if tok not in _STOPWORDS} for q in questions
    ]

    def _relevance(tokens: Set[str]) -> float:
        if not tokens or not context_tokens:
            return 0.0
        return len(tokens & context_tokens) / float(len(tokens))

    def _redundancy(idx: int, chosen: List[int]) -> float:
        if not chosen:
            return 0.0
        current = question_tokens[idx]
        if not current:
            return 0.0
        best = 0.0
        for other in chosen:
            other_tokens = question_tokens[other]
            if not other_tokens:
                continue
            union = len(current | other_tokens)
            if union == 0:
                continue
            overlap = len(current & other_tokens) / union
            if overlap > best:
                best = overlap
        return best

    selected: List[int] = []
    remaining = list(range(len(questions)))

    while remaining and len(selected) < k:
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            relevance = _relevance(question_tokens[idx])
            redundancy = _redundancy(idx, selected)
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [questions[idx] for idx in selected]
