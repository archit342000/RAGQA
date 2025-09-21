"""Quality checks for synthesized question-answer pairs."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    context: Mapping[str, str] | str,
    window_text: str | None = None,
) -> bool:
    """Return ``True`` when question terms overlap with answer/context tokens."""

    q_tokens = [tok for tok in _tokenize(question) if tok not in _STOPWORDS]
    if not q_tokens:
        return False

    sources: List[str] = []
    if isinstance(context, Mapping):
        sources.extend(str(value) for value in context.values())
    else:
        sources.append(str(context))
    if window_text:
        sources.append(window_text)

    for source in sources:
        if not source:
            continue
        tokens = [tok for tok in _tokenize(source) if tok not in _STOPWORDS]
        if set(q_tokens) & set(tokens):
            return True
    return False


def has_banned_opening(question: str, banned: Sequence[str]) -> bool:
    prefix = question.strip().lower()
    return any(prefix.startswith(entry.lower()) for entry in banned)


def no_vague_pronoun(question: str) -> bool:
    tokens = _tokenize(question)
    if not tokens:
        return False
    return tokens[0] not in _PRONOUN_HEADS


def no_vague_pronouns(question: str) -> bool:
    """Compatibility wrapper returning :func:`no_vague_pronoun`."""

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
    span: Tuple[int, int],
    question: str,
    slots: Mapping[str, str] | None,
) -> bool:
    """Return ``True`` when the span/question pair is answerable on the page.

    The heuristics intentionally stay light-weight so they can run during test
    fixtures without additional dependencies. We require:

    * a valid, non-empty span inside ``page_text``;
    * a question containing at least one meaningful token; and
    * lexical overlap between the question and either the answer span or the
      extracted slot metadata.

    These checks catch obvious failures (e.g. span mismatch, vague questions)
    while remaining permissive enough for synthetic fixtures.
    """

    if not isinstance(span, (tuple, list)) or len(span) != 2:
        return False
    try:
        start = int(span[0])
        end = int(span[1])
    except (TypeError, ValueError):
        return False
    if start < 0 or end <= start:
        return False
    if start >= len(page_text) or end > len(page_text):
        return False

    answer = page_text[start:end].strip()
    if not answer:
        return False

    q_tokens = [tok for tok in _tokenize(question) if tok not in _STOPWORDS]
    if not q_tokens:
        return False

    slots = slots or {}
    slot_tokens = [tok for tok in _tokenize(" ".join(slots.values())) if tok not in _STOPWORDS]
    answer_tokens = [tok for tok in _tokenize(answer) if tok not in _STOPWORDS]

    vocabulary = set(slot_tokens) | set(answer_tokens)
    if vocabulary & set(q_tokens):
        return True

    doc_hint = slots.get("doc_name") or slots.get("heading")
    if doc_hint and doc_hint.lower() in question.lower():
        return True

    return False


def _normalise_question(question: str) -> str:
    return question.strip()


def _unique_questions(questions: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for question in questions:
        normalised = _normalise_question(question)
        if not normalised:
            continue
        key = normalised.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalised)
    return ordered


def enforce_wh_distribution(
    questions: Sequence[str],
    targets: Mapping[str, float],
    seed: int = 0,
) -> List[str]:
    """Trim questions so WH buckets respect the configured proportions."""

    unique_questions = _unique_questions(questions)
    if not unique_questions or not targets:
        return unique_questions

    total = len(unique_questions)
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, question in enumerate(unique_questions):
        grouped[detect_wh(question)].append(idx)

    rng = random.Random(seed)
    selected_indices: Set[int] = set()

    for wh, indices in grouped.items():
        share = targets.get(wh)
        if share is None:
            selected_indices.update(indices)
            continue
        limit = int(math.ceil(total * max(share, 0.0)))
        if limit <= 0:
            continue
        if len(indices) <= limit:
            selected_indices.update(indices)
            continue
        shuffled = indices[:]
        rng.shuffle(shuffled)
        chosen = sorted(shuffled[:limit])
        selected_indices.update(chosen)

    if not selected_indices:
        return unique_questions[:1]

    return [unique_questions[idx] for idx in sorted(selected_indices)]


def mmr_select(
    candidates: Sequence[str],
    reference_text: str,
    *,
    k: int,
    lambda_: float = 0.7,
) -> List[str]:
    """Select up to ``k`` diverse questions using Maximal Marginal Relevance."""

    unique_candidates = _unique_questions(candidates)
    if not unique_candidates or k <= 0:
        return []
    if len(unique_candidates) <= k:
        return unique_candidates

    lambda_ = float(lambda_)
    if not (0.0 < lambda_ <= 1.0):
        lambda_ = 0.7

    corpus = unique_candidates + [reference_text or ""]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    questions_matrix = matrix[:-1]
    reference_vector = matrix[-1]

    if reference_vector.nnz:
        relevance_scores = cosine_similarity(questions_matrix, reference_vector).ravel().tolist()
    else:
        relevance_scores = [0.0] * len(unique_candidates)

    candidate_indices = list(range(len(unique_candidates)))
    selected: List[int] = []

    while candidate_indices and len(selected) < k:
        best_idx = None
        best_score = -float("inf")
        for idx in candidate_indices:
            relevance_score = float(relevance_scores[idx])
            redundancy = 0.0
            if selected:
                sims = cosine_similarity(questions_matrix[idx], questions_matrix[selected])
                redundancy = float(sims.max()) if sims.size else 0.0
            score = lambda_ * relevance_score - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    return [unique_candidates[idx] for idx in selected]


__all__ = [
    "answerability_check",
    "detect_wh",
    "enforce_wh_distribution",
    "has_banned_opening",
    "is_entity_anchored",
    "mmr_select",
    "no_vague_pronoun",
    "no_vague_pronouns",
    "readability_bounds",
]
