"""Quality checks for synthesized question-answer pairs."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

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


def is_entity_anchored(question: str, answer_text: str, window_text: str) -> bool:
    q_tokens = [tok for tok in _tokenize(question) if tok not in _STOPWORDS]
    if not q_tokens:
        return False
    answer_tokens = [tok for tok in _tokenize(answer_text) if tok not in _STOPWORDS]
    overlap = set(q_tokens) & set(answer_tokens)
    if overlap:
        return True
    # Fall back to local context around the first occurrence of the answer
    answer_norm = answer_text.strip().lower()
    if not answer_norm:
        return False
    idx = window_text.lower().find(answer_norm)
    if idx == -1:
        return False
    radius = 100
    start = max(0, idx - radius)
    end = min(len(window_text), idx + len(answer_text) + radius)
    context_tokens = [tok for tok in _tokenize(window_text[start:end]) if tok not in _STOPWORDS]
    return bool(set(q_tokens) & set(context_tokens))


def has_banned_opening(question: str, banned: Sequence[str]) -> bool:
    prefix = question.strip().lower()
    return any(prefix.startswith(entry.lower()) for entry in banned)


def no_vague_pronoun(question: str) -> bool:
    tokens = _tokenize(question)
    if not tokens:
        return False
    return tokens[0] not in _PRONOUN_HEADS


def readability_bounds(question: str, min_len: int = 8, max_len: int = 160) -> bool:
    stripped = question.strip()
    if not stripped.endswith("?"):
        return False
    if len(stripped) < min_len or len(stripped) > max_len:
        return False
    words = stripped.split()
    return len(words) >= 3
