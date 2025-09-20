"""Quality filters and diversity utilities for gold-set question generation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import regex as re

_SIMPLE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_PRONOUN_HEADS = {"it", "they", "this", "that", "these", "those", "its", "their"}
_BANNED_SUBPHRASES = re.compile(r"\b(?:thing|stuff|something|anything)\b", re.IGNORECASE)

_WH_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "how_many": ("how many",),
    "how_much": ("how much",),
    "how": ("how",),
    "why": ("why",),
    "which": ("which",),
    "who": ("who",),
    "when": ("when",),
    "where": ("where",),
    "what": ("what",),
    "aux": (
        "does",
        "do",
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
    ),
}


def _simple_tokenize(text: str) -> List[str]:
    return _SIMPLE_TOKEN_RE.findall(text.lower())


def is_entity_anchored(question: str, slots: Dict[str, str]) -> bool:
    """Ensure the question references at least one concrete slot value."""

    q_norm = question.lower()
    slot_values = [value.strip() for value in slots.values() if isinstance(value, str) and value.strip()]
    slot_values.sort(key=len, reverse=True)
    for value in slot_values:
        if len(value) < 3:
            continue
        if value.lower() in q_norm:
            return True
        tokens = _simple_tokenize(value)
        if tokens and all(token in q_norm for token in tokens[:2]):
            return True
    return False


def has_banned_opening(question: str, banned: Sequence[str]) -> bool:
    prefix = question.strip().lower()
    return any(prefix.startswith(entry.lower()) for entry in banned)


def no_vague_pronouns(question: str) -> bool:
    tokens = _simple_tokenize(question)
    if not tokens:
        return False
    for idx, token in enumerate(tokens[:4]):
        if token in _PRONOUN_HEADS and idx < 3:
            return False
    return _BANNED_SUBPHRASES.search(question) is None


def readability_bounds(question: str, min_len: int = 8, max_len: int = 160) -> bool:
    stripped = question.strip()
    if not stripped.endswith("?"):
        return False
    char_len = len(stripped)
    if char_len < min_len or char_len > max_len:
        return False
    words = stripped.split()
    return len(words) >= 3


def answerability_check(
    page_text: str,
    answer_span: Tuple[int, int],
    question: str,
    slots: Dict[str, str],
) -> bool:
    if not page_text:
        return False
    start, end = answer_span
    if not (0 <= start < end <= len(page_text)):
        return False
    answer_text = page_text[start:end].strip()
    if not answer_text:
        return False
    answer_tokens = set(_simple_tokenize(answer_text))
    question_tokens = set(_simple_tokenize(question))
    if not question_tokens:
        return False
    if answer_tokens & question_tokens:
        return True
    span_window = page_text[max(0, start - 120) : min(len(page_text), end + 120)]
    context_tokens = set(_simple_tokenize(span_window))
    slot_tokens = {token for value in slots.values() for token in _simple_tokenize(value)}
    anchor_overlap = question_tokens & slot_tokens & context_tokens
    return bool(anchor_overlap)


def detect_wh(question: str) -> str:
    lowered = question.strip().lower()
    for label, prefixes in _WH_PREFIXES.items():
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return label
    return "what"


def enforce_wh_distribution(questions: List[str], targets: Dict[str, float], seed: int) -> List[str]:
    if not questions:
        return []
    total = len(questions)
    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for idx, question in enumerate(questions):
        buckets[detect_wh(question)].append((idx, question))
    for entries in buckets.values():
        rng.shuffle(entries)

    desired_counts: Dict[str, int] = {}
    consumed = 0
    remainders: List[Tuple[float, str]] = []
    for label, share in targets.items():
        if share <= 0:
            desired_counts[label] = 0
            continue
        raw = share * total
        base = int(math.floor(raw))
        consumed += base
        desired_counts[label] = base
        remainders.append((raw - base, label))
    remaining = max(0, total - consumed)
    remainders.sort(reverse=True)
    for frac, label in remainders:
        if remaining <= 0:
            break
        if label in buckets:
            desired_counts[label] += 1
            remaining -= 1

    selected_pairs: List[Tuple[int, str]] = []
    unused_pairs: List[Tuple[int, str, str]] = []
    for label, entries in buckets.items():
        limit = desired_counts.get(label, 0)
        take = min(len(entries), limit)
        selected_pairs.extend(entries[:take])
        if take < len(entries):
            unused_pairs.extend((idx, question, label) for idx, question in entries[take:])

    leftover_needed = total - len(selected_pairs)
    if leftover_needed > 0:
        fill_pool: List[Tuple[int, str]] = []
        for idx, question, label in unused_pairs:
            if label not in targets or desired_counts.get(label, 0) < len(buckets[label]):
                fill_pool.append((idx, question))
        if not fill_pool and unused_pairs:
            fill_pool = [(idx, question) for idx, question, _ in unused_pairs]
        rng.shuffle(fill_pool)
        selected_pairs.extend(fill_pool[:leftover_needed])

    selected_pairs.sort(key=lambda pair: pair[0])
    return [question for _, question in selected_pairs]


def mmr_select(questions: List[str], page_text: str, k: int, lambda_: float = 0.7) -> List[str]:
    if k <= 0 or not questions:
        return []
    tokens_page = set(_simple_tokenize(page_text)) if page_text else set()
    question_tokens = [set(_simple_tokenize(q)) for q in questions]
    relevance_scores = [len(tokens & tokens_page) for tokens in question_tokens]
    selected: List[int] = []
    candidates = list(range(len(questions)))
    while candidates and len(selected) < k:
        best_idx = None
        best_score = -float("inf")
        for idx in candidates:
            redundancy = 0.0
            for chosen in selected:
                overlap = question_tokens[idx] & question_tokens[chosen]
                union = question_tokens[idx] | question_tokens[chosen]
                if union:
                    redundancy = max(redundancy, len(overlap) / len(union))
            mmr_value = lambda_ * relevance_scores[idx] - (1 - lambda_) * redundancy
            if mmr_value > best_score:
                best_score = mmr_value
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
    return [questions[idx] for idx in selected]


__all__ = [
    "answerability_check",
    "detect_wh",
    "enforce_wh_distribution",
    "has_banned_opening",
    "is_entity_anchored",
    "mmr_select",
    "no_vague_pronouns",
    "readability_bounds",
]
