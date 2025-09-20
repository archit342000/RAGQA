"""Answer span alignment utilities."""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein



def _build_ws_pattern(text: str) -> Optional[re.Pattern[str]]:
    stripped = text.strip()
    if not stripped:
        return None
    parts: List[str] = []
    in_space = False
    for char in stripped:
        if char.isspace():
            if not in_space:
                parts.append(r"\s+")
                in_space = True
            continue
        parts.append(re.escape(char))
        in_space = False
    pattern = "".join(parts)
    if not pattern:
        return None
    return re.compile(pattern, flags=re.IGNORECASE)


def _distance_to_evidence(
    start: int,
    end: int,
    evidence_sentence_spans: Optional[Sequence[Tuple[int, int]]],
) -> float:
    if not evidence_sentence_spans:
        return 0.0
    center = (start + end) / 2.0
    distances = []
    for span_start, span_end in evidence_sentence_spans:
        if span_start <= center <= span_end:
            return 0.0
        distances.append(min(abs(center - span_start), abs(center - span_end)))
    if not distances:
        return math.inf
    return min(distances)


def find_span(
    window_text: str,
    answer_text: str,
    evidence_sentence_spans: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Tuple[int, int]]:
    """Locate the best matching span of ``answer_text`` inside ``window_text``."""

    if not window_text or not answer_text:
        return None
    answer = answer_text.strip()
    if not answer:
        return None

    candidates: List[Tuple[int, int, int, float]] = []  # (priority, start, end, score)
    seen: set[Tuple[int, int]] = set()

    def register(start: int, end: int, priority: int, score: float = 1.0) -> None:
        if start < 0 or end <= start or end > len(window_text):
            return
        key = (start, end)
        if key in seen:
            return
        seen.add(key)
        candidates.append((priority, start, end, score))

    # Strategy 1: exact match
    idx = window_text.find(answer)
    while idx != -1:
        register(idx, idx + len(answer), 0)
        idx = window_text.find(answer, idx + 1)

    if candidates:
        return _select_candidate(candidates, evidence_sentence_spans)

    # Strategy 2: case-insensitive match
    lower_text = window_text.lower()
    lower_answer = answer.lower()
    idx = lower_text.find(lower_answer)
    while idx != -1:
        register(idx, idx + len(answer), 1)
        idx = lower_text.find(lower_answer, idx + 1)

    if candidates:
        return _select_candidate(candidates, evidence_sentence_spans)

    # Strategy 3: whitespace-normalised match
    pattern = _build_ws_pattern(answer)
    if pattern is not None:
        for match in pattern.finditer(window_text):
            register(match.start(), match.end(), 2)
    if candidates:
        return _select_candidate(candidates, evidence_sentence_spans)

    # Strategy 4: fuzzy match with rapidfuzz
    fuzzy_candidates = _fuzzy_candidates(window_text, answer, evidence_sentence_spans)
    for start, end, score in fuzzy_candidates:
        register(start, end, 3, score)

    if not candidates:
        return None
    return _select_candidate(candidates, evidence_sentence_spans)


def _fuzzy_candidates(
    window_text: str,
    answer: str,
    evidence_sentence_spans: Optional[Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int, float]]:
    spans: List[Tuple[int, int]] = []
    if evidence_sentence_spans:
        for span_start, span_end in evidence_sentence_spans:
            start = max(0, span_start - 200)
            end = min(len(window_text), span_end + 200)
            spans.append((start, end))
    if not spans:
        spans.append((0, len(window_text)))

    results: List[Tuple[int, int, float]] = []
    for span_start, span_end in spans:
        segment = window_text[span_start:span_end]
        if not segment.strip():
            continue
        ratio = fuzz.partial_ratio(answer, segment)
        if ratio < 92:
            continue
        ops = list(Levenshtein.opcodes(answer, segment))
        seg_start: Optional[int] = None
        seg_end: Optional[int] = None
        for op in ops:
            if op.tag == "insert":
                continue
            if seg_start is None:
                seg_start = op.dest_start
            seg_end = op.dest_end
        if seg_start is None or seg_end is None or seg_end <= seg_start:
            continue
        matched = segment[seg_start:seg_end]
        fine = max(fuzz.ratio(answer, matched), fuzz.partial_ratio(answer, matched))
        if fine < 92:
            continue
        results.append((span_start + seg_start, span_start + seg_end, fine))
    return results


def _select_candidate(
    candidates: Sequence[Tuple[int, int, int, float]],
    evidence_sentence_spans: Optional[Sequence[Tuple[int, int]]],
) -> Tuple[int, int]:
    best = None
    best_key = None
    for priority, start, end, score in candidates:
        distance = _distance_to_evidence(start, end, evidence_sentence_spans)
        key = (priority, distance, -score, start)
        if best_key is None or key < best_key:
            best = (start, end)
            best_key = key
    return best  # type: ignore[return-value]
