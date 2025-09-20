"""Dataclasses used across gold-set utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class ParsedPage:
    """Single page extracted from a parsed document."""

    doc_id: str
    doc_name: str
    page_num: int
    text: str
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None


@dataclass(slots=True)
class ParsedDoc:
    """Parsed document containing one or more pages."""

    doc_id: str
    doc_name: str
    pages: List[ParsedPage]
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class CandidateItem:
    """Intermediate candidate produced by span extraction/paraphrasing."""

    id: str
    question: str
    answer_text: str
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    tags: List[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class Evidence:
    """Evidence span supporting a final gold answer."""

    doc_id: str
    page: int
    char_start: int
    char_end: int


@dataclass(slots=True)
class GoldItem:
    """Final gold QA instance used for evaluation."""

    id: str
    question: str
    answer_text: str
    doc_id: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    tags: List[str] = field(default_factory=list)
    hard_negative_ids: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    program: Optional[str] = None


__all__ = [
    "ParsedDoc",
    "ParsedPage",
    "CandidateItem",
    "GoldItem",
    "Evidence",
]
