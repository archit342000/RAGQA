"""Validation utilities for mined atomic facts."""

from __future__ import annotations

import hashlib
from typing import Any, List

from pydantic import BaseModel, Field, ValidationError, model_validator


class Atom(BaseModel):
    """Single mined atomic fact."""

    kind: str
    text: str
    char_start: int
    char_end: int
    labels: List[str] = Field(default_factory=list)
    evidence: List[dict] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _check_spans(self) -> "Atom":
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("invalid character span")
        if len(self.text) > 300:
            raise ValueError("text exceeds 300 characters")
        return self


def strict_span_match(window_text: str, start: int, end: int, text: str) -> bool:
    """Return True when indices match the verbatim substring in window_text."""

    if start < 0 or end > len(window_text) or end <= start:
        return False
    return window_text[start:end] == text


def validate_atoms(raw: Any, window_text: str) -> List[Atom]:
    """Validate raw LLM output against the schema and the supplied window text."""

    if not isinstance(raw, list):
        return []
    valid: List[Atom] = []
    for entry in raw:
        try:
            atom = Atom.model_validate(entry)
        except ValidationError:
            continue
        if not strict_span_match(window_text, atom.char_start, atom.char_end, atom.text):
            continue
        valid.append(atom)
    return valid


def hash_atom(doc_id: str, window_id: str, start: int, end: int) -> str:
    """Stable identifier hash for a mined atom."""

    payload = f"{doc_id}|{window_id}|{start}|{end}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()
