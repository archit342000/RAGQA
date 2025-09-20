"""Validation helpers for LLM-generated synthesis and mining payloads."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, List

import orjson
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

_ALLOWED_WH = {
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "how_many",
    "how_much",
    "aux",
}


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*", "", text, count=1).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def parse_json_array(text: str) -> List[Any]:
    """Parse a JSON array emitted by the LLM."""

    cleaned = _strip_code_fences(text)
    if not cleaned:
        return []
    try:
        data = orjson.loads(cleaned)
    except orjson.JSONDecodeError as exc:
        raise ValueError("Failed to parse JSON array") from exc
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array")
    return data


class SynthItem(BaseModel):
    question: str
    wh: str
    type: str
    answer_text: str
    evidence: List[dict] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("question", "wh", "type", "answer_text", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("question")
    @classmethod
    def _check_question(cls, value: str) -> str:
        if not value or len(value) > 320:
            raise ValueError("invalid question length")
        return value

    @field_validator("wh")
    @classmethod
    def _normalize_wh(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in _ALLOWED_WH:
            # Allow custom WH but ensure non-empty
            if not lowered:
                raise ValueError("wh must be provided")
        return lowered

    @field_validator("answer_text")
    @classmethod
    def _check_answer(cls, value: str) -> str:
        if not value:
            raise ValueError("answer_text cannot be empty")
        if len(value) > 300:
            raise ValueError("answer_text too long")
        return value


class Atom(BaseModel):
    """Schema for atomic facts produced during the mining pass."""

    kind: str
    text: str
    char_start: int
    char_end: int
    labels: List[str] = Field(default_factory=list)
    evidence: List[dict] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("kind", "text", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("kind")
    @classmethod
    def _ensure_kind(cls, value: str) -> str:
        if not value:
            raise ValueError("kind must be provided")
        return value

    @field_validator("char_start", "char_end", mode="before")
    @classmethod
    def _coerce_int(cls, value: Any) -> Any:
        if isinstance(value, bool):  # bool is subclass of int
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return value
            try:
                return int(float(stripped))
            except ValueError:
                return value
        return value

    @field_validator("labels", "tags", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Iterable):
            return []
        result: List[str] = []
        for item in value:
            if isinstance(item, str):
                candidate = item.strip()
            else:
                candidate = str(item).strip()
            if candidate:
                result.append(candidate)
        return result

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, value: Any) -> List[dict]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (str, bytes)):
            return []
        if not isinstance(value, Iterable):
            return []
        result: List[dict] = []
        for item in value:
            if isinstance(item, dict):
                result.append(item)
        return result

    @model_validator(mode="after")
    def _validate_span(self) -> "Atom":
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("invalid span indices")
        if not self.text:
            raise ValueError("text cannot be empty")
        return self


def validate_synth_items(raw: List[Any]) -> List[SynthItem]:
    """Validate raw decoded items and keep only those adhering to the schema."""

    if not isinstance(raw, list):
        return []
    valid: List[SynthItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            item = SynthItem.model_validate(entry)
        except ValidationError:
            continue
        if not item.question.endswith("?"):
            continue
        valid.append(item)
    return valid


def hash_atom(doc_id: str, window_id: str, char_start: int, char_end: int) -> str:
    """Derive a stable identifier for a mined atom span."""

    payload = f"{doc_id}|{window_id}|{char_start}|{char_end}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def strict_span_match(window_text: str, char_start: int, char_end: int, text: str) -> bool:
    """Return True if the character offsets cleanly align with the provided text."""

    if char_start < 0 or char_end <= char_start:
        return False
    if char_end > len(window_text):
        return False
    extracted = window_text[char_start:char_end]
    if extracted == text:
        return True
    return _normalize(extracted) == _normalize(text)


__all__ = [
    "SynthItem",
    "Atom",
    "parse_json_array",
    "validate_synth_items",
    "hash_atom",
    "strict_span_match",
]
