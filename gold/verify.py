"""Validation utilities for synthesized QA items."""

from __future__ import annotations

import re
from typing import Any, List

import orjson
from pydantic import BaseModel, Field, ValidationError, field_validator

ALLOWED_WH = {
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "how_many",
    "how_much",
}

ALLOWED_TYPES = {
    "numeric",
    "comparison",
    "procedural",
    "temporal",
    "definitional",
    "multi-hop",
    "location",
    "cause-effect",
    "verification",
}

ALLOWED_EVIDENCE_TYPES = {
    "sentence",
    "list_item",
    "table_cell",
    "figure_caption",
}


def canonicalize_wh(value: str) -> str:
    """Normalize WH values emitted by the LLM prompt."""

    if not isinstance(value, str):
        return ""
    lowered = value.strip().lower()
    if not lowered:
        return ""
    normalized = re.sub(r"[\s-]+", "_", lowered)
    return normalized


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


class EvidenceItem(BaseModel):
    type: str
    index: int

    model_config = {"extra": "forbid"}

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("evidence.type must be a string")
        normalized = value.strip().lower()
        normalized = normalized.replace("-", "_")
        normalized = re.sub(r"\s+", "_", normalized)
        if normalized not in ALLOWED_EVIDENCE_TYPES:
            raise ValueError("unsupported evidence type")
        return normalized

    @field_validator("index")
    @classmethod
    def _validate_index(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("evidence.index must be an integer")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("evidence.index must be an integer") from exc
        if index < 0:
            raise ValueError("evidence.index must be non-negative")
        return index


class SynthItem(BaseModel):
    question: str
    wh: str
    type: str
    answer_text: str
    evidence: List[EvidenceItem] = Field(default_factory=list)

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
        if value.count("?") != 1 or not value.endswith("?"):
            raise ValueError("question must end with a single question mark")
        return value

    @field_validator("wh")
    @classmethod
    def _normalize_wh(cls, value: str) -> str:
        lowered = canonicalize_wh(value)
        if lowered not in ALLOWED_WH:
            # Allow custom WH but ensure non-empty
            if not lowered:
                raise ValueError("wh must be provided")
        return lowered

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("type must be a string")
        lowered = value.strip().lower()
        normalized = re.sub(r"[\s_]+", "-", lowered)
        if normalized not in ALLOWED_TYPES:
            raise ValueError("unsupported question type")
        return normalized

    @field_validator("answer_text")
    @classmethod
    def _check_answer(cls, value: str) -> str:
        if not value:
            raise ValueError("answer_text cannot be empty")
        if len(value) > 300:
            raise ValueError("answer_text too long")
        return value

    @field_validator("evidence")
    @classmethod
    def _check_evidence(cls, value: List[EvidenceItem]) -> List[EvidenceItem]:
        if not value:
            raise ValueError("evidence cannot be empty")
        return value


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
