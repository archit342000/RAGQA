"""Validation utilities for synthesized QA items."""

from __future__ import annotations

import re
from typing import Any, List

import orjson
from pydantic import BaseModel, Field, ValidationError, field_validator

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
