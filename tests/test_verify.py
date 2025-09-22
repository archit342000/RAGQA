import pytest
from pydantic import ValidationError

from gold.verify import SynthItem


def _make_payload(question: str) -> dict:
    return {
        "question": question,
        "wh": "what",
        "type": "definitional",
        "answer_text": "Data retention policy",
        "evidence": ["The policy covers data retention requirements."],
    }


def test_synth_item_accepts_valid_question() -> None:
    payload = _make_payload("What is the retention period for backups?")
    item = SynthItem.model_validate(payload)
    assert item.question == payload["question"]


def test_synth_item_rejects_first_person_language() -> None:
    payload = _make_payload("What should I do about the outage?")
    with pytest.raises(ValidationError):
        SynthItem.model_validate(payload)


def test_synth_item_rejects_second_person_language() -> None:
    payload = _make_payload("Where should you store the audit logs?")
    with pytest.raises(ValidationError):
        SynthItem.model_validate(payload)


def test_synth_item_rejects_vague_placeholder() -> None:
    payload = _make_payload("Who alerted someone about the incident?")
    with pytest.raises(ValidationError):
        SynthItem.model_validate(payload)


def test_synth_item_rejects_meta_reference() -> None:
    payload = _make_payload("According to the text, when was the policy updated?")
    with pytest.raises(ValidationError):
        SynthItem.model_validate(payload)
