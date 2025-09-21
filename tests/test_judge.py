from __future__ import annotations

from typing import Any, List

import pytest

from gold.judge import LLMJudge, build_judge


class DummyClient:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[dict[str, Any]] = []

    def chat(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        seed: int | None,
        response_format_json: bool = True,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "response_format_json": response_format_json,
            }
        )
        if not self.responses:
            raise RuntimeError("No responses remaining")
        return self.responses.pop(0)


def make_record() -> dict:
    window_text = "The policy covers data retention requirements."
    return {
        "doc_id": "doc-1",
        "doc_name": "policy.pdf",
        "page_start": 1,
        "page_end": 1,
        "window_id": "doc-1:0",
        "question": "What does the policy cover?",
        "wh": "what",
        "type": "definitional",
        "answer_text": "It covers data retention requirements.",
        "evidence": [{"type": "sentence", "index": 0}],
        "char_start": 4,
        "char_end": 42,
        "window_text": window_text,
    }


def test_llm_judge_accepts_valid_pair() -> None:
    client = DummyClient(['{"decision": "pass", "violations": []}'])
    judge = LLMJudge(client, temperature=0.1, max_tokens=256, seed=123)
    record = make_record()
    verdict = judge.evaluate(record, [(0, len(record["window_text"]))])

    assert verdict.passed is True
    assert verdict.error is False
    assert verdict.reasons == []
    assert client.calls, "Expected the judge to call the client"
    assert client.calls[0]["messages"][0]["role"] == "system"


def test_llm_judge_handles_invalid_json() -> None:
    client = DummyClient(["not-json"])
    judge = LLMJudge(client)
    record = make_record()
    verdict = judge.evaluate(record, [(0, len(record["window_text"]))])

    assert verdict.passed is False
    assert verdict.error is True
    assert "invalid_json" in verdict.reasons


def test_build_judge_respects_config() -> None:
    client = DummyClient([])
    config = {
        "seed": 7,
        "judge": {
            "enabled": True,
            "temperature": 0.25,
            "max_tokens": 512,
            "seed": 99,
            "requirements": ["Rule A", "Rule B"],
        },
    }

    judge = build_judge(config, client)
    assert isinstance(judge, LLMJudge)
    assert judge.temperature == pytest.approx(0.25)
    assert judge.max_tokens == 512
    assert judge.seed == 99
    assert "Rule A" in judge.requirements
    assert "Rule B" in judge.requirements

    disabled = build_judge({"judge": {"enabled": False}}, client)
    assert disabled is None
