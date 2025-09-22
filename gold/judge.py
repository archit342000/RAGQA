"""LLM-based judge for verifying synthesized question quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import orjson

from .llm_client import VLLMClient
from .prompts import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE
from .spans import resolve_evidence_spans as _resolve_evidence_spans
from .spans import sentence_spans as _sentence_spans


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _format_template(template: str, **values) -> str:
    safe_values = {}
    for key, value in values.items():
        if isinstance(value, str):
            safe_values[key] = _escape_braces(value)
        else:
            safe_values[key] = value
    return template.format(**safe_values)


def _render_snippets(window_text: str, spans: Sequence[Tuple[int, int]]) -> List[str]:
    text_len = len(window_text)
    seen: set[Tuple[int, int]] = set()
    snippets: List[str] = []

    for start, end in spans:
        start = max(0, min(text_len, start))
        end = max(start, min(text_len, end))
        key = (start, end)
        if key in seen or start >= end:
            continue
        snippet = window_text[start:end].strip()
        if not snippet:
            continue
        seen.add(key)
        snippets.append(snippet)

    return [f"{idx}. {snippet}" for idx, snippet in enumerate(snippets, start=1)]


def _summarize_evidence(
    window_text: str,
    evidence_entries: Optional[Sequence[Any]],
    fallback_spans: Optional[Sequence[Tuple[int, int]]] = None,
) -> str:
    sentence_bounds = _sentence_spans(window_text)
    spans = _resolve_evidence_spans(evidence_entries or [], sentence_bounds)
    snippets = _render_snippets(window_text, spans)

    if not snippets and fallback_spans:
        snippets = _render_snippets(window_text, fallback_spans)

    if not snippets:
        return "No evidence snippets available."
    return "\n".join(snippets)


@dataclass
class JudgeVerdict:
    """Result returned by the LLM judge."""

    passed: bool
    reasons: List[str]
    raw: Optional[Dict] = None
    error: bool = False


class LLMJudge:
    """Use an LLM to validate synthesized question-answer pairs."""

    def __init__(
        self,
        client: VLLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        requirements: str = "",
    ) -> None:
        self._client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.requirements = requirements

    def evaluate(
        self,
        record: Dict,
        evidence_spans: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> JudgeVerdict:
        window_text = record.get("window_text", "")
        candidate_payload = {
            "doc_id": record.get("doc_id"),
            "doc_name": record.get("doc_name"),
            "page_start": record.get("page_start"),
            "page_end": record.get("page_end"),
            "window_id": record.get("window_id"),
            "question": record.get("question"),
            "wh": record.get("wh"),
            "type": record.get("type"),
            "answer_text": record.get("answer_text"),
            "evidence": record.get("evidence", []),
        }
        candidate_json = orjson.dumps(candidate_payload, option=orjson.OPT_INDENT_2).decode("utf-8")
        evidence_text = _summarize_evidence(
            window_text,
            candidate_payload.get("evidence") or [],
            evidence_spans,
        )
        user_prompt = _format_template(
            JUDGE_USER_TEMPLATE,
            requirements=self.requirements,
            doc_id=str(candidate_payload.get("doc_id", "")),
            doc_name=str(candidate_payload.get("doc_name", "")),
            page_start=str(candidate_payload.get("page_start", "")),
            page_end=str(candidate_payload.get("page_end", "")),
            candidate_json=candidate_json,
            evidence_text=evidence_text,
            window_text=window_text,
        )
        system_prompt = JUDGE_SYSTEM
        extra_requirements = self.requirements.strip()
        if (
            extra_requirements
            and "{requirements}" not in JUDGE_SYSTEM
            and "{requirements}" not in JUDGE_USER_TEMPLATE
        ):
            system_prompt = f"{system_prompt.rstrip()}\n\nAdditional requirements:\n{extra_requirements}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_text = self._client.chat(
            messages,
            self.temperature,
            self.max_tokens,
            self.seed,
            response_format_json=True,
        )
        response = self._parse_response(response_text)
        return response

    def _parse_response(self, response_text: str) -> JudgeVerdict:
        try:
            data = orjson.loads(response_text)
        except orjson.JSONDecodeError:
            return JudgeVerdict(False, ["invalid_json"], None, True)
        if not isinstance(data, dict):
            return JudgeVerdict(False, ["invalid_payload"], None, True)

        decision = data.get("decision")
        passed: Optional[bool]
        error = False

        if isinstance(decision, str):
            normalized = decision.strip().lower()
            if normalized in {"pass", "accept", "keep", "valid", "yes"}:
                passed = True
            elif normalized in {"fail", "reject", "invalid", "no"}:
                passed = False
            else:
                return JudgeVerdict(False, ["unknown_decision"], data, True)
        elif isinstance(decision, bool):
            passed = bool(decision)
        else:
            return JudgeVerdict(False, ["missing_decision"], data, True)

        reasons: List[str] = []
        reasons_field = data.get("violations")
        if reasons_field is None:
            reasons_field = data.get("reasons") or data.get("issues")
        if isinstance(reasons_field, list):
            for entry in reasons_field:
                if isinstance(entry, str):
                    trimmed = entry.strip()
                    if trimmed:
                        reasons.append(trimmed)
                elif isinstance(entry, (int, float)):
                    reasons.append(str(entry))
        elif isinstance(reasons_field, str):
            trimmed = reasons_field.strip()
            if trimmed:
                reasons.append(trimmed)

        if not passed and not reasons:
            reasons = ["unspecified_violation"]

        return JudgeVerdict(passed, reasons, data, error)


def build_judge(config: Dict, client: VLLMClient) -> Optional[LLMJudge]:
    """Create an ``LLMJudge`` instance when enabled in configuration."""

    judge_cfg = config.get("judge")
    if not isinstance(judge_cfg, dict):
        return None
    if not judge_cfg.get("enabled", False):
        return None

    temperature = float(judge_cfg.get("temperature", 0.0))
    max_tokens = int(judge_cfg.get("max_tokens", 768))
    seed: Optional[int]
    if "seed" in judge_cfg and judge_cfg["seed"] is not None:
        seed = int(judge_cfg["seed"])
    else:
        seed = config.get("seed")
        if seed is not None:
            seed = int(seed)

    requirements = judge_cfg.get("requirements")
    if isinstance(requirements, list):
        requirements_text = "\n".join(str(item) for item in requirements)
    elif isinstance(requirements, str) and requirements.strip():
        requirements_text = requirements
    else:
        requirements_text = ""

    return LLMJudge(
        client,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        requirements=requirements_text,
    )
