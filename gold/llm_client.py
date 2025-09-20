"""Thin OpenAI-compatible client for vLLM-backed models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class VLLMClient:
    """HTTP client wrapper around the OpenAI-compatible chat completions API."""

    base_url: str
    api_key: str
    model: str
    timeout_s: int = 60
    max_retries: int = 5
    backoff_base: float = 1.5

    def __post_init__(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self._client = httpx.Client(base_url=self.base_url.rstrip("/"), headers=headers, timeout=self.timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VLLMClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.close()

    def chat(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        response_format_json: bool = True,
    ) -> str:
        """Issue a chat completion call and return the model's text content."""

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.RequestError as exc:  # network issue
                logger.warning("LLM request error on attempt %d/%d: %s", attempt, self.max_retries, exc)
                self._sleep(attempt)
                continue

            if response.status_code == 200:
                try:
                    data = response.json()
                    choice = data["choices"][0]["message"]["content"]
                    if not isinstance(choice, str):
                        raise ValueError("Unexpected response format: content is not a string")
                    return choice
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(f"Malformed completion payload: {response.text}") from exc

            if response.status_code in {429, 500, 502, 503, 504}:
                logger.warning(
                    "LLM returned status %s on attempt %d/%d; retrying",
                    response.status_code,
                    attempt,
                    self.max_retries,
                )
                self._sleep(attempt)
                continue

            raise RuntimeError(
                f"Unexpected LLM status {response.status_code}: {response.text[:400]}"
            )

        raise RuntimeError("Exceeded maximum retries for LLM request")

    def _sleep(self, attempt: int) -> None:
        delay = min(self.backoff_base**attempt, 30.0)
        time.sleep(delay)
