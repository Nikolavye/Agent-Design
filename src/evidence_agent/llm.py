from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any


class LLMClient:
    def __init__(self, enabled: bool = True):
        self.model = os.getenv("EVIDENCE_AGENT_MODEL", "gpt-5.1")
        self.enabled = enabled and bool(os.getenv("OPENAI_API_KEY"))
        self._client = None
        self.last_error: str | None = None
        if self.enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except Exception as exc:
                self.enabled = False
                self.last_error = f"{type(exc).__name__}: {exc}"

    def text(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1200) -> str | None:
        if not self.enabled or self._client is None:
            return None
        self.last_error = None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def json(self, system: str, user: str, default_response: dict[str, Any] | None = None) -> dict[str, Any]:
        default_response = default_response or {}
        if not self.enabled or self._client is None:
            return default_response
        self.last_error = None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return self._parse_json_object(content)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return default_response

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        stripped = content.strip()
        try:
            parsed = json.loads(stripped)
        except JSONDecodeError:
            parsed, _end = json.JSONDecoder().raw_decode(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was valid JSON but not an object.")
        return parsed
