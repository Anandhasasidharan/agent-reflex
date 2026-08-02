from __future__ import annotations

import json
import os
import re
from typing import Any

from agent_reflex.common.config import Settings

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def resolve_api_key(settings: Settings) -> str:
    """Resolve the LLM provider API key from settings or common env vars."""
    for candidate in (
        settings.llm_api_key,
        settings.openai_api_key,
        os.environ.get("AGENT_REFLEX_LLM_API_KEY", ""),
        os.environ.get("AGENT_REFLEX_OPENAI_API_KEY", ""),
        os.environ.get("DEEPSEEK_API_KEY", ""),
        os.environ.get("OPENAI_API_KEY", ""),
    ):
        if candidate:
            return candidate
    return ""


def extract_json(content: str) -> dict[str, Any]:
    """Robustly extract a JSON object from an LLM response.

    Handles markdown code fences, prose-wrapped JSON, and trailing text.
    """
    if not content:
        raise ValueError("empty LLM response")

    fence_match = _JSON_FENCE_RE.search(content)
    candidate = fence_match.group(1) if fence_match else content

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {content[:200]!r}")

    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        raise ValueError(f"invalid JSON in response: {candidate[start : end + 1][:200]!r}")


class LLMClient:
    """Thin wrapper around the openai SDK for any OpenAI-compatible provider.

    DeepSeek, Ollama, vLLM, OpenRouter, etc. all speak the OpenAI protocol,
    so only base_url and model need to change.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: Any | None = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=resolve_api_key(self._settings),
                base_url=self._settings.llm_base_url,
            )
        return self._client

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self._settings.llm_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
        except Exception:
            response = self.client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
        return extract_json(content)

    def embed(self, text: str) -> list[float]:
        if not self._settings.llm_embedding_base_url:
            raise RuntimeError(
                "No embedding provider configured: set AGENT_REFLEX_LLM_EMBEDDING_BASE_URL"
            )
        import openai
        embedding_client = openai.OpenAI(
            api_key=resolve_api_key(self._settings),
            base_url=self._settings.llm_embedding_base_url,
        )
        response = embedding_client.embeddings.create(
            model=self._settings.llm_embedding_model,
            input=text,
        )
        return response.data[0].embedding
