"""
Resilient LLM client.

The system is OpenAI-compatible, so one client (the `openai` SDK) talks to any
provider that exposes a /v1 endpoint. Providers are tried in priority order:
Gemini first (free tier, reliable, native JSON mode), then Hugging Face Inference
(open models, free). On any error from the primary, we transparently fall back.

Usage:
    from app.llm.client import llm
    text = llm.chat(system="...", user="...")
    obj = llm.extract_json(system="...", user="...")
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import LLMProvider, settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when all configured providers fail."""


class LLMClient:
    """OpenAI-compatible client with ordered provider fallback."""

    def __init__(self, providers: Optional[list[LLMProvider]] = None) -> None:
        self.providers: list[LLMProvider] = providers if providers is not None else settings.llm_providers
        # One OpenAI SDK client per provider (cheap; just config).
        self._clients: dict[str, OpenAI] = {
            p.name: OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=settings.llm_timeout_seconds)
            for p in self.providers
        }
        if not self.providers:
            logger.warning(
                "No LLM provider configured. Set GEMINI_API_KEY or HF_TOKEN in .env. "
                "AI features (extraction, generation) will raise LLMError until configured."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        """Plain text completion. Tries providers in order."""
        return self._complete(system, user, temperature or settings.extraction_temperature)

    def extract_json(self, system: str, user: str, temperature: Optional[float] = None) -> Any:
        """Completion that is expected to be JSON. Parses and returns the object."""
        raw = self._complete(
            system, user, temperature or settings.extraction_temperature, json_mode=True
        )
        return self._parse_json(raw)

    @property
    def is_configured(self) -> bool:
        return len(self.providers) > 0

    @property
    def primary_provider_name(self) -> str:
        return self.providers[0].name if self.providers else "none"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _complete(
        self, system: str, user: str, temperature: float, json_mode: bool = False
    ) -> str:
        if not self.providers:
            raise LLMError(
                "No LLM provider configured. Set GEMINI_API_KEY and/or HF_TOKEN in your .env file."
            )

        last_error: Optional[Exception] = None
        for provider in self.providers:
            try:
                client = self._clients[provider.name]
                kwargs: dict[str, Any] = {
                    "model": provider.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                logger.debug("LLM %s responded (%d chars)", provider.name, len(content))
                return content
            except Exception as exc:  # noqa: BLE001 — fall through to next provider
                last_error = exc
                logger.warning(
                    "Provider %s failed: %s. Trying next provider if available.",
                    provider.name,
                    exc,
                )

        raise LLMError(f"All LLM providers failed. Last error: {last_error}")

    @staticmethod
    def _parse_json(raw: str) -> Any:
        """Parse JSON, tolerating markdown fences and surrounding prose."""
        # Strip ```json ... ``` fences if present.
        fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw, re.DOTALL)
        candidate = fenced.group(1) if fenced else raw
        # Otherwise try to locate the first {...} or [...].
        if not fenced:
            obj_match = re.search(r"(\{.*\}|\[.*\])", candidate, re.DOTALL)
            if obj_match:
                candidate = obj_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Could not parse LLM output as JSON: {exc}\nRaw: {raw[:500]}") from exc


# Module-level singleton.
llm = LLMClient()
