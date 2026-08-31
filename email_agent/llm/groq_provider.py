"""Groq-backed classifier (default cloud provider)."""

from __future__ import annotations

import logging
from typing import Any

from email_agent.llm.base import (
    LLMConfigError,
    LLMError,
    LLMProvider,
    LLMTransientError,
    truncate,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.1-8b-instant"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _import_groq() -> Any:
    try:
        import groq
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise LLMConfigError(
            "The 'groq' package is required for LLM_PROVIDER=groq. "
            "Install it with: pip install groq"
        ) from exc
    return groq


class GroqProvider(LLMProvider):
    """Classifies emails with Groq's OpenAI-compatible chat completions API."""

    name = "groq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model or DEFAULT_MODEL, **kwargs)
        self._client = client
        self._api_key = api_key
        self._base_url = base_url

    def warmup(self) -> None:
        """Build the client so misconfiguration fails fast (no network call)."""
        _ = self.client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        if not self._api_key:
            raise LLMConfigError(
                "GROQ_API_KEY is not set. Export it or add it to your .env file."
            )
        groq = _import_groq()
        options: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": self.timeout_s,
            # Retries are handled by the provider layer so behaviour is uniform.
            "max_retries": 0,
        }
        if self._base_url:
            options["base_url"] = self._base_url
        return groq.Groq(**options)

    def _complete_json(self, *, system: str, user: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
                response_format={"type": "json_object"},
                timeout=self.timeout_s,
            )
        except Exception as exc:  # mapped to the provider-agnostic error types
            raise self._map_error(exc) from exc

        return self._extract_content(response)

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMTransientError("groq returned no choices")
        message = getattr(choices[0], "message", None)
        return getattr(message, "content", None) or ""

    def _map_error(self, exc: Exception) -> LLMError:
        groq_module = None
        try:
            groq_module = _import_groq()
        except LLMConfigError:  # pragma: no cover - only when groq is missing
            pass

        if groq_module is not None:
            if isinstance(exc, groq_module.AuthenticationError):
                return LLMConfigError("groq rejected the API key (check GROQ_API_KEY)")
            if isinstance(
                exc,
                (
                    groq_module.APITimeoutError,
                    groq_module.APIConnectionError,
                    groq_module.RateLimitError,
                    groq_module.InternalServerError,
                ),
            ):
                return LLMTransientError(f"{type(exc).__name__}: {truncate(str(exc))}")

        status_code = getattr(exc, "status_code", None)
        if status_code in RETRYABLE_STATUS_CODES:
            return LLMTransientError(f"groq HTTP {status_code}: {truncate(str(exc))}")
        if isinstance(exc, TimeoutError):
            return LLMTransientError(f"groq timeout: {truncate(str(exc))}")
        if isinstance(exc, LLMError):
            return exc
        return LLMError(f"groq call failed ({type(exc).__name__}): {truncate(str(exc))}")
