"""Ollama-backed classifier, kept for local/offline compatibility."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from email_agent.llm.base import LLMError, LLMProvider, LLMTransientError, truncate
from email_agent.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1"
DEFAULT_BASE_URL = "http://localhost:11434"
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class OllamaProvider(LLMProvider):
    """Runs the same prompts against a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("timeout_s", 180.0)
        super().__init__(model=model or DEFAULT_MODEL, **kwargs)
        self._client = client or OllamaClient(base_url=base_url, model=self.model)

    def warmup(self) -> None:
        try:
            self._client.warmup()
        except Exception as exc:
            logger.warning("ollama warmup failed: %s", truncate(str(exc)))

    def _complete_json(self, *, system: str, user: str) -> str:
        try:
            return self._client.chat(
                system=system,
                user=user,
                temperature=self.temperature,
                num_predict=self.max_output_tokens,
                timeout_s=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"ollama timeout: {truncate(str(exc))}") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in RETRYABLE_STATUS_CODES:
                raise LLMTransientError(f"ollama HTTP {status}") from exc
            raise LLMError(f"ollama HTTP {status}") from exc
        except httpx.HTTPError as exc:
            raise LLMTransientError(f"ollama transport error: {truncate(str(exc))}") from exc
