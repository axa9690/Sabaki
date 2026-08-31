"""Environment-driven provider selection."""

from __future__ import annotations

import logging
from typing import Any

from email_agent.config import Settings, settings as default_settings
from email_agent.llm.base import LLMConfigError, LLMProvider

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("groq", "ollama")


def create_provider(
    provider: str | None = None,
    *,
    model: str | None = None,
    settings: Settings | None = None,
    **overrides: Any,
) -> LLMProvider:
    """Build the provider named by ``LLM_PROVIDER`` (or the explicit argument)."""
    cfg = settings or default_settings
    name = (provider or cfg.llm_provider or "").strip().lower()

    shared: dict[str, Any] = {
        "model": model or cfg.llm_model,
        "temperature": cfg.llm_temperature,
        "max_output_tokens": cfg.llm_max_output_tokens,
        "max_retries": cfg.llm_max_retries,
        "retry_backoff_s": cfg.llm_retry_backoff_s,
        **overrides,
    }

    if name == "groq":
        from email_agent.llm.groq_provider import GroqProvider

        shared.setdefault("timeout_s", cfg.llm_timeout_s)
        return GroqProvider(
            api_key=cfg.groq_api_key,
            base_url=cfg.groq_base_url,
            **shared,
        )

    if name == "ollama":
        from email_agent.llm.ollama_provider import OllamaProvider

        shared.setdefault("timeout_s", cfg.ollama_timeout_s)
        return OllamaProvider(base_url=cfg.ollama_base_url, **shared)

    raise LLMConfigError(
        f"Unknown LLM_PROVIDER={name!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
