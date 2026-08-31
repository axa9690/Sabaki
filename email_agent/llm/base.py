"""Provider-agnostic LLM contract for email classification.

Providers implement a single method (``_complete_json``); JSON extraction,
Pydantic validation and retries live here so every provider behaves the same.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import ValidationError

from email_agent.llm.prompts import (
    SYSTEM_PROMPT,
    build_retry_prompt,
    build_user_prompt,
)
from email_agent.schemas import EmailAnalysis

logger = logging.getLogger(__name__)

# Keep log lines useful but free of full email content / model transcripts.
LOG_SNIPPET_CHARS = 200


class LLMError(RuntimeError):
    """Base error for LLM provider failures."""


class LLMTransientError(LLMError):
    """Temporary provider failure (timeout, rate limit, 5xx): worth retrying."""


class LLMResponseError(LLMError):
    """The provider replied, but the payload was not a valid EmailAnalysis."""


class LLMConfigError(LLMError):
    """Provider is misconfigured (missing API key, unknown provider, ...)."""


@dataclass(frozen=True)
class EmailContext:
    """Everything a provider is allowed to see about one email."""

    from_email: str
    subject: str
    date: str
    text: str


def truncate(value: str, limit: int = LOG_SNIPPET_CHARS) -> str:
    value = (value or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[:limit] + "…"


def extract_json_object(text: str) -> dict | None:
    """Parse JSON, falling back to the first ``{...}`` block in the text."""
    text = (text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


class LLMProvider(ABC):
    """Common interface for analysing a single email."""

    name: str = "base"

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 200,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self.retry_backoff_s = max(0.0, retry_backoff_s)

    @abstractmethod
    def _complete_json(self, *, system: str, user: str) -> str:
        """Call the provider and return the raw response text.

        Implementations must raise :class:`LLMTransientError` for retryable
        failures and :class:`LLMError` for permanent ones.
        """

    def warmup(self) -> None:
        """Optional hook for providers with cold-start cost (e.g. Ollama)."""

    def analyze_email(self, ctx: EmailContext) -> EmailAnalysis:
        """Classify one email, retrying malformed output and transient errors."""
        user_prompt = build_user_prompt(
            from_email=ctx.from_email,
            subject=ctx.subject,
            date=ctx.date,
            text=ctx.text,
        )
        last_error = "no attempt was made"
        last_was_transient = False

        for attempt in range(1, self.max_retries + 2):
            try:
                raw = self._complete_json(system=SYSTEM_PROMPT, user=user_prompt)
            except LLMTransientError as exc:
                last_error = f"transient provider error: {exc}"
                last_was_transient = True
                logger.warning(
                    "%s attempt %s/%s failed: %s",
                    self.name,
                    attempt,
                    self.max_retries + 1,
                    truncate(str(exc)),
                )
                self._sleep_before_retry(attempt)
                continue

            analysis, last_error = self._parse(raw)
            if analysis is not None:
                return analysis

            last_was_transient = False
            logger.warning(
                "%s attempt %s/%s returned unusable output: %s",
                self.name,
                attempt,
                self.max_retries + 1,
                truncate(last_error),
            )
            user_prompt = build_retry_prompt(user_prompt, last_error)

        error_type = LLMTransientError if last_was_transient else LLMResponseError
        raise error_type(
            f"{self.name}: no valid EmailAnalysis after "
            f"{self.max_retries + 1} attempt(s). Last error: {truncate(last_error)}"
        )

    def analyze(
        self, *, from_email: str, subject: str, date: str, text: str
    ) -> EmailAnalysis:
        """Convenience wrapper around :meth:`analyze_email`."""
        return self.analyze_email(
            EmailContext(from_email=from_email, subject=subject, date=date, text=text)
        )

    def _parse(self, raw: str) -> tuple[EmailAnalysis | None, str]:
        payload = extract_json_object(raw)
        if payload is None:
            return None, f"response was not JSON: {truncate(raw)}"
        try:
            return EmailAnalysis.model_validate(payload), ""
        except ValidationError as exc:
            summary = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            return None, f"schema validation failed: {truncate(summary)}"

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_s and attempt <= self.max_retries:
            time.sleep(self.retry_backoff_s * attempt)
