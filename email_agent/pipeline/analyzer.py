"""LLM fallback stage of the pipeline.

Only reached when :mod:`email_agent.pipeline.rules` cannot classify an email.
"""

from __future__ import annotations

from email_agent.llm.base import EmailContext, LLMProvider
from email_agent.schemas import EmailAnalysis
from email_agent.text.normalize import normalize_email_text

# Cap prompt size to keep latency and token usage low.
MAX_PROMPT_CHARS = 4000


def analyze_email_with_llm(
    *,
    provider: LLMProvider,
    subject: str,
    from_email: str,
    date: str,
    text: str,
    max_chars: int = MAX_PROMPT_CHARS,
) -> EmailAnalysis:
    """Normalise the email, then classify it with the configured provider."""
    normalized = normalize_email_text(
        subject=subject,
        snippet=text,
        max_chars=max_chars,
    )
    return provider.analyze_email(
        EmailContext(
            from_email=from_email,
            subject=subject,
            date=date,
            text=normalized,
        )
    )
