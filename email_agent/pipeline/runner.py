"""Analyse-and-label workflow.

Shared by the local script and the HTTP entrypoint so both behave identically.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Protocol

from email_agent.config import JOB_LABELS, PROCESSED_LABEL, settings
from email_agent.gmail.fetch import fetch_recent_emails
from email_agent.gmail.fetch_body import fetch_email_body_text
from email_agent.gmail.labels import apply_labels, ensure_labels
from email_agent.llm.base import LLMProvider
from email_agent.pipeline.analyzer import analyze_email_with_llm
from email_agent.pipeline.rules import needs_body_fetch, short_circuit_label
from email_agent.schemas import JobLabel

logger = logging.getLogger(__name__)

# Low-priority outcomes that no longer need the user's attention.
MARK_READ_LABELS = frozenset(
    {JobLabel.APPLIED, JobLabel.REJECTED, JobLabel.ADVERTISEMENTS}
)

BodyFetcher = Callable[[str], str]


class EmailLike(Protocol):
    message_id: str
    from_email: str
    subject: str
    date: str
    snippet: str
    label_ids: list[str]


@dataclass
class RunStats:
    checked: int = 0
    labeled: int = 0
    skipped: int = 0
    failed: int = 0
    rule_hits: int = 0
    llm_calls: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class Decision:
    label: JobLabel
    reason: str
    used_llm: bool = False


def _body_text(email: EmailLike, body_fetcher: BodyFetcher) -> str:
    """Full body text, reusing what the fetch already returned when possible."""
    cached = getattr(email, "body_text", "") or ""
    return cached if cached.strip() else body_fetcher(email.message_id)


def classify_email(
    *,
    email: EmailLike,
    provider: LLMProvider,
    body_fetcher: BodyFetcher,
) -> Decision:
    """Rules first; the LLM only runs when the rules are not confident."""
    forced = short_circuit_label(email.subject, email.snippet, email.from_email)

    body = ""
    if forced is None or needs_body_fetch(email.subject, email.snippet):
        body = _body_text(email, body_fetcher)

    if body:
        from_body = short_circuit_label(
            email.subject, f"{email.snippet}\n{body}", email.from_email
        )
        forced = from_body or forced

    if forced is not None:
        return Decision(label=forced, reason="rule_short_circuit")

    analysis = analyze_email_with_llm(
        provider=provider,
        subject=email.subject,
        from_email=email.from_email,
        date=email.date,
        text=f"{email.snippet}\n{body}" if body else email.snippet,
    )
    return Decision(label=analysis.label, reason=analysis.reasoning_brief, used_llm=True)


def process_emails(
    *,
    service,
    provider: LLMProvider,
    emails: Iterable[EmailLike],
    label_ids: dict[str, str],
    processed_label_id: str,
    body_fetcher: BodyFetcher | None = None,
) -> RunStats:
    """Label every email, isolating failures so one bad email is not fatal."""
    fetcher: BodyFetcher = body_fetcher or (
        lambda message_id: fetch_email_body_text(service, message_id)
    )
    stats = RunStats()

    for email in emails:
        stats.checked += 1
        subject = (email.subject or "")[:70]

        try:
            if processed_label_id in email.label_ids or PROCESSED_LABEL in email.label_ids:
                stats.skipped += 1
                continue

            decision = classify_email(
                email=email, provider=provider, body_fetcher=fetcher
            )
            if decision.used_llm:
                stats.llm_calls += 1
            else:
                stats.rule_hits += 1

            if decision.label == JobLabel.OTHERS:
                # Leave it for manual review: PROCESSED only, no job label.
                apply_labels(
                    service, email.message_id, add_label_ids=[processed_label_id]
                )
                stats.skipped += 1
                logger.info("⚠️ Unclassified (PROCESSED only): %s", subject)
                continue

            add_ids = [label_ids[decision.label.value], processed_label_id]
            remove_ids = ["UNREAD"] if decision.label in MARK_READ_LABELS else []
            apply_labels(
                service,
                email.message_id,
                add_label_ids=add_ids,
                remove_label_ids=remove_ids,
            )

            stats.labeled += 1
            logger.info(
                "✅ Labeled: %s -> %s (+%s) [%s]",
                subject,
                decision.label.value,
                PROCESSED_LABEL,
                decision.reason,
            )

        except Exception as exc:
            stats.failed += 1
            logger.error("❌ Error processing email '%s': %s", subject, exc)
            continue

    return stats


def run_agent(
    *,
    service,
    provider: LLMProvider,
    max_emails: int | None = None,
) -> RunStats:
    """Ensure labels exist, fetch recent emails and process them."""
    limit = max_emails or settings.max_emails
    label_ids = ensure_labels(service, JOB_LABELS + [PROCESSED_LABEL])
    emails = fetch_recent_emails(service, max_results=limit)

    return process_emails(
        service=service,
        provider=provider,
        emails=emails,
        label_ids=label_ids,
        processed_label_id=label_ids[PROCESSED_LABEL],
    )
