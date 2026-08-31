"""Local entrypoint: label recent Gmail emails using rules + the LLM fallback."""

from __future__ import annotations

import logging

from email_agent.config import settings
from email_agent.gmail.service import build_gmail_service
from email_agent.llm.factory import create_provider
from email_agent.pipeline.runner import run_agent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    service = build_gmail_service()
    provider = create_provider()
    provider.warmup()

    print(f"Provider: {provider.name} | model: {provider.model}")

    stats = run_agent(service=service, provider=provider, max_emails=settings.max_emails)

    print(
        f"\nDone. checked={stats.checked}, labeled={stats.labeled}, "
        f"skipped={stats.skipped}, failed={stats.failed} "
        f"(rules={stats.rule_hits}, llm={stats.llm_calls})"
    )


if __name__ == "__main__":
    main()
