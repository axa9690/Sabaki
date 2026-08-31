"""End-to-end workflow behaviour with a fake Gmail service and stub provider."""

from __future__ import annotations

from conftest import FakeEmail, FakeGmailService

from email_agent.config import JOB_LABELS, PROCESSED_LABEL
from email_agent.llm.base import LLMProvider, LLMTransientError
from email_agent.pipeline.runner import process_emails
from email_agent.schemas import JobLabel

LABEL_IDS = {name: f"id_{name}" for name in JOB_LABELS + [PROCESSED_LABEL]}
PROCESSED_ID = LABEL_IDS[PROCESSED_LABEL]


class StubProvider(LLMProvider):
    """Returns canned JSON and counts how often the LLM was consulted."""

    name = "stub"

    def __init__(self, script: list | None = None) -> None:
        super().__init__(model="stub-model", max_retries=0, retry_backoff_s=0.0)
        self.script = list(script or [])
        self.calls = 0

    def _complete_json(self, *, system: str, user: str) -> str:
        self.calls += 1
        item = self.script.pop(0) if self.script else '{"label":"OTHERS",' \
            '"urgency":"low","reasoning_brief":"unknown","needs_reply":false}'
        if isinstance(item, Exception):
            raise item
        return item


def run(emails, provider=None, body_fetcher=None):
    service = FakeGmailService()
    provider = provider or StubProvider()
    stats = process_emails(
        service=service,
        provider=provider,
        emails=emails,
        label_ids=LABEL_IDS,
        processed_label_id=PROCESSED_ID,
        body_fetcher=body_fetcher or (lambda _msg_id: ""),
    )
    return stats, service, provider


def test_rule_hit_labels_email_without_calling_the_llm():
    email = FakeEmail(
        subject="Application received",
        snippet="Thank you for applying to the Data Engineer role.",
    )

    stats, service, provider = run([email])

    assert provider.calls == 0
    assert stats.labeled == 1
    assert stats.rule_hits == 1
    modification = service.modifications[0]
    assert modification["addLabelIds"] == [LABEL_IDS["APPLIED"], PROCESSED_ID]
    assert modification["removeLabelIds"] == ["UNREAD"]


def test_already_processed_email_is_skipped():
    email = FakeEmail(label_ids=[PROCESSED_ID], snippet="Thank you for applying.")

    stats, service, provider = run([email])

    assert stats.skipped == 1
    assert stats.labeled == 0
    assert service.modifications == []
    assert provider.calls == 0


def test_llm_is_used_only_when_rules_are_unsure():
    email = FakeEmail(subject="Quick question", snippet="Are you available next week?")
    provider = StubProvider(
        ['{"label":"IN PROCESS","urgency":"medium",'
         '"reasoning_brief":"Recruiter follow-up.","needs_reply":true}']
    )

    stats, service, _ = run([email], provider=provider)

    assert provider.calls == 1
    assert stats.llm_calls == 1
    assert service.modifications[0]["addLabelIds"] == [
        LABEL_IDS["IN PROCESS"],
        PROCESSED_ID,
    ]
    # IN PROCESS still needs attention, so it stays unread.
    assert service.modifications[0]["removeLabelIds"] == []


def test_others_only_gets_processed_label():
    email = FakeEmail(subject="Voluntary questionnaire", snippet="Optional survey.")

    stats, service, provider = run([email])

    assert provider.calls == 1
    assert stats.labeled == 0
    assert stats.skipped == 1
    assert service.modifications == [
        {"id": "m1", "addLabelIds": [PROCESSED_ID], "removeLabelIds": []}
    ]


def test_body_is_fetched_when_snippet_is_inconclusive():
    email = FakeEmail(subject="Update on your application", snippet="See below.")
    fetched = {"count": 0}

    def fetcher(message_id: str) -> str:
        fetched["count"] += 1
        return "After careful consideration, we decided not to move forward."

    stats, service, provider = run([email], body_fetcher=fetcher)

    assert fetched["count"] == 1
    assert provider.calls == 0
    assert service.modifications[0]["addLabelIds"][0] == LABEL_IDS["REJECTED"]
    assert stats.labeled == 1


def test_cached_body_text_avoids_an_extra_api_call():
    email = FakeEmail(
        subject="Update on your application",
        snippet="See below.",
        body_text="The position has been filled.",
    )

    def fetcher(message_id: str) -> str:
        raise AssertionError("body should not be re-fetched")

    stats, service, _ = run([email], body_fetcher=fetcher)

    assert service.modifications[0]["addLabelIds"][0] == LABEL_IDS["REJECTED"]
    assert stats.labeled == 1


def test_one_failing_email_does_not_stop_the_others():
    emails = [
        FakeEmail(message_id="bad", subject="Mystery", snippet="???"),
        FakeEmail(
            message_id="good",
            subject="Application received",
            snippet="Thank you for applying.",
        ),
    ]
    provider = StubProvider([LLMTransientError("provider down")])

    stats, service, _ = run(emails, provider=provider)

    assert stats.checked == 2
    assert stats.failed == 1
    assert stats.labeled == 1
    assert [m["id"] for m in service.modifications] == ["good"]


def test_stats_are_serialisable():
    stats, _, _ = run([FakeEmail(snippet="Thank you for applying.")])

    assert stats.as_dict()["labeled"] == 1


def test_rejected_and_advertisements_are_marked_read():
    emails = [
        FakeEmail(
            message_id="rej",
            subject="Update",
            snippet="We regret to inform you that we will not be moving forward.",
        ),
        FakeEmail(
            message_id="ad",
            subject="Sale",
            snippet="Get 30% off today only. Shop now!",
        ),
    ]

    _, service, _ = run(emails)

    assert all(m["removeLabelIds"] == ["UNREAD"] for m in service.modifications)
    assert service.modifications[0]["addLabelIds"][0] == LABEL_IDS[JobLabel.REJECTED.value]
    assert service.modifications[1]["addLabelIds"][0] == LABEL_IDS[
        JobLabel.ADVERTISEMENTS.value
    ]
