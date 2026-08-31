"""Shared provider behaviour: JSON extraction, retries and the common contract."""

from __future__ import annotations

import pytest
from conftest import VALID_JSON

from email_agent.llm.base import (
    EmailContext,
    LLMProvider,
    LLMResponseError,
    LLMTransientError,
    extract_json_object,
    truncate,
)
from email_agent.schemas import JobLabel

CTX = EmailContext(
    from_email="a@b.com", subject="Subject", date="today", text="body text"
)


class ScriptedProvider(LLMProvider):
    """Provider whose raw responses (or errors) are supplied by the test."""

    name = "scripted"

    def __init__(self, script: list, **kwargs) -> None:
        kwargs.setdefault("retry_backoff_s", 0.0)
        super().__init__(model="test-model", **kwargs)
        self.script = list(script)
        self.prompts: list[str] = []

    def _complete_json(self, *, system: str, user: str) -> str:
        self.prompts.append(user)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.parametrize(
    "raw, expected_label",
    [
        (VALID_JSON, "APPLIED"),
        ('```json\n{"label":"OTHERS","urgency":"low","reasoning_brief":"x",'
         '"needs_reply":false}\n```', "OTHERS"),
        ('Here: {"label":"REJECTED","urgency":"low","reasoning_brief":"x",'
         '"needs_reply":false} done', "REJECTED"),
    ],
)
def test_extract_json_object_handles_wrapped_payloads(raw, expected_label):
    payload = extract_json_object(raw)

    assert payload is not None
    assert payload["label"] == expected_label


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "[1, 2, 3]", "{broken"])
def test_extract_json_object_returns_none_for_unusable_text(raw):
    assert extract_json_object(raw) is None


def test_truncate_keeps_logs_short():
    assert truncate("a" * 500).endswith("…")
    assert len(truncate("a" * 500)) == 201


def test_provider_returns_validated_analysis():
    provider = ScriptedProvider([VALID_JSON])

    analysis = provider.analyze_email(CTX)

    assert analysis.label is JobLabel.APPLIED
    assert analysis.reasoning_brief


def test_provider_stops_at_first_valid_response():
    provider = ScriptedProvider([VALID_JSON, VALID_JSON], max_retries=2)

    provider.analyze_email(CTX)

    assert len(provider.prompts) == 1


def test_provider_retries_transient_errors_then_succeeds():
    provider = ScriptedProvider(
        [LLMTransientError("boom"), VALID_JSON], max_retries=2
    )

    assert provider.analyze_email(CTX).label is JobLabel.APPLIED
    assert len(provider.prompts) == 2


def test_provider_raises_transient_when_retries_exhausted():
    provider = ScriptedProvider(
        [LLMTransientError("boom")] * 3, max_retries=2
    )

    with pytest.raises(LLMTransientError):
        provider.analyze_email(CTX)

    assert len(provider.prompts) == 3


def test_zero_retries_means_single_attempt():
    provider = ScriptedProvider(["garbage"], max_retries=0)

    with pytest.raises(LLMResponseError):
        provider.analyze_email(CTX)

    assert len(provider.prompts) == 1


def test_analyze_convenience_wrapper():
    provider = ScriptedProvider([VALID_JSON])

    analysis = provider.analyze(
        from_email="a@b.com", subject="s", date="d", text="t"
    )

    assert analysis.label is JobLabel.APPLIED


def test_prompt_contains_email_fields_and_allowed_labels():
    provider = ScriptedProvider([VALID_JSON])
    provider.analyze_email(CTX)

    prompt = provider.prompts[0]
    assert CTX.subject in prompt
    assert CTX.from_email in prompt
    assert CTX.text in prompt
