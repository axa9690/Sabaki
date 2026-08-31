"""Groq provider behaviour, driven entirely by mocked SDK responses.

No test in this file may reach the real Groq API.
"""

from __future__ import annotations

import logging

import groq
import pytest
from conftest import OTHERS_JSON, VALID_JSON, FakeGroqClient, http_error

from email_agent.llm.base import (
    EmailContext,
    LLMConfigError,
    LLMResponseError,
    LLMTransientError,
)
from email_agent.llm.groq_provider import DEFAULT_MODEL, GroqProvider
from email_agent.schemas import JobLabel, Urgency

CTX = EmailContext(
    from_email="talent@example.com",
    subject="Regarding your application",
    date="Mon, 1 Sep 2025 10:00:00 -0500",
    text="thanks for applying to the data engineer role",
)


def make_provider(responses: list, **kwargs) -> tuple[GroqProvider, FakeGroqClient]:
    client = FakeGroqClient(responses)
    provider = GroqProvider(
        api_key="test-key",
        model=DEFAULT_MODEL,
        client=client,
        retry_backoff_s=0.0,
        **kwargs,
    )
    return provider, client


def test_valid_response_is_parsed_and_validated():
    provider, client = make_provider([VALID_JSON])

    analysis = provider.analyze_email(CTX)

    assert analysis.label is JobLabel.APPLIED
    assert analysis.urgency is Urgency.LOW
    assert analysis.needs_reply is False
    assert len(client.calls) == 1


def test_request_is_deterministic_json_mode_and_token_capped():
    provider, client = make_provider([VALID_JSON], max_output_tokens=200)

    provider.analyze_email(CTX)
    call = client.calls[0]

    assert call["model"] == "llama-3.1-8b-instant"
    assert call["temperature"] == 0.0
    assert call["response_format"] == {"type": "json_object"}
    assert call["max_completion_tokens"] == 200
    assert call["timeout"] == provider.timeout_s
    assert [m["role"] for m in call["messages"]] == ["system", "user"]


def test_json_embedded_in_prose_is_recovered():
    provider, _ = make_provider([f"Sure, here you go:\n{VALID_JSON}\nThanks!"])

    assert provider.analyze_email(CTX).label is JobLabel.APPLIED


def test_others_result_is_valid():
    provider, _ = make_provider([OTHERS_JSON])

    analysis = provider.analyze_email(CTX)

    assert analysis.label is JobLabel.OTHERS
    assert analysis.needs_reply is False


def test_malformed_json_is_retried_then_raises():
    provider, client = make_provider(
        ["not json at all", "still nonsense", "<html>nope</html>"], max_retries=2
    )

    with pytest.raises(LLMResponseError) as excinfo:
        provider.analyze_email(CTX)

    assert len(client.calls) == 3
    assert "3 attempt" in str(excinfo.value)


def test_retry_succeeds_after_malformed_json():
    provider, client = make_provider(["{oops", VALID_JSON], max_retries=2)

    analysis = provider.analyze_email(CTX)

    assert analysis.label is JobLabel.APPLIED
    assert len(client.calls) == 2
    # The retry prompt tells the model what was wrong with the previous answer.
    assert "previous response was rejected" in client.calls[1]["messages"][1]["content"]


def test_invalid_label_is_rejected():
    invented = (
        '{"label":"SUPER_INTERVIEW","urgency":"low",'
        '"reasoning_brief":"made up","needs_reply":false}'
    )
    provider, client = make_provider([invented, invented], max_retries=1)

    with pytest.raises(LLMResponseError) as excinfo:
        provider.analyze_email(CTX)

    assert "schema validation failed" in str(excinfo.value)
    assert len(client.calls) == 2


def test_invalid_label_then_valid_label_succeeds():
    invalid = (
        '{"label":"MAYBE","urgency":"low",'
        '"reasoning_brief":"made up","needs_reply":false}'
    )
    provider, client = make_provider([invalid, OTHERS_JSON], max_retries=1)

    assert provider.analyze_email(CTX).label is JobLabel.OTHERS
    assert len(client.calls) == 2


def test_api_timeout_is_retried_and_surfaces_as_transient(groq_request):
    timeout = groq.APITimeoutError(request=groq_request)
    provider, client = make_provider([timeout, timeout], max_retries=1)

    with pytest.raises(LLMTransientError) as excinfo:
        provider.analyze_email(CTX)

    assert "APITimeoutError" in str(excinfo.value)
    assert len(client.calls) == 2


def test_temporary_api_failure_then_retry_succeeds(groq_request):
    provider, client = make_provider(
        [http_error(503, "service unavailable", groq_request), VALID_JSON],
        max_retries=2,
    )

    assert provider.analyze_email(CTX).label is JobLabel.APPLIED
    assert len(client.calls) == 2


def test_rate_limit_is_transient(groq_request):
    provider, _ = make_provider(
        [http_error(429, "rate limited", groq_request), VALID_JSON], max_retries=1
    )

    assert provider.analyze_email(CTX).label is JobLabel.APPLIED


def test_connection_error_retries_exhausted(groq_request):
    error = groq.APIConnectionError(request=groq_request)
    provider, client = make_provider([error, error, error], max_retries=2)

    with pytest.raises(LLMTransientError):
        provider.analyze_email(CTX)

    assert len(client.calls) == 3


def test_client_error_is_not_retried(groq_request):
    provider, client = make_provider(
        [http_error(400, "bad request", groq_request)], max_retries=2
    )

    with pytest.raises(Exception) as excinfo:
        provider.analyze_email(CTX)

    assert not isinstance(excinfo.value, LLMTransientError)
    assert len(client.calls) == 1


def test_empty_choices_is_transient():
    from types import SimpleNamespace

    class EmptyClient(FakeGroqClient):
        def __init__(self):
            super().__init__([VALID_JSON])
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(choices=[])
                )
            )

    provider = GroqProvider(
        api_key="test-key", client=EmptyClient(), max_retries=0, retry_backoff_s=0.0
    )

    with pytest.raises(LLMTransientError):
        provider.analyze_email(CTX)


def test_missing_api_key_raises_config_error():
    provider = GroqProvider(api_key=None)

    with pytest.raises(LLMConfigError) as excinfo:
        provider.analyze_email(CTX)

    assert "GROQ_API_KEY" in str(excinfo.value)


def test_warmup_fails_fast_without_an_api_key():
    with pytest.raises(LLMConfigError):
        GroqProvider(api_key=None).warmup()


def test_warmup_is_a_noop_when_configured():
    provider, client = make_provider([VALID_JSON])

    provider.warmup()

    assert client.calls == []


def test_logs_never_contain_the_api_key_or_full_email(caplog):
    secret = "gsk_supersecrettoken"
    client = FakeGroqClient(["nope", "nope"])
    provider = GroqProvider(
        api_key=secret, client=client, max_retries=1, retry_backoff_s=0.0
    )

    with caplog.at_level(logging.WARNING), pytest.raises(LLMResponseError):
        provider.analyze_email(CTX)

    logged = caplog.text
    assert secret not in logged
    assert CTX.text not in logged
