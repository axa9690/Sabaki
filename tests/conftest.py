from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

VALID_JSON = (
    '{"label":"APPLIED","urgency":"low",'
    '"reasoning_brief":"Application confirmation.","needs_reply":false}'
)
OTHERS_JSON = (
    '{"label":"OTHERS","urgency":"low",'
    '"reasoning_brief":"Unrelated questionnaire.","needs_reply":false}'
)


def chat_response(content: str) -> SimpleNamespace:
    """Mimic the shape of a Groq chat completion response."""
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    """Replays queued responses/exceptions and records the request kwargs."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("provider called more times than expected")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return chat_response(item)


class FakeGroqClient:
    def __init__(self, responses: list[Any]) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


@pytest.fixture
def groq_request() -> httpx.Request:
    return httpx.Request("POST", GROQ_URL)


def http_error(status_code: int, message: str, request: httpx.Request) -> Exception:
    import groq

    response = httpx.Response(status_code=status_code, request=request)
    return groq.APIStatusError(message, response=response, body=None)


@dataclass
class FakeEmail:
    """Stands in for gmail.fetch.SimpleEmail."""

    message_id: str = "m1"
    thread_id: str = "t1"
    from_email: str = "recruiter@example.com"
    subject: str = "Update"
    date: str = "Mon, 1 Sep 2025 10:00:00 -0500"
    snippet: str = ""
    body_text: str = ""
    label_ids: list[str] = field(default_factory=list)


class _Request:
    def __init__(self, run) -> None:
        self._run = run

    def execute(self) -> dict[str, Any]:
        return self._run()


class FakeGmailService:
    """Minimal stand-in for the Gmail API client used by apply_labels."""

    def __init__(self) -> None:
        self.modifications: list[dict[str, Any]] = []

    def users(self) -> "FakeGmailService":
        return self

    def messages(self) -> "FakeGmailService":
        return self

    def modify(self, *, userId: str, id: str, body: dict[str, Any]) -> _Request:
        def run() -> dict[str, Any]:
            self.modifications.append({"id": id, **body})
            return {"id": id}

        return _Request(run)
