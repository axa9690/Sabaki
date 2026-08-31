"""Provider selection through environment variables."""

from __future__ import annotations

import pytest

from email_agent.config import Settings
from email_agent.llm.base import LLMConfigError
from email_agent.llm.factory import create_provider

ENV_KEYS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_RETRIES",
    "LLM_TIMEOUT_S",
    "GROQ_API_KEY",
    "OLLAMA_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_groq_is_the_default_provider(clean_env):
    clean_env.setenv("GROQ_API_KEY", "test-key")

    provider = create_provider(settings=Settings.from_env())

    assert provider.name == "groq"
    assert provider.model == "llama-3.1-8b-instant"
    assert provider.temperature == 0.0


def test_model_and_tuning_come_from_env(clean_env):
    clean_env.setenv("LLM_PROVIDER", "groq")
    clean_env.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    clean_env.setenv("LLM_MAX_RETRIES", "4")
    clean_env.setenv("LLM_TIMEOUT_S", "12.5")
    clean_env.setenv("GROQ_API_KEY", "test-key")

    provider = create_provider(settings=Settings.from_env())

    assert provider.model == "llama-3.3-70b-versatile"
    assert provider.max_retries == 4
    assert provider.timeout_s == 12.5


def test_ollama_provider_still_selectable(clean_env):
    clean_env.setenv("LLM_PROVIDER", "ollama")
    clean_env.setenv("OLLAMA_MODEL", "llama3.1")

    provider = create_provider(settings=Settings.from_env())

    assert provider.name == "ollama"
    assert provider.model == "llama3.1"


def test_api_key_is_never_hardcoded(clean_env):
    clean_env.setenv("LLM_PROVIDER", "groq")

    settings = Settings.from_env()

    assert settings.groq_api_key is None


def test_unknown_provider_raises(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(LLMConfigError) as excinfo:
        create_provider(settings=Settings.from_env())

    assert "openai" in str(excinfo.value)


def test_explicit_arguments_override_env(clean_env):
    clean_env.setenv("LLM_PROVIDER", "ollama")

    provider = create_provider(
        "groq", model="llama-3.1-8b-instant", settings=Settings.from_env()
    )

    assert provider.name == "groq"
