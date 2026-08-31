from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Default model per provider, used when LLM_MODEL is not set.
DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.1-8b-instant",
    "ollama": "llama3.1",
}

DEFAULT_PROVIDER = "groq"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_optional(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


class Settings(BaseModel):
    """Runtime configuration, read once from the environment (or .env)."""

    # Where local OAuth artifacts live (never commit these)
    gmail_client_secret_path: str = "secrets/gmail_oauth_client.json"
    gmail_token_path: str = "secrets/gmail_token.json"
    # Cloud deployments pass the token contents instead of a file path
    gmail_token_json: str | None = None

    # LLM selection
    llm_provider: str = DEFAULT_PROVIDER
    llm_model: str = DEFAULT_MODELS[DEFAULT_PROVIDER]
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 200
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_s: float = 0.5

    # Groq
    groq_api_key: str | None = None
    groq_base_url: str | None = None

    # Ollama (optional local/offline mode)
    ollama_base_url: str = "http://localhost:11434"
    # Local models are much slower to warm up than a hosted API.
    ollama_timeout_s: float = 180.0

    # Workflow
    max_emails: int = 50
    run_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env_str("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
        fallback_model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS[DEFAULT_PROVIDER])
        # OLLAMA_MODEL stays supported so existing local setups keep working.
        if provider == "ollama":
            fallback_model = _env_str("OLLAMA_MODEL", fallback_model)

        return cls(
            gmail_client_secret_path=_env_str(
                "GMAIL_CLIENT_SECRET_PATH", "secrets/gmail_oauth_client.json"
            ),
            gmail_token_path=_env_str("GMAIL_TOKEN_PATH", "secrets/gmail_token.json"),
            gmail_token_json=_env_optional("GMAIL_TOKEN_JSON"),
            llm_provider=provider,
            llm_model=_env_str("LLM_MODEL", fallback_model),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
            llm_max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 200),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", 30.0),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
            llm_retry_backoff_s=_env_float("LLM_RETRY_BACKOFF_S", 0.5),
            groq_api_key=_env_optional("GROQ_API_KEY"),
            groq_base_url=_env_optional("GROQ_BASE_URL"),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_timeout_s=_env_float("OLLAMA_TIMEOUT_S", 180.0),
            max_emails=_env_int("MAX_EMAILS", 50),
            run_api_key=_env_optional("RUN_API_KEY"),
        )


settings = Settings.from_env()


# Gmail labels the agent manages. OTHERS is deliberately absent: unclassified
# emails only get PROCESSED so they can be reviewed manually.
JOB_LABELS = [
    "APPLIED",
    "ASSESSMENTS",
    "IN PROCESS",
    "INTERVIEWS",
    "REJECTED",
    "OTP_SECURITY",
    "RECOMMENDATIONS",
    "JOB_ALERTS",
    "ADVERTISEMENTS",
]

PROCESSED_LABEL = "PROCESSED"
