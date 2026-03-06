from __future__ import annotations

from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    # Where local OAuth artifacts live (do not commit these)
    gmail_client_secret_path: str = os.getenv("GMAIL_CLIENT_SECRET_PATH", "secrets/gmail_oauth_client.json")
    gmail_token_path: str = os.getenv("GMAIL_TOKEN_PATH", "secrets/gmail_token.json")

    # LLM choice for now (ollama by default to keep $0)
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "gemini"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # If you later use Gemini:
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")


settings = Settings()


JOB_LABELS = [
    "APPLIED",
    "ASSESSMENTS",
    "IN PROCESS",
    "INTERVIEWS",
    "REJECTED",
    "OTP_SECURITY",
    "RECOMMENDATIONS",
    "JOB_ALERTS",
    "ADVERTISEMENTS"
]

PROCESSED_LABEL = "PROCESSED"   