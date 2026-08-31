"""HTTP entrypoint for running the agent as a small cloud application."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Header, HTTPException
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from email_agent.config import settings
from email_agent.gmail.service import SCOPES
from email_agent.llm.factory import create_provider
from email_agent.pipeline.runner import run_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="Sabaki AI Email Agent")


def _build_service():
    """Gmail client from GMAIL_TOKEN_JSON (cloud) or the local token file."""
    if settings.gmail_token_json:
        token_info = json.loads(settings.gmail_token_json)
        creds = Credentials.from_authorized_user_info(token_info, SCOPES)
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    from email_agent.gmail.service import build_gmail_service

    return build_gmail_service()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }


@app.post("/run")
def run(x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    if settings.run_api_key and x_api_key != settings.run_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        provider = create_provider()
        provider.warmup()
        service = _build_service()
    except Exception as exc:
        logger.error("startup failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    stats = run_agent(
        service=service, provider=provider, max_emails=settings.max_emails
    )

    return {
        "ok": True,
        "provider": provider.name,
        "model": provider.model,
        **stats.as_dict(),
    }
