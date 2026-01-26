from __future__ import annotations

import json
import os
from fastapi import FastAPI, Header, HTTPException

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from email_agent.gmail.fetch_meta import fetch_recent_email_meta
from email_agent.gmail.labels import ensure_label
from email_agent.pipeline.label_router import label_for_category, processed_label
from email_agent.llm.gemini_client import analyze_with_gemini

app = FastAPI()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

def _env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/run")
def run_agent(x_api_key: str | None = Header(default=None)):
    # Simple protection so strangers can't hit your endpoint
    expected = os.getenv("RUN_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    gemini_api_key = _env("GEMINI_API_KEY")
    gmail_token_json = _env("GMAIL_TOKEN_JSON")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    max_emails = int(os.getenv("MAX_EMAILS", "5"))

    token_info = json.loads(gmail_token_json)

    creds = Credentials.from_authorized_user_info(token_info, SCOPES)
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    processed_id = ensure_label(service, processed_label())
    emails = fetch_recent_email_meta(service, max_results=max_emails)

    labeled = 0
    skipped = 0

    for e in emails:
        if processed_id in e.label_ids:
            skipped += 1
            continue

        analysis = analyze_with_gemini(
            api_key=gemini_api_key,
            model=model,
            from_email=e.from_email,
            subject=e.subject,
            date=e.date,
            snippet=e.snippet,
        )

        cat_label_name = label_for_category(analysis.category)
        cat_id = ensure_label(service, cat_label_name)

        service.users().messages().modify(
            userId="me",
            id=e.message_id,
            body={"addLabelIds": [cat_id, processed_id], "removeLabelIds": []},
        ).execute()

        labeled += 1

    return {"ok": True, "checked": len(emails), "labeled": labeled, "skipped": skipped, "model": model}