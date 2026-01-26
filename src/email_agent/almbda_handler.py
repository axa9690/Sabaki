from __future__ import annotations

import json
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from email_agent.aws.secrets import get_secret_json
from email_agent.gmail.fetch_meta import fetch_recent_email_meta
from email_agent.gmail.labels import ensure_label
from email_agent.pipeline.label_router import label_for_category, processed_label
from email_agent.llm.gemini_client import analyze_with_gemini

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

def lambda_handler(event, context):
    region = os.getenv("AWS_REGION", "us-east-1")

    gmail_secret_id = os.getenv("GMAIL_SECRET_ID", "ai-email-agent/gmail")
    gemini_secret_id = os.getenv("GEMINI_SECRET_ID", "ai-email-agent/gemini")

    gmail_secret = get_secret_json(gmail_secret_id, region)
    gemini_secret = get_secret_json(gemini_secret_id, region)

    # Store gmail_token_json as JSON object OR JSON string in Secrets Manager.
    token_val = gmail_secret["gmail_token_json"]
    token_info = token_val if isinstance(token_val, dict) else json.loads(token_val)

    gemini_api_key = gemini_secret["gemini_api_key"]
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    creds = Credentials.from_authorized_user_info(token_info, SCOPES)
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    processed_name = processed_label()
    processed_id = ensure_label(service, processed_name)

    max_emails = int(os.getenv("MAX_EMAILS", "10"))
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
