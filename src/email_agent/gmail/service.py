from __future__ import annotations

import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]


def build_gmail_service():
    """
    Builds an authenticated Gmail API service.

    Uses env var:
      GMAIL_TOKEN_PATH (default: secrets/gmail_token.json)
    Optional:
      GMAIL_OAUTH_CLIENT_PATH (default: secrets/gmail_oauth_client.json)
    """
    token_path = os.getenv("GMAIL_TOKEN_PATH", "secrets/gmail_token.json")
    client_path = os.getenv("GMAIL_OAUTH_CLIENT_PATH", "secrets/gmail_oauth_client.json")

    if not os.path.exists(token_path):
        raise FileNotFoundError(
            f"Missing token file: {token_path}. Run scripts/auth_gmail_local.py to generate it."
        )

    creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # refresh if needed
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # write back refreshed token
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service
