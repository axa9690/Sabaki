from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EmailMeta:
    message_id: str
    thread_id: str
    from_email: str
    subject: str
    date: str
    snippet: str
    label_ids: list[str]


def _get_header(headers: List[Dict[str, str]], name: str) -> str:
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def fetch_recent_email_meta(service, max_results: int = 10) -> list[EmailMeta]:
    """
    Fast: list IDs -> get METADATA only (no body).
    """
    resp = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = resp.get("messages", []) or []

    out: list[EmailMeta] = []
    for m in messages:
        msg_id = m["id"]
        full = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=msg_id,
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )

        payload = full.get("payload", {}) or {}
        headers = payload.get("headers", []) or []

        out.append(
            EmailMeta(
                message_id=full.get("id", ""),
                thread_id=full.get("threadId", ""),
                from_email=_get_header(headers, "From"),
                subject=_get_header(headers, "Subject"),
                date=_get_header(headers, "Date"),
                snippet=full.get("snippet", "") or "",
                label_ids=full.get("labelIds", []) or [],
            )
        )

    return out
