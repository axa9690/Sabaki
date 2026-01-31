from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class SimpleEmail:
    message_id: str
    thread_id: str
    from_email: str
    subject: str
    date: str
    snippet: str
    body_text: str
    label_ids: list[str]


def _get_header(headers: List[Dict[str, str]], name: str) -> str:
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _decode_base64url(data: str) -> str:
    if not data:
        return ""
    # Gmail uses base64url without padding
    missing_padding = (-len(data)) % 4
    if missing_padding:
        data += "=" * missing_padding
    return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")


def _extract_text_from_payload(payload: Dict[str, Any]) -> str:
    """
    Prefer text/plain. If multipart, walk parts recursively.
    """
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {}) or {}
    data = body.get("data")

    if mime_type == "text/plain" and data:
        return _decode_base64url(data)

    # multipart: recurse into parts
    parts = payload.get("parts", []) or []
    for part in parts:
        text = _extract_text_from_payload(part)
        if text.strip():
            return text

    # fallback: if top-level has data even when not text/plain
    if data:
        return _decode_base64url(data)

    return ""


def fetch_recent_emails(service, max_results: int = 5) -> list[SimpleEmail]:
    """
    Fetch recent messages and return a clean, minimal representation.
    """
    resp = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = resp.get("messages", []) or []

    results: list[SimpleEmail] = []
    for m in messages:
        msg_id = m["id"]
        full = (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )

        payload = full.get("payload", {}) or {}
        headers = payload.get("headers", []) or []

        results.append(
            SimpleEmail(
                message_id=full.get("id", ""),
                thread_id=full.get("threadId", ""),
                from_email=_get_header(headers, "From"),
                subject=_get_header(headers, "Subject"),
                date=_get_header(headers, "Date"),
                snippet=full.get("snippet", "") or "",
                body_text=_extract_text_from_payload(payload).strip(),
                label_ids=full.get("labelIds", []) or [],
            )
        )

    return results
