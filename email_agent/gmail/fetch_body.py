from __future__ import annotations

import base64
from typing import Any, Dict, List


def _decode_base64url(data: str) -> str:
    if not data:
        return ""
    missing_padding = (-len(data)) % 4
    if missing_padding:
        data += "=" * missing_padding
    return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")


def _extract_text_from_payload(payload: Dict[str, Any]) -> str:
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {}) or {}
    data = body.get("data")

    if mime_type == "text/plain" and data:
        return _decode_base64url(data)

    parts = payload.get("parts", []) or []
    for part in parts:
        text = _extract_text_from_payload(part)
        if text.strip():
            return text

    if data:
        return _decode_base64url(data)

    return ""


def fetch_email_body_text(service, message_id: str) -> str:
    """
    Slower: fetch full message and extract text/plain if possible.
    """
    full = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )
    payload = full.get("payload", {}) or {}
    return _extract_text_from_payload(payload).strip()
