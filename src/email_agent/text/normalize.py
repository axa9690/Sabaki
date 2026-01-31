# src/email_agent/text/normalize.py
from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\uFEFF]")


def html_to_text(s: str) -> str:
    """Best-effort HTML -> plain text without extra dependencies."""
    if not s:
        return ""
    s = html.unescape(s)
    # remove script/style blocks first (common in emails)
    s = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", s)
    # strip tags
    s = _TAG_RE.sub(" ", s)
    return s


def normalize_email_text(
    *,
    subject: str,
    snippet: str,
    max_chars: int = 6000,
) -> str:
    subject = subject or ""
    snippet = snippet or ""

    body_txt = html_to_text(snippet)

    text = f"{subject}\n{snippet}\n{body_txt}".lower()
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "…"

    return text
