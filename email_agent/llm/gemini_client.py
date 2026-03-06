from __future__ import annotations

import json
from typing import Optional

from google import genai
from email_agent.schemas import EmailAnalysis


SYSTEM = """Return ONLY valid JSON matching this schema:
{
  "category": "recruiting|interview|job_application|work|finance|school|personal|newsletter|promotions|social|spam|other",
  "urgency": "critical|high|medium|low",
  "draft_reply": string|null,
  "reasoning_brief": string,
  "needs_reply": boolean
}
Rules:
- Output JSON only. No markdown. No extra text.
- Never invent categories. If unsure, use "other".
- Keep reasoning_brief to 1–2 sentences.
- draft_reply must be null for alerts/newsletters/promotions.
"""

def _safe_json_extract(text: str) -> Optional[dict]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            return None
    return None


def analyze_with_gemini(
    *,
    api_key: str,
    model: str,
    from_email: str,
    subject: str,
    date: str,
    snippet: str,
) -> EmailAnalysis:
    client = genai.Client(api_key=api_key)  # matches official quickstart pattern :contentReference[oaicite:2]{index=2}

    prompt = f"""{SYSTEM}

From: {from_email}
Date: {date}
Subject: {subject}
Snippet: {snippet}
"""

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": 0.2},
    )
    text = (resp.text or "").strip()

    obj = _safe_json_extract(text)
    if obj is None:
        raise ValueError(f"Gemini returned non-JSON: {text[:400]}")

    return EmailAnalysis.model_validate(obj)
