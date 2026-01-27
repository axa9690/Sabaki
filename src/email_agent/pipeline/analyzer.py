from __future__ import annotations

import json
from typing import Optional

from email_agent.schemas import EmailAnalysis
from email_agent.llm.ollama_client import OllamaClient

SYSTEM_PROMPT = """You are an AI email assistant for a job-application inbox.
You MUST output ONLY valid JSON. No markdown. No extra text.

You MUST choose "label" from EXACTLY this list (case-sensitive):
APPLIED
ASSESSMENTS
IN PROCESS
INTERVIEWS
REJECTED
OTP_SECURITY
RECOMMENDATIONS
JOB_ALERTS
ADVERTISEMENTS

If UNSURE about the label, choose OTHERS

You MUST choose "urgency" from EXACTLY this list:
low
medium
high

Schema (return EXACTLY these keys, no more, no less):
{
  "label": "<ONE of the allowed label strings>",
  "urgency": "<low|medium|high>",
  "reasoning_brief": "<1 short sentence>",
  "needs_reply": boolean
}

Rules:
- Output JSON only.
- Use ONLY the key "label" (never "category").
- Never invent new labels.
- Keep reasoning_brief to 1 sentence.
"""


def _safe_json_extract(text: str) -> Optional[dict]:
    """
    Tries to parse JSON strictly; if the model adds extra text,
    extract the first {...} block and parse that.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # best-effort extraction
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def analyze_email_with_ollama(
    *,
    subject: str,
    from_email: str,
    date: str,
    snippet: str,
    client: OllamaClient,
    max_retries: int = 2,
) -> EmailAnalysis:

    user_prompt = f"""Classify this email for a job-application inbox.

From: {from_email}
Date: {date}
Subject: {subject}
Snippet: {snippet}

Return ONLY JSON with EXACT keys: label, urgency, reasoning_brief, needs_reply.
The key must be "label" (NOT category).
Example:
{{"label":"APPLIED","urgency":"low","reasoning_brief":"Application confirmation.","needs_reply":false}}
"""


    last_err = None

    for attempt in range(max_retries + 1):
        raw = client.chat(system=SYSTEM_PROMPT, user=user_prompt, temperature=0.2)

        obj = _safe_json_extract(raw)
        if obj is None:
            last_err = f"Could not parse JSON. Raw output:\n{raw[:500]}"
        else:
            try:
                return EmailAnalysis.model_validate(obj)
            except Exception as e:
                last_err = f"Pydantic validation failed: {e}. Raw JSON: {obj}"

        # Retry by telling the model exactly what failed
        user_prompt = (
            user_prompt
            + "\n\nYour previous output was invalid.\n"
            + f"Error: {last_err}\n"
            + "Return ONLY corrected JSON matching the schema."
        )

    raise ValueError(f"Failed to produce valid EmailAnalysis after retries. Last error: {last_err}")
