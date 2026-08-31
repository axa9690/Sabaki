"""Centralised prompts for the LLM fallback classifier.

The LLM is only reached when deterministic rules cannot classify an email, so
the prompt is intentionally short: classification only, no long-form reasoning.
"""

from __future__ import annotations

from email_agent.schemas import JobLabel, Urgency

ALLOWED_LABELS: tuple[str, ...] = tuple(label.value for label in JobLabel)
ALLOWED_URGENCIES: tuple[str, ...] = tuple(urgency.value for urgency in Urgency)

_LABEL_LIST = "\n".join(ALLOWED_LABELS)
_URGENCY_LIST = " | ".join(ALLOWED_URGENCIES)

SYSTEM_PROMPT = f"""You are the fallback classifier for a job-application inbox.
Deterministic rules already handled the obvious emails; you only see ambiguous ones.

Output ONLY valid JSON. No markdown, no commentary.

"label" MUST be EXACTLY one of these (case-sensitive):
{_LABEL_LIST}

"urgency" MUST be exactly one of: {_URGENCY_LIST}

Schema (exactly these four keys):
{{
  "label": "<one allowed label>",
  "urgency": "<{_URGENCY_LIST}>",
  "reasoning_brief": "<one short sentence, max 15 words>",
  "needs_reply": <true|false>
}}

Rules:
- Never invent labels and never use the key "category".
- Use OTHERS when the email is unrelated to the job search or cannot be mapped confidently.
- An application acknowledgement is APPLIED even if it mentions a possible future interview.
- INTERVIEWS requires a real interview scheduling or confirmation intent.
- Do not explain your thinking; return the JSON object only.
"""

_USER_TEMPLATE = """Classify this email.

From: {from_email}
Date: {date}
Subject: {subject}
Body: {text}

Return ONLY JSON with the keys: label, urgency, reasoning_brief, needs_reply.
Example: {{"label":"APPLIED","urgency":"low","reasoning_brief":"Application confirmation.","needs_reply":false}}
"""

RETRY_HINT = """
Your previous response was rejected: {error}
Return ONLY corrected JSON matching the schema.
"""


def build_user_prompt(*, from_email: str, subject: str, date: str, text: str) -> str:
    return _USER_TEMPLATE.format(
        from_email=from_email or "(unknown)",
        date=date or "(unknown)",
        subject=subject or "(no subject)",
        text=text or "(empty)",
    )


def build_retry_prompt(user_prompt: str, error: str) -> str:
    return user_prompt + RETRY_HINT.format(error=error)
