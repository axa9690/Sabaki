"""Deterministic classification rules.

These run before any LLM call. Order matters: obvious job-state emails must be
resolved before the generic job-alert / advertisement heuristics, otherwise a
recruiting email with a marketing footer would be mislabelled.
"""

from __future__ import annotations

import re
from typing import Callable

from email_agent.schemas import JobLabel
from email_agent.text.normalize import normalize_email_text

# Rules read the whole body, so allow more text than the LLM prompt does.
RULE_MAX_CHARS = 20_000

Predicate = Callable[[str], bool]


def _re(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


OTP_RE = _re(
    r"\botp\b",
    r"\bverification code\b",
    r"\bsecurity code\b",
    r"\bpasscode\b",
    r"\bone[- ]time (code|password|passcode|pin)\b",
    r"\byour (login|sign[- ]in) code\b",
)

REJECTED_RE = _re(
    r"decided not to (move forward|proceed|continue|pursue)",
    r"not (be )?(moving|move) forward",
    r"not to move forward",
    r"will not be (moving forward|proceeding|progressing)",
    r"we regret to inform you",
    r"we have decided not to (proceed|move forward)",
    r"your (application|candidacy) (was|has been) (unsuccessful|not selected|declined)",
    r"your application was not selected",
    r"you (were|have) not been selected",
    r"(pursuing|moving forward with) other (candidates|applicants)",
    r"(position|role|job|vacancy) (has (now )?been|is|was|has been) (filled|closed)",
    r"(position|role) is no longer (available|open|being considered)",
    r"(interview )?process has (officially )?(concluded|ended|closed)",
    r"\binterview process\b.{0,40}\b(concluded|ended|closed)\b",
)

# Explicit scheduling intent. Checked before APPLIED so "schedule your interview"
# wins, while an acknowledgement mentioning "if selected for an interview" does not.
INTERVIEW_SCHEDULING_RE = _re(
    r"(schedule|book|confirm|reschedule)\s+(your|an|a|the)\s+(\w+\s+){0,2}"
    r"(interview|phone screen|screening call|onsite)",
    r"(select|choose|pick|book)\s+(an?|your|the)\s+(\w+\s+){0,2}"
    r"(interview\s+)?(time|timeslot|time slot|slot|date)",
    r"interview (invitation|invite|scheduling|availability)",
    r"invitation to interview",
    r"interview (has been |is )?(scheduled|confirmed|booked)",
    r"(self[- ]schedule|schedule your own)",
    r"\bcalendly\b",
)

# Phrasing that keeps an application acknowledgement in APPLIED even when it
# mentions a possible future interview.
CONDITIONAL_INTERVIEW_RE = _re(
    r"if (you are |you're |you |your )?(select|shortlist|chosen|move)",
    r"if selected",
    r"should (you|your) (be )?(select|shortlist|advance|progress|qualif)",
    r"if your (profile|qualifications|background|application|experience|skills)",
    r"if (there is|there's) a (match|fit)",
    r"(will|may|would) (contact|reach out to|be in touch with)? ?(you)?( if| should| to schedule| to arrange)",
    r"in the event (that )?you",
)

INTERVIEW_TOPIC_RE = _re(
    r"\b(interview|phone screen|screening call|onsite|final round)\b",
)

SCHEDULING_INTENT_RE = _re(
    r"\b(schedule|scheduled|reschedule|confirm|confirmation|book|booking|"
    r"calendar|invite|invitation|availability|time slot|timeslot)\b",
)

MEETING_LINK_RE = _re(
    r"\bzoom\b", r"\bgoogle meet\b", r"\bmicrosoft teams\b", r"\bteams meeting\b"
)

ASSESSMENT_VENDOR_RE = _re(
    r"\bhackerrank\b",
    r"\bcodility\b",
    r"\bcodesignal\b",
    r"\bcoderbyte\b",
    r"\btestgorilla\b",
    r"\bkarat\b",
    r"\bshl\b",
    r"\bimocha\b",
)

ASSESSMENT_TOPIC_RE = _re(
    r"\bassessment\b",
    r"\bcoding (challenge|test|exercise)\b",
    r"\btake[- ]home (test|assignment|challenge)\b",
    r"\bskills? (test|assessment)\b",
    r"\bonline (test|exam)\b",
    r"\btechnical (test|screen|screening)\b",
)

ASSESSMENT_ACTION_RE = _re(
    r"\b(start|begin|complete|invite|invitation|link|timed|deadline|expires|take|attempt)\b",
)

APPLIED_RE = _re(
    r"thank(s| you) for (applying|your application)",
    r"thank you for (submitting|completing) your application",
    r"thank you for your interest in",
    r"we (have |just |also )*received your (application|resume|cv|submission)",
    r"we've received your (application|resume|cv)",
    r"your application (has been|was) (received|submitted)",
    r"application (received|submitted) successfully",
    r"successfully (submitted|applied)",
    r"confirm(ing|ation)?( that)? we (have )?received your (application|resume)",
    r"we are (currently )?reviewing your application",
    r"we will (be )?review(ing)? your (application|qualifications)",
    r"application (confirmation|received)",
)

RECOMMENDATIONS_RE = _re(
    r"recommended you for (this|the|a) (role|position|job)",
    r"you (have been|were) referred",
    r"referred you (for|to)",
    r"\breferral\b",
    r"\bwas referred\b",
)

JOB_ALERTS_RE = _re(
    r"\bjob alert\b",
    r"\bnew jobs?\b",
    r"\bjobs you may (like|be interested in)\b",
    r"\brecommended jobs\b",
    r"\bjob matches\b",
    r"\bjobs for you\b",
    r"\b(job|career)s? newsletter\b",
    r"\bjobs? (posted|matching your)\b",
)

# Marketing signals only. "unsubscribe" and a bare "offer" are deliberately
# excluded: recruiting mail carries unsubscribe footers, and a job offer is real.
ADVERTISEMENTS_RE = _re(
    r"\b\d+% off\b",
    r"%\s?off\b",
    r"\bdiscount(ed|s)?\b",
    r"\bcoupon\b",
    r"\bpromo code\b",
    r"\bpromotional\b",
    r"\b(flash|clearance)? ?sale\b",
    r"\blimited[- ]time\b",
    r"\b(shop|buy|order) now\b",
    r"\bfree trial\b",
    r"\b(special|exclusive|introductory) offer\b",
    r"\bbest deals?\b",
    r"\bdeal of the (day|week)\b",
    r"\bblack friday\b",
    r"\bcyber monday\b",
    r"\bupgrade (now|today|to pro)\b",
    r"\bsubscribe (now|today)\b",
)


def _matches(pattern: re.Pattern[str]) -> Predicate:
    return lambda text: bool(pattern.search(text))


def _matches_all(*patterns: re.Pattern[str]) -> Predicate:
    return lambda text: all(p.search(text) for p in patterns)


def _any_of(*predicates: Predicate) -> Predicate:
    return lambda text: any(p(text) for p in predicates)


def _is_interview_scheduling(text: str) -> bool:
    """Actionable interview scheduling, not a conditional mention in an ack."""
    if not INTERVIEW_SCHEDULING_RE.search(text):
        return False
    is_conditional_ack = bool(
        APPLIED_RE.search(text) and CONDITIONAL_INTERVIEW_RE.search(text)
    )
    return not is_conditional_ack


_is_assessment = _any_of(
    _matches(ASSESSMENT_VENDOR_RE),
    _matches_all(ASSESSMENT_TOPIC_RE, ASSESSMENT_ACTION_RE),
)

_is_interview_conversation = _any_of(
    _matches_all(INTERVIEW_TOPIC_RE, SCHEDULING_INTENT_RE),
    _matches_all(MEETING_LINK_RE, INTERVIEW_TOPIC_RE),
)


# Highest priority first.
RULES: tuple[tuple[JobLabel, Predicate], ...] = (
    (JobLabel.OTP_SECURITY, _matches(OTP_RE)),
    (JobLabel.REJECTED, _matches(REJECTED_RE)),
    (JobLabel.INTERVIEWS, _is_interview_scheduling),
    (JobLabel.ASSESSMENTS, _is_assessment),
    (JobLabel.APPLIED, _matches(APPLIED_RE)),
    (JobLabel.INTERVIEWS, _is_interview_conversation),
    (JobLabel.RECOMMENDATIONS, _matches(RECOMMENDATIONS_RE)),
    (JobLabel.JOB_ALERTS, _matches(JOB_ALERTS_RE)),
    (JobLabel.ADVERTISEMENTS, _matches(ADVERTISEMENTS_RE)),
)


def prepare_rule_text(subject: str, snippet: str, from_email: str) -> str:
    """Lowercased, HTML-stripped text used by every rule."""
    return normalize_email_text(
        subject=subject,
        snippet=f"{snippet}\n{from_email}",
        max_chars=RULE_MAX_CHARS,
    )


def short_circuit_label(subject: str, snippet: str, from_email: str) -> JobLabel | None:
    """Return a label when the rules are confident, else None for LLM fallback."""
    text = prepare_rule_text(subject, snippet, from_email)
    for label, predicate in RULES:
        if predicate(text):
            return label
    return None


NEEDS_BODY_RE = _re(
    r"\b(status|update|interest|next step|next steps|moving forward|decision|regarding your)\b"
)


def needs_body_fetch(subject: str, snippet: str) -> bool:
    """True for templates whose snippet usually hides the actual outcome."""
    return bool(NEEDS_BODY_RE.search(f"{subject}\n{snippet}"))
