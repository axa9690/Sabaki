"""Deterministic rule layer: priority and the known tricky emails."""

from __future__ import annotations

import pytest

from email_agent.pipeline.rules import needs_body_fetch, short_circuit_label
from email_agent.schemas import JobLabel

SENDER = "talent@example.com"

UNSUBSCRIBE_FOOTER = (
    "\n\nYou are receiving this email because you applied on our site. "
    "Unsubscribe from these emails."
)


def classify(subject: str, body: str, sender: str = SENDER) -> JobLabel | None:
    return short_circuit_label(subject, body, sender)


@pytest.mark.parametrize(
    "subject, body, expected",
    [
        (
            "Update on your application",
            "After careful consideration, we decided not to move forward with your "
            "candidacy at this time.",
            JobLabel.REJECTED,
        ),
        (
            "Software Engineer role",
            "Thank you for your patience. The position has been filled.",
            JobLabel.REJECTED,
        ),
        (
            "Application received",
            "Thank you for applying to the Data Engineer position.",
            JobLabel.APPLIED,
        ),
        (
            "Your application",
            "Thank you for your application. Our team will review it shortly.",
            JobLabel.APPLIED,
        ),
        (
            "We got it",
            "We received your application for the Backend Engineer opening.",
            JobLabel.APPLIED,
        ),
        (
            "Interview scheduling",
            "Please schedule your interview using the link below.",
            JobLabel.INTERVIEWS,
        ),
        (
            "Next steps",
            "Select an interview time slot that works for you.",
            JobLabel.INTERVIEWS,
        ),
        (
            "Coding challenge",
            "Your HackerRank test is ready. The link expires in 7 days.",
            JobLabel.ASSESSMENTS,
        ),
        (
            "Technical assessment invitation",
            "Please complete the online assessment before Friday.",
            JobLabel.ASSESSMENTS,
        ),
        (
            "A referral for you",
            "You have been referred to this role by a colleague.",
            JobLabel.RECOMMENDATIONS,
        ),
        (
            "Your weekly job alert",
            "New jobs matching your profile are available.",
            JobLabel.JOB_ALERTS,
        ),
        (
            "Flash sale inside",
            "Get 40% off our annual plan. Shop now!",
            JobLabel.ADVERTISEMENTS,
        ),
        (
            "Your verification code",
            "Your one-time code is 483920. Do not share it.",
            JobLabel.OTP_SECURITY,
        ),
    ],
)
def test_expected_rule_outcomes(subject, body, expected):
    assert classify(subject, body) is expected


def test_application_ack_mentioning_a_future_interview_stays_applied():
    body = (
        "Thank you for applying to the Analyst position. "
        "If selected for an interview, a recruiter will contact you to schedule a time."
    )

    assert classify("Application received", body) is JobLabel.APPLIED


def test_application_ack_with_conditional_scheduling_stays_applied():
    body = (
        "We received your application. If your qualifications match our needs, "
        "we will contact you to schedule an interview."
    )

    assert classify("Thanks for your interest", body) is JobLabel.APPLIED


def test_actionable_interview_wins_over_acknowledgement():
    body = (
        "Thank you for applying. Good news: please schedule your interview with the "
        "hiring manager using this Calendly link."
    )

    assert classify("Interview invitation", body) is JobLabel.INTERVIEWS


def test_recruiting_email_with_unsubscribe_footer_is_not_an_advertisement():
    body = "We received your application for the Platform Engineer role." + UNSUBSCRIBE_FOOTER

    assert classify("Application received", body) is JobLabel.APPLIED


def test_unsubscribe_footer_alone_is_not_an_advertisement():
    body = (
        "A member of our talent team would like to connect about an opportunity."
        + UNSUBSCRIBE_FOOTER
    )

    assert classify("Quick question", body) is not JobLabel.ADVERTISEMENTS


def test_the_word_offer_alone_is_not_an_advertisement():
    body = "We are pleased to extend an offer of employment for the SDE II role."

    assert classify("Your offer", body) is not JobLabel.ADVERTISEMENTS


def test_promo_code_email_is_not_an_assessment():
    body = "Use promo code SAVE20 for a discount on your next order."

    assert classify("Deal inside", body) is JobLabel.ADVERTISEMENTS


def test_rejection_wins_over_acknowledgement_wording():
    body = (
        "Thank you for applying to the Data Analyst role. "
        "Unfortunately, we have decided not to proceed with your application."
    )

    assert classify("Application update", body) is JobLabel.REJECTED


def test_voluntary_tax_credit_survey_is_left_to_the_llm():
    body = (
        "Please complete this voluntary Work Opportunity Tax Credit questionnaire. "
        "Your response is optional and does not affect your candidacy."
    )

    assert classify("Voluntary questionnaire", body) is None


def test_unclassifiable_email_returns_none():
    assert classify("Lunch tomorrow?", "Are we still on for noon?") is None


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("Status of your application", True),
        ("An update for you", True),
        ("Next steps", True),
        ("Your weekly newsletter", False),
    ],
)
def test_needs_body_fetch(subject, expected):
    assert needs_body_fetch(subject, "") is expected


def test_rules_read_html_bodies():
    body = "<html><body><p>Thank you for <b>applying</b> to our team.</p></body></html>"

    assert classify("Application", body) is JobLabel.APPLIED
