from __future__ import annotations

import os
import re

from email_agent.config import JOB_LABELS, PROCESSED_LABEL
from email_agent.gmail import labels
from email_agent.gmail import service
from email_agent.gmail.fetch_meta import fetch_recent_email_meta
from email_agent.gmail.fetch import fetch_recent_emails
from email_agent.gmail.labels import ensure_labels, apply_labels
from email_agent.gmail.service import build_gmail_service
from email_agent.gmail.fetch_body import fetch_email_body_text
from email_agent.pipeline.analyzer import analyze_email_with_ollama
from email_agent.llm.ollama_client import OllamaClient
from email_agent.schemas import JobLabel, EmailAnalysis


def short_circuit_label(subject: str, snippet: str, from_email: str) -> JobLabel | None:
    text = f"{subject}\n{snippet}\n{from_email}".lower()

    # Advertisements / promotions / marketing
    if re.search(r"\bunsubscribe\b|\bpromo\b|\bpromotion\b|\bdeal\b|\boffer\b|\bdiscount\b|\bsale\b|\b% off\b|\bvaibhav sisinity\b|\bextern\b", text):
        return JobLabel.ADVERTISEMENTS
    
    # Job alerts (LinkedIn/Indeed/company alerts)
    if re.search(r"\bjob alert\b|\bnew job(s)?\b|\bjobs you may like\b|\brecommended jobs\b|\bjob matches\b", text):
        return JobLabel.JOB_ALERTS
    
    # OTP/security codes
    if re.search(r"\botp\b|\bverification code\b|\bsecurity code\b|\bpasscode\b|\bone[- ]time\b", text):
        return JobLabel.OTP_SECURITY

    # Rejection
    if re.search(r"\bunfortunately\b|\bregret to inform\b|\bwe regret\b|\bnot selected\b|\bdeclined\b|\bmoving forward with other candidates?\b|\bnot to move forward\b", text):
        return JobLabel.REJECTED

    # Interview
    if re.search(r"\binterview\b|\bschedule\b|\bcalendly\b|\bzoom\b|\bgoogle meet\b|\bteams meeting\b", text):
        return JobLabel.INTERVIEWS

    # Applied/confirmation
    if re.search(
        r"\bwe (just )?(have )?received your (application|resume)\b"
        r"|\bconfirm(ing)? that we (have )?received your (application|resume)\b"
        r"|\bthank you for (your )?interest\b"
        r"|\bthanks for (your )?interest\b"
        r"|\bthanks for applying\b"
        r"|\bwe received your application\b"
        r"|\byour application\b.*\b(received|submitted)\b",
        text):
        return JobLabel.APPLIED

    # Assessment (invite + action OR known platform)
    if (
        re.search(r"\b(assessment|coding challenge|skill assessment)\b", text)
        and re.search(r"\b(start|click|begin|complete|link|timed)\b", text)
    ) or re.search(r"\bhackerrank\b|\bshl\b|\bcodility\b|\bkarat\b|\bcode(signal)?\b", text):
        return JobLabel.ASSESSMENTS

    # Recommendations (role recommendations / similar jobs)
    if re.search(r"\brecommended for you\b|\byou might be interested\b|\bsuggested (role|job|position)\b|\bsimilar jobs\b", text):
        return JobLabel.RECOMMENDATIONS

    

    return None

def needs_body_fetch(subject: str, snippet: str) -> bool:
    s = f"{subject}\n{snippet}".lower()
    # subjects/templates where snippet often hides the outcome
    return bool(re.search(r"\b(status|update|interest|next step|moving forward)\b", s))


def debug_others(email, combined_text):
    print("\n" + "=" * 80)
    print("⚠️ DEBUG: CLASSIFIED AS OTHERS")
    print(f"From   : {email.from_email}")
    print(f"Subject: {email.subject}")
    print("----- TEXT SENT TO CLASSIFIER -----")
    print(combined_text)  # cap to avoid terminal spam
    print("=" * 80 + "\n")


def main():
    max_emails = int(os.getenv("MAX_EMAILS", "50"))

    service = build_gmail_service()

    # Ensure labels exist in Gmail
    wanted = JOB_LABELS + [PROCESSED_LABEL]
    label_ids = ensure_labels(service, wanted)

    processed_label_id = label_ids[PROCESSED_LABEL]

    #emails = fetch_recent_email_meta(service, max_results=max_emails)
    emails = fetch_recent_emails(service, max_results=max_emails)

    checked = labeled = skipped = 0

    for e in emails:
        checked += 1

        # skip already processed
        if processed_label_id in e.label_ids or PROCESSED_LABEL in e.label_ids:
            skipped += 1
            continue


        # short-circuit first (snippet)
        forced = short_circuit_label(e.subject, e.snippet, e.from_email)

        # If risky template OR forced=APPLIED but could be rejection later, fetch body and re-check
        body_text = ""
        if (forced is None) or needs_body_fetch(e.subject, e.snippet):
            body_text = fetch_email_body_text(service, e.message_id)

        if body_text:
            forced2 = short_circuit_label(e.subject, f"{e.snippet}\n{body_text}", e.from_email)
            if forced2:
                forced = forced2

        if forced:
            final_label = forced
            reasoning = "rule_short_circuit"
        else:
            # LLM fallback (Ollama)
            analysis: EmailAnalysis = analyze_email_with_ollama(
                subject=e.subject,
                from_email=e.from_email,
                snippet=(f"{e.snippet}\n{body_text}" if body_text else e.snippet),
                date=e.date,
                client=OllamaClient()
            )
            final_label = analysis.label
            reasoning = analysis.reasoning_brief
        
        if final_label == JobLabel.OTHERS:
            combined_text = f"{e.snippet}".lower()
            debug_others(e, combined_text)
            
            apply_labels(service, e.message_id, add_label_ids=[processed_label_id])
            print(f"⚠️ Unclassified (PROCESSED only): {e.subject[:70]}")
            skipped += 1
            continue
        
        add_ids = [label_ids[final_label.value], processed_label_id]

        combined_text = f"{e.snippet}".lower()
        debug_others(e, combined_text)
            
            
        remove_ids = []
        if final_label in (JobLabel.APPLIED, JobLabel.REJECTED, JobLabel.ADVERTISEMENTS):
            remove_ids.append("UNREAD")  # Gmail system label

        apply_labels(service, e.message_id, add_label_ids=add_ids, remove_label_ids=remove_ids)

        labeled += 1
        print(f"✅ Labeled: {e.subject[:70]} -> {final_label.value} (+PROCESSED) [{reasoning}]")

    print(f"\nDone. checked={checked}, labeled={labeled}, skipped={skipped}")


if __name__ == "__main__":
    main()
