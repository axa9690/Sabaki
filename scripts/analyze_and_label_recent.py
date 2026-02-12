from __future__ import annotations

import os
import re

from httpx import ReadTimeout

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

    # 1️⃣ OTP / security (highest priority)
    if re.search(r"\botp\b|\bverification code\b|\bsecurity code\b|\bpasscode\b|\bone[- ]time\b", text):
        return JobLabel.OTP_SECURITY

    # 2️⃣ REJECTED (must be early)
    if re.search(
        r"(not move forward with your candidacy|"
        r"not move forward at this time|"
        r"decided not to move forward|"
        r"we have decided not to proceed|"
        r"we regret to inform you|"
        r"after careful consideration.*not move forward|"
        r"unfortunately.*not (moving forward|move forward|proceed)|"
        r"your application was not selected|"
        r"you were not selected|"
        r"\b(position|role)\b.*\bhas (now )?been filled\b|"
        r"\binterview process\b.*\b(concluded|ended|closed)\b|"
        r"\bprocess has officially concluded\b|"
        r"will not be moving forward)|"
        r"position has been closed|"
        r"position is no longer available",
        text,
    ):
        return JobLabel.REJECTED

    # 3️⃣ INTERVIEWS (require scheduling intent)
    if (
        re.search(r"\b(interview|phone screen|screening call|onsite|final round)\b", text)
        and re.search(r"\b(schedule|scheduled|reschedule|confirm|confirmation|book|booking|calendar|invite|invitation|availability|time slot)\b", text)
    ) or re.search(r"\bcalendly\b|\bzoom\b|\bgoogle meet\b|\bmicrosoft teams\b|\bteams meeting\b", text):
        return JobLabel.INTERVIEWS

    # 4️⃣ ASSESSMENTS
    if (
        re.search(r"\b(assessment|coding challenge|skill assessment)\b", text)
        and re.search(r"\b(start|click|begin|complete|link|timed)\b", text)
    ) or re.search(r"\bhackerrank\b|\bshl\b|\bcodility\b|\bkarat\b|\bcode(signal)?\b", text):
        return JobLabel.ASSESSMENTS

    # 5️⃣ APPLIED / acknowledgements
    if re.search(
        r"\bthank you for applying\b|"
        r"\bthank you for your interest\b|"
        r"\bthank you for submitting your application\b|"
        r"\bwe (just )?(have )?received your (application|resume)\b|"
        r"\bconfirm(ing)? that we (have )?received your (application|resume)\b|"
        r"\byour application\b.*\b(received|submitted)\b|"
        r"\bwe are currently reviewing your application\b|"
        r"\bwe will be reviewing your qualifications\b",
        text,
    ):
        return JobLabel.APPLIED

    # 6️⃣ RECOMMENDATIONS (referrals)
    if re.search(
        r"\brecommended you for this role\b|"
        r"\byou have been referred\b|"
        r"\breferred you\b|"
        r"\breferral\b",
        text,
    ):
        return JobLabel.RECOMMENDATIONS

    # 7️⃣ JOB ALERTS
    if re.search(
        r"\bjob alert\b|"
        r"\bnew job(s)?\b|"
        r"\bjobs you may like\b|"
        r"\brecommended jobs\b|"
        r"\bjob matches\b",
        text,
    ):
        return JobLabel.JOB_ALERTS

    # 8️⃣ ADVERTISEMENTS (always last)
    if re.search(
        r"\bunsubscribe\b|"
        r"\bpromo\b|"
        r"\bpromotion\b|"
        r"\bdeal\b|"
        r"\boffer\b|"
        r"\bdiscount\b|"
        r"\bsale\b|"
        r"\b% off\b|"
        r"\bextern\b",
        text,
    ):
        return JobLabel.ADVERTISEMENTS

    # IMPORTANT: allow LLM fallback
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
    print(combined_text)  
    print("=" * 80 + "\n")


def main():
    max_emails = 500

    service = build_gmail_service()

    # Ensure labels exist in Gmail
    wanted = JOB_LABELS + [PROCESSED_LABEL]
    label_ids = ensure_labels(service, wanted)

    processed_label_id = label_ids[PROCESSED_LABEL]

    #emails = fetch_recent_email_meta(service, max_results=max_emails)
    emails = fetch_recent_emails(service, max_results=max_emails)

    checked = labeled = skipped = 0
    
    client = OllamaClient()
    client.warmup()

    for e in emails:
        checked += 1

        try:
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
                    client=client
                )
                final_label = analysis.label
                reasoning = analysis.reasoning_brief
            
            if final_label == JobLabel.OTHERS:
                combined_text = f"{e.snippet}".lower()
                
                apply_labels(service, e.message_id, add_label_ids=[processed_label_id])
                print(f"⚠️ Unclassified (PROCESSED only): {e.subject[:70]}")
                skipped += 1
                continue
            
            add_ids = [label_ids[final_label.value], processed_label_id]

            combined_text = f"{e.snippet}".lower()
            
            remove_ids = []
            if final_label in (JobLabel.APPLIED, JobLabel.REJECTED, JobLabel.ADVERTISEMENTS):
                remove_ids.append("UNREAD")  # Gmail system label

            apply_labels(service, e.message_id, add_label_ids=add_ids, remove_label_ids=remove_ids)

            labeled += 1
            print(f"✅ Labeled: {e.subject[:70]} -> {final_label.value} (+PROCESSED) [{reasoning}]")
        
        except ReadTimeout:
            print(f"⏰ Timeout, Skipping email: '{e.subject[:70]}'")
            skipped += 1
            continue
        
        except Exception as ex:
            print(f"❌ Error processing email '{e.subject[:70]}': {ex}")
            skipped += 1
            continue

    print(f"\nDone. checked={checked}, labeled={labeled}, skipped={skipped}")


if __name__ == "__main__":
    main()
