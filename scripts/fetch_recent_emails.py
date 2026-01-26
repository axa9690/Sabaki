from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from email_agent.gmail.fetch import fetch_recent_emails

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.labels",
]

def main():
    repo_root = Path(__file__).resolve().parents[1]
    token_path = repo_root / "secrets" / "gmail_token.json"

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    emails = fetch_recent_emails(service, max_results=1)

    for i, e in enumerate(emails, start=1):
        print("\n" + "=" * 80)
        print(f"{i}. Subject: {e.subject}")
        print(f"   From: {e.from_email}")
        print(f"   Date: {e.date}")
        print(f"   Snippet: {e.snippet}")
        preview = e.body_text[:300].replace("\n", " ")
        print(f"   Body Preview: {preview}")

if __name__ == "__main__":
    main()
