from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.labels",
]

def main():
    repo_root = Path(__file__).resolve().parents[1]
    token_path = repo_root / "secrets" / "gmail_token.json"

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    profile = service.users().getProfile(userId="me").execute()
    print("✅ Auth OK")
    print("Email:", profile.get("emailAddress"))
    print("Messages total:", profile.get("messagesTotal"))
    print("Threads total:", profile.get("threadsTotal"))

if __name__ == "__main__":
    main()
