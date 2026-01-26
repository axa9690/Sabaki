from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    secrets_path = repo_root / "secrets" / "gmail_oauth_client.json"
    token_path = repo_root / "secrets" / "gmail_token.json"

    if not secrets_path.exists():
        raise FileNotFoundError(
            f"Missing: {secrets_path}\n"
            "Download OAuth Desktop client JSON from Google Cloud Console."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
    creds = flow.run_local_server(port=0)

    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"✅ Saved token to: {token_path}")

if __name__ == "__main__":
    main()
