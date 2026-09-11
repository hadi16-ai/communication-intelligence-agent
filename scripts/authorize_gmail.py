"""One-time interactive OAuth flow to generate a local Gmail token.json.

Run this manually from a terminal — it is NOT invoked by the Streamlit
app, since the interactive browser consent flow (opening a browser,
listening for a local redirect) doesn't fit inside a Streamlit script
rerun. Run it once; the Streamlit app's Gmail tab then reuses the saved
token (refreshing it automatically when it expires).

Requests ONLY the read-only Gmail scope
(https://www.googleapis.com/auth/gmail.readonly) — this application can
never be granted permission to send, delete, modify, or label mail
through this flow.

Setup (see README "V1 Gmail Integration" for full details):
  1. In Google Cloud Console, create an OAuth Client ID of type
     "Desktop app" and download its JSON as your client secret file.
  2. Place it locally at the path configured by GMAIL_CLIENT_SECRET_PATH
     in your .env (default: credentials/client_secret.json). This path
     is gitignored — never commit this file.
  3. Run:
         python scripts/authorize_gmail.py
     A browser window opens for you to sign in and consent. On success,
     a token is saved at GMAIL_TOKEN_PATH (default: credentials/token.json)
     — also gitignored, also never committed.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.gmail.client import READONLY_SCOPE  # noqa: E402


def main() -> None:
    settings = Settings()
    client_secret_path = Path(settings.gmail_client_secret_path)
    token_path = Path(settings.gmail_token_path)

    if not client_secret_path.exists():
        print(f"ERROR: no OAuth client secret file found at '{client_secret_path}'.")
        print(
            "Create an OAuth Client ID (type: Desktop app) in Google Cloud Console, "
            "download its JSON, and save it at that path (see README)."
        )
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), [READONLY_SCOPE])
    credentials = flow.run_local_server(port=0)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")

    print(f"Gmail connected (read-only). Token saved to '{token_path}'.")
    print("You can now use the Gmail tab in the Streamlit dashboard.")


if __name__ == "__main__":
    main()
