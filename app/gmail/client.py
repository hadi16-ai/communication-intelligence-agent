"""Read-only Gmail API client.

This is the ONLY module in the application that talks to the Gmail API.
It exposes exactly two operations — list message ids, get one message —
and nothing else. There is no send/modify/delete/label/settings method
anywhere in this class: it isn't merely policy that callers must not
misuse it, the class is structurally incapable of those operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.core.models import EmailMessage
from app.gmail.parser import GmailParsingError, parse_gmail_message

READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
"""The narrowest Gmail scope that allows reading messages. Deliberately
NOT `gmail.modify`, `gmail.compose`, `gmail.send`, `gmail.labels`,
`gmail.settings.basic`, or the full `mail.google.com` scope — this
application must be structurally incapable of changing anything in the
user's mailbox, not just conventionally well-behaved.
"""


class GmailAuthError(RuntimeError):
    """Missing/invalid/expired OAuth credentials. Messages here describe
    the problem and the fix (re-run the authorization script) — never the
    token or client secret contents.
    """


class GmailAPIError(RuntimeError):
    """The Gmail API itself returned an error (permissions, rate
    limiting, service unavailable, etc.).
    """


@dataclass(frozen=True)
class FetchResult:
    """The result of one fetch_recent_messages() call. A per-message
    failure (a message that couldn't be fetched or parsed) never aborts
    the whole batch, and is never silently dropped — it's recorded here
    so the caller (the Streamlit Gmail tab) can show it.
    """

    emails: list[EmailMessage] = field(default_factory=list)
    failed_message_ids: list[tuple[str, str]] = field(default_factory=list)


class GmailClient:
    """A thin, read-only wrapper around the Gmail API."""

    def __init__(self, credentials: Credentials):
        self._service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    @classmethod
    def from_token_file(cls, token_path: str | Path, client_secret_path: str | Path) -> GmailClient:
        """Loads previously saved credentials (see scripts/authorize_gmail.py)
        and refreshes them if expired, persisting the refreshed token.

        Never performs the interactive OAuth consent flow itself — that is
        a deliberate, one-time, manual step run from a terminal, not from
        inside a Streamlit script (see scripts/authorize_gmail.py docstring).
        """
        token_path = Path(token_path)
        if not token_path.exists():
            raise GmailAuthError(
                f"No Gmail token found at '{token_path}'. Run "
                "'python scripts/authorize_gmail.py' once to connect your Gmail account "
                f"(client secret expected at '{client_secret_path}')."
            )

        try:
            credentials = Credentials.from_authorized_user_file(str(token_path), [READONLY_SCOPE])
        except (ValueError, OSError) as exc:
            raise GmailAuthError(f"Gmail token file at '{token_path}' is invalid or unreadable.") from exc

        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except RefreshError as exc:
                raise GmailAuthError(
                    "Gmail credentials expired and could not be refreshed. "
                    "Run 'python scripts/authorize_gmail.py' again to reconnect."
                ) from exc
            token_path.write_text(credentials.to_json(), encoding="utf-8")

        if not credentials.valid:
            raise GmailAuthError(
                "Gmail credentials are not valid. Run 'python scripts/authorize_gmail.py' to reconnect."
            )

        return cls(credentials)

    def list_message_ids(self, max_results: int) -> list[str]:
        """IDs of the most recent inbox messages — read-only `messages.list`."""
        try:
            response = (
                self._service.users()
                .messages()
                .list(userId="me", maxResults=max_results, labelIds=["INBOX"])
                .execute()
            )
        except HttpError as exc:
            raise GmailAPIError(f"Gmail API error while listing messages: {exc}") from exc

        return [message["id"] for message in response.get("messages", [])]

    def get_message(self, message_id: str) -> dict:
        """The full raw message resource for one id — read-only `messages.get`."""
        try:
            return (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:
            raise GmailAPIError(f"Gmail API error while fetching message '{message_id}': {exc}") from exc


def fetch_recent_messages(client: GmailClient, max_results: int) -> FetchResult:
    """Lists and fetches up to `max_results` recent inbox messages,
    normalized into EmailMessage. The single entry point the rest of the
    application should use — callers then hand the resulting emails to
    the existing EmailProcessor exactly as they would any other
    EmailMessage source (see app/core/pipeline.py).
    """
    message_ids = client.list_message_ids(max_results)

    emails: list[EmailMessage] = []
    failures: list[tuple[str, str]] = []
    for message_id in message_ids:
        try:
            raw = client.get_message(message_id)
            emails.append(parse_gmail_message(raw))
        except (GmailAPIError, GmailParsingError) as exc:
            failures.append((message_id, str(exc)))

    return FetchResult(emails=emails, failed_message_ids=failures)
