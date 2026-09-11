"""Parses raw Gmail API message resources into the existing EmailMessage
model. Pure and deterministic — no network calls, no Gmail API client
here (see app/gmail/client.py for that).

Gmail message content (sender, subject, body) is UNTRUSTED INPUT, exactly
like the synthetic dataset emails: this module only extracts text, never
interprets or executes anything found inside a message (no HTML
rendering/scripting, no attachment execution). Nothing here treats email
content as instructions — that boundary is enforced again, independently,
by the classifier's prompt design (app/ai/prompts.py).
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

from app.core.models import EmailMessage


class GmailParsingError(RuntimeError):
    """Raised when a Gmail message resource can't be turned into a valid
    EmailMessage — e.g. no id, or no timestamp derivable from either
    `internalDate` or the `Date` header. Never silently fabricated (a
    message truly missing every usable field is a real data problem, not
    something to paper over with a fake "now" timestamp).
    """


class _TextExtractingHTMLParser(HTMLParser):
    """Best-effort HTML -> plain text using only the stdlib. Skips
    `<script>`/`<style>` contents; does not execute or evaluate anything
    — this is text extraction, not rendering.
    """

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._chunks)


def _html_to_text(html_content: str) -> str:
    parser = _TextExtractingHTMLParser()
    try:
        parser.feed(html_content)
    except Exception:
        # Malformed HTML: fall back to the raw markup rather than losing
        # the content entirely — still just text, never executed.
        return html_content
    return parser.get_text()


def _decode_base64url(data: str) -> str:
    """Gmail encodes body content as URL-safe base64 without padding.

    Uses strict validation (`base64.b64decode(..., validate=True)`, after
    translating the URL-safe alphabet) rather than the lenient
    `urlsafe_b64decode`, which would otherwise silently ignore invalid
    characters and produce garbled output instead of raising — defeating
    the point of the except block below.
    """
    try:
        padded = data + "=" * (-len(data) % 4)
        translated = padded.translate(str.maketrans("-_", "+/"))
        decoded_bytes = base64.b64decode(translated, validate=True)
        return decoded_bytes.decode("utf-8", errors="replace")
    except Exception:
        # Corrupt/undecodable body content degrades to empty text rather
        # than blocking the whole message — body is best-effort, unlike
        # message_id/timestamp which are structurally required.
        return ""


def _find_body_by_mime_type(payload: dict, mime_type: str) -> str | None:
    """Depth-first search through the MIME tree for the first part whose
    mimeType matches. Handles a plain single-part message, as well as
    multipart/alternative and multipart/mixed trees (attachments simply
    never match text/plain or text/html and are naturally skipped).
    """
    if payload.get("mimeType") == mime_type:
        data = (payload.get("body") or {}).get("data")
        if data:
            return _decode_base64url(data)
        return None

    for part in payload.get("parts") or []:
        result = _find_body_by_mime_type(part, mime_type)
        if result is not None:
            return result
    return None


def _extract_body(payload: dict) -> str:
    """Prefers text/plain; falls back to text/html converted to plain
    text; otherwise returns "" (e.g. an attachment-only message).
    """
    plain = _find_body_by_mime_type(payload, "text/plain")
    if plain:
        return plain

    html_content = _find_body_by_mime_type(payload, "text/html")
    if html_content:
        return _html_to_text(html_content)

    return ""


def _extract_timestamp(raw: dict, header_map: dict[str, str]) -> datetime:
    internal_date = raw.get("internalDate")
    if internal_date:
        try:
            return datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc)
        except (ValueError, TypeError, OverflowError, OSError):
            pass

    date_header = header_map.get("date")
    if date_header:
        try:
            parsed = parsedate_to_datetime(date_header)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed

    raise GmailParsingError(
        "Gmail message has no usable timestamp (missing/invalid internalDate and Date header)."
    )


def parse_gmail_message(raw: dict) -> EmailMessage:
    """Converts one raw Gmail `messages.get` response into an EmailMessage.

    Defensive about malformed input: missing sender/subject default to
    safe placeholders (an email is still classifiable without them), a
    corrupt/undecodable body degrades to empty text, but a missing
    message id or a wholly unrecoverable timestamp raise `GmailParsingError`
    clearly rather than being silently guessed at.
    """
    message_id = raw.get("id")
    if not message_id:
        raise GmailParsingError("Gmail message is missing its 'id' field.")

    payload = raw.get("payload") or {}
    headers = payload.get("headers") or []
    header_map = {
        h["name"].lower(): h.get("value", "")
        for h in headers
        if isinstance(h, dict) and h.get("name")
    }

    sender = header_map.get("from") or "unknown-sender"
    subject = header_map.get("subject") or ""
    received_at = _extract_timestamp(raw, header_map)
    body = _extract_body(payload)

    return EmailMessage(
        message_id=message_id,
        thread_id=raw.get("threadId"),
        sender=sender,
        subject=subject,
        body=body,
        received_at=received_at,
    )
