"""Pure, streamlit-free helpers backing the Gmail tab: connection-status
checks and fetch/process orchestration.

Contains NO Gmail API calls of its own (those live in app.gmail.client)
and NO classification/decision logic of its own (that stays in
app.core.pipeline.EmailProcessor) — this module only wires the two
together, using the exact same cache-lookup -> classify -> decide -> store
path every other EmailMessage source goes through, and shapes the result
for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.pipeline import ClassificationSource, EmailProcessor, ProcessingResult, run_batch
from app.gmail.client import FetchResult, GmailClient, fetch_recent_messages


def gmail_is_connected(token_path: str | Path) -> bool:
    """Whether a saved Gmail token exists — i.e. whether
    scripts/authorize_gmail.py has been run. Does not validate or refresh
    the token itself (that happens lazily in
    `GmailClient.from_token_file` when actually used); this is only a
    quick check for which UI state to show.
    """
    return Path(token_path).exists()


@dataclass(frozen=True)
class GmailSyncResult:
    """Everything the UI needs to summarize one fetch-and-process run."""

    fetch: FetchResult
    processing: list[ProcessingResult]

    @property
    def fetched_count(self) -> int:
        return len(self.fetch.emails)

    @property
    def failed_count(self) -> int:
        return len(self.fetch.failed_message_ids)

    @property
    def cache_hit_count(self) -> int:
        return sum(1 for r in self.processing if r.classification_source == ClassificationSource.CACHE)

    @property
    def newly_classified_count(self) -> int:
        return len(self.processing) - self.cache_hit_count


def sync_gmail(client: GmailClient, processor: EmailProcessor, max_results: int) -> GmailSyncResult:
    """Fetches up to `max_results` recent Gmail messages and runs them
    through the existing `EmailProcessor` — the same pipeline used
    everywhere else in the app. No parallel processing path, no
    duplicated caching logic.
    """
    fetch_result = fetch_recent_messages(client, max_results)
    processing_results = run_batch(processor, fetch_result.emails)
    return GmailSyncResult(fetch=fetch_result, processing=processing_results)


def format_sync_summary(result: GmailSyncResult) -> str:
    parts = [f"Fetched {result.fetched_count} message(s)."]
    if result.processing:
        parts.append(
            f"{result.newly_classified_count} newly classified, {result.cache_hit_count} served from cache."
        )
    if result.failed_count:
        parts.append(f"{result.failed_count} message(s) could not be processed.")
    return " ".join(parts)
