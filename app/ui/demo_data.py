"""One-click demo data for the Inbox tab's empty state.

Reuses the existing M3/M9 ground-truth dataset (data/eval_dataset.json) as
a small set of realistic synthetic emails, and the existing M7
`EmailProcessor` to classify/decide/store them — no new email content, no
parallel classification path, no new persistence. A real (small, one-time,
user-triggered) Gemini call per not-yet-cached email; already-cached ones
are served from the existing message_id cache exactly as everywhere else.
"""

from __future__ import annotations

from app.core.models import EmailMessage
from app.core.pipeline import EmailProcessor, ProcessingResult, run_batch
from app.evaluation.dataset import load_ground_truth

DEFAULT_DEMO_COUNT = 5


def load_demo_emails(limit: int = DEFAULT_DEMO_COUNT) -> list[EmailMessage]:
    """The first `limit` emails from the shipped ground-truth dataset,
    chosen only for a mix of categories — not an evaluation run.
    """
    records = load_ground_truth()
    return [record.to_email_message() for record in records[:limit]]


def run_demo_batch(processor: EmailProcessor, limit: int = DEFAULT_DEMO_COUNT) -> list[ProcessingResult]:
    return run_batch(processor, load_demo_emails(limit))
