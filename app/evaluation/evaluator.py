"""Ties ground truth (app.evaluation.dataset), stored predictions
(app.storage.repositories.EmailRepository), and metric computation
(app.evaluation.metrics) together into a single EvaluationReport.

Makes NO Gemini/network calls whatsoever. This module only:
  - reads data/eval_dataset.json (ground truth),
  - reads whatever has already been stored in SQLite (actual predictions),
  - runs the real, unmodified DecisionEngine + PreferenceManager on a
    deterministic stand-in classification to derive an "expected
    decision" — never from Gemini's output, never invented ad hoc.

A record with no stored classification is excluded from both evaluation
axes and counted only in coverage — never fabricated. This is what makes
partial evaluation (today: 6 of 41 emails) safe and honest.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.decisions import DecisionEngine
from app.core.models import Category, ClassificationResult, Decision, EmailMessage, RiskFlag, UrgencyLevel
from app.evaluation.dataset import GroundTruthRecord, load_ground_truth
from app.evaluation.metrics import (
    ClassificationMetrics,
    DecisionMetrics,
    compute_classification_metrics,
    compute_decision_metrics,
)
from app.preferences.manager import PreferenceManager
from app.storage.repositories import EmailRepository


@dataclass(frozen=True)
class EvaluationCoverage:
    """How much of the dataset has actually been evaluated.

    Never conflate `evaluated_count` with `total_count`: a partial
    evaluation must always be presented as partial, never as if it
    represented the full dataset.
    """

    evaluated_count: int
    total_count: int
    missing_message_ids: tuple[str, ...]

    @property
    def missing_count(self) -> int:
        return self.total_count - self.evaluated_count

    @property
    def coverage_percent(self) -> float:
        return (self.evaluated_count / self.total_count * 100) if self.total_count > 0 else 0.0

    @property
    def is_partial(self) -> bool:
        return self.evaluated_count < self.total_count


@dataclass(frozen=True)
class MismatchedRecord:
    """One record where expected and actual disagreed, on either
    evaluation axis."""

    message_id: str
    expected: Category
    actual: Category
    notes: str = ""


@dataclass(frozen=True)
class EvaluationReport:
    """The full evaluation result: coverage, both metric axes, and the
    specific records that disagreed on each axis.

    `notify_recall_*` and `quarantine_recall_*` are exposed directly as
    the two safety/product-critical numbers this project has explicitly
    prioritized from the start: missing a genuinely important
    job/interview/security email is worse than an extra notification, and
    failing to quarantine something dangerous is a distinct, equally
    serious failure mode. Both are surfaced at classification level (did
    the AI understand the email correctly) and decision level (did the
    personalized system act on it correctly) since those are different
    questions with different failure causes.
    """

    coverage: EvaluationCoverage
    classification: ClassificationMetrics
    decision: DecisionMetrics
    classification_mismatches: tuple[MismatchedRecord, ...]
    decision_mismatches: tuple[MismatchedRecord, ...]

    @property
    def notify_recall_classification(self) -> float:
        return self.classification.per_category[Category.NOTIFY].recall

    @property
    def notify_recall_decision(self) -> float:
        return self.decision.per_category[Category.NOTIFY].recall

    @property
    def quarantine_recall_classification(self) -> float:
        return self.classification.per_category[Category.QUARANTINE].recall

    @property
    def quarantine_recall_decision(self) -> float:
        return self.decision.per_category[Category.QUARANTINE].recall


def derive_ground_truth_classification(email: EmailMessage, expected_category: Category) -> ClassificationResult:
    """A deterministic stand-in `ClassificationResult` representing "what
    a perfect classifier would have said" for a ground-truth category.

    Used ONLY to derive an "expected decision" via the real
    `DecisionEngine` below — it is never stored and never compared
    against as if it were an actual prediction.

    The dataset labels only a category, not urgency or risk flags
    independently, so this mirrors the exact convention already
    established and tested in the M5/M6/M7 dataset-compatibility tests:
    QUARANTINE implies a generic risk flag (so the decision engine's
    security check fires, matching its own real behavior), NOTIFY implies
    HIGH urgency, everything else LOW.
    """
    risk_flags = [RiskFlag.OTHER] if expected_category == Category.QUARANTINE else []
    urgency = UrgencyLevel.HIGH if expected_category == Category.NOTIFY else UrgencyLevel.LOW
    return ClassificationResult(
        message_id=email.message_id,
        category=expected_category,
        urgency=urgency,
        confidence=1.0,
        risk_flags=risk_flags,
        summary="Ground-truth stand-in for evaluation only; never stored.",
        reasoning="Deterministically derived from data/eval_dataset.json expected_category for decision evaluation.",
    )


def derive_expected_decision(
    email: EmailMessage,
    expected_category: Category,
    preferences: PreferenceManager,
    decision_engine: DecisionEngine,
) -> Decision:
    """What the EXISTING, unmodified decision policy (DecisionEngine +
    the real preferences.json) says should happen, given a perfect
    classification of `expected_category`. This is "expected decision"
    per the M9 requirement — deterministic, derived from the existing
    architecture, never from Gemini.
    """
    ground_truth_classification = derive_ground_truth_classification(email, expected_category)
    return decision_engine.decide(email, ground_truth_classification, preferences)


def evaluate(
    repository: EmailRepository,
    preferences: PreferenceManager,
    decision_engine: DecisionEngine | None = None,
    ground_truth: list[GroundTruthRecord] | None = None,
) -> EvaluationReport:
    """Builds a full `EvaluationReport` from whatever has actually been
    stored in `repository`. Never calls a classifier and never invents a
    result for a message_id that hasn't been processed yet.

    `ground_truth` defaults to the full shipped dataset; tests pass a
    smaller, controlled list instead. Duplicate message_ids in
    `ground_truth` are de-duplicated (last one wins) — defensive, since
    the shipped dataset is already tested elsewhere to have none.
    """
    decision_engine = decision_engine or DecisionEngine()
    records = ground_truth if ground_truth is not None else load_ground_truth()
    records_by_id = {record.message_id: record for record in records}

    classification_pairs: list[tuple[Category, Category]] = []
    decision_pairs: list[tuple[Category, Category]] = []
    classification_mismatches: list[MismatchedRecord] = []
    decision_mismatches: list[MismatchedRecord] = []
    evaluated_ids: set[str] = set()

    for message_id, record in records_by_id.items():
        actual_classification = repository.get_classification(message_id)
        if actual_classification is None:
            continue  # never fabricate a missing prediction

        evaluated_ids.add(message_id)
        email = record.to_email_message()

        classification_pairs.append((record.expected_category, actual_classification.category))
        if record.expected_category != actual_classification.category:
            classification_mismatches.append(
                MismatchedRecord(
                    message_id=message_id,
                    expected=record.expected_category,
                    actual=actual_classification.category,
                    notes=record.notes,
                )
            )

        actual_decision = repository.get_decision(message_id)
        if actual_decision is not None:
            expected_decision = derive_expected_decision(
                email, record.expected_category, preferences, decision_engine
            )
            decision_pairs.append((expected_decision.category, actual_decision.category))
            if expected_decision.category != actual_decision.category:
                decision_mismatches.append(
                    MismatchedRecord(
                        message_id=message_id,
                        expected=expected_decision.category,
                        actual=actual_decision.category,
                        notes=record.notes,
                    )
                )

    coverage = EvaluationCoverage(
        evaluated_count=len(evaluated_ids),
        total_count=len(records_by_id),
        missing_message_ids=tuple(sorted(set(records_by_id) - evaluated_ids)),
    )

    return EvaluationReport(
        coverage=coverage,
        classification=compute_classification_metrics(classification_pairs),
        decision=compute_decision_metrics(decision_pairs),
        classification_mismatches=tuple(classification_mismatches),
        decision_mismatches=tuple(decision_mismatches),
    )
