"""Runs the evaluation module against the CURRENT SQLite database and
prints a text report.

Makes ZERO Gemini/network calls — it only reads whatever classifications
and decisions are already stored via EmailRepository, and compares them
against data/eval_dataset.json ground truth using the deterministic
app.evaluation module. Safe to run at any time, including with an
incomplete/partial database.

Usage:
    python scripts/run_evaluation.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.decisions import DecisionEngine  # noqa: E402
from app.core.models import Category  # noqa: E402
from app.evaluation.evaluator import evaluate  # noqa: E402
from app.preferences.manager import PreferenceManager  # noqa: E402
from app.storage.repositories import EmailRepository  # noqa: E402
from app.ui.evaluation_view import format_recall  # noqa: E402

PREFERENCES_PATH = REPO_ROOT / "data" / "preferences.json"


def main() -> None:
    settings = Settings()
    repository = EmailRepository(settings.database_path)
    repository.initialize()
    preferences = PreferenceManager.from_file(PREFERENCES_PATH)

    report = evaluate(repository, preferences, DecisionEngine())
    coverage = report.coverage

    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Coverage: {coverage.evaluated_count}/{coverage.total_count} ({coverage.coverage_percent:.1f}%)")
    if coverage.is_partial:
        print(
            f"PARTIAL EVALUATION — {coverage.missing_count} email(s) not yet processed "
            "(no Gemini call was made to fill them in):"
        )
        print(f"  {', '.join(coverage.missing_message_ids)}")
    else:
        print("Full evaluation — every dataset email has a stored result.")

    if coverage.evaluated_count == 0:
        print("\nNothing evaluated yet — no metrics to report.")
        return

    print()
    print(
        f"Classification accuracy: {report.classification.accuracy * 100:.1f}% "
        f"({report.classification.correct}/{report.classification.total})"
    )
    print(
        f"Decision accuracy:       {report.decision.accuracy * 100:.1f}% "
        f"({report.decision.correct}/{report.decision.total})"
    )

    print("\nSafety-critical metrics:")
    print(f"  NOTIFY recall     (classification): {format_recall(report.classification.per_category[Category.NOTIFY])}")
    print(f"  NOTIFY recall     (decision):        {format_recall(report.decision.per_category[Category.NOTIFY])}")
    print(f"  QUARANTINE recall (classification): {format_recall(report.classification.per_category[Category.QUARANTINE])}")
    print(f"  QUARANTINE recall (decision):        {format_recall(report.decision.per_category[Category.QUARANTINE])}")

    print("\nPer-category classification metrics:")
    for category, m in report.classification.per_category.items():
        print(f"  {category.value:12s} precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} support={m.support}")

    print("\nPer-category decision metrics:")
    for category, m in report.decision.per_category.items():
        print(f"  {category.value:12s} precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} support={m.support}")

    if report.classification_mismatches:
        print(f"\nClassification mismatches ({len(report.classification_mismatches)}):")
        for m in report.classification_mismatches:
            print(f"  {m.message_id}: expected={m.expected.value} actual={m.actual.value}  ({m.notes})")

    if report.decision_mismatches:
        print(f"\nDecision mismatches ({len(report.decision_mismatches)}):")
        for m in report.decision_mismatches:
            print(f"  {m.message_id}: expected={m.expected.value} actual={m.actual.value}  ({m.notes})")


if __name__ == "__main__":
    main()
