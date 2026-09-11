# Communication Intelligence Agent

A personalized AI communication assistant that reads a Gmail inbox and decides,
per message, what should happen to it for *this* user — not just what the
message is about.

## The four decisions

- 🔴 **NOTIFY** — important enough to interrupt the user now
- 🟡 **DIGEST** — useful but not urgent; included in a later digest
- ⚪ **MUTE** — low-value, repetitive, promotional, or irrelevant
- 🛡️ **QUARANTINE** — suspicious, phishing, scam, or potentially unsafe

`CLASSIFICATION` (what a message is about) is kept separate from `DECISION`
(what should happen to it for this user). MUTE never means delete, and
QUARANTINE is kept distinct from MUTE because low-value is not the same
thing as dangerous.

## Status

**Version 1 (Intelligence Core) — in progress.**

Build order: synthetic emails → understand → classify → personalize →
decide → store → display → evaluate → *then* connect Gmail. See the
milestone table below.

| # | Milestone | Status |
|---|---|---|
| M0 | Repo bootstrap | ✅ |
| M1 | Core config and data models | ✅ |
| M2 | Structured preference system | ✅ |
| M3 | Synthetic email fixtures / test data | ✅ |
| M4 | Gemini classification with structured output | ✅ |
| M5 | Personalized deterministic decision engine | ✅ |
| M6 | SQLite storage and caching | ✅ |
| M7 | End-to-end synthetic pipeline | ✅ |
| M8 | Streamlit dashboard | ✅ |
| M9 | Evaluation and quality metrics | ✅ (architecture complete; **data partial**, see below) |
| M10 | Gmail read-only integration | pending |

**A note on real Gemini usage:** a live 41-email evaluation run against the
real Gemini API was started and hit a free-tier rate limit (5 requests/min)
after 6 successful calls, before billing was fully configured. Those 6 real
classifications remain cached in `data/app.db` (never overwritten or
deleted). The evaluation architecture built in M9 was deliberately designed
to report this honestly as a **partial** evaluation (6 of 41) rather than
hide or extrapolate from it — see "Evaluation" below. The remaining 35
emails will be processed once billing is confirmed working; no code changes
will be needed to do that.

Version 1 is strictly read-only with respect to Gmail: it only **READ**s,
**CLASSIFY**s, and **DECIDE**s — it never acts on the mailbox (no send, no
label, no delete). Versions 2 and 3 are scoped separately and are not part
of this build.

## Architecture

```
Gmail (read-only, M10) / synthetic fixtures (M3)
    → app/gmail or fixtures        (READ)
    → app/ai (Gemini, structured)  (CLASSIFY)
    → app/core/decisions           (DECIDE — deterministic, preference-driven)
    → app/storage (SQLite)         (STORE — metadata + derived fields only)
    → app/streamlit_app            (DISPLAY)

data/eval_dataset.json (ground truth) + app/storage (stored results)
    → app/evaluation                (EVALUATE — classification vs. decision, never calls Gemini)
    → app/streamlit_app             (DISPLAY — separate "Evaluation" tab)
```

The AI layer sits behind a small interface (`app/ai/classifier.py`) so the
LLM provider can change without touching the rest of the application. The
default model is configured, not hard-coded:

```
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"   # overridable via GEMINI_MODEL
```

## Project structure

```
app/
    streamlit_app.py       # dashboard entry point: Inbox + Evaluation tabs (M8/M9)
    core/
        config.py          # settings, incl. GEMINI_MODEL (M1)
        models.py          # Category enum, EmailMessage, ClassificationResult, Decision (M1)
        decisions.py        # deterministic preference-override logic (M5)
        pipeline.py          # cache-lookup -> CLASSIFY -> DECIDE -> STORE orchestration (M7)
    gmail/
        client.py          # read-only Gmail API client (M10)
        parser.py          # raw Gmail message -> EmailMessage (M10)
    ai/
        classifier.py      # BaseClassifier interface + factory (M4)
        gemini.py           # Gemini implementation (M4)
        prompts.py          # prompt templates (M4)
    preferences/
        manager.py         # structured preference loading + matching (M2)
    storage/
        database.py        # SQLite schema (M6)
        repositories.py    # save/query, message_id dedupe, list_records (M6/M9)
    evaluation/
        dataset.py         # typed ground-truth loader for eval_dataset.json (M9)
        metrics.py          # confusion matrix, precision/recall/F1 (M9)
        evaluator.py         # combines ground truth + stored results into an EvaluationReport (M9)
    ui/
        dashboard_data.py   # pure Inbox-tab data shaping, no streamlit import (M8)
        evaluation_view.py   # pure Evaluation-tab data shaping, no streamlit import (M9)
    utils/
        logging.py         # structured, privacy-safe logging (M1+)

scripts/
    authorize_gmail.py         # one-time OAuth consent flow (M10)
    run_synthetic_pipeline.py  # manual full-dataset run against the REAL Gemini API (M7)
    run_evaluation.py           # prints an evaluation report from the current DB; zero API calls (M9)

tests/                     # mirrors app/
data/
    sample_emails/         # synthetic fixtures (M3)
    eval_dataset.json      # evaluation set, 41 examples (M3/M9)
    preferences.json       # structured user preferences (M2)
```

## Privacy and security posture

- Version 1 uses the `gmail.readonly` OAuth scope only — no send/modify/delete
  capability exists in the codebase.
- No secrets are hard-coded. API keys, OAuth client secrets, and tokens are
  read from environment variables / local files listed in `.gitignore` and
  are never committed.
- Local storage (SQLite) holds metadata and derived fields only — message_id,
  sender, subject, timestamp, summary, category, confidence, reasoning,
  risk_flags, decision. Raw email bodies are never stored locally; Gmail
  remains the source of truth for the original message.
- Logs contain decision metadata only, never full email content.

## Evaluation (M9)

The evaluation layer (`app/evaluation/`) measures how well the system is
doing, using only data that already exists — it never calls Gemini or any
external API.

**Classification vs. decision are evaluated separately, on purpose.**
`ClassificationResult.category` (what the AI thinks an email is) and
`Decision.category` (what the personalized system finally does about it)
are different questions with different failure causes, so they get
independent accuracy, precision/recall/F1, and confusion matrices:

- **Classification metrics** compare each email's stored
  `ClassificationResult.category` against `expected_category` from
  `data/eval_dataset.json`.
- **Decision metrics** compare each email's stored `Decision.category`
  against an **expected decision**, which is derived deterministically —
  never from Gemini — by running the real, unmodified `DecisionEngine` and
  the real `data/preferences.json` against a ground-truth stand-in
  classification built from `expected_category` (see
  `derive_expected_decision` in `app/evaluation/evaluator.py`). This is the
  same "what would the existing policy do with a perfect classification?"
  convention already used internally by the M5–M7 dataset-compatibility
  tests, just made reusable and typed for M9.

**Safety-critical metrics** — surfaced prominently, both in the CLI report
and in the dashboard's Evaluation tab — are **NOTIFY recall** and
**QUARANTINE recall** (each shown at both the classification and decision
level). The reasoning, unchanged from the project's original design intent:
missing a genuinely important job/interview/security email is more costly
than one extra notification, and failing to quarantine dangerous content is
a distinct, equally serious failure mode. When a category has zero
evaluated examples so far, its recall is shown as `N/A (0 examples)` rather
than a bare `0.0%`, which would misleadingly read as "the system failed
every case."

**Partial evaluation is a first-class, honestly-reported state.** The
evaluator only scores emails that already have a stored classification in
SQLite — it never calls Gemini to fill gaps and never fabricates a result
for a missing one. It always reports `evaluated_count`, `total_count`,
`missing_count`, and `coverage_percent` alongside the metrics, and the UI
shows a clear "Partial evaluation — N of 41 emails evaluated" warning
banner (with the specific missing `message_id`s available in an expander)
whenever coverage is incomplete — metrics from a partial run are never
presented as if they represent the full 41-email dataset.

**Running the evaluator locally:**

```bash
python scripts/run_evaluation.py
```

Prints coverage, overall and per-category classification/decision metrics,
both confusion matrices, the safety-critical recall figures, and any
mismatched records — reading only the existing SQLite database
(`Settings.database_path`, default `data/app.db`). Makes zero network
calls, so it's safe to run at any time regardless of Gemini/billing status.

The same report also renders as a "📊 Evaluation" tab in the Streamlit
dashboard (`streamlit run app/streamlit_app.py`), alongside the existing
"📥 Inbox" tab.

**Once Gemini access is restored:** running the existing M7 pipeline
(`scripts/run_synthetic_pipeline.py`, or `EmailProcessor` directly) against
the remaining 35 emails will populate the rest of `data/app.db` — the
already-cached 6 are never re-classified, since caching is keyed by
`message_id` (M6). No change to the evaluation architecture is needed: the
same `python scripts/run_evaluation.py` command, and the same dashboard
tab, will automatically report full 41/41 coverage once those rows exist.

## Setup

Setup instructions (virtual environment, dependencies, Gemini API key, Gmail
OAuth) will be filled in as the corresponding milestones land. For now:

```bash
python -m venv venv
```

## License

TBD.
