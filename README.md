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
| M1 | Core config and data models | pending |
| M2 | Structured preference system | pending |
| M3 | Synthetic email fixtures / test data | pending |
| M4 | Gemini classification with structured output | pending |
| M5 | Personalized deterministic decision engine | pending |
| M6 | SQLite storage and caching | pending |
| M7 | End-to-end synthetic pipeline | pending |
| M8 | Streamlit dashboard | pending |
| M9 | Evaluation dataset and metrics | pending |
| M10 | Gmail read-only integration | pending |

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
    streamlit_app.py       # dashboard entry point (M8)
    core/
        config.py          # settings, incl. GEMINI_MODEL (M1)
        models.py          # Category enum, EmailMessage, ClassificationResult, Decision (M1)
        decisions.py        # deterministic preference-override logic (M5)
        pipeline.py          # READ -> CLASSIFY -> DECIDE -> STORE orchestration (M7)
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
        repositories.py    # save/query, message_id dedupe (M6)
    evaluation/
        dataset.py         # synthetic evaluation dataset loader (M9)
        metrics.py          # accuracy / confusion matrix (M9)
    utils/
        logging.py         # structured, privacy-safe logging (M1+)

scripts/
    authorize_gmail.py     # one-time OAuth consent flow (M10)

tests/                     # mirrors app/
data/
    sample_emails/         # synthetic fixtures (M3)
    eval_dataset.json      # evaluation set, 30-50+ examples (M3/M9)
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

## Setup

Setup instructions (virtual environment, dependencies, Gemini API key, Gmail
OAuth) will be filled in as the corresponding milestones land. For now:

```bash
python -m venv venv
```

## License

TBD.
