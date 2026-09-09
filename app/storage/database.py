"""SQLite schema and connection management (implemented in M6).

Stores metadata and derived fields only (message_id, sender, subject,
timestamp, summary, category, confidence, reasoning, risk_flags, decision) —
never raw email body content. Gmail remains the source of truth.
"""
