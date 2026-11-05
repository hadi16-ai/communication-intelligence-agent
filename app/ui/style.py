"""Shared, scoped visual system for the Streamlit presentation layer."""

from __future__ import annotations

import streamlit as st


def apply_design_system() -> None:
    """Install the app's visual tokens without changing application logic."""
    st.markdown(
        """
        <style>
        :root { --ink:#10233f; --muted:#60708a; --surface:#ffffff; --canvas:#f6f8fc;
                --line:#dce3ef; --coral:#e85d5d; --blue:#3577d4; --grey:#7c8797;
                --amber:#d98b20; --shadow:0 10px 28px rgba(16,35,63,.08); }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(246,248,252,.92); }
        .block-container { max-width: 1280px; padding-top: 2.2rem; }
        .cia-eyebrow { color:var(--blue); font-size:.73rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
        .cia-hero { background:var(--ink); color:white; border-radius:20px; padding:2rem 2.2rem; margin:.2rem 0 1.4rem; box-shadow:var(--shadow); }
        .cia-hero h1 { color:white !important; font-size:2.2rem !important; margin:0 0 .35rem; }
        .cia-hero p { color:#c7d3e4; margin:0; max-width:720px; }
        .cia-card { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:1rem 1.15rem; min-height:104px; box-shadow:0 3px 12px rgba(16,35,63,.035); }
        .cia-card-label { color:var(--muted); font-size:.76rem; font-weight:750; letter-spacing:.05em; text-transform:uppercase; }
        .cia-card-value { color:var(--ink); font-size:1.7rem; font-weight:800; line-height:1.35; }
        .cia-card-note { color:var(--muted); font-size:.82rem; }
        .cia-card--notify { border-top:3px solid var(--coral); }
        .cia-card--digest { border-top:3px solid var(--blue); }
        .cia-card--mute { border-top:3px solid var(--grey); }
        .cia-card--quarantine { border-top:3px solid var(--amber); }
        .cia-section-kicker { color:var(--blue); font-size:.73rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin-top:.5rem; }
        .stButton > button { border-radius:9px; border:1px solid #1d5fb8; background:#1d5fb8; color:white; font-weight:700; }
        .stButton > button:hover { border-color:#154987; background:#154987; color:white; }
        [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:.8rem; }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stTabs"] [role="tablist"] { gap:.35rem; border-bottom:1px solid var(--line); }
        [data-testid="stTabs"] [role="tab"] { height:42px; border-radius:9px 9px 0 0; color:var(--muted); font-weight:700; }
        [data-testid="stTabs"] [aria-selected="true"] { color:var(--ink); background:#eaf1fc; }
        .stAlert { border-radius:12px; }
        @media (max-width: 700px) { .block-container { padding:1.1rem .85rem; } .cia-hero { padding:1.45rem; } .cia-hero h1 { font-size:1.75rem !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, description: str, eyebrow: str = "Communication intelligence") -> None:
    st.markdown(
        f'<div class="cia-hero"><div class="cia-eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def card(label: str, value: str | int, note: str = "", tone: str = "") -> None:
    note_markup = f'<div class="cia-card-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="cia-card cia-card--{tone}"><div class="cia-card-label">{label}</div><div class="cia-card-value">{value}</div>{note_markup}</div>',
        unsafe_allow_html=True,
    )
