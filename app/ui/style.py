"""Visual primitives for the Streamlit presentation layer."""

from __future__ import annotations

import streamlit as st


def apply_design_system() -> None:
    """Install responsive product tokens without changing application logic."""
    st.markdown(
        """
        <style>
        :root { --cia-ink:#10243f; --cia-ink-2:#1a365d; --cia-canvas:#f5f6f8; --cia-surface:#fff; --cia-line:#dbe3ee; --cia-muted:#65738a; --cia-blue:#316fe6; --cia-teal:#168c92; --cia-coral:#e75d63; --cia-slate:#7a8798; --cia-amber:#d88926; --cia-shadow:0 12px 30px rgba(16,36,63,.09); }
        .stApp { background:var(--cia-canvas); color:var(--cia-ink); }
        .block-container { max-width:1320px; padding:1.4rem 2rem 3rem; }
        [data-testid="stHeader"] { background:rgba(245,246,248,.92); }
        [data-testid="stSidebar"] { background:#10243f; }
        [data-testid="stSidebar"] * { color:#edf3fb; }
        [data-testid="stSidebar"] [data-testid="stAlert"] { background:rgba(255,255,255,.09); border-color:rgba(255,255,255,.18); }
        .cia-eyebrow { color:var(--cia-blue); font-size:.72rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
        .cia-hero { display:flex; justify-content:space-between; align-items:center; gap:1.5rem; background:var(--cia-ink); border:1px solid #1d3a60; border-radius:18px; padding:1.25rem 1.5rem; margin:0 0 1.15rem; box-shadow:var(--cia-shadow); }
        .cia-hero h1 { color:#fff!important; font-size:1.72rem!important; letter-spacing:-.035em; margin:.2rem 0!important; }
        .cia-hero p { color:#bfcee2; font-size:.94rem; margin:0; max-width:670px; }
        .cia-hero-mark { display:grid; place-items:center; height:44px; width:44px; flex:0 0 44px; background:rgba(79,136,244,.18); border:1px solid rgba(141,180,255,.35); border-radius:13px; color:#dce9ff; font-size:1.25rem; }
        .cia-section-kicker { color:var(--cia-blue); font-size:.71rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-top:.35rem; }
        .cia-metric-card { height:100%; min-height:118px; position:relative; overflow:hidden; background:var(--cia-surface); border:1px solid var(--cia-line); border-radius:15px; padding:1rem 1.05rem; box-shadow:0 3px 12px rgba(16,36,63,.035); }
        .cia-metric-card:before { content:""; position:absolute; inset:0 auto 0 0; width:4px; background:var(--cia-blue); }.cia-metric-card--notify:before { background:var(--cia-coral); }.cia-metric-card--digest:before { background:var(--cia-blue); }.cia-metric-card--mute:before { background:var(--cia-slate); }.cia-metric-card--quarantine:before { background:var(--cia-amber); }.cia-metric-card--teal:before { background:var(--cia-teal); }
        .cia-card-top { display:flex; justify-content:space-between; gap:.75rem; align-items:start; }.cia-card-icon { color:var(--cia-muted); font-size:1rem; }.cia-card-label { color:var(--cia-muted); font-size:.72rem; font-weight:800; letter-spacing:.065em; text-transform:uppercase; }.cia-card-value { color:var(--cia-ink); font-size:1.72rem; font-weight:800; line-height:1.2; margin:.45rem 0 .35rem; }.cia-card-note { color:var(--cia-muted); font-size:.8rem; }
        /* Responsive segmented navigation: wraps on tablets and scrolls, never clips, on phones. */
        [data-testid="stTabs"] [role="tablist"] { display:flex; flex-wrap:wrap; gap:.55rem; padding:.55rem; margin:0 0 1.25rem; background:var(--cia-surface); border:1px solid var(--cia-line); border-radius:16px; box-shadow:0 3px 12px rgba(16,36,63,.035); }
        [data-testid="stTabs"] [role="tab"] { flex:1 1 auto; min-width:128px; height:44px; padding:0 1rem; border:1px solid transparent; border-radius:10px; color:var(--cia-muted); font-size:.86rem; font-weight:750; letter-spacing:-.01em; transition:background .16s ease,border-color .16s ease,color .16s ease,transform .16s ease; }
        [data-testid="stTabs"] [role="tab"]:hover { background:#edf3ff; border-color:#c8dafb; color:var(--cia-ink-2); transform:translateY(-1px); }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] { background:var(--cia-ink); border-color:var(--cia-ink); color:#fff; box-shadow:0 5px 12px rgba(16,36,63,.18); }
        [data-testid="stTabs"] [role="tab"]::after { display:none!important; }
        .stButton > button { min-height:40px; border-radius:10px; border:1px solid #255dc6; background:#316fe6; color:#fff; font-weight:750; transition:transform .16s ease,background .16s ease,box-shadow .16s ease; }.stButton > button:hover { background:#235cc7; border-color:#235cc7; box-shadow:0 6px 14px rgba(49,111,230,.24); transform:translateY(-1px); color:#fff; }
        [data-testid="stMetric"] { background:var(--cia-surface); border:1px solid var(--cia-line); border-radius:13px; padding:.85rem; }.stAlert { border-radius:12px; } [data-testid="stDataFrame"] { border:1px solid var(--cia-line); border-radius:13px; overflow:hidden; }
        .cia-empty { background:var(--cia-surface); border:1px dashed #b7c7dc; border-radius:16px; padding:1.35rem; text-align:left; }.cia-empty strong { color:var(--cia-ink); font-size:1.05rem; }.cia-empty p { color:var(--cia-muted); margin:.35rem 0 0; }
        .cia-distribution-row { display:flex; align-items:center; gap:.6rem; margin:.55rem 0; }.cia-distribution-label { width:104px; color:var(--cia-muted); font-size:.8rem; font-weight:700; }.cia-distribution-track { height:8px; flex:1; overflow:hidden; background:#e8edf4; border-radius:999px; }.cia-distribution-fill { height:100%; border-radius:999px; background:var(--cia-blue); }.cia-distribution-count { width:24px; color:var(--cia-ink); font-size:.8rem; font-weight:800; text-align:right; }
        .cia-list-item { background:var(--cia-surface); border:1px solid var(--cia-line); border-radius:12px; padding:.8rem; margin:.5rem 0; }.cia-list-item:hover { border-color:#a9c3ef; box-shadow:0 4px 10px rgba(16,36,63,.06); }
        @media (max-width:760px) { .block-container { padding:1rem .85rem 2rem; }.cia-hero { padding:1rem; align-items:flex-start; }.cia-hero h1 { font-size:1.35rem!important; }.cia-hero-mark { height:38px; width:38px; flex-basis:38px; }.cia-metric-card { min-height:104px; } [data-testid="stTabs"] [role="tablist"] { flex-wrap:nowrap; overflow-x:auto; scrollbar-width:thin; } [data-testid="stTabs"] [role="tab"] { flex:0 0 auto; min-width:max-content; padding:0 .85rem; font-size:.8rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, description: str, eyebrow: str = "Communication intelligence") -> None:
    st.markdown(f'<section class="cia-hero"><div><div class="cia-eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div><div class="cia-hero-mark" aria-hidden="true">✦</div></section>', unsafe_allow_html=True)


def card(label: str, value: str | int, note: str = "", tone: str = "", icon: str = "") -> None:
    """Render a metric card from trusted labels and computed values."""
    st.markdown(f'<article class="cia-metric-card cia-metric-card--{tone}"><div class="cia-card-top"><div class="cia-card-label">{label}</div><div class="cia-card-icon">{icon}</div></div><div class="cia-card-value">{value}</div><div class="cia-card-note">{note}</div></article>', unsafe_allow_html=True)


def empty_state(title: str, message: str) -> None:
    st.markdown(f'<div class="cia-empty"><strong>{title}</strong><p>{message}</p></div>', unsafe_allow_html=True)
