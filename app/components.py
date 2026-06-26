from __future__ import annotations

import html
import pandas as pd
import streamlit as st


def safe(x) -> str:
    if pd.isna(x):
        return ""
    return html.escape(str(x))


def hero() -> None:
    st.markdown("""
    <div class="ewc-hero">
        <h1>The Eternal World Cup</h1>
        <p>Football refuses to say goodbye to its greats. This app uses neural-network player embeddings — a form of football DNA — to explore which modern players resemble past stars, which profiles cluster together, and who might become the players we are still talking about decades from now.</p>
        <div>
            <span class="pill">Neural embeddings</span>
            <span class="pill">Player similarity</span>
            <span class="pill">World Cup storytelling</span>
            <span class="pill">Talent ID lens</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_grid(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f"<div class='metric-card'><div class='metric-label'>{safe(label)}</div><div class='metric-value'>{safe(value)}</div><div class='metric-note'>{safe(note)}</div></div>"
        for label, value, note in items
    )
    st.markdown(f"<div class='metric-grid'>{cards}</div>", unsafe_allow_html=True)


def player_card(rank: int, name: str, meta: str, score: float | str, flag: str = "") -> None:
    st.markdown(f"""
    <div class="player-card">
        <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start;">
            <div>
                <div class="player-title">{rank}. {safe(flag)} {safe(name)}</div>
                <div class="player-meta">{safe(meta)}</div>
            </div>
            <div class="score-chip">{safe(score)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
