from __future__ import annotations

from pathlib import Path
import html

import pandas as pd
import streamlit as st


def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "styles.css"

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


def hero() -> None:
    st.markdown(
        """
        <section class="ewc-hero">
          <h1>The Eternal World Cup</h1>
          <p>
            Football refuses to say goodbye to its greats. This app uses
            football-DNA player embeddings to explore which modern players
            resemble past stars, which profiles cluster together, and who
            might become the players we are still talking about decades from now.
          </p>
          <div class="ewc-badges">
            <span class="ewc-badge">Neural-style embeddings</span>
            <span class="ewc-badge">Player similarity</span>
            <span class="ewc-badge">World Cup storytelling</span>
            <span class="ewc-badge">Talent ID lens</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metrics_grid(metrics: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f"""
        <div class="ewc-metric">
          <div class="label">{html.escape(label)}</div>
          <div class="value">{html.escape(value)}</div>
          <div class="label">{html.escape(note)}</div>
        </div>
        """
        for label, value, note in metrics
    )

    st.markdown(
        f"<div class='ewc-metric-grid'>{cards}</div>",
        unsafe_allow_html=True,
    )


def player_cards(df: pd.DataFrame, max_cards: int = 8) -> None:
    if df.empty:
        st.info("No matching players found.")
        return

    cards: list[str] = []

    for _, row in df.head(max_cards).iterrows():
        flag = html.escape(str(row.get("flag", "") or ""))
        name = html.escape(
            str(row.get("short_name", row.get("player_name", "Unknown")))
        )
        season = html.escape(
            str(row.get("season_label", row.get("fifa_version", "")) or "")
        )
        club = html.escape(
            str(row.get("club_name", row.get("club", "")) or "")
        )
        nation = html.escape(
            str(row.get("nationality_name", row.get("country", "")) or "")
        )
        position = html.escape(
            str(row.get("player_positions", row.get("position", "")) or "")
        )
        overall = html.escape(
            str(row.get("overall", row.get("ovr", "")) or "")
        )
        age = html.escape(str(row.get("age", "") or ""))
        cluster = html.escape(str(row.get("archetype_id", "") or ""))

        why = html.escape(
            str(row.get("why_similar", "") or "")
        )
        differences = html.escape(
            str(row.get("main_differences", "") or "")
        )

        similarity = row.get("similarity", None)
        score = (
            f"{float(similarity) * 100:.1f}%"
            if pd.notna(similarity)
            else ""
        )

        explanation_html = ""

        if why:
            explanation_html += (
                f'<div class="ewc-player-why">{why}</div>'
            )

        if differences:
            explanation_html += (
                f'<div class="ewc-player-differences">{differences}</div>'
            )

        cards.append(
            f"""
            <article class="ewc-player-card">
              <div class="ewc-player-top">
                <div>
                  <div class="ewc-player-name">{flag} {name}</div>
                  <div class="ewc-player-meta">{season} · {club}</div>
                  <div class="ewc-player-meta">{nation} · {position}</div>
                </div>
                <div class="ewc-score">{score}</div>
              </div>

              <div class="ewc-pill-row">
                <span class="ewc-pill">Overall {overall}</span>
                <span class="ewc-pill">Age {age}</span>
                <span class="ewc-pill">Cluster {cluster}</span>
              </div>

              {explanation_html}
            </article>
            """
        )

    st.markdown(
        f"<div class='ewc-card-grid'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )