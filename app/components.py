from pathlib import Path

import html
import pandas as pd
import streamlit as st


from pathlib import Path
import streamlit as st

def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "styles.css"
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <section class="ewc-hero">
          <h1>The Eternal World Cup</h1>
          <p>Football refuses to say goodbye to its greats. This app uses football-DNA player embeddings to explore which modern players resemble past stars, which profiles cluster together, and who might become the players we are still talking about decades from now.</p>
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
    st.markdown(f"<div class='ewc-metric-grid'>{cards}</div>", unsafe_allow_html=True)


def player_cards(df: pd.DataFrame, max_cards: int = 8) -> None:
    if df.empty:
        st.info("No matching players found.")
        return
    cards = []
    for _, r in df.head(max_cards).iterrows():
        flag = str(r.get("flag", "") or "")
        name = html.escape(str(r.get("short_name", r.get("player_name", "Unknown"))))
        season = html.escape(str(r.get("season_label", r.get("fifa_version", ""))))
        club = html.escape(str(r.get("club_name", r.get("club", "")) or ""))
        nation = html.escape(str(r.get("nationality_name", r.get("country", "")) or ""))
        pos = html.escape(str(r.get("player_positions", r.get("position", "")) or ""))
        overall = r.get("overall", r.get("ovr", ""))
        sim = r.get("similarity", None)
        score = f"{sim*100:.1f}%" if pd.notna(sim) else ""
        cards.append(
            f"""
            <article class="ewc-player-card">
              <div class="ewc-player-top">
                <div>
                  <div class="ewc-player-name">{flag} {name}</div>
                  <div class="ewc-player-meta">{season} · {club}</div>
                  <div class="ewc-player-meta">{nation} · {pos}</div>
                </div>
                <div class="ewc-score">{score}</div>
              </div>
              <span class="ewc-pill">Overall {overall}</span>
              <span class="ewc-pill">Age {html.escape(str(r.get('age', '')))}</span>
              <span class="ewc-pill">Cluster {html.escape(str(r.get('archetype_id', '')))}</span>
            </article>
            """
        )
    st.markdown(f"<div class='ewc-card-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
