from __future__ import annotations

from pathlib import Path
from textwrap import dedent
import html

import pandas as pd
import streamlit as st


def render_html(content: str) -> None:
    """Render dedented HTML so Markdown does not interpret it as a code block."""
    st.markdown(dedent(content).strip(), unsafe_allow_html=True)


def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "styles.css"

    with css_path.open("r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True,
        )


def valid_image_url(value: object) -> str:
    """Return a usable HTTP(S) image URL, otherwise an empty string."""
    url = str(value or "").strip()

    if url.lower() in {"", "nan", "none", "<na>"}:
        return ""

    if not url.startswith(("http://", "https://")):
        return ""

    return url


def player_initials(name: str) -> str:
    """Create compact initials for the image fallback."""
    parts = [
        part
        for part in html.unescape(name).replace(".", " ").split()
        if part
    ]

    initials = "".join(part[0].upper() for part in parts[:2])
    return initials or "⚽"


def hero() -> None:
    render_html(
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
        """
    )


def metrics_grid(metrics: list[tuple[str, str, str]]) -> None:
    cards = []

    for label, value, note in metrics:
        cards.append(
            dedent(
                f"""
                <div class="ewc-metric">
                  <div class="label">{html.escape(str(label))}</div>
                  <div class="value">{html.escape(str(value))}</div>
                  <div class="label">{html.escape(str(note))}</div>
                </div>
                """
            ).strip()
        )

    render_html(
        f"""
        <div class="ewc-metric-grid">
          {''.join(cards)}
        </div>
        """
    )


def player_cards(df: pd.DataFrame, max_cards: int = 8) -> None:
    if df.empty:
        st.info("No matching players found.")
        return

    cards: list[str] = []

    for _, row in df.head(max_cards).iterrows():
        flag = html.escape(str(row.get("flag", "") or ""))

        raw_name = str(
            row.get("short_name", row.get("player_name", "Unknown"))
            or "Unknown"
        )
        name = html.escape(raw_name)

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

        archetype_name = str(row.get("archetype_name", "") or "").strip()
        archetype_id = str(row.get("archetype_id", "") or "").strip()

        if archetype_name:
            archetype_label = html.escape(archetype_name)
        elif archetype_id:
            archetype_label = html.escape(f"Cluster {archetype_id}")
        else:
            archetype_label = "Unclassified profile"

        why = html.escape(
            str(row.get("why_similar", "") or "")
        )

        differences = html.escape(
            str(row.get("main_differences", "") or "")
        )

        image_url = valid_image_url(row.get("image_url"))
        initials = html.escape(player_initials(raw_name))

        if image_url:
            safe_image_url = html.escape(image_url, quote=True)

            image_html = dedent(
                f"""
                <div class="ewc-player-image-wrap">
                  <img
                    class="ewc-player-image"
                    src="{safe_image_url}"
                    alt="{name} player portrait"
                    loading="lazy"
                    onerror="
                      this.style.display='none';
                      this.nextElementSibling.style.display='flex';
                    "
                  >
                  <div
                    class="ewc-player-image-placeholder"
                    style="display:none;"
                  >
                    {initials}
                  </div>
                </div>
                """
            ).strip()
        else:
            image_html = dedent(
                f"""
                <div class="ewc-player-image-wrap">
                  <div class="ewc-player-image-placeholder">
                    {initials}
                  </div>
                </div>
                """
            ).strip()

        similarity = pd.to_numeric(
            pd.Series([row.get("similarity")]),
            errors="coerce",
        ).iloc[0]

        score = (
            f"{float(similarity) * 100:.1f}%"
            if pd.notna(similarity)
            else ""
        )

        explanation_parts: list[str] = []

        if why:
            explanation_parts.append(
                f'<div class="ewc-player-why">{why}</div>'
            )

        if differences:
            explanation_parts.append(
                f'<div class="ewc-player-differences">{differences}</div>'
            )

        explanation_html = "".join(explanation_parts)

        cards.append(
            dedent(
                f"""
                <article class="ewc-player-card">
                  <div class="ewc-player-card-layout">
                    {image_html}

                    <div class="ewc-player-content">
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
                        <span class="ewc-pill">{archetype_label}</span>
                      </div>

                      {explanation_html}
                    </div>
                  </div>
                </article>
                """
            ).strip()
        )

    render_html(
        f"""
        <div class="ewc-card-grid">
          {''.join(cards)}
        </div>
        """
    )