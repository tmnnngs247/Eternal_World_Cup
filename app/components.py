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


def clean_text(value: object) -> str:
    """Return clean display text, removing common missing-value strings."""
    text = str(value or "").strip()

    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""

    return text


def valid_image_url(value: object) -> str:
    """Return a usable HTTP(S) image URL, otherwise an empty string."""
    url = clean_text(value)

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
            (
                '<div class="ewc-metric">'
                f'<div class="label">{html.escape(str(label))}</div>'
                f'<div class="value">{html.escape(str(value))}</div>'
                f'<div class="label">{html.escape(str(note))}</div>'
                "</div>"
            )
        )

    grid_html = (
        '<div class="ewc-metric-grid">'
        + "".join(cards)
        + "</div>"
    )

    st.markdown(
        grid_html,
        unsafe_allow_html=True,
    )


def player_cards(df: pd.DataFrame, max_cards: int = 8) -> None:
    if df.empty:
        st.info("No matching players found.")
        return

    cards: list[str] = []

    for _, row in df.head(max_cards).iterrows():
        raw_name = (
            clean_text(
                row.get(
                    "short_name",
                    row.get("player_name", "Unknown"),
                )
            )
            or "Unknown"
        )

        name = html.escape(raw_name)
        flag = html.escape(clean_text(row.get("flag")))

        season = html.escape(
            clean_text(
                row.get(
                    "season_label",
                    row.get("fifa_version"),
                )
            )
        )

        club = html.escape(
            clean_text(
                row.get(
                    "club_name",
                    row.get("club"),
                )
            )
        )

        nation = html.escape(
            clean_text(
                row.get(
                    "nationality_name",
                    row.get("country"),
                )
            )
        )

        position = html.escape(
            clean_text(
                row.get(
                    "player_positions",
                    row.get("position"),
                )
            )
        )

        overall = html.escape(
            clean_text(
                row.get(
                    "overall",
                    row.get("ovr"),
                )
            )
        )

        age = html.escape(clean_text(row.get("age")))

        archetype_name = clean_text(row.get("archetype_name"))
        archetype_id = clean_text(row.get("archetype_id"))

        if archetype_name:
            archetype_label = html.escape(archetype_name)
        elif archetype_id:
            archetype_label = html.escape(f"Cluster {archetype_id}")
        else:
            archetype_label = "Unclassified profile"

        why = html.escape(
            clean_text(row.get("why_similar"))
        )

        differences = html.escape(
            clean_text(row.get("main_differences"))
        )

        image_url = valid_image_url(row.get("image_url"))
        initials = html.escape(player_initials(raw_name))

        if image_url:
            safe_image_url = html.escape(
                image_url,
                quote=True,
            )

            image_html = (
                '<div class="ewc-player-image-wrap">'
                f'<img class="ewc-player-image" '
                f'src="{safe_image_url}" '
                f'alt="{name} player portrait" '
                'loading="lazy">'
                "</div>"
            )
        else:
            image_html = (
                '<div class="ewc-player-image-wrap">'
                '<div class="ewc-player-image-placeholder">'
                f"{initials}"
                "</div>"
                "</div>"
            )

        similarity = pd.to_numeric(
            pd.Series([row.get("similarity")]),
            errors="coerce",
        ).iloc[0]

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
                '<div class="ewc-player-differences">'
                f"{differences}"
                "</div>"
            )

        card_html = (
            '<article class="ewc-player-card">'
            '<div class="ewc-player-card-layout">'
            f"{image_html}"
            '<div class="ewc-player-content">'
            '<div class="ewc-player-top">'
            "<div>"
            f'<div class="ewc-player-name">{flag} {name}</div>'
            f'<div class="ewc-player-meta">{season} · {club}</div>'
            f'<div class="ewc-player-meta">{nation} · {position}</div>'
            "</div>"
            f'<div class="ewc-score">{score}</div>'
            "</div>"
            '<div class="ewc-pill-row">'
            f'<span class="ewc-pill">Overall {overall}</span>'
            f'<span class="ewc-pill">Age {age}</span>'
            f'<span class="ewc-pill">{archetype_label}</span>'
            "</div>"
            f"{explanation_html}"
            "</div>"
            "</div>"
            "</article>"
        )

        cards.append(card_html)

    grid_html = (
        '<div class="ewc-card-grid">'
        + "".join(cards)
        + "</div>"
    )

    st.markdown(
        grid_html,
        unsafe_allow_html=True,
    )