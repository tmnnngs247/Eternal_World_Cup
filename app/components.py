from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from textwrap import dedent
import base64
import html
import re

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PLAYER_FACE_DIR = ROOT / "assets" / "player_faces"


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


_SOFIFA_ID_PATTERN = re.compile(r"/players/(\d+)/(\d+)/")


def sofifa_id_from_image_url(value: object) -> str:
    """Recover the numeric SoFIFA id embedded in a cdn.sofifa.net image URL."""
    match = _SOFIFA_ID_PATTERN.search(clean_text(value))
    return f"{match.group(1)}{match.group(2)}" if match else ""


def player_initials(name: str) -> str:
    """Create compact initials for the image fallback."""
    parts = [
        part
        for part in html.unescape(name).replace(".", " ").split()
        if part
    ]

    initials = "".join(part[0].upper() for part in parts[:2])
    return initials or "⚽"


def get_local_image_path(value: object) -> str:
    """Return the downloaded portrait path for a SoFIFA ID if it exists."""
    sofifa_id = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(sofifa_id):
        return ""

    image_path = PLAYER_FACE_DIR / f"{int(sofifa_id)}.png"

    if not image_path.is_file():
        return ""

    return str(image_path)


@lru_cache(maxsize=5000)
def image_data_uri(path_string: str) -> str:
    """Convert a downloaded portrait to an embeddable base64 data URI."""
    if not path_string:
        return ""

    image_path = Path(path_string)

    if not image_path.is_file():
        return ""

    try:
        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode("utf-8")
    except OSError:
        return ""

    return f"data:image/png;base64,{encoded}"


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


def _build_card_html(row: pd.Series, score_label: str) -> str:
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

    shares = row.get("shares")
    differences = row.get("differences")

    successor_score_value = pd.to_numeric(
        pd.Series([row.get("successor_score")]),
        errors="coerce",
    ).iloc[0]

    local_image_path = get_local_image_path(
        sofifa_id_from_image_url(row.get("image_url"))
    )

    image_source = image_data_uri(
        local_image_path
    )

    if not image_source:
        image_source = valid_image_url(
            row.get("image_url")
        )

    initials = html.escape(
        player_initials(raw_name)
    )

    if image_source:
        safe_image_source = html.escape(
            image_source,
            quote=True,
        )

        image_html = (
            '<div class="ewc-player-image-wrap">'
            f'<img class="ewc-player-image" '
            f'src="{safe_image_source}" '
            f'alt="{name} player portrait" '
            'loading="lazy" '
            'referrerpolicy="no-referrer">'
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

    if isinstance(shares, list) and shares:
        items = "".join(f"<li>{html.escape(str(item))}</li>" for item in shares)
        explanation_html += (
            '<div class="ewc-player-shares">'
            '<div class="ewc-player-section-label">Shares</div>'
            f"<ul>{items}</ul>"
            "</div>"
        )

    if isinstance(differences, list) and differences:
        items = "".join(f"<li>{html.escape(str(item))}</li>" for item in differences)
        explanation_html += (
            '<div class="ewc-player-differences">'
            '<div class="ewc-player-section-label">Differences</div>'
            f"<ul>{items}</ul>"
            "</div>"
        )

    if pd.notna(successor_score_value):
        explanation_html += (
            '<div class="ewc-successor-score">Modern Successor Score: '
            f"<strong>{successor_score_value:.1f}/10</strong></div>"
        )

    score_html = (
        '<div class="ewc-score-wrap">'
        f'<div class="ewc-score-label">{html.escape(score_label)}</div>'
        f'<div class="ewc-score">{score}</div>'
        "</div>"
        if score
        else ""
    )

    return (
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
        f"{score_html}"
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


def player_cards(df: pd.DataFrame, max_cards: int = 8, score_label: str = "DNA Match") -> None:
    if df.empty:
        st.info("No matching players found.")
        return

    cards = [
        _build_card_html(row, score_label)
        for _, row in df.head(max_cards).iterrows()
    ]

    grid_html = (
        '<div class="ewc-card-grid">'
        + "".join(cards)
        + "</div>"
    )

    st.markdown(
        grid_html,
        unsafe_allow_html=True,
    )


def dna_pathway(rows: list[pd.Series], step_labels: list[str] | None = None) -> None:
    """Render a vertical chain of player cards connected by arrows.

    The first row is treated as the starting point (no DNA Match badge);
    each subsequent row is expected to carry a "similarity" value against
    the row before it, which _build_card_html renders as the score badge.
    """
    if not rows:
        st.info("No pathway to show.")
        return

    blocks: list[str] = []

    for i, row in enumerate(rows):
        if step_labels and i < len(step_labels):
            blocks.append(
                f'<div class="ewc-pathway-step-label">{html.escape(step_labels[i])}</div>'
            )

        blocks.append(_build_card_html(row, score_label="DNA Match"))

        if i < len(rows) - 1:
            blocks.append('<div class="ewc-pathway-connector">&darr;</div>')

    st.markdown(
        '<div class="ewc-pathway">' + "".join(blocks) + "</div>",
        unsafe_allow_html=True,
    )