from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from textwrap import dedent
import base64
import html
import re

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


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


_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


@lru_cache(maxsize=5000)
def image_data_uri(path_string: str) -> str:
    """Convert a local image to an embeddable base64 data URI.

    Streamlit doesn't reliably serve files from the app directory as plain
    URLs once deployed, so anything rendered via raw HTML has to be inlined.
    """
    if not path_string:
        return ""

    image_path = Path(path_string)

    if not image_path.is_file():
        return ""

    mime_type = _IMAGE_MIME_TYPES.get(image_path.suffix.lower(), "image/png")

    try:
        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode("utf-8")
    except OSError:
        return ""

    return f"data:{mime_type};base64,{encoded}"


HERO_IMAGE_PATH = ROOT / "assets" / "football_genome_header_v2.jpg"


def hero() -> None:
    hero_image_uri = image_data_uri(str(HERO_IMAGE_PATH))

    # The banner art already carries the title, tagline and hook line as
    # part of the design, so there's no separate text overlay here anymore --
    # duplicating it in HTML would just repeat what the image already says.
    hero_image_html = (
        f'<div class="ewc-hero-image-wrap"><img class="ewc-hero-image" '
        f'src="{html.escape(hero_image_uri, quote=True)}" alt="The Football Genome -- '
        f'every footballer has a DNA profile. Compare generations, discover future '
        f'successors, reveal football\'s hidden connections." loading="lazy"></div>'
        if hero_image_uri
        else ""
    )

    render_html(
        f"""
        <section class="ewc-hero">
          {hero_image_html}
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


def sidebar_stats(stats: list[tuple[str, str]]) -> None:
    """A compact 2-column stat list sized for the sidebar's narrow width --
    metrics_grid's 4-column cards don't fit there."""
    items = "".join(
        '<div class="ewc-sidebar-stat">'
        f'<div class="value">{html.escape(str(value))}</div>'
        f'<div class="label">{html.escape(str(label))}</div>'
        "</div>"
        for label, value in stats
    )

    st.markdown(
        f'<div class="ewc-sidebar-stats">{items}</div>',
        unsafe_allow_html=True,
    )


# Thresholds checked high-to-low; first match wins. Lets the eye jump
# straight to "this is why he's a match" instead of reading every number.
_TRAIT_BAR_THRESHOLDS = [
    (90, "#4ADE80"),
    (80, "#A3E635"),
    (70, "#FDE047"),
    (60, "#FB923C"),
    (0, "#F87171"),
]


def _trait_bar_color(pct: float) -> str:
    for threshold, color in _TRAIT_BAR_THRESHOLDS:
        if pct >= threshold:
            return color

    return _TRAIT_BAR_THRESHOLDS[-1][1]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in range(0, 6, 2))


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
    narrative = clean_text(row.get("narrative"))
    trait_breakdown = row.get("trait_breakdown")

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

    narrative_html = (
        '<div class="ewc-player-verdict">'
        '<div class="ewc-player-section-label">DNA Verdict</div>'
        f'<p class="ewc-player-narrative">{html.escape(narrative)}</p>'
        "</div>"
        if narrative
        else ""
    )

    breakdown_html = ""

    if isinstance(trait_breakdown, list) and trait_breakdown:
        bars = []

        for trait_label, trait_pct in trait_breakdown:
            pct = max(0.0, min(100.0, float(trait_pct)))
            bar_color = _trait_bar_color(pct)
            bars.append(
                '<div class="ewc-trait-row">'
                f'<span class="ewc-trait-label">{html.escape(str(trait_label))}</span>'
                '<span class="ewc-trait-bar">'
                f'<span class="ewc-trait-fill" style="width:{pct:.0f}%;background:{bar_color}"></span>'
                "</span>"
                f'<span class="ewc-trait-pct" style="color:{bar_color}">{pct:.0f}%</span>'
                "</div>"
            )

        breakdown_html = (
            '<div class="ewc-trait-breakdown">'
            '<div class="ewc-player-section-label">DNA breakdown</div>'
            + "".join(bars)
            + "</div>"
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
            '<div class="ewc-successor-score">DNA Fit: '
            f"<strong>{successor_score_value:.1f}/10</strong></div>"
        )

    match_context = clean_text(row.get("match_context"))
    match_context_html = (
        f'<div class="ewc-score-context">{html.escape(match_context)}</div>'
        if match_context
        else ""
    )

    score_html = (
        '<div class="ewc-score-wrap">'
        f"{match_context_html}"
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
        '<div class="ewc-player-identity">'
        f'<div class="ewc-player-name">{flag} {name}</div>'
        f'<div class="ewc-player-meta">{season} · {club}</div>'
        f'<div class="ewc-player-meta">{nation} · {position}</div>'
        "</div>"
        f"{score_html}"
        "</div>"
        f'<div class="ewc-archetype-chip">{archetype_label}</div>'
        '<div class="ewc-pill-row">'
        f'<span class="ewc-pill">Overall {overall}</span>'
        f'<span class="ewc-pill">Age {age}</span>'
        "</div>"
        f"{narrative_html}"
        f"{breakdown_html}"
        f"{explanation_html}"
        "</div>"
        "</div>"
        "</article>"
    )


_EXPORT_BG = (11, 16, 32)
_EXPORT_PANEL = (26, 38, 62)
_EXPORT_TRACK = (40, 50, 72)
_EXPORT_TEXT = (248, 250, 252)
_EXPORT_MUTED = (175, 193, 212)
_EXPORT_ACCENT = (125, 211, 252)
_EXPORT_PURPLE = (192, 132, 252)
_EXPORT_BORDER = (90, 104, 132)
_EXPORT_WATERMARK = (120, 132, 156)


@lru_cache(maxsize=8)
def _export_font(size: int) -> "ImageFont.FreeTypeFont":
    return ImageFont.load_default(size=size)


def _wrap_export_text(
    draw: "ImageDraw.ImageDraw",
    text: str,
    font: "ImageFont.FreeTypeFont",
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]

        while draw.textlength(f"{last}…", font=font) > max_width and len(last) > 1:
            last = last[:-1]

        lines[-1] = f"{last.rstrip()}…"

    return lines


def render_card_image(row: pd.Series, score_label: str = "DNA Match") -> bytes:
    """Server-side render of a shareable PNG mirroring the on-screen card.

    Streamlit can't reliably screenshot arbitrary HTML client-side, so this
    redraws the same fields with Pillow instead of trying to rasterize the DOM.
    """
    width, height = 1000, 640
    image = Image.new("RGB", (width, height), _EXPORT_BG)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        [20, 20, width - 20, height - 20],
        radius=28,
        outline=_EXPORT_BORDER,
        width=2,
    )

    raw_name = clean_text(row.get("short_name", row.get("player_name", "Unknown"))) or "Unknown"
    season = clean_text(row.get("season_label", row.get("fifa_version")))
    club = clean_text(row.get("club_name", row.get("club")))
    nation = clean_text(row.get("nationality_name", row.get("country")))
    position = clean_text(row.get("player_positions", row.get("position")))
    overall = clean_text(row.get("overall", row.get("ovr")))
    age = clean_text(row.get("age"))
    archetype_name = clean_text(row.get("archetype_name"))
    archetype_id = clean_text(row.get("archetype_id"))
    archetype_label = archetype_name or (f"Cluster {archetype_id}" if archetype_id else "Unclassified profile")
    narrative = clean_text(row.get("narrative"))
    trait_breakdown = row.get("trait_breakdown")
    successor_score_value = pd.to_numeric(
        pd.Series([row.get("successor_score")]), errors="coerce"
    ).iloc[0]
    similarity = pd.to_numeric(pd.Series([row.get("similarity")]), errors="coerce").iloc[0]

    portrait_box = (56, 56, 316, 560)
    draw.rounded_rectangle(portrait_box, radius=20, fill=_EXPORT_PANEL)

    local_image_path = get_local_image_path(sofifa_id_from_image_url(row.get("image_url")))
    pasted = False

    if local_image_path:
        try:
            portrait_img = Image.open(local_image_path).convert("RGBA")
            box_w = portrait_box[2] - portrait_box[0] - 24
            box_h = portrait_box[3] - portrait_box[1] - 24
            portrait_img.thumbnail((box_w, box_h))
            px = portrait_box[0] + (portrait_box[2] - portrait_box[0] - portrait_img.width) // 2
            py = portrait_box[3] - 12 - portrait_img.height
            image.paste(portrait_img, (px, py), portrait_img)
            pasted = True
        except OSError:
            pasted = False

    if not pasted:
        initials = player_initials(raw_name)
        side = portrait_box[2] - portrait_box[0] - 80
        cx0 = portrait_box[0] + 40
        cy0 = portrait_box[1] + 150
        draw.ellipse([cx0, cy0, cx0 + side, cy0 + side], fill=_EXPORT_TRACK)
        font_initials = _export_font(48)
        tw = draw.textlength(initials, font=font_initials)
        draw.text(
            (cx0 + side / 2 - tw / 2, cy0 + side / 2 - 28),
            initials,
            font=font_initials,
            fill=_EXPORT_MUTED,
        )

    x0 = 348

    font_name = _export_font(36)
    draw.text((x0, 52), raw_name, font=font_name, fill=_EXPORT_TEXT)

    font_meta = _export_font(19)
    meta_line = " · ".join(part for part in [season, club] if part)
    draw.text((x0, 100), meta_line, font=font_meta, fill=_EXPORT_MUTED)
    meta_line2 = " · ".join(part for part in [nation, position] if part)
    draw.text((x0, 126), meta_line2, font=font_meta, fill=_EXPORT_MUTED)

    if pd.notna(similarity):
        score_text = f"{float(similarity) * 100:.1f}%"
        font_score = _export_font(38)
        badge_cx, badge_cy, badge_r = width - 130, 116, 62
        draw.ellipse(
            [badge_cx - badge_r, badge_cy - badge_r, badge_cx + badge_r, badge_cy + badge_r],
            outline=_EXPORT_ACCENT,
            width=3,
            fill=(17, 33, 51),
        )
        tw = draw.textlength(score_text, font=font_score)
        draw.text((badge_cx - tw / 2, badge_cy - 21), score_text, font=font_score, fill=_EXPORT_ACCENT)
        font_label = _export_font(13)
        label_text = score_label.upper()
        lw = draw.textlength(label_text, font=font_label)
        draw.text(
            (badge_cx - lw / 2, badge_cy - badge_r - 20),
            label_text,
            font=font_label,
            fill=_EXPORT_MUTED,
        )

    font_chip = _export_font(18)
    chip_w = draw.textlength(archetype_label, font=font_chip) + 28
    chip_box = (x0, 168, x0 + chip_w, 202)
    draw.rounded_rectangle(chip_box, radius=17, outline=_EXPORT_PURPLE, width=2, fill=(43, 26, 61))
    draw.text((x0 + 14, 176), archetype_label, font=font_chip, fill=_EXPORT_PURPLE)

    pill_x, pill_y = x0, 216
    font_pill = _export_font(16)

    for value, pill_text in [(overall, f"Overall {overall}"), (age, f"Age {age}")]:
        if not value:
            continue

        pw = draw.textlength(pill_text, font=font_pill) + 24
        draw.rounded_rectangle(
            (pill_x, pill_y, pill_x + pw, pill_y + 30), radius=15, outline=_EXPORT_BORDER, width=1
        )
        draw.text((pill_x + 12, pill_y + 6), pill_text, font=font_pill, fill=_EXPORT_TEXT)
        pill_x += pw + 10

    y = 266

    if narrative:
        font_narrative = _export_font(17)
        max_text_width = width - x0 - 40

        for line in _wrap_export_text(draw, narrative, font_narrative, max_text_width, max_lines=3):
            draw.text((x0, y), line, font=font_narrative, fill=_EXPORT_MUTED)
            y += 24

        y += 12

    if isinstance(trait_breakdown, list) and trait_breakdown:
        font_trait = _export_font(14)
        bar_w = 200

        for trait_label, trait_pct in trait_breakdown[:6]:
            pct = max(0.0, min(100.0, float(trait_pct)))
            bar_color = _hex_to_rgb(_trait_bar_color(pct))
            draw.text((x0, y + 2), str(trait_label), font=font_trait, fill=_EXPORT_MUTED)
            track_x0 = x0 + 110
            track_x1 = track_x0 + bar_w
            draw.rounded_rectangle((track_x0, y + 4, track_x1, y + 12), radius=4, fill=_EXPORT_TRACK)
            fill_x1 = track_x0 + bar_w * (pct / 100.0)

            if fill_x1 > track_x0:
                draw.rounded_rectangle((track_x0, y + 4, fill_x1, y + 12), radius=4, fill=bar_color)

            draw.text((track_x1 + 12, y + 1), f"{pct:.0f}%", font=font_trait, fill=bar_color)
            y += 24

    if pd.notna(successor_score_value):
        font_footer = _export_font(18)
        draw.text(
            (x0, 566),
            f"DNA Fit: {successor_score_value:.1f}/10",
            font=font_footer,
            fill=_EXPORT_MUTED,
        )

    font_watermark = _export_font(15)
    watermark = "The Eternal World Cup"
    ww = draw.textlength(watermark, font=font_watermark)
    draw.text((width - 40 - ww, height - 40), watermark, font=font_watermark, fill=_EXPORT_WATERMARK)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _download_card_button(row: pd.Series, score_label: str, key: str) -> None:
    file_stub = re.sub(r"[^a-z0-9]+", "_", clean_text(row.get("short_name", "player")).lower()).strip("_")

    st.download_button(
        "⬇ Download DNA card",
        data=render_card_image(row, score_label),
        file_name=f"{file_stub or 'player'}_dna_card.png",
        mime="image/png",
        key=key,
        width="stretch",
    )


def player_cards(
    df: pd.DataFrame,
    max_cards: int = 8,
    score_label: str = "DNA Match",
    key_prefix: str = "cards",
) -> None:
    if df.empty:
        st.info("No matching players found.")
        return

    rows = list(df.head(max_cards).iterrows())

    for start in range(0, len(rows), 2):
        chunk = rows[start:start + 2]
        columns = st.columns(2)

        for position, (column, (row_index, row)) in enumerate(zip(columns, chunk)):
            with column:
                st.markdown(_build_card_html(row, score_label), unsafe_allow_html=True)
                _download_card_button(
                    row,
                    score_label,
                    key=f"card_dl_{key_prefix}_{start + position}",
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