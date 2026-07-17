from __future__ import annotations

from pathlib import Path
import html
import math
import random
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from components import clean_text, dna_pathway, hero, load_css, metrics_grid, player_cards

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="The Football Genome", page_icon="🧬", layout="wide")
load_css()

@st.cache_data
def load_players() -> pd.DataFrame:
    path = PROCESSED / "app_players.csv"
    if not path.exists():
        st.error(
            "Processed data not found. Run `python src/run_pipeline.py` locally "
            "and commit data/processed outputs."
        )
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_archetypes() -> pd.DataFrame:
    path = PROCESSED / "archetypes.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_scores() -> pd.DataFrame:
    path = PROCESSED / "legend_scores.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# Each descriptor averages a small group of related FIFA attributes into one
# "profile" reading, then compares query vs candidate on that reading. Using
# groups (not single stats) means a bullet like "elite creative passing
# profile" reflects passing + vision + long/short passing together, not a
# single noisy attribute.
PROFILE_DESCRIPTORS = [
    # key, columns, strong-shared label, ordinary-shared label, "more" label, "less" label
    ("passing", ["passing", "vision", "long_passing", "short_passing"],
     "Elite creative passing profile", "Similar passing profile",
     "More creative passing", "Less creative passing"),
    ("shooting", ["shooting", "finishing", "shot_power"],
     "High attacking output", "Similar shooting profile",
     "More shooting threat", "Less shooting threat"),
    ("dribbling", ["dribbling", "ball_control", "agility"],
     "Elite ball-carrying ability", "Similar dribbling profile",
     "More ball carrying", "Less ball carrying"),
    ("pace", ["pace", "acceleration", "sprint_speed"],
     "Elite pace profile", "Similar pace profile",
     "More raw pace", "Less raw pace"),
    ("physic", ["physic", "strength", "stamina"],
     "Dominant physical profile", "Similar physical profile",
     "More physical presence", "Less physical presence"),
    ("defending", ["defending", "interceptions", "standing_tackle"],
     "Elite defensive output", "Similar defensive profile",
     "More defensive involvement", "Less defensive involvement"),
]

SHARE_DIFF_THRESHOLD = 6
SHARE_MIN_VALUE = 55
STRONG_VALUE_THRESHOLD = 80
DIFFERENCE_MIN_THRESHOLD = 8

# Story-driven display labels for search modes -- the underlying values used
# throughout the matching logic stay unchanged, only what the user reads changes.
SEARCH_MODE_LABELS = {
    "Young successors": "Who's next? (U23 successors)",
    "Current replacements": "Playing today",
    "Historical lookalikes": "Modern lookalikes",
    "All similar players": "Unlimited search",
}

# Short context line shown above the DNA Match score on each result card, so
# the score reads with its search intent rather than as a bare percentage.
SEARCH_MODE_CONTEXT_LABELS = {
    "Young successors": "Modern Successor",
    "Current replacements": "Current Replacement",
    "Historical lookalikes": "Historical Match",
    "All similar players": "",
}

# Short display labels for the compact per-trait score breakdown.
TRAIT_BREAKDOWN_LABELS = {
    "passing": "Passing",
    "shooting": "Shooting",
    "dribbling": "Dribbling",
    "pace": "Pace",
    "physic": "Physicality",
    "defending": "Defending",
}

# Sentence-friendly phrasing for the templated "Why this player?" narrative --
# distinct from the bullet-point labels above, which read fine as list items
# but not as flowing prose.
# Each trait maps to several synonym phrasings -- build_narrative_sentence
# picks one deterministically per candidate so repeated top traits (pace and
# physicality dominate most searches) don't read identically across every
# card in a results grid, while staying stable across reruns.
NARRATIVE_SHARE_PHRASES = {
    "passing": ["creative passing range", "vision and range of pass", "eye for a killer ball"],
    "shooting": ["attacking output", "finishing instinct", "goal threat"],
    "dribbling": [
        "ability to carry the ball into dangerous areas",
        "close control in tight spaces",
        "knack for beating a man one-on-one",
    ],
    "pace": ["electric pace", "raw speed in behind", "gas to burn defenders"],
    "physic": ["physical profile", "power and presence", "physical duels edge"],
    "defending": ["defensive contribution", "work rate off the ball", "positional discipline"],
}

NARRATIVE_MORE_PHRASES = {
    "passing": ["even more creative passing", "a wider range of pass", "sharper vision"],
    "shooting": ["a sharper shooting threat", "a more clinical edge in front of goal", "extra firepower"],
    "dribbling": ["more direct ball-carrying", "sharper close control", "a bigger dribbling threat"],
    "pace": ["extra pace", "a real speed advantage", "more explosiveness in behind"],
    "physic": ["a more physical presence", "extra power in duels", "a stronger physical edge"],
    "defending": ["more defensive work", "a heavier defensive workload", "more discipline without the ball"],
}

NARRATIVE_LESS_PHRASES = {
    "passing": ["a less creative passing profile", "a tighter passing range", "less range of pass"],
    "shooting": ["less of a shooting threat", "a lighter goal threat", "less firepower in front of goal"],
    "dribbling": ["less ball-carrying threat", "a lighter dribbling load", "less of a one-on-one threat"],
    "pace": ["less raw pace", "a shade less speed", "less gas in behind"],
    "physic": ["a lighter physical profile", "less physical presence", "a softer edge in duels"],
    "defending": ["less defensive involvement", "a lighter defensive workload", "less positional discipline"],
}

# Opening claim for the DNA Verdict, picked by similarity tier -- this is
# the "commit to the storytelling" sentence, not a hedge-everything summary.
NARRATIVE_TIER_OPENERS = {
    "elite": [
        "{candidate} isn't the next {query} -- nobody is. But strip {query}'s game down to {trait}, and this is roughly what you'd get.",
        "{candidate} is about as close as the dataset gets to a modern {query}, built around {trait}.",
        "Rebuild {query} around {trait} alone, and {candidate} is close to what comes out the other side.",
    ],
    "strong": [
        "{candidate} occupies a genuinely similar football neighbourhood to {query}, built around {trait}.",
        "{candidate} carries a strong dose of {query}'s DNA, anchored by {trait}.",
        "There's a real family resemblance here -- both {candidate} and {query} lean on {trait}.",
    ],
    "moderate": [
        "{candidate} is a plausible stylistic cousin of {query}, sharing {trait} even as the rest of the profile diverges.",
        "{candidate} and {query} overlap around {trait}, without matching everywhere else.",
    ],
    "distant": [
        "{candidate} is a distant echo of {query} -- {trait} links them, but the profiles part ways from there.",
        "The connection to {query} is real but thin: mostly {trait}.",
    ],
}

NARRATIVE_VERDICT_CLOSERS = [
    "{candidate} sacrifices some of that for {diff}, but retains enough of the underlying profile to make the comparison stick.",
    "The trade-off shows up in {diff}, though the core of the profile still holds together.",
    "That said, {diff} keeps this from being a perfect match -- just a very convincing one.",
]

NARRATIVE_FALLBACK_SHARES = [
    "{candidate} carries a broadly similar football DNA profile to {query}.",
    "{candidate} lands in the same broad territory as {query}'s profile.",
    "{candidate}'s overall football DNA sits close to {query}'s.",
]

NARRATIVE_AGE_CLAUSES = [
    " At {gap:.0f} years younger, there's real time to grow into it.",
    " {gap:.0f} years {query}'s junior, with plenty of runway left.",
    " A {gap:.0f}-year head start on development still to come.",
]


def _pick_variant(options: object, seed_parts: tuple) -> str:
    if isinstance(options, str):
        return options

    seed = "|".join(str(part) for part in seed_parts)
    return random.Random(seed).choice(options)


def descriptor_value(row_like: pd.Series, columns: list[str]) -> float:
    values = [
        pd.to_numeric(row_like.get(column), errors="coerce")
        for column in columns
    ]
    values = [float(value) for value in values if pd.notna(value)]
    return float(np.mean(values)) if values else float("nan")


def build_profile_comparison(candidate: pd.Series, query: pd.Series) -> dict:
    share_candidates = []
    difference_candidates = []
    trait_breakdown = []

    for key, columns, strong_label, similar_label, more_label, less_label in PROFILE_DESCRIPTORS:
        query_value = descriptor_value(query, columns)
        candidate_value = descriptor_value(candidate, columns)

        if pd.isna(query_value) or pd.isna(candidate_value):
            continue

        diff = candidate_value - query_value
        abs_diff = abs(diff)

        trait_breakdown.append((
            TRAIT_BREAKDOWN_LABELS.get(key, key.title()),
            max(0.0, 100.0 - abs_diff),
        ))

        if abs_diff <= SHARE_DIFF_THRESHOLD and min(query_value, candidate_value) >= SHARE_MIN_VALUE:
            average_value = (query_value + candidate_value) / 2
            label = strong_label if average_value >= STRONG_VALUE_THRESHOLD else similar_label
            share_candidates.append((abs_diff, label, key))
        elif abs_diff >= DIFFERENCE_MIN_THRESHOLD:
            direction = "more" if diff > 0 else "less"
            difference_candidates.append((abs_diff, more_label if diff > 0 else less_label, key, direction))

    share_candidates.sort(key=lambda item: item[0])
    difference_candidates.sort(key=lambda item: -item[0])

    shares = [label for _, label, _ in share_candidates[:4]]
    share_keys = [key for _, _, key in share_candidates[:2]]
    differences = [label for _, label, _, _ in difference_candidates[:3]]
    difference_keys = [(key, direction) for _, _, key, direction in difference_candidates[:2]]

    query_age = pd.to_numeric(query.get("age"), errors="coerce")
    candidate_age = pd.to_numeric(candidate.get("age"), errors="coerce")
    age_gap = float("nan")

    if pd.notna(query_age) and pd.notna(candidate_age):
        age_gap = float(query_age) - float(candidate_age)

        if abs(age_gap) >= 2:
            qualifier = "Significantly" if abs(age_gap) >= 6 else "Slightly"
            direction = "younger" if age_gap > 0 else "older"
            differences.append(f"{qualifier} {direction} profile")

    if not shares:
        shares = ["Broadly similar overall football DNA profile"]

    if not differences:
        differences = ["No standout attribute differences"]

    return {
        "shares": shares,
        "differences": differences,
        "share_keys": share_keys,
        "difference_keys": difference_keys,
        "age_gap": age_gap,
        "trait_breakdown": trait_breakdown,
    }


def _narrative_tier(similarity: float) -> str:
    if similarity >= 0.93:
        return "elite"
    if similarity >= 0.85:
        return "strong"
    if similarity >= 0.70:
        return "moderate"
    return "distant"


def build_narrative_sentence(
    candidate_name: str,
    query_name: str,
    comparison: dict,
    similarity: float = float("nan"),
) -> str:
    """Template a confident 'DNA verdict' from the same comparison data that
    drives the Shares/Differences bullets -- no external model call."""
    share_keys = comparison.get("share_keys", [])
    difference_keys = comparison.get("difference_keys", [])
    age_gap = comparison.get("age_gap", float("nan"))

    if share_keys:
        share_phrases = [
            _pick_variant(
                NARRATIVE_SHARE_PHRASES.get(key, key.replace("_", " ")),
                (candidate_name, key, "share"),
            )
            for key in share_keys
        ]
        tier = _narrative_tier(float(similarity)) if pd.notna(similarity) else "moderate"
        opener = _pick_variant(
            NARRATIVE_TIER_OPENERS[tier], (candidate_name, tier, "opener")
        ).format(candidate=candidate_name, query=query_name, trait=share_phrases[0])
    else:
        opener = _pick_variant(
            NARRATIVE_FALLBACK_SHARES, (candidate_name, query_name, "fallback")
        ).format(candidate=candidate_name, query=query_name)

    difference_sentence = ""

    if difference_keys:
        difference_phrases = []

        for key, direction in difference_keys:
            bank = NARRATIVE_MORE_PHRASES if direction == "more" else NARRATIVE_LESS_PHRASES
            difference_phrases.append(
                _pick_variant(
                    bank.get(key, key.replace("_", " ")),
                    (candidate_name, key, direction),
                )
            )

        difference_sentence = _pick_variant(
            NARRATIVE_VERDICT_CLOSERS, (candidate_name, "closer")
        ).format(candidate=candidate_name, diff=" and ".join(difference_phrases))

    age_clause = ""

    if pd.notna(age_gap) and age_gap >= 4:
        age_clause = _pick_variant(
            NARRATIVE_AGE_CLAUSES, (candidate_name, "age")
        ).format(gap=age_gap, query=query_name)

    return f"{opener} {difference_sentence}{age_clause}".strip()


def successor_score(similarity: float, query: pd.Series, candidate: pd.Series) -> float:
    """A narrative 0-10 score: DNA similarity, weighted toward candidates who
    represent a younger generation rather than a same-era peer. Deliberately
    leaves headroom below 10 -- reaching the ceiling requires both a
    near-perfect DNA match and a large generational gap, not just one or the
    other, so results don't all peg at the maximum within a single search."""
    similarity_component = similarity * 8.5
    query_age = pd.to_numeric(query.get("age"), errors="coerce")
    candidate_age = pd.to_numeric(candidate.get("age"), errors="coerce")
    age_component = 0.0

    if pd.notna(query_age) and pd.notna(candidate_age):
        age_gap = float(query_age) - float(candidate_age)
        age_factor = max(0.0, min(1.0, age_gap / 15))
        age_component = age_factor * 1.5

    return max(0.0, min(10.0, similarity_component + age_component))


def add_similarity_reasons(res: pd.DataFrame, query: pd.Series) -> pd.DataFrame:
    res = res.copy()
    shares_column = []
    differences_column = []
    score_column = []
    narrative_column = []
    trait_breakdown_column = []

    query_name = clean_text(query.get("short_name", query.get("player_name", "This player"))) or "This player"

    for _, row in res.iterrows():
        comparison = build_profile_comparison(row, query)
        shares_column.append(comparison["shares"])
        differences_column.append(comparison["differences"])
        trait_breakdown_column.append(comparison["trait_breakdown"])

        candidate_name = clean_text(row.get("short_name", row.get("player_name", "This player"))) or "This player"

        similarity_value = pd.to_numeric(
            pd.Series([row.get("similarity")]),
            errors="coerce",
        ).iloc[0]

        narrative_column.append(
            build_narrative_sentence(candidate_name, query_name, comparison, float(similarity_value))
        )

        score_column.append(
            successor_score(float(similarity_value), query, row)
            if pd.notna(similarity_value)
            else float("nan")
        )

    res["shares"] = shares_column
    res["differences"] = differences_column
    res["successor_score"] = score_column
    res["narrative"] = narrative_column
    res["trait_breakdown"] = trait_breakdown_column

    return res


def broad_position_group(position_string: object) -> str:
    positions = str(position_string or "").upper()

    if "GK" in positions:
        return "Goalkeeper"

    if any(position in positions for position in ["CB", "LB", "RB", "LWB", "RWB"]):
        return "Defender"

    if any(position in positions for position in ["CDM", "CM", "CAM", "LM", "RM"]):
        return "Midfielder"

    if any(position in positions for position in ["LW", "RW", "CF", "ST"]):
        return "Forward"

    return "Other"


def primary_position(position_string: object) -> str:
    positions = str(position_string or "")
    return positions.split(",")[0].strip().upper()


players = load_players()
archetypes = load_archetypes()
scores = load_scores()

emb_cols = [c for c in players.columns if c.startswith("emb_")]
if not emb_cols:
    st.error("No embedding columns found in data/processed/app_players.csv. Re-run the pipeline.")
    st.stop()

PAGE_NAV_KEY = "page_nav_radio"
PENDING_NAV_KEY = "_pending_page_nav"

# A widget's session_state key can't be reassigned once that widget has
# rendered in the current script run -- Streamlit raises StreamlitAPIException.
# Jump helpers below stash the target page here and call st.rerun(); applying
# it here, before the sidebar radio (bound to PAGE_NAV_KEY) ever instantiates,
# is what makes the reassignment legal.
if PENDING_NAV_KEY in st.session_state:
    st.session_state[PAGE_NAV_KEY] = st.session_state.pop(PENDING_NAV_KEY)

with st.sidebar:
    st.markdown("## 🧬 The Football Genome")
    st.caption("Football DNA, decoded.")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Successor Finder", "Compare Players", "Evolution", "Pathways", "DNA Map", "Legend Score", "Archetypes", "Method"],
        key=PAGE_NAV_KEY,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.success("Autoencoder embeddings live")
    st.caption("Trained on FIFA attributes + FBRef performance data across 10 FIFA editions.")

hero()
metrics_grid([
    ("Player-seasons", f"{len(players):,}", "Historical + current records"),
    ("DNA dimensions", str(len(emb_cols)), "Compressed profile space"),
    ("Archetypes", f"{players['archetype_id'].nunique() if 'archetype_id' in players.columns else 0}", "Profile clusters"),
    ("Latest player pool", f"{players['name_key'].nunique():,}", "Unique players"),
    ("Countries", f"{players['nationality_name'].nunique():,}", "Nations on the map"),
    ("Clubs", f"{players['club_name'].nunique():,}", "Clubs represented"),
    ("DNA comparisons", f"{math.comb(len(players), 2) / 1_000_000:.0f}M+", "Possible player-pairs"),
    ("FIFA editions", f"{players['season_label'].nunique()}", "Seasons of data"),
])

players = players.copy()
latest = players.sort_values("season_year").groupby("name_key", as_index=False).tail(1).copy()

players["display_name"] = (
    players["short_name"].astype(str)
    + " — "
    + players["season_label"].astype(str)
    + " — "
    + players["club_name"].fillna("").astype(str)
)

latest["display_name"] = (
    latest["short_name"].astype(str)
    + " — "
    + latest["season_label"].astype(str)
    + " — "
    + latest["club_name"].fillna("").astype(str)
)

REFERENCE_KEY = "successor_reference_select"
SEARCH_MODE_KEY = "successor_search_mode_select"
COMPARE_A_KEY = "compare_player_a_select"
COMPARE_B_KEY = "compare_player_b_select"

reference_options = players.sort_values(
    ["overall", "season_year"],
    ascending=[False, False],
)["display_name"].head(5000)


def jump_to_player(name_substring: str, mode: str) -> None:
    """Safely pre-seed the Successor Finder reference + mode and switch to it.

    Only ever sets REFERENCE_KEY to a value that's actually a valid option in
    the reference selectbox -- pre-seeding a selectbox's session_state with a
    value outside its options list raises StreamlitAPIException.
    """
    matches = reference_options[
        reference_options.str.contains(name_substring, case=False, na=False, regex=False)
    ]

    if not matches.empty:
        st.session_state[REFERENCE_KEY] = matches.iloc[0]

    st.session_state[SEARCH_MODE_KEY] = mode
    st.session_state[PENDING_NAV_KEY] = "Successor Finder"
    st.rerun()


def jump_to_compare(name_a_substring: str, name_b_substring: str) -> None:
    """Safely pre-seed both Compare Players selections and switch to it."""
    pool = players["display_name"]
    match_a = pool[pool.str.contains(name_a_substring, case=False, na=False, regex=False)]
    match_b = pool[pool.str.contains(name_b_substring, case=False, na=False, regex=False)]

    if not match_a.empty:
        st.session_state[COMPARE_A_KEY] = match_a.iloc[0]

    if not match_b.empty:
        st.session_state[COMPARE_B_KEY] = match_b.iloc[0]

    st.session_state[PENDING_NAV_KEY] = "Compare Players"
    st.rerun()


st.markdown("#### Try these questions")

hook_questions = [
    ("Can anyone replace Lionel Messi?", "player", "L. Messi", "Current replacements"),
    ("Who is the next Kevin De Bruyne?", "player", "K. De Bruyne", "Young successors"),
    ("What type of midfielder is Jude Bellingham?", "player", "J. Bellingham", "All similar players"),
    ("Which players share Ronaldinho's DNA?", "player", "Ronaldinho", "Historical lookalikes"),
    ("How similar is Erling Haaland to Ronaldo?", "compare", "Haaland", "Cristiano Ronaldo"),
]

hook_cols = st.columns(2)

for i, hook_question in enumerate(hook_questions):
    label, kind = hook_question[0], hook_question[1]

    with hook_cols[i % 2]:
        if st.button(label, key=f"hook_question_{i}", width="stretch"):
            if kind == "player":
                _, _, target, mode = hook_question
                jump_to_player(target, mode)
            else:
                _, _, name_a, name_b = hook_question
                jump_to_compare(name_a, name_b)

if page == "Successor Finder":
    st.header("Find a player's closest football DNA matches")

    st.markdown("#### Popular searches")

    popular_searches = [
        ("Lionel Messi", "L. Messi"),
        ("Cristiano Ronaldo", "Cristiano Ronaldo"),
        ("Kevin De Bruyne", "K. De Bruyne"),
        ("Jude Bellingham", "J. Bellingham"),
        ("Lamine Yamal", "Lamine Yamal"),
        ("Erling Haaland", "Haaland"),
        ("Virgil van Dijk", "van Dijk"),
        ("Bukayo Saka", "B. Saka"),
    ]

    popular_cols = st.columns(4)

    for i, (label, target) in enumerate(popular_searches):
        with popular_cols[i % 4]:
            if st.button(label, key=f"popular_search_{i}", width="stretch"):
                jump_to_player(target, "Young successors")

    st.markdown("#### Start here")

    start_here_prompts = [
        ("Can anyone replace Kevin De Bruyne?", "K. De Bruyne", "Young successors"),
        ("Who carries Cristiano Ronaldo's football DNA?", "Cristiano Ronaldo", "Young successors"),
        ("Which archetype does Cole Palmer belong to?", "Cole Palmer", "All similar players"),
        ("Find football's next great playmaker", "L. Modrić", "Young successors"),
        ("Explore every player, unfiltered", "L. Messi", "All similar players"),
    ]

    start_here_cols = st.columns(3)

    for i, (label, target, mode) in enumerate(start_here_prompts):
        with start_here_cols[i % 3]:
            if st.button(label, key=f"start_here_{i}", width="stretch"):
                jump_to_player(target, mode)

    st.markdown(
        "<p class='ewc-hint'>Pick a reference player-season below, or use one of the "
        "questions above. Choose whether you want young successors, current replacements, "
        "historical lookalikes or an unrestricted DNA search.</p>",
        unsafe_allow_html=True,
    )

    control_row_1 = st.columns([2.3, 1.3])

    with control_row_1[0]:
        selected = st.selectbox(
            "Reference player-season",
            reference_options,
            key=REFERENCE_KEY,
        )

    with control_row_1[1]:
        search_mode = st.selectbox(
            "Search mode",
            [
                "Young successors",
                "Current replacements",
                "Historical lookalikes",
                "All similar players",
            ],
            format_func=lambda mode: SEARCH_MODE_LABELS.get(mode, mode),
            key=SEARCH_MODE_KEY,
        )

    with st.expander("Advanced filters"):
        advanced_row = st.columns(4)

        with advanced_row[0]:
            position_match = st.selectbox(
                "Position match",
                [
                    "Same broad position",
                    "Exact position",
                    "Any position",
                ],
            )

        with advanced_row[1]:
            maximum_age = st.selectbox(
                "Maximum age",
                ["Mode default", 21, 23, 25, 27, "No limit"],
            )

        with advanced_row[2]:
            minimum_overall = st.slider(
                "Minimum overall",
                min_value=50,
                max_value=95,
                value=70,
                step=1,
            )

        with advanced_row[3]:
            n = st.slider(
                "Matches",
                min_value=5,
                max_value=20,
                value=8,
            )

    query = players.loc[
        players["display_name"].eq(selected)
    ].iloc[0]

    query_archetype = query.get("archetype_name")
    query_archetype = query_archetype if pd.notna(query_archetype) else "Unclassified profile"

    elite_traits = []

    for key, columns, *_ in PROFILE_DESCRIPTORS:
        trait_value = descriptor_value(query, columns)

        if pd.notna(trait_value) and trait_value >= STRONG_VALUE_THRESHOLD:
            elite_traits.append((trait_value, TRAIT_BREAKDOWN_LABELS.get(key, key.title())))

    elite_traits.sort(key=lambda item: -item[0])
    elite_traits_label = ", ".join(label for _, label in elite_traits[:3]) or "No standout elite traits at this threshold"

    season_label = clean_text(query.get("season_label", query.get("fifa_version")))
    search_mode_summary = {
        "Young successors": "Searching for U23 successors carrying similar DNA.",
        "Current replacements": "Searching for players active today who could replace this profile.",
        "Historical lookalikes": "Searching across all eras for historical lookalikes.",
        "All similar players": "Searching the full, unrestricted player pool.",
    }.get(search_mode, "")

    st.markdown(
        "<div class='ewc-callout ewc-reference-callout'>"
        f"<div class='ewc-reference-header'>🧬 {html.escape(str(query['short_name']))}"
        f"{f' &middot; {html.escape(season_label)}' if season_label else ''}</div>"
        "<div class='ewc-reference-row'>"
        f"<span class='ewc-reference-pill'>{html.escape(str(query_archetype))}</span>"
        f"<span class='ewc-reference-pill'>Overall {html.escape(str(query.get('overall', '')))}</span>"
        f"<span class='ewc-reference-pill'>Age {html.escape(str(query.get('age', '')))}</span>"
        "</div>"
        f"<div class='ewc-reference-traits'>Elite traits: <strong>{html.escape(elite_traits_label)}</strong></div>"
        f"<div class='ewc-reference-mode'>{html.escape(search_mode_summary)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    pool = players.copy()

    if search_mode in {"Young successors", "Current replacements"}:
        pool = latest.copy()

    if search_mode == "Young successors":
        pool = pool[
            pd.to_numeric(pool["age"], errors="coerce").le(23)
        ]

    elif search_mode == "Current replacements":
        query_age = pd.to_numeric(
            pd.Series([query.get("age")]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(query_age):
            pool = pool[
                pd.to_numeric(pool["age"], errors="coerce").lt(query_age)
            ]

    elif search_mode == "Historical lookalikes":
        pool = players.copy()

    if maximum_age not in {"Mode default", "No limit"}:
        pool = pool[
            pd.to_numeric(pool["age"], errors="coerce").le(int(maximum_age))
        ]

    pool = pool[
        pd.to_numeric(pool["overall"], errors="coerce").ge(minimum_overall)
    ]

    if position_match == "Same broad position":
        query_group = broad_position_group(query.get("player_positions"))
        pool = pool[
            pool["player_positions"]
            .map(broad_position_group)
            .eq(query_group)
        ]

    elif position_match == "Exact position":
        query_primary_position = primary_position(
            query.get("player_positions")
        )
        pool = pool[
            pool["player_positions"]
            .map(primary_position)
            .eq(query_primary_position)
        ]

    pool = pool[
        pool["player_season_id"].ne(query["player_season_id"])
    ]

    pool = pool[
        pool["name_key"].ne(query["name_key"])
    ]

    if pool.empty:
        st.warning(
            "No players match the selected filters. Try widening the age, "
            "overall or position settings."
        )
    else:
        with st.spinner("Searching football DNA..."):
            Xq = query[emb_cols].to_numpy(float).reshape(1, -1)
            X = pool[emb_cols].to_numpy(float)

            res = pool.copy()
            res["similarity"] = cosine_similarity(Xq, X).ravel()

            res = (
                res.sort_values("similarity", ascending=False)
                .head(n)
            )

            res = add_similarity_reasons(res, query)
            res["match_context"] = SEARCH_MODE_CONTEXT_LABELS.get(search_mode, "")

        st.caption(
            f"Showing {SEARCH_MODE_LABELS.get(search_mode, search_mode).lower()} using "
            f"{position_match.lower()} filtering."
        )

        player_cards(res, max_cards=n, key_prefix="successor")

        with st.expander("Show table"):
            table_res = res.copy()
            table_res["shares"] = table_res["shares"].apply("; ".join)
            table_res["differences"] = table_res["differences"].apply("; ".join)

            cols = [
                "short_name",
                "season_label",
                "club_name",
                "nationality_name",
                "overall",
                "age",
                "player_positions",
                "similarity",
                "successor_score",
                "shares",
                "differences",
                "archetype_name",
            ]

            st.dataframe(
                table_res[[column for column in cols if column in table_res.columns]],
                width="stretch",
            )

elif page == "Compare Players":
    st.header("Compare two player-seasons")
    st.caption(
        "Pick any two player-seasons to see the story behind their DNA "
        "overlap, not just the raw attribute gap."
    )

    c1, c2 = st.columns(2)

    with c1:
        a = st.selectbox("Player A", players["display_name"].sort_values(), index=0, key=COMPARE_A_KEY)

    with c2:
        b = st.selectbox("Player B", players["display_name"].sort_values(), index=1, key=COMPARE_B_KEY)

    pa = players.loc[players["display_name"].eq(a)].iloc[0]
    pb = players.loc[players["display_name"].eq(b)].iloc[0]

    sim = cosine_similarity(
        pa[emb_cols].to_numpy(float).reshape(1, -1),
        pb[emb_cols].to_numpy(float).reshape(1, -1),
    )[0, 0]

    pb_row = add_similarity_reasons(pd.DataFrame([pb]), pa).iloc[0].copy()
    pb_row["similarity"] = sim

    combined = pd.concat(
        [pd.DataFrame([pa]), pd.DataFrame([pb_row])],
        ignore_index=True,
    )

    player_cards(combined, max_cards=2, score_label="DNA Match", key_prefix="compare")

    st.subheader("Attribute radar")

    attrs = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

    comp = pd.DataFrame({
        "attribute": attrs,
        str(pa["short_name"]): [pa.get(x, np.nan) for x in attrs],
        str(pb["short_name"]): [pb.get(x, np.nan) for x in attrs],
    })

    fig = px.line_polar(
        comp.melt("attribute", var_name="player", value_name="value"),
        r="value",
        theta="attribute",
        color="player",
        line_close=True,
        range_r=[0, 100],
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, width="stretch")

elif page == "Evolution":
    st.header("How a player's football DNA has evolved")

    st.markdown(
        "<p class='ewc-hint'>Track a single player across FIFA editions to see how "
        "their overall rating, attributes, archetype and football DNA have shifted over "
        "time. Some editions may be missing for a player if they weren't in the dataset "
        "that season.</p>",
        unsafe_allow_html=True,
    )

    season_counts = players.groupby("name_key")["season_year"].nunique()
    eligible_name_keys = season_counts[season_counts.ge(2)].index

    timeline_candidates = players[players["name_key"].isin(eligible_name_keys)].copy()

    player_lookup = (
        timeline_candidates.sort_values("season_year")
        .groupby("name_key")
        .tail(1)[["name_key", "short_name", "nationality_name"]]
        .copy()
    )

    player_lookup["display_label"] = (
        player_lookup["short_name"].astype(str)
        + " — "
        + player_lookup["nationality_name"].fillna("").astype(str)
    )

    player_lookup = player_lookup.sort_values("display_label")

    default_index = 0
    default_matches = player_lookup[
        player_lookup["short_name"].astype(str).str.contains("Saka", na=False)
    ]

    if not default_matches.empty:
        display_labels = player_lookup["display_label"].tolist()
        target_label = default_matches.iloc[0]["display_label"]

        if target_label in display_labels:
            default_index = display_labels.index(target_label)

    selected_evolution_label = st.selectbox(
        "Choose a player",
        player_lookup["display_label"],
        index=default_index,
    )

    selected_name_key = player_lookup.loc[
        player_lookup["display_label"].eq(selected_evolution_label), "name_key"
    ].iloc[0]

    timeline = (
        players[players["name_key"].eq(selected_name_key)]
        .sort_values(["season_year", "overall"], ascending=[True, False])
        .drop_duplicates("season_year", keep="first")
        .reset_index(drop=True)
    )

    if len(timeline) < 2:
        st.info("Not enough seasons on record to show an evolution timeline for this player.")
    else:
        rating_long = timeline.melt(
            id_vars="season_label",
            value_vars=["overall", "potential"],
            var_name="metric",
            value_name="value",
        )

        fig_rating = px.line(
            rating_long,
            x="season_label",
            y="value",
            color="metric",
            markers=True,
            title="Overall & potential by FIFA edition",
        )

        fig_rating.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_rating, width="stretch")

        attr_cols = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

        attr_long = timeline.melt(
            id_vars="season_label",
            value_vars=[c for c in attr_cols if c in timeline.columns],
            var_name="attribute",
            value_name="value",
        )

        fig_attrs = px.line(
            attr_long,
            x="season_label",
            y="value",
            color="attribute",
            markers=True,
            title="Attribute development",
        )

        fig_attrs.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(fig_attrs, width="stretch")

        timeline_emb_cols = [c for c in timeline.columns if c.startswith("emb_")]

        if timeline_emb_cols and timeline[timeline_emb_cols].notna().all(axis=None):
            baseline_vector = timeline.iloc[0][timeline_emb_cols].to_numpy(float).reshape(1, -1)
            current_vectors = timeline[timeline_emb_cols].to_numpy(float)
            dna_similarity = cosine_similarity(baseline_vector, current_vectors).ravel() * 100
            timeline["dna_similarity_to_debut"] = dna_similarity

            debut_label = timeline.iloc[0]["season_label"]

            fig_dna = px.line(
                timeline,
                x="season_label",
                y="dna_similarity_to_debut",
                markers=True,
                title=f"Football DNA similarity to {debut_label} debut",
                range_y=[0, 100],
            )

            fig_dna.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_dna, width="stretch")

        archetype_journey = []
        previous_archetype = None

        for _, row in timeline.iterrows():
            current_archetype = row.get("archetype_name")

            if pd.isna(current_archetype):
                continue

            if current_archetype != previous_archetype:
                archetype_journey.append((row["season_label"], current_archetype))
                previous_archetype = current_archetype

        if archetype_journey:
            journey_steps_html = []

            for i, (season_label, archetype_name) in enumerate(archetype_journey):
                journey_steps_html.append(
                    '<div class="ewc-journey-step">'
                    f'<div class="ewc-journey-season">{html.escape(str(season_label))}</div>'
                    f'<div class="ewc-journey-archetype">{html.escape(str(archetype_name))}</div>'
                    "</div>"
                )

                if i < len(archetype_journey) - 1:
                    journey_steps_html.append('<div class="ewc-journey-arrow">&darr;</div>')

            st.subheader("Archetype journey")
            st.markdown(
                '<div class="ewc-journey">' + "".join(journey_steps_html) + "</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Closest players by season")
        st.caption(
            "Who this player most resembled in football DNA that same FIFA edition -- "
            "not who they resemble today."
        )

        closest_by_season = []

        for _, row in timeline.iterrows():
            season_pool = players[
                players["season_year"].eq(row["season_year"])
                & players["name_key"].ne(row["name_key"])
            ].dropna(subset=emb_cols)

            if season_pool.empty:
                continue

            Xq = row[emb_cols].to_numpy(float).reshape(1, -1)
            X = season_pool[emb_cols].to_numpy(float)
            season_similarities = cosine_similarity(Xq, X).ravel()
            best_position = season_similarities.argmax()
            best_match = season_pool.iloc[best_position]

            closest_by_season.append({
                "season_label": row["season_label"],
                "name": best_match["short_name"],
                "similarity": season_similarities[best_position],
            })

        if closest_by_season:
            items_html = "".join(
                "<li>"
                f"<strong>{html.escape(str(item['season_label']))}</strong> — "
                f"{html.escape(str(item['name']))} "
                f"<span class='ewc-closest-sim'>({item['similarity'] * 100:.0f}% DNA match)</span>"
                "</li>"
                for item in closest_by_season
            )

            st.markdown(
                f"<div class='ewc-callout'><ul class='ewc-closest-by-season'>{items_html}</ul></div>",
                unsafe_allow_html=True,
            )

        st.subheader("Career snapshots")
        player_cards(timeline, max_cards=len(timeline), key_prefix="evolution")

elif page == "Pathways":
    st.header("Football DNA succession pathways")

    st.markdown(
        "<p class='ewc-hint'>Start from a player and follow the chain: each step finds "
        "the closest football DNA match among today's players who are meaningfully younger "
        "than the step before it -- tracing a generational lineage rather than a single list "
        "of matches.</p>",
        unsafe_allow_html=True,
    )

    pathway_options = latest.sort_values("overall", ascending=False)["display_name"].head(3000)

    control_row = st.columns([2.2, 1])

    with control_row[0]:
        default_pathway_index = 0
        debruyne_matches = pathway_options[
            pathway_options.str.contains("De Bruyne", case=False, na=False)
        ]

        if not debruyne_matches.empty:
            option_list = pathway_options.tolist()
            default_pathway_index = option_list.index(debruyne_matches.iloc[0])

        selected_pathway_player = st.selectbox(
            "Start from",
            pathway_options,
            index=default_pathway_index,
        )

    with control_row[1]:
        pathway_steps = st.slider("Generations", min_value=2, max_value=5, value=4)

    with st.expander("Advanced options"):
        min_age_gap = st.slider(
            "Minimum age gap per step",
            min_value=1,
            max_value=6,
            value=3,
        )

    start_row = latest.loc[
        latest["display_name"].eq(selected_pathway_player)
    ].iloc[0]

    chain_rows = [start_row]
    step_labels = ["Starting point"]
    excluded_name_keys = {start_row["name_key"]}
    current_row = start_row

    for step in range(pathway_steps):
        current_age = pd.to_numeric(
            pd.Series([current_row.get("age")]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(current_age):
            break

        candidate_pool = latest[
            pd.to_numeric(latest["age"], errors="coerce").le(current_age - min_age_gap)
            & ~latest["name_key"].isin(excluded_name_keys)
        ]

        if candidate_pool.empty:
            break

        Xq = current_row[emb_cols].to_numpy(float).reshape(1, -1)
        X = candidate_pool[emb_cols].to_numpy(float)

        similarities = cosine_similarity(Xq, X).ravel()
        best_position = similarities.argmax()

        next_row = candidate_pool.iloc[best_position].copy()
        next_row["similarity"] = similarities[best_position]

        chain_rows.append(next_row)
        step_labels.append(f"Generation {step + 1}")
        excluded_name_keys.add(next_row["name_key"])
        current_row = next_row

    if len(chain_rows) < 2:
        st.warning(
            "Couldn't extend a pathway from this player -- try a smaller minimum age gap "
            "or a different starting player."
        )
    else:
        dna_pathway(chain_rows, step_labels)

        if len(chain_rows) - 1 < pathway_steps:
            st.caption(
                f"Pathway stopped after {len(chain_rows) - 1} generation(s): no further "
                "meaningfully younger DNA match was found. Try a smaller minimum age gap."
            )
        else:
            st.markdown(
                "<div class='ewc-callout'>🧬 Who comes next?</div>",
                unsafe_allow_html=True,
            )

elif page == "DNA Map":
    st.header("Explore football's DNA landscape")

    st.markdown(
        "<p class='ewc-hint'>Filter by position, age, overall, archetype, nationality "
        "or season -- or jump straight to a preset like wonderkids or physical monsters. "
        "Click a point on the map to see who it is.</p>",
        unsafe_allow_html=True,
    )

    season_options = ["Latest season only"] + sorted(
        players["season_label"].dropna().unique().tolist(),
        reverse=True,
    )

    quick_filter = st.selectbox(
        "Quick filter",
        ["None", "Wonderkids", "Elite creators", "Physical monsters", "Playmakers"],
    )

    filter_row_1 = st.columns(3)

    with filter_row_1[0]:
        season_choice = st.selectbox("Season", season_options)

    with filter_row_1[1]:
        position_choice = st.selectbox(
            "Position",
            ["All", "Goalkeeper", "Defender", "Midfielder", "Forward"],
        )

    with filter_row_1[2]:
        archetype_choice = st.selectbox(
            "Archetype",
            ["All"] + sorted(players["archetype_name"].dropna().unique().tolist()),
        )

    filter_row_2 = st.columns(2)

    with filter_row_2[0]:
        age_range = st.slider("Age range", 15, 45, (15, 45))

    with filter_row_2[1]:
        overall_range = st.slider("Overall range", 40, 99, (40, 99))

    with st.expander("More filters"):
        nationality_choice = st.multiselect(
            "Nationality",
            sorted(players["nationality_name"].dropna().unique().tolist()),
        )

    if season_choice == "Latest season only":
        map_df = latest.dropna(subset=["map_x", "map_y"]).copy()
    else:
        map_df = players[
            players["season_label"].eq(season_choice)
        ].dropna(subset=["map_x", "map_y"]).copy()

    if quick_filter == "Wonderkids":
        map_df = map_df[
            pd.to_numeric(map_df["age"], errors="coerce").le(21)
            & pd.to_numeric(map_df["potential"], errors="coerce").ge(82)
        ]
    elif quick_filter == "Elite creators":
        map_df = map_df[pd.to_numeric(map_df["passing"], errors="coerce").ge(82)]
    elif quick_filter == "Physical monsters":
        map_df = map_df[pd.to_numeric(map_df["physic"], errors="coerce").ge(85)]
    elif quick_filter == "Playmakers":
        map_df = map_df[
            pd.to_numeric(map_df["passing"], errors="coerce").ge(78)
            & pd.to_numeric(map_df["dribbling"], errors="coerce").ge(78)
        ]

    if position_choice != "All":
        map_df = map_df[
            map_df["player_positions"].map(broad_position_group).eq(position_choice)
        ]

    if archetype_choice != "All":
        map_df = map_df[map_df["archetype_name"].eq(archetype_choice)]

    map_df = map_df[
        pd.to_numeric(map_df["age"], errors="coerce").between(age_range[0], age_range[1])
        & pd.to_numeric(map_df["overall"], errors="coerce").between(overall_range[0], overall_range[1])
    ]

    if nationality_choice:
        map_df = map_df[map_df["nationality_name"].isin(nationality_choice)]

    map_df = map_df.sort_values("overall", ascending=False)

    if len(map_df) > 500:
        max_points = st.slider(
            "Max players shown",
            500,
            min(10000, len(map_df)),
            min(2500, len(map_df)),
            500,
        )
        map_df = map_df.head(max_points)

    st.caption(f"Showing {len(map_df):,} player-seasons.")

    if map_df.empty:
        st.warning("No players match these filters. Try widening them.")
    else:
        hover_fields = [
            "short_name", "club_name", "nationality_name",
            "player_positions", "overall", "age", "archetype_name",
        ]

        fig = px.scatter(
            map_df,
            x="map_x",
            y="map_y",
            color="archetype_name" if "archetype_name" in map_df else None,
            custom_data=["player_season_id"] + hover_fields,
            title="Football DNA landscape",
            labels={"map_x": "DNA axis 1", "map_y": "DNA axis 2"},
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "%{customdata[2]} · %{customdata[3]}<br>"
                "%{customdata[4]} · Overall %{customdata[5]} · Age %{customdata[6]}<br>"
                "%{customdata[7]}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=700,
        )

        selection_event = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="dna_map_chart",
        )

        selected_points = []

        if selection_event and selection_event.get("selection"):
            selected_points = selection_event["selection"].get("points", [])

        if selected_points:
            selected_ids = [
                point["customdata"][0]
                for point in selected_points
                if point.get("customdata")
            ]

            selected_rows = map_df[map_df["player_season_id"].isin(selected_ids)]

            if not selected_rows.empty:
                st.subheader("Selected")
                player_cards(selected_rows, max_cards=len(selected_rows), key_prefix="dnamap")
        else:
            st.caption("Click a point above to see who it is.")

elif page == "Legend Score":
    st.header("Prototype Legend Style Score")
    st.markdown(
        "<p class='ewc-hint'>This is a narrative ranking, not a prediction model. "
        "It combines current quality, potential, age curve, reputation and World Cup metadata "
        "where available.</p>",
        unsafe_allow_html=True,
    )

    if scores.empty:
        st.info("No legend score file found.")
    else:
        legend_cards = (
            scores.sort_values("legend_style_score", ascending=False)
            .rename(columns={"legend_style_score": "similarity"})
            .assign(similarity=lambda d: d["similarity"] / 100)
        )

        player_cards(legend_cards, max_cards=12, score_label="Legend Score", key_prefix="legend")

        with st.expander("Show full ranking"):
            cols = [
                "short_name", "season_label", "club_name", "nationality_name",
                "overall", "potential", "age", "legend_style_score",
                "archetype_name",
            ]
            st.dataframe(scores[[c for c in cols if c in scores.columns]].head(250), width="stretch")

elif page == "Archetypes":
    st.header("Football DNA archetypes")

    if archetypes.empty:
        st.info("No archetype summary found.")
    else:
        archetype_order = archetypes.sort_values(
            "player_count", ascending=False
        )["archetype_name"].tolist()

        ARCHETYPE_SELECT_KEY = "archetypes_page_select"

        if ARCHETYPE_SELECT_KEY not in st.session_state:
            st.session_state[ARCHETYPE_SELECT_KEY] = archetype_order[0]

        st.markdown(
            f"<p class='ewc-hint'>There are {len(archetype_order)} football DNA archetypes in this "
            "dataset -- every profile in football boils down to one of these. Pick one to explore its "
            "all-time greats, modern examples and young prospects.</p>",
            unsafe_allow_html=True,
        )

        archetype_tile_cols = st.columns(4)

        for i, tile_name in enumerate(archetype_order):
            tile_count = int(
                archetypes.loc[archetypes["archetype_name"].eq(tile_name), "player_count"].iloc[0]
            )

            with archetype_tile_cols[i % 4]:
                if st.button(
                    f"{tile_name} ({tile_count:,})",
                    key=f"archetype_tile_{i}",
                    width="stretch",
                ):
                    st.session_state[ARCHETYPE_SELECT_KEY] = tile_name

        selected_archetype_name = st.selectbox(
            "Or jump directly to an archetype",
            archetype_order,
            key=ARCHETYPE_SELECT_KEY,
        )

        archetype_row = archetypes.loc[
            archetypes["archetype_name"].eq(selected_archetype_name)
        ].iloc[0]

        st.markdown(
            "<div class='ewc-callout'>"
            f"{html.escape(str(archetype_row.get('definition', '')))}"
            "</div>",
            unsafe_allow_html=True,
        )

        key_traits = [
            trait.strip()
            for trait in str(archetype_row.get("key_traits", "")).split(",")
            if trait.strip()
        ]

        if key_traits:
            trait_pills = "".join(
                f'<span class="ewc-pill">{html.escape(trait)}</span>'
                for trait in key_traits
            )
            st.markdown(
                f"<div style='margin-bottom:1rem;'>{trait_pills}</div>",
                unsafe_allow_html=True,
            )

        metrics_grid([
            ("Players", f"{int(archetype_row['player_count']):,}", "Player-seasons in this archetype"),
            ("Avg overall", f"{float(archetype_row['avg_overall']):.1f}", "Across all seasons"),
        ])

        radar_attrs = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
        radar_values = [archetype_row.get(f"avg_{attr}") for attr in radar_attrs]

        if all(pd.notna(value) for value in radar_values):
            radar_df = pd.DataFrame({"attribute": radar_attrs, "value": radar_values})

            fig = px.line_polar(
                radar_df,
                r="value",
                theta="attribute",
                line_close=True,
                range_r=[0, 100],
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )

            st.plotly_chart(fig, width="stretch")
        else:
            st.caption(
                "No attribute radar for this archetype -- goalkeepers use a "
                "different rating set than pace/shooting/passing/dribbling/defending/physic."
            )

        cluster_players = players[
            players["archetype_id"].eq(archetype_row["archetype_id"])
        ].copy()

        max_season = pd.to_numeric(players["season_year"], errors="coerce").max()
        recent_cutoff = max_season - 3

        st.subheader("All-time greats")
        st.caption("The highest peak overall rating ever recorded for this archetype.")

        all_time_examples = (
            cluster_players.sort_values("overall", ascending=False)
            .drop_duplicates("name_key")
            .head(5)
        )

        player_cards(all_time_examples, max_cards=5, key_prefix="archetype_alltime")

        st.subheader("Modern examples")
        st.caption(f"Peak season-{int(recent_cutoff)} onward, by overall rating.")

        best_pool = cluster_players[
            pd.to_numeric(cluster_players["season_year"], errors="coerce").ge(recent_cutoff)
        ]

        best_examples = (
            best_pool.sort_values("overall", ascending=False)
            .drop_duplicates("name_key")
            .head(5)
        )

        player_cards(best_examples, max_cards=5, key_prefix="archetype_best")

        st.subheader("Young prospects")
        st.caption(f"FIFA {int(max_season) - 2000}, age 22 or under, by potential.")

        emerging_pool = cluster_players[
            pd.to_numeric(cluster_players["season_year"], errors="coerce").eq(max_season)
            & pd.to_numeric(cluster_players["age"], errors="coerce").le(22)
        ]

        emerging_examples = (
            emerging_pool.sort_values("potential", ascending=False)
            .drop_duplicates("name_key")
            .head(5)
        )

        if emerging_examples.empty:
            st.info("No emerging (age 22 or under) examples found for this archetype in the latest season.")
        else:
            player_cards(emerging_examples, max_cards=5, key_prefix="archetype_emerging")

        with st.expander("Show full archetype table"):
            st.dataframe(archetypes, width="stretch")

else:
    st.header("How this works")
    st.markdown(
        """
        <div class="ewc-section-card">
        <p>Every player-season in football history gets compressed into a football DNA fingerprint -- a signature built from their attributes and playing style. Players who land in a similar region of that DNA space tend to play a similar role on the pitch, however different their nationality, era or reputation.</p>
        <p>That's what lets us ask questions like:</p>
        <ul>
        <li>Who carries Lionel Messi's football DNA today?</li>
        <li>Which modern player resembles a prime Luka Modri&#263;?</li>
        <li>Which archetypes have quietly gone extinct?</li>
        <li>What might football look like in 2035?</li>
        </ul>
        <h3>What this isn't</h3>
        <p>This is a resemblance engine, not a crystal ball. A high DNA match means two players occupy a similar footballing profile -- it isn't a prediction that one will become as good as the other. Coverage also isn't even: detailed match-performance stats exist for only a minority of player-seasons, so older or less-tracked players lean more on their FIFA ratings alone.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )