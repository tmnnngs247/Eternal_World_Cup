from __future__ import annotations

from pathlib import Path
import html
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from components import dna_pathway, hero, load_css, metrics_grid, player_cards

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="The Eternal World Cup", page_icon="⚽", layout="wide")
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

    for _, columns, strong_label, similar_label, more_label, less_label in PROFILE_DESCRIPTORS:
        query_value = descriptor_value(query, columns)
        candidate_value = descriptor_value(candidate, columns)

        if pd.isna(query_value) or pd.isna(candidate_value):
            continue

        diff = candidate_value - query_value
        abs_diff = abs(diff)

        if abs_diff <= SHARE_DIFF_THRESHOLD and min(query_value, candidate_value) >= SHARE_MIN_VALUE:
            average_value = (query_value + candidate_value) / 2
            label = strong_label if average_value >= STRONG_VALUE_THRESHOLD else similar_label
            share_candidates.append((abs_diff, label))
        elif abs_diff >= DIFFERENCE_MIN_THRESHOLD:
            difference_candidates.append((abs_diff, more_label if diff > 0 else less_label))

    share_candidates.sort(key=lambda item: item[0])
    difference_candidates.sort(key=lambda item: -item[0])

    shares = [label for _, label in share_candidates[:4]]
    differences = [label for _, label in difference_candidates[:3]]

    query_age = pd.to_numeric(query.get("age"), errors="coerce")
    candidate_age = pd.to_numeric(candidate.get("age"), errors="coerce")

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

    return {"shares": shares, "differences": differences}


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

    for _, row in res.iterrows():
        comparison = build_profile_comparison(row, query)
        shares_column.append(comparison["shares"])
        differences_column.append(comparison["differences"])

        similarity_value = pd.to_numeric(
            pd.Series([row.get("similarity")]),
            errors="coerce",
        ).iloc[0]

        score_column.append(
            successor_score(float(similarity_value), query, row)
            if pd.notna(similarity_value)
            else float("nan")
        )

    res["shares"] = shares_column
    res["differences"] = differences_column
    res["successor_score"] = score_column

    return res


players = load_players()
archetypes = load_archetypes()
scores = load_scores()

emb_cols = [c for c in players.columns if c.startswith("emb_")]
if not emb_cols:
    st.error("No embedding columns found in data/processed/app_players.csv. Re-run the pipeline.")
    st.stop()

with st.sidebar:
    st.markdown("## ⚽ Eternal World Cup")
    st.caption("Football DNA, decoded.")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Successor Finder", "Compare Players", "Evolution", "Pathways", "DNA Map", "Legend Score", "Archetypes", "Method"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.success("Autoencoder embeddings live")
    st.caption("Trained on FIFA attributes + FBRef performance data across 8 seasons.")

hero()
metrics_grid([
    ("Player-seasons", f"{len(players):,}", "Historical + current records"),
    ("DNA dimensions", str(len(emb_cols)), "Compressed profile space"),
    ("Archetypes", f"{players['archetype_id'].nunique() if 'archetype_id' in players.columns else 0}", "Profile clusters"),
    ("Latest player pool", f"{players['name_key'].nunique():,}", "Unique players"),
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

if page == "Successor Finder":
    st.header("Find a player's closest football DNA matches")

    reference_options = players.sort_values(
        ["overall", "season_year"],
        ascending=[False, False],
    )["display_name"].head(5000)

    REFERENCE_KEY = "successor_reference_select"
    SEARCH_MODE_KEY = "successor_search_mode_select"

    def jump_to_player(name_substring: str, mode: str) -> None:
        matches = reference_options[
            reference_options.str.contains(name_substring, case=False, na=False, regex=False)
        ]

        if not matches.empty:
            st.session_state[REFERENCE_KEY] = matches.iloc[0]

        st.session_state[SEARCH_MODE_KEY] = mode

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
        "<div class='ewc-callout'>Pick a reference player-season below, or use one of the "
        "questions above. Choose whether you want young successors, current replacements, "
        "historical lookalikes or an unrestricted DNA search.</div>",
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

    st.markdown(
        "<div class='ewc-callout'>🧬 "
        f"<strong>{html.escape(str(query['short_name']))}</strong>'s football DNA profile: "
        f"<strong>{html.escape(str(query_archetype))}</strong> "
        f"&middot; Overall {html.escape(str(query.get('overall', '')))} "
        f"&middot; Age {html.escape(str(query.get('age', '')))}"
        "</div>",
        unsafe_allow_html=True,
    )

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
        Xq = query[emb_cols].to_numpy(float).reshape(1, -1)
        X = pool[emb_cols].to_numpy(float)

        res = pool.copy()
        res["similarity"] = cosine_similarity(Xq, X).ravel()

        res = (
            res.sort_values("similarity", ascending=False)
            .head(n)
        )

        res = add_similarity_reasons(res, query)

        st.caption(
            f"Showing {search_mode.lower()} using "
            f"{position_match.lower()} filtering."
        )

        player_cards(res, max_cards=n)

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

    c1, c2 = st.columns(2)

    with c1:
        a = st.selectbox("Player A", players["display_name"].sort_values(), index=0)

    with c2:
        b = st.selectbox("Player B", players["display_name"].sort_values(), index=1)

    pa = players.loc[players["display_name"].eq(a)].iloc[0]
    pb = players.loc[players["display_name"].eq(b)].iloc[0]

    sim = cosine_similarity(
        pa[emb_cols].to_numpy(float).reshape(1, -1),
        pb[emb_cols].to_numpy(float).reshape(1, -1),
    )[0, 0]

    st.metric("Football DNA Match", f"{sim * 100:.1f}%")

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
        "<div class='ewc-callout'>Track a single player across FIFA editions to see how "
        "their overall rating, attributes, archetype and football DNA have shifted over "
        "time. Some editions may be missing for a player if they weren't in the dataset "
        "that season.</div>",
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
                archetype_journey.append(f"{row['season_label']}: {current_archetype}")
                previous_archetype = current_archetype

        if archetype_journey:
            st.markdown(
                "<div class='ewc-callout'>🧬 Archetype journey: "
                + " &rarr; ".join(html.escape(step) for step in archetype_journey)
                + "</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Career snapshots")
        player_cards(timeline, max_cards=len(timeline))

elif page == "Pathways":
    st.header("Football DNA succession pathways")

    st.markdown(
        "<div class='ewc-callout'>Start from a player and follow the chain: each step finds "
        "the closest football DNA match among today's players who are meaningfully younger "
        "than the step before it -- tracing a generational lineage rather than a single list "
        "of matches.</div>",
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
    st.header("Football DNA map")

    plot_df = latest.dropna(subset=[emb_cols[0], emb_cols[1]]).copy()
    max_points = st.slider("Number of players", 500, min(10000, len(plot_df)), 2500, 500)
    plot_df = plot_df.sort_values("overall", ascending=False).head(max_points)

    fig = px.scatter(
        plot_df,
        x=emb_cols[0],
        y=emb_cols[1],
        color="archetype_name" if "archetype_name" in plot_df else None,
        hover_data=[
            "short_name", "club_name", "nationality_name",
            "overall", "age", "player_positions",
        ],
        title="Latest player-seasons projected onto first two DNA dimensions",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=700,
    )

    st.plotly_chart(fig, width="stretch")

elif page == "Legend Score":
    st.header("Prototype Legend Style Score")
    st.markdown(
        "<div class='ewc-callout'>This is a narrative ranking, not a prediction model. "
        "It combines current quality, potential, age curve, reputation and World Cup metadata "
        "where available.</div>",
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

        player_cards(legend_cards, max_cards=12, score_label="Legend Score")

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

        selected_archetype_name = st.selectbox("Choose an archetype", archetype_order)

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

        st.subheader("Best modern examples")
        st.caption(f"Peak season-{int(recent_cutoff)} onward, by overall rating.")

        best_pool = cluster_players[
            pd.to_numeric(cluster_players["season_year"], errors="coerce").ge(recent_cutoff)
        ]

        best_examples = (
            best_pool.sort_values("overall", ascending=False)
            .drop_duplicates("name_key")
            .head(5)
        )

        player_cards(best_examples, max_cards=5)

        st.subheader("Emerging examples")
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
            player_cards(emerging_examples, max_cards=5)

        with st.expander("Show full archetype table"):
            st.dataframe(archetypes, width="stretch")

else:
    st.header("Method & caveats")
    st.markdown(
        """
        <div class="ewc-section-card">
        <h3>What the model does</h3>
        <p>The football-DNA embedding is produced by a trained autoencoder (a bottlenecked neural network) over standardised FIFA attribute ratings and, where available, FBRef per-90 performance stats such as xG, progressive passes and tackles. Similarity is calculated with cosine similarity in that embedding space.</p>
        <h3>What it does not do yet</h3>
        <p>FBRef performance data currently covers roughly 10% of player-seasons, concentrated in FIFA 18-25 for players FBRef tracks; the rest rely on FIFA attributes alone. This is a similarity model, not a predictive one: it does not forecast future performance.</p>
        <h3>Next upgrades</h3>
        <p>Improve player identity matching to raise FBRef coverage, broaden flag and nationality coverage, and explore predictive modelling of future performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )