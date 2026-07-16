from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from components import hero, load_css, metrics_grid, player_cards

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


def add_similarity_reasons(res: pd.DataFrame, query: pd.Series) -> pd.DataFrame:
    key_attrs = [
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physic",
    ]

    def explain(row: pd.Series) -> pd.Series:
        comparisons = []

        for attr in key_attrs:
            if attr not in row.index or attr not in query.index:
                continue

            row_val = pd.to_numeric(row[attr], errors="coerce")
            query_val = pd.to_numeric(query[attr], errors="coerce")

            if pd.notna(row_val) and pd.notna(query_val):
                comparisons.append({
                    "attribute": attr,
                    "difference": abs(float(row_val) - float(query_val)),
                    "signed_difference": float(row_val) - float(query_val),
                })

        if not comparisons:
            return pd.Series({
                "why_similar": "Similar overall football DNA profile.",
                "main_differences": "No comparable headline attributes available.",
            })

        comparisons = sorted(comparisons, key=lambda x: x["difference"])

        closest = comparisons[:3]
        furthest = sorted(
            comparisons,
            key=lambda x: x["difference"],
            reverse=True,
        )[:2]

        similar_text = ", ".join(
            item["attribute"].replace("_", " ").title()
            for item in closest
        )

        difference_parts = []

        for item in furthest:
            attr = item["attribute"].replace("_", " ").title()
            signed_diff = item["signed_difference"]

            if signed_diff > 0:
                difference_parts.append(f"more {attr.lower()}")
            elif signed_diff < 0:
                difference_parts.append(f"less {attr.lower()}")
            else:
                difference_parts.append(f"similar {attr.lower()}")

        return pd.Series({
            "why_similar": f"Closest shared traits: {similar_text}.",
            "main_differences": "Main differences: " + " and ".join(difference_parts) + ".",
        })

    explanations = res.apply(explain, axis=1)

    res = res.copy()
    res["why_similar"] = explanations["why_similar"]
    res["main_differences"] = explanations["main_differences"]

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
        ["Successor Finder", "Compare Players", "DNA Map", "Legend Score", "Archetypes", "Method"],
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

    st.markdown(
        "<div class='ewc-callout'>Choose whether you want young successors, "
        "current replacements, historical lookalikes or an unrestricted similarity search.</div>",
        unsafe_allow_html=True,
    )

    control_row_1 = st.columns([2.3, 1.3])

    with control_row_1[0]:
        selected = st.selectbox(
            "Reference player-season",
            players.sort_values(
                ["overall", "season_year"],
                ascending=[False, False],
            )["display_name"].head(5000),
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
            cols = [
                "short_name",
                "season_label",
                "club_name",
                "nationality_name",
                "overall",
                "age",
                "player_positions",
                "similarity",
                "why_similar",
                "main_differences",
                "archetype_name",
            ]

            st.dataframe(
                res[[column for column in cols if column in res.columns]],
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

    st.metric("Football DNA similarity", f"{sim * 100:.1f}%")

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

        player_cards(legend_cards, max_cards=12)

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