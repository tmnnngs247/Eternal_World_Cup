import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="The Eternal World Cup",
    page_icon="⚽",
    layout="wide",
)

DATA_DIR = "data"

@st.cache_data
def load_data():
    embeddings = pd.read_csv(f"{DATA_DIR}/player_dna_embeddings_fifa_fbref_v2.csv")
    scores = pd.read_csv(f"{DATA_DIR}/current_players_legend_style_score.csv")
    sims = pd.read_csv(f"{DATA_DIR}/modern_successors_similarity_examples.csv")
    clusters = pd.read_csv(f"{DATA_DIR}/archetype_cluster_summary.csv")
    world_cup = pd.read_csv(f"{DATA_DIR}/world_cup_appearances_summary.csv")
    dna_map = pd.read_csv(f"{DATA_DIR}/player_dna_map_2d.csv")
    return embeddings, scores, sims, clusters, world_cup, dna_map

embeddings, scores, sims, clusters, world_cup, dna_map = load_data()
emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]

# Basic cleaning for display
for df in [embeddings, scores, sims, dna_map]:
    for col in ["short_name", "long_name", "club_name", "nationality_name", "player_positions", "archetype_label", "season_label"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")


def player_label(row):
    return f"{row['short_name']} — {row['season_label']} — {row.get('club_name', 'Unknown')}"


def cosine_similarity_matrix(query_vec, matrix):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m @ q


def find_similar(query_index, candidate_df, top_n=10):
    query_vec = embeddings.loc[query_index, emb_cols].astype(float).values
    matrix = candidate_df[emb_cols].astype(float).values
    similarities = cosine_similarity_matrix(query_vec, matrix)
    out = candidate_df.copy()
    out["similarity"] = similarities
    query_name = embeddings.loc[query_index, "short_name"]
    query_season = embeddings.loc[query_index, "season_label"]
    out = out[~((out["short_name"] == query_name) & (out["season_label"] == query_season))]
    out = out.sort_values("similarity", ascending=False).head(top_n)
    display_cols = [
        "similarity", "season_label", "short_name", "long_name", "age",
        "player_positions", "overall", "potential", "club_name",
        "nationality_name", "archetype_label"
    ]
    display_cols = [c for c in display_cols if c in out.columns]
    out = out[display_cols].copy()
    out["similarity"] = (out["similarity"] * 100).round(1)
    return out


st.markdown("""
# ⚽ The Eternal World Cup
**Using neural-network football DNA to find the modern successors to football's greats.**

This app uses a neural autoencoder trained on FIFA-style attributes and FBRef-style performance data. Each player-season is compressed into a 32-dimensional embedding, allowing us to compare players by football profile rather than name recognition alone.
""")

metric_cols = st.columns(4)
metric_cols[0].metric("Player-seasons", f"{len(embeddings):,}")
metric_cols[1].metric("Embedding dimensions", len(emb_cols))
metric_cols[2].metric("Archetypes", clusters["archetype_cluster"].nunique())
metric_cols[3].metric("Current score pool", f"{len(scores):,}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Find successors", "Compare two players", "Legend score", "Football DNA map", "Method & caveats"
])

with tab1:
    st.subheader("Find a player's closest football DNA matches")
    st.write("Choose any player-season, then decide whether to search across all seasons or only the current player pool.")

    labels = embeddings.apply(player_label, axis=1)
    label_to_idx = dict(zip(labels, embeddings.index))

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        selected = st.selectbox("Choose a reference player-season", labels, index=0)
    with c2:
        pool = st.selectbox("Comparison pool", ["Current players only", "All seasons"])
    with c3:
        top_n = st.slider("Number of matches", 5, 25, 10)

    query_idx = label_to_idx[selected]
    if pool == "Current players only":
        candidate_df = embeddings[embeddings["season_label"].isin(["FIFA 25", "FIFA 26"])].copy()
    else:
        candidate_df = embeddings.copy()

    query_row = embeddings.loc[query_idx]
    st.markdown(f"### Reference: {query_row['short_name']} ({query_row['season_label']})")
    st.caption(f"{query_row.get('club_name', 'Unknown')} · {query_row.get('nationality_name', 'Unknown')} · {query_row.get('player_positions', 'Unknown')} · Archetype: {query_row.get('archetype_label', 'Unknown')}")

    similar = find_similar(query_idx, candidate_df, top_n=top_n)
    st.dataframe(similar, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Compare two player-seasons")
    labels = embeddings.apply(player_label, axis=1)
    label_to_idx = dict(zip(labels, embeddings.index))
    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("Player A", labels, index=0, key="p1")
    with c2:
        default_idx = min(1, len(labels) - 1)
        p2 = st.selectbox("Player B", labels, index=default_idx, key="p2")

    i1, i2 = label_to_idx[p1], label_to_idx[p2]
    v1 = embeddings.loc[i1, emb_cols].astype(float).values
    v2 = embeddings.loc[i2, emb_cols].astype(float).values
    sim_score = float(cosine_similarity_matrix(v1, v2.reshape(1, -1))[0] * 100)

    a, b, c = st.columns(3)
    a.metric("Football DNA similarity", f"{sim_score:.1f}%")
    b.metric("Player A overall", int(embeddings.loc[i1, "overall"]) if pd.notna(embeddings.loc[i1, "overall"]) else "—")
    c.metric("Player B overall", int(embeddings.loc[i2, "overall"]) if pd.notna(embeddings.loc[i2, "overall"]) else "—")

    compare_cols = ["short_name", "season_label", "age", "player_positions", "overall", "potential", "club_name", "league_name", "nationality_name", "archetype_label"]
    st.dataframe(embeddings.loc[[i1, i2], compare_cols], use_container_width=True, hide_index=True)

    emb_compare = pd.DataFrame({
        "dimension": emb_cols,
        embeddings.loc[i1, "short_name"]: v1,
        embeddings.loc[i2, "short_name"]: v2,
    })
    fig = px.line(emb_compare, x="dimension", y=[embeddings.loc[i1, "short_name"], embeddings.loc[i2, "short_name"]], markers=True,
                  title="Embedding profile comparison")
    fig.update_layout(legend_title_text="Player", xaxis_title="Football DNA dimension", yaxis_title="Embedding value")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Current players ranked by Legend Style Score")
    st.write("This is a prototype score based on how strongly a current player resembles historically elite football DNA profiles. Treat it as a storytelling device, not a scouting truth.")

    c1, c2, c3 = st.columns(3)
    with c1:
        nationality = st.selectbox("Filter nationality", ["All"] + sorted(scores["nationality_name"].dropna().unique().tolist()))
    with c2:
        position_text = st.text_input("Position contains", "")
    with c3:
        max_age = st.slider("Maximum age", 16, 45, 28)

    score_df = scores.copy()
    if nationality != "All":
        score_df = score_df[score_df["nationality_name"] == nationality]
    if position_text.strip():
        score_df = score_df[score_df["player_positions"].str.contains(position_text.strip(), case=False, na=False)]
    score_df = score_df[pd.to_numeric(score_df["age"], errors="coerce") <= max_age]
    score_df = score_df.sort_values("legend_style_score", ascending=False)

    fig = px.bar(score_df.head(20), x="legend_style_score", y="short_name", orientation="h",
                 hover_data=["age", "club_name", "nationality_name", "player_positions", "archetype_label"],
                 title="Top Legend Style Score players")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Legend Style Score", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(score_df.head(100), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Football DNA map")
    st.write("A 2D projection of the 32-dimensional neural embedding. Nearby dots have similar player profiles.")

    c1, c2, c3 = st.columns(3)
    with c1:
        map_season = st.selectbox("Season", ["All"] + sorted(dna_map["season_label"].dropna().unique().tolist()), index=0)
    with c2:
        map_arch = st.selectbox("Archetype", ["All"] + sorted(dna_map["archetype_label"].dropna().unique().tolist()), index=0)
    with c3:
        min_overall = st.slider("Minimum overall", 40, 95, 75)

    map_df = dna_map.copy()
    map_df["overall"] = pd.to_numeric(map_df["overall"], errors="coerce")
    if map_season != "All":
        map_df = map_df[map_df["season_label"] == map_season]
    if map_arch != "All":
        map_df = map_df[map_df["archetype_label"] == map_arch]
    map_df = map_df[map_df["overall"] >= min_overall]
    map_df = map_df.head(8000)

    fig = px.scatter(
        map_df, x="x", y="y", color="archetype_label",
        hover_name="short_name",
        hover_data=["season_label", "age", "overall", "potential", "club_name", "nationality_name", "player_positions"],
        title="Football DNA space"
    )
    fig.update_layout(xaxis_title="DNA component 1", yaxis_title="DNA component 2", legend_title_text="Archetype")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Method & caveats")
    st.markdown("""
    **What the model does**

    - Combines FIFA-style player attributes with FBRef-style performance data.
    - Trains an autoencoder to compress each player-season into a 32-dimensional representation.
    - Uses cosine similarity to find players with similar football DNA.
    - Clusters embeddings into broad archetypes.
    - Creates a prototype Legend Style Score for current players.

    **Important caveats**

    - The app compares player profiles, not career achievements.
    - Similarity does not mean equal quality.
    - FIFA 26 player records are useful for current-player comparisons, but true 25/26 FBRef production should be added when available.
    - Player-name matching across datasets can introduce noise.
    - The Legend Style Score should be framed as exploratory and narrative-led, not definitive recruitment evidence.

    **Good portfolio framing**

    This project demonstrates representation learning, football domain knowledge, player similarity, talent identification, data storytelling and interactive deployment.
    """)

    st.subheader("Archetype cluster summary")
    st.dataframe(clusters, use_container_width=True, hide_index=True)

    st.subheader("World Cup appearances context")
    st.dataframe(world_cup.head(100), use_container_width=True, hide_index=True)
