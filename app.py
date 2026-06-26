import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="The Eternal World Cup",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --ewc-bg: #070A13;
        --ewc-panel: #101827;
        --ewc-panel-2: #151F32;
        --ewc-text: #F8FAFC;
        --ewc-muted: #94A3B8;
        --ewc-border: rgba(148, 163, 184, 0.22);
        --ewc-accent: #7DD3FC;
        --ewc-accent-2: #F472B6;
        --ewc-gold: #FACC15;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(125, 211, 252, 0.13), transparent 28rem),
            radial-gradient(circle at top right, rgba(244, 114, 182, 0.10), transparent 28rem),
            var(--ewc-bg);
        color: var(--ewc-text);
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.92);
        border-right: 1px solid var(--ewc-border);
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    .hero {
        padding: 2rem 2rem 1.6rem 2rem;
        border: 1px solid var(--ewc-border);
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(125, 211, 252, 0.16), rgba(244, 114, 182, 0.08)),
            rgba(15, 23, 42, 0.72);
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: clamp(2.2rem, 5vw, 4.6rem);
        line-height: 0.95;
        margin-bottom: 0.75rem;
    }

    .hero p {
        max-width: 820px;
        color: var(--ewc-muted);
        font-size: 1.08rem;
        line-height: 1.65;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
    }

    .pill {
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        border: 1px solid var(--ewc-border);
        background: rgba(255, 255, 255, 0.04);
        color: var(--ewc-text);
        font-size: 0.88rem;
    }

    .stat-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid var(--ewc-border);
        background: rgba(15, 23, 42, 0.68);
        min-height: 112px;
    }

    .stat-label {
        color: var(--ewc-muted);
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .stat-note {
        margin-top: 0.4rem;
        color: var(--ewc-muted);
        font-size: 0.82rem;
    }

    .section-card {
        border: 1px solid var(--ewc-border);
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.68);
        padding: 1.1rem 1.25rem;
        margin: 0.6rem 0 1rem 0;
    }

    .player-card {
        border: 1px solid var(--ewc-border);
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.025));
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .player-name {
        font-weight: 800;
        font-size: 1.1rem;
        margin-bottom: 0.15rem;
    }

    .player-meta {
        color: var(--ewc-muted);
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: rgba(125, 211, 252, 0.16);
        color: #BAE6FD;
        border: 1px solid rgba(125, 211, 252, 0.34);
        font-weight: 700;
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }

    .callout {
        border-left: 4px solid var(--ewc-accent);
        background: rgba(125, 211, 252, 0.08);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        color: var(--ewc-text);
        margin: 0.6rem 0 1rem 0;
    }

    .small-muted {
        color: var(--ewc-muted);
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid var(--ewc-border);
        border-radius: 18px;
        padding: 1rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 999px;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(125, 211, 252, 0.20) !important;
        color: #E0F2FE !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data loading
# -----------------------------
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

for df in [embeddings, scores, sims, dna_map]:
    for col in [
        "short_name", "long_name", "club_name", "league_name", "nationality_name",
        "player_positions", "archetype_label", "season_label"
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

for df in [embeddings, scores, dna_map]:
    for col in ["age", "overall", "potential"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

if "legend_style_score" in scores.columns:
    scores["legend_style_score"] = pd.to_numeric(scores["legend_style_score"], errors="coerce")

# -----------------------------
# Helper functions
# -----------------------------
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


def render_stat(label, value, note=""):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_player_card(rank, row, score_col="similarity"):
    score_value = row.get(score_col, np.nan)
    if pd.notna(score_value):
        score_text = f"{score_value:.1f}%" if score_col == "similarity" else f"{score_value:.1f}"
    else:
        score_text = "—"
    st.markdown(
        f"""
        <div class="player-card">
            <div class="small-muted">#{rank}</div>
            <div class="player-name">{row.get('short_name', 'Unknown')}</div>
            <div class="player-meta">
                {row.get('season_label', 'Unknown')} · {row.get('club_name', 'Unknown')} · {row.get('nationality_name', 'Unknown')}<br>
                {row.get('player_positions', 'Unknown')} · Overall {row.get('overall', '—')} · {row.get('archetype_label', 'Unknown')}
            </div>
            <div class="score-badge">{score_col.replace('_', ' ').title()}: {score_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        margin=dict(l=20, r=20, t=55, b=35),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## ⚽ Eternal World Cup")
st.sidebar.markdown(
    "Neural-network football DNA for player similarity, successor finding and World Cup storytelling."
)
st.sidebar.divider()
st.sidebar.markdown("**Project status**")
st.sidebar.success("Phase 2: polished public app")
st.sidebar.caption("Next data upgrade: fresh 25/26 FBRef performance data.")
st.sidebar.divider()
st.sidebar.markdown("**How to read this**")
st.sidebar.caption(
    "Similarity = profile resemblance, not guaranteed quality. Legend Style Score is exploratory and narrative-led."
)

# -----------------------------
# Main app
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>The Eternal World Cup</h1>
        <p>
        Football refuses to say goodbye to its greats. This app uses neural-network player embeddings — a form of
        football DNA — to explore which modern players resemble past stars, which profiles cluster together, and who
        might become the players we are still talking about decades from now.
        </p>
        <div class="pill-row">
            <span class="pill">Neural embeddings</span>
            <span class="pill">Player similarity</span>
            <span class="pill">World Cup storytelling</span>
            <span class="pill">Talent ID lens</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_stat("Player-seasons", f"{len(embeddings):,}", "Historical + current records")
with m2:
    render_stat("DNA dimensions", f"{len(emb_cols)}", "Compressed neural profile")
with m3:
    render_stat("Archetypes", f"{clusters['archetype_cluster'].nunique()}", "Profile clusters")
with m4:
    render_stat("Current score pool", f"{len(scores):,}", "Prototype ranked players")

st.write("")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Successor finder", "Compare players", "Legend score", "DNA map", "Archetypes", "Method"
])

# -----------------------------
# Tab 1: successor finder
# -----------------------------
with tab1:
    st.markdown("### Find a player's closest football DNA matches")
    st.markdown(
        "<div class='callout'>Pick a reference player-season and search for modern successors or historical lookalikes. The output is based on neural embedding similarity.</div>",
        unsafe_allow_html=True,
    )

    labels = embeddings.apply(player_label, axis=1)
    label_to_idx = dict(zip(labels, embeddings.index))

    # Make a sensible default if Messi is present
    default_idx = 0
    messi_matches = [i for i, x in enumerate(labels) if "Messi" in x and ("FIFA 22" in x or "FIFA 21" in x)]
    if messi_matches:
        default_idx = messi_matches[-1]

    c1, c2, c3 = st.columns([2.3, 1, 1])
    with c1:
        selected = st.selectbox("Reference player-season", labels, index=default_idx)
    with c2:
        pool = st.selectbox("Comparison pool", ["Current players only", "All seasons", "Same archetype"])
    with c3:
        top_n = st.slider("Matches", 5, 20, 8)

    query_idx = label_to_idx[selected]
    query_row = embeddings.loc[query_idx]

    if pool == "Current players only":
        candidate_df = embeddings[embeddings["season_label"].isin(["FIFA 25", "FIFA 26"])].copy()
    elif pool == "Same archetype":
        candidate_df = embeddings[embeddings["archetype_label"] == query_row["archetype_label"]].copy()
    else:
        candidate_df = embeddings.copy()

    st.markdown(
        f"""
        <div class="section-card">
            <div class="small-muted">Reference profile</div>
            <h3>{query_row['short_name']} · {query_row['season_label']}</h3>
            <p class="small-muted">
                {query_row.get('club_name', 'Unknown')} · {query_row.get('nationality_name', 'Unknown')} ·
                {query_row.get('player_positions', 'Unknown')} · Overall {query_row.get('overall', '—')} ·
                {query_row.get('archetype_label', 'Unknown')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    similar = find_similar(query_idx, candidate_df, top_n=top_n)

    card_cols = st.columns(4)
    for i, (_, row) in enumerate(similar.head(4).iterrows(), start=1):
        with card_cols[i - 1]:
            render_player_card(i, row, "similarity")

    st.markdown("#### Full similarity table")
    st.dataframe(similar, use_container_width=True, hide_index=True)

# -----------------------------
# Tab 2: compare players
# -----------------------------
with tab2:
    st.markdown("### Compare two player-seasons")
    labels = embeddings.apply(player_label, axis=1)
    label_to_idx = dict(zip(labels, embeddings.index))

    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("Player A", labels, index=0, key="p1")
    with c2:
        p2 = st.selectbox("Player B", labels, index=min(1, len(labels) - 1), key="p2")

    i1, i2 = label_to_idx[p1], label_to_idx[p2]
    v1 = embeddings.loc[i1, emb_cols].astype(float).values
    v2 = embeddings.loc[i2, emb_cols].astype(float).values
    sim_score = float(cosine_similarity_matrix(v1, v2.reshape(1, -1))[0] * 100)

    a, b, c = st.columns(3)
    a.metric("Football DNA similarity", f"{sim_score:.1f}%")
    b.metric(f"{embeddings.loc[i1, 'short_name']} overall", "—" if pd.isna(embeddings.loc[i1, "overall"]) else f"{int(embeddings.loc[i1, 'overall'])}")
    c.metric(f"{embeddings.loc[i2, 'short_name']} overall", "—" if pd.isna(embeddings.loc[i2, "overall"]) else f"{int(embeddings.loc[i2, 'overall'])}")

    compare_cols = ["short_name", "season_label", "age", "player_positions", "overall", "potential", "club_name", "league_name", "nationality_name", "archetype_label"]
    st.dataframe(embeddings.loc[[i1, i2], compare_cols], use_container_width=True, hide_index=True)

    emb_compare = pd.DataFrame({
        "dimension": emb_cols,
        embeddings.loc[i1, "short_name"]: v1,
        embeddings.loc[i2, "short_name"]: v2,
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=emb_compare["dimension"], y=emb_compare[embeddings.loc[i1, "short_name"]], mode="lines+markers", name=embeddings.loc[i1, "short_name"]))
    fig.add_trace(go.Scatter(x=emb_compare["dimension"], y=emb_compare[embeddings.loc[i2, "short_name"]], mode="lines+markers", name=embeddings.loc[i2, "short_name"]))
    fig.update_layout(title="Embedding profile comparison", xaxis_title="Football DNA dimension", yaxis_title="Embedding value")
    st.plotly_chart(plotly_layout(fig, height=430), use_container_width=True)

# -----------------------------
# Tab 3: legend score
# -----------------------------
with tab3:
    st.markdown("### Current players ranked by Legend Style Score")
    st.markdown(
        "<div class='callout'>This prototype score measures resemblance to historically elite football-DNA profiles. It is useful for storytelling and hypothesis generation, not definitive scouting.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        nationality = st.selectbox("Nationality", ["All"] + sorted(scores["nationality_name"].dropna().unique().tolist()))
    with c2:
        position_text = st.text_input("Position contains", placeholder="e.g. RW, CM, ST")
    with c3:
        max_age = st.slider("Maximum age", 16, 45, 28)

    score_df = scores.copy()
    if nationality != "All":
        score_df = score_df[score_df["nationality_name"] == nationality]
    if position_text.strip():
        score_df = score_df[score_df["player_positions"].str.contains(position_text.strip(), case=False, na=False)]
    score_df = score_df[pd.to_numeric(score_df["age"], errors="coerce") <= max_age]
    score_df = score_df.sort_values("legend_style_score", ascending=False)

    top_cards = st.columns(3)
    for i, (_, row) in enumerate(score_df.head(3).iterrows(), start=1):
        with top_cards[i - 1]:
            render_player_card(i, row, "legend_style_score")

    fig = px.bar(
        score_df.head(20),
        x="legend_style_score",
        y="short_name",
        orientation="h",
        hover_data=["age", "club_name", "nationality_name", "player_positions", "archetype_label"],
        title="Top Legend Style Score players",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Legend Style Score", yaxis_title="")
    st.plotly_chart(plotly_layout(fig, height=620), use_container_width=True)
    st.dataframe(score_df.head(100), use_container_width=True, hide_index=True)

# -----------------------------
# Tab 4: map
# -----------------------------
with tab4:
    st.markdown("### Football DNA map")
    st.write("A 2D projection of the neural embedding. Nearby dots have similar player profiles.")

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
        map_df,
        x="x",
        y="y",
        color="archetype_label",
        hover_name="short_name",
        hover_data=["season_label", "age", "overall", "potential", "club_name", "nationality_name", "player_positions"],
        title="Football DNA space",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.72, line=dict(width=0)))
    fig.update_layout(xaxis_title="DNA component 1", yaxis_title="DNA component 2", legend_title_text="Archetype")
    st.plotly_chart(plotly_layout(fig, height=720), use_container_width=True)

# -----------------------------
# Tab 5: archetypes
# -----------------------------
with tab5:
    st.markdown("### Archetype clusters")
    st.write("These clusters summarise broad profile groups learned from the embedding space.")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.bar(
            clusters.sort_values("players", ascending=True),
            x="players",
            y="archetype_label",
            orientation="h",
            title="Players per archetype",
            hover_data=["avg_overall", "avg_age"],
        )
        fig.update_layout(xaxis_title="Player-seasons", yaxis_title="")
        st.plotly_chart(plotly_layout(fig, height=520), use_container_width=True)
    with c2:
        st.dataframe(clusters.sort_values("players", ascending=False), use_container_width=True, hide_index=True)

# -----------------------------
# Tab 6: method
# -----------------------------
with tab6:
    st.markdown("### Method & caveats")
    st.markdown(
        """
        <div class="section-card">
        <h3>What the model does</h3>
        <p class="small-muted">
        The app combines FIFA-style player attributes with FBRef-style performance data, then trains an autoencoder to compress each player-season into a neural representation. Similarity is calculated in that football-DNA space using cosine similarity.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Profile inputs**")
        st.caption("FIFA attributes + available FBRef-style production metrics.")
    with c2:
        st.markdown("**2. Neural compression**")
        st.caption("Autoencoder creates lower-dimensional football DNA embeddings.")
    with c3:
        st.markdown("**3. Storytelling layer**")
        st.caption("Similarity, archetypes, current successor search and prototype Legend Style Score.")

    st.markdown("#### Caveats")
    st.markdown(
        """
        - Similarity means profile resemblance, not equal ability or career value.
        - The Legend Style Score is exploratory and should be framed as a narrative device.
        - FIFA 26 records are useful for current-player comparisons, but true 25/26 FBRef data should be added later.
        - Player-name matching across datasets can introduce noise.
        - A polished app should explain *why* a player matches, not just show the score. That is the next major modelling upgrade.
        """
    )

    st.markdown("#### Training loss")
    try:
        loss = pd.read_csv(f"{DATA_DIR}/training_loss_v2.csv")
        fig = px.line(loss, x="epoch", y="reconstruction_mse", markers=True, title="Autoencoder reconstruction loss")
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Reconstruction MSE")
        st.plotly_chart(plotly_layout(fig, height=360), use_container_width=True)
    except Exception:
        st.info("Training loss file not available.")
