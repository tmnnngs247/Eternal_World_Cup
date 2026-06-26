from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from app.components import hero, metric_grid, player_card
from app.styles import inject_css

st.set_page_config(page_title="The Eternal World Cup", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")
inject_css()

DATA_DIR = Path("data/processed")

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)

@st.cache_data
def load_data():
    embeddings = load_csv("player_dna_embeddings_fifa_fbref_v2.csv")
    if embeddings.empty:
        embeddings = load_csv("player_embeddings.csv")
    scores = load_csv("current_players_legend_style_score.csv")
    if scores.empty:
        scores = load_csv("legend_scores.csv")
    sims = load_csv("modern_successors_similarity_examples.csv")
    if sims.empty:
        sims = load_csv("similarity_index.csv")
    archetypes = load_csv("archetypes.csv")
    if archetypes.empty:
        archetypes = load_csv("archetype_cluster_summary.csv")
    dna_map = load_csv("player_dna_map_2d.csv")
    roster = load_csv("player_roster_metadata_2026.csv")
    return embeddings, scores, sims, archetypes, dna_map, roster

embeddings, scores, sims, archetypes, dna_map, roster = load_data()

# Sidebar
with st.sidebar:
    st.markdown("### ⚽ Eternal World Cup")
    st.caption("Football DNA, decoded.")
    st.divider()
    st.markdown("**Project status**")
    st.success("Real-project scaffold")
    st.caption("App reads processed files. ML pipeline lives in `src/`.")
    st.divider()
    st.markdown("**How to read this**")
    st.caption("Similarity = profile resemblance, not guaranteed quality. Legend Style Score is exploratory.")

hero()

n_players = len(embeddings) if not embeddings.empty else 0
n_dims = len([c for c in embeddings.columns if c.startswith("dna_")]) if not embeddings.empty else 0
n_arch = len(archetypes) if not archetypes.empty else 0
n_scores = len(scores) if not scores.empty else 0
metric_grid([
    ("Player-seasons", f"{n_players:,}", "Historical + current records"),
    ("DNA dimensions", str(n_dims), "Compressed player profile"),
    ("Archetypes", str(n_arch), "Profile clusters"),
    ("Current score pool", f"{n_scores:,}", "Ranked players"),
])

tabs = st.tabs(["Successor finder", "Compare players", "Legend score", "DNA map", "Archetypes", "Method"])

# Helpers
def display_name(row):
    club = row.get("club_name", row.get("match_club", ""))
    year = row.get("season_year", row.get("match_season", ""))
    name = row.get("short_name", row.get("match_player", ""))
    return f"{name} — {year} — {club}"

def get_flag(name: str, nation: str = "") -> str:
    if roster.empty:
        return ""
    # crude exact match first
    m = roster[roster["short_name"].astype(str).str.lower().eq(str(name).lower())]
    if not m.empty and "nationality_flag" in m.columns:
        return str(m.iloc[0].get("nationality_flag", ""))
    return ""

with tabs[0]:
    st.header("Find a player's closest football DNA matches")
    st.markdown('<div class="info-callout">Pick a reference player-season and search for modern successors or historical lookalikes. The output is based on embedding similarity.</div>', unsafe_allow_html=True)
    if embeddings.empty:
        st.warning("No embeddings file found in data/processed.")
    else:
        options_df = embeddings.copy()
        options_df["label"] = options_df.apply(display_name, axis=1)
        col1, col2, col3 = st.columns([2.5, 1.1, .9])
        with col1:
            default_idx = int(options_df[options_df["short_name"].astype(str).str.contains("Messi", case=False, na=False)].index[0]) if options_df["short_name"].astype(str).str.contains("Messi", case=False, na=False).any() else 0
            label = st.selectbox("Reference player-season", options_df["label"].tolist(), index=min(default_idx, len(options_df)-1))
        with col2:
            pool = st.selectbox("Comparison pool", ["Current players only", "All seasons"])
        with col3:
            topn = st.slider("Matches", 3, 15, 8)
        query = options_df[options_df["label"].eq(label)].iloc[0]
        emb_cols = [c for c in embeddings.columns if c.startswith("dna_")]
        pool_df = embeddings.copy()
        if pool == "Current players only" and "season_year" in pool_df.columns:
            pool_df = pool_df[pool_df["season_year"].eq(pool_df["season_year"].max())]
        Xq = query[emb_cols].to_numpy(float).reshape(1, -1)
        X = pool_df[emb_cols].to_numpy(float)
        sim = cosine_similarity(Xq, X).ravel()
        res = pool_df.copy()
        res["similarity"] = sim
        res = res[res.get("player_season_id", pd.Series(index=res.index)) != query.get("player_season_id", None)]
        res = res.sort_values("similarity", ascending=False).head(topn)
        st.subheader(f"Closest matches for {query.get('short_name')}")
        for i, (_, r) in enumerate(res.iterrows(), 1):
            pct = f"{r['similarity']*100:.1f}%"
            meta = f"{r.get('season_year','')} · {r.get('club_name','')} · {r.get('nationality_name','')} · OVR {r.get('overall','')}"
            player_card(i, r.get("short_name", "Unknown"), meta, pct, get_flag(r.get("short_name", ""), r.get("nationality_name", "")))

with tabs[1]:
    st.header("Compare two player-seasons")
    if embeddings.empty:
        st.warning("No embeddings available.")
    else:
        options_df = embeddings.copy()
        options_df["label"] = options_df.apply(display_name, axis=1)
        a, b = st.columns(2)
        with a:
            p1 = st.selectbox("Player A", options_df["label"].tolist(), key="p1")
        with b:
            p2 = st.selectbox("Player B", options_df["label"].tolist(), index=min(10, len(options_df)-1), key="p2")
        emb_cols = [c for c in embeddings.columns if c.startswith("dna_")]
        r1 = options_df[options_df["label"].eq(p1)].iloc[0]
        r2 = options_df[options_df["label"].eq(p2)].iloc[0]
        sim = cosine_similarity(r1[emb_cols].to_numpy(float).reshape(1,-1), r2[emb_cols].to_numpy(float).reshape(1,-1))[0,0]
        st.metric("Football DNA similarity", f"{sim*100:.1f}%")
        attrs = [c for c in ["pace","shooting","passing","dribbling","defending","physic","overall"] if c in embeddings.columns]
        if attrs:
            comp = pd.DataFrame({"attribute": attrs, str(r1.get("short_name")): [r1.get(c) for c in attrs], str(r2.get("short_name")): [r2.get(c) for c in attrs]})
            st.dataframe(comp, use_container_width=True, hide_index=True)

with tabs[2]:
    st.header("Legend Style Score")
    st.markdown('<div class="info-callout">A narrative ranking of current players. Treat it as a discussion starter, not a validated projection model.</div>', unsafe_allow_html=True)
    if scores.empty:
        st.warning("No legend score file found.")
    else:
        score_col = "legend_style_score" if "legend_style_score" in scores.columns else scores.select_dtypes(include=np.number).columns[-1]
        for i, (_, r) in enumerate(scores.sort_values(score_col, ascending=False).head(20).iterrows(), 1):
            meta = f"{r.get('age','')} · {r.get('club_name','')} · {r.get('nationality_name','')} · {r.get('player_positions','')}"
            player_card(i, r.get("short_name", r.get("long_name", "Unknown")), meta, r.get(score_col), get_flag(r.get("short_name", ""), r.get("nationality_name", "")))

with tabs[3]:
    st.header("Football DNA map")
    if dna_map.empty:
        st.info("Run the projection step to create player_dna_map_2d.csv.")
    else:
        x = "x" if "x" in dna_map.columns else dna_map.select_dtypes(include=np.number).columns[0]
        y = "y" if "y" in dna_map.columns else dna_map.select_dtypes(include=np.number).columns[1]
        color = "archetype_name" if "archetype_name" in dna_map.columns else None
        fig = px.scatter(dna_map, x=x, y=y, color=color, hover_name="short_name" if "short_name" in dna_map.columns else None,
                         hover_data=[c for c in ["club_name","nationality_name","overall","age"] if c in dna_map.columns], height=680)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.header("Archetypes")
    if archetypes.empty:
        st.warning("No archetype file found.")
    else:
        st.dataframe(archetypes, use_container_width=True, hide_index=True)

with tabs[5]:
    st.header("Method & caveats")
    st.markdown("""
    <div class="section-card">
    <h3>What the model does</h3>
    <p>The pipeline standardises FIFA-style attributes, optionally enriches them with FBRef production metrics, compresses player-seasons into a lower-dimensional football-DNA space, and calculates similarity with cosine similarity.</p>
    </div>
    <div class="section-card">
    <h3>What it does not do</h3>
    <p>It does not understand tactical role, injuries, transfers, coaching environment, or future development. It should support discussion, not replace scouting judgement.</p>
    </div>
    """, unsafe_allow_html=True)
