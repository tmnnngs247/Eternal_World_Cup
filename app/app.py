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
    path = PROCESSED / "player_embeddings.csv"
    if not path.exists():
        st.error("Processed data not found. Run `python src/run_pipeline.py` locally and commit data/processed outputs.")
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

players = load_players()
archetypes = load_archetypes()
scores = load_scores()
emb_cols = [c for c in players.columns if c.startswith("emb_")]
if not emb_cols:
    st.error("No embedding columns found in data/processed/player_embeddings.csv. Re-run the pipeline.")
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
    st.success("Stable v1: real processed pipeline")
    st.caption("Next: autoencoder training + richer 25/26 FBRef data + player images.")

hero()
metrics_grid([
    ("Player-seasons", f"{len(players):,}", "Historical + current records"),
    ("DNA dimensions", str(len(emb_cols)), "Compressed profile space"),
    ("Archetypes", f"{players['archetype_id'].nunique() if 'archetype_id' in players.columns else 0}", "Profile clusters"),
    ("Latest player pool", f"{players['name_key'].nunique():,}", "Unique players"),
])

latest = players.sort_values("season_year").groupby("name_key", as_index=False).tail(1).copy()
players["display_name"] = players["short_name"].astype(str) + " — " + players["season_label"].astype(str) + " — " + players["club_name"].fillna("").astype(str)
latest["display_name"] = latest["short_name"].astype(str) + " — " + latest["season_label"].astype(str) + " — " + latest["club_name"].fillna("").astype(str)

if page == "Successor Finder":
    st.header("Find a player's closest football DNA matches")
    st.markdown("<div class='ewc-callout'>Pick a reference player-season and search for modern successors or historical lookalikes. Similarity is calculated live in embedding space.</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2.4, 1.2, .8])
    with c1:
        selected = st.selectbox("Reference player-season", players.sort_values(["overall","season_year"], ascending=[False,False])["display_name"].head(5000))
    with c2:
        pool_choice = st.selectbox("Comparison pool", ["Latest player-season only", "All player-seasons", "Same season only"])
    with c3:
        n = st.slider("Matches", 5, 20, 8)
    query = players.loc[players["display_name"].eq(selected)].iloc[0]
    if pool_choice == "Latest player-season only":
        pool = latest.copy()
    elif pool_choice == "Same season only":
        pool = players[players["season_year"].eq(query["season_year"])].copy()
    else:
        pool = players.copy()
    Xq = query[emb_cols].to_numpy(float).reshape(1, -1)
    X = pool[emb_cols].to_numpy(float)
    res = pool.copy()
    res["similarity"] = cosine_similarity(Xq, X).ravel()
    res = res[res["player_season_id"] != query["player_season_id"]].sort_values("similarity", ascending=False).head(n)
    player_cards(res, max_cards=n)
    with st.expander("Show table"):
        st.dataframe(res[["short_name","season_label","club_name","nationality_name","overall","age","player_positions","similarity","archetype_name"]], width="stretch")

elif page == "Compare Players":
    st.header("Compare two player-seasons")
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Player A", players["display_name"].sort_values(), index=0)
    with c2:
        b = st.selectbox("Player B", players["display_name"].sort_values(), index=1)
    pa = players.loc[players["display_name"].eq(a)].iloc[0]
    pb = players.loc[players["display_name"].eq(b)].iloc[0]
    sim = cosine_similarity(pa[emb_cols].to_numpy(float).reshape(1,-1), pb[emb_cols].to_numpy(float).reshape(1,-1))[0,0]
    st.metric("Football DNA similarity", f"{sim*100:.1f}%")
    attrs = ["pace","shooting","passing","dribbling","defending","physic"]
    comp = pd.DataFrame({"attribute": attrs, str(pa["short_name"]): [pa.get(x, np.nan) for x in attrs], str(pb["short_name"]): [pb.get(x, np.nan) for x in attrs]})
    fig = px.line_polar(comp.melt("attribute", var_name="player", value_name="value"), r="value", theta="attribute", color="player", line_close=True, range_r=[0,100])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

elif page == "DNA Map":
    st.header("Football DNA map")
    plot_df = latest.dropna(subset=[emb_cols[0], emb_cols[1]]).copy()
    max_points = st.slider("Number of players", 500, min(10000, len(plot_df)), 2500, 500)
    plot_df = plot_df.sort_values("overall", ascending=False).head(max_points)
    fig = px.scatter(
        plot_df,
        x=emb_cols[0], y=emb_cols[1], color="archetype_name" if "archetype_name" in plot_df else None,
        hover_data=["short_name", "club_name", "nationality_name", "overall", "age", "player_positions"],
        title="Latest player-seasons projected onto first two DNA dimensions",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=700)
    st.plotly_chart(fig, width="stretch")

elif page == "Legend Score":
    st.header("Prototype Legend Style Score")
    st.markdown("<div class='ewc-callout'>This is a narrative ranking, not a prediction model. It combines current quality, potential, age curve, reputation and World Cup metadata where available.</div>", unsafe_allow_html=True)
    if scores.empty:
        st.info("No legend score file found.")
    else:
        player_cards(scores.sort_values("legend_style_score", ascending=False).rename(columns={"legend_style_score":"similarity"}).assign(similarity=lambda d: d["similarity"]/100), max_cards=12)
        with st.expander("Show full ranking"):
            st.dataframe(scores[["short_name","season_label","club_name","nationality_name","overall","potential","age","legend_style_score","archetype_name"]].head(250), width="stretch")

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
        <p>The current stable build creates a football-DNA embedding from standardised FIFA/FBRef-style numerical player features. Similarity is calculated with cosine similarity in that embedding space.</p>
        <h3>What it does not do yet</h3>
        <p>This version is not yet a fully trained autoencoder and does not predict future performance. It is a stable foundation for the real ML build: clean data, reproducible outputs and a working app shell.</p>
        <h3>Next upgrades</h3>
        <p>Train an autoencoder, add richer current-season FBRef data, improve player identity matching, source Wikimedia-compatible player images and write interpretable successor explanations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
