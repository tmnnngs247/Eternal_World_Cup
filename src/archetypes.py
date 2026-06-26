"""Cluster player embeddings into interpretable archetypes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import PROCESSED_DIR

ARCHETYPE_NAMES = [
    "Ball-Carrying Creator", "Elite Final-Third Forward", "Creative Wide Threat", "Tempo-Setting Midfielder",
    "Deep Progressor", "Box-Crashing Midfielder", "Ball-Winning Six", "High-Volume Fullback",
    "Inverted Fullback", "Sweeping Centre-Back", "Aggressive Stopper", "Aerial Box Defender",
    "Pressing Forward", "Penalty-Box Striker", "Transition Runner", "Inside Forward",
    "Set-Piece Creator", "Defensive Utility Player", "Possession Goalkeeper", "Shot-Stopping Goalkeeper",
    "Developmental Prospect", "Physical Carrier", "Low-Touch Finisher", "Hybrid Playmaker",
]


def main(k: int = 24) -> None:
    df = pd.read_csv(PROCESSED_DIR / "player_embeddings.csv", low_memory=False)
    emb_cols = [c for c in df.columns if c.startswith("dna_")]
    X = df[emb_cols].to_numpy(float)
    k = min(k, max(2, len(df) // 25))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    df["archetype_id"] = labels
    names = {i: ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)] for i in range(k)}
    df["archetype_name"] = df["archetype_id"].map(names)
    df.to_csv(PROCESSED_DIR / "player_embeddings_with_archetypes.csv", index=False)
    summary = (
        df.groupby(["archetype_id", "archetype_name"])
          .agg(players=("short_name", "count"), avg_overall=("overall", "mean"), avg_age=("age", "mean"))
          .reset_index()
          .sort_values("players", ascending=False)
    )
    examples = df.sort_values("overall", ascending=False).groupby("archetype_id").head(5)
    ex = examples.groupby("archetype_id")["short_name"].apply(lambda s: ", ".join(s.astype(str).head(5))).reset_index(name="example_players")
    summary = summary.merge(ex, on="archetype_id", how="left")
    summary.to_csv(PROCESSED_DIR / "archetypes.csv", index=False)
    print(f"Wrote archetypes: {summary.shape}")

if __name__ == "__main__":
    main()
