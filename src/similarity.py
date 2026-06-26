"""Build nearest-neighbour similarity outputs from player embeddings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import PROCESSED_DIR


def main(top_n: int = 20, max_reference_players: int = 5000) -> None:
    df = pd.read_csv(PROCESSED_DIR / "player_embeddings.csv", low_memory=False)
    emb_cols = [c for c in df.columns if c.startswith("dna_")]
    if not emb_cols:
        raise ValueError("No dna_ columns found")
    # Keep the public index small enough for Streamlit/GitHub.
    ref = df.sort_values(["overall", "season_year"], ascending=[False, False]).head(max_reference_players).copy()
    X = ref[emb_cols].to_numpy(float)
    sims = cosine_similarity(X)
    rows = []
    for i, row in ref.reset_index(drop=True).iterrows():
        order = np.argsort(-sims[i])
        rank = 0
        for j in order:
            if i == j:
                continue
            rank += 1
            other = ref.iloc[j]
            rows.append({
                "query_player": row.get("short_name"), "query_season": row.get("season_year"), "query_club": row.get("club_name"),
                "match_rank": rank, "match_player": other.get("short_name"), "match_season": other.get("season_year"),
                "match_club": other.get("club_name"), "match_nation": other.get("nationality_name"),
                "similarity": round(float(sims[i, j]), 4), "match_overall": other.get("overall"), "match_age": other.get("age"),
            })
            if rank >= top_n:
                break
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED_DIR / "similarity_index.csv", index=False)
    print(f"Wrote similarity index: {out.shape}")

if __name__ == "__main__":
    main()
