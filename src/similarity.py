from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

QUERIES = ["L. Messi", "Cristiano Ronaldo", "Neymar Jr", "K. Mbappé", "L. Modrić", "J. Bellingham"]


def main():
    df = pd.read_csv(PROCESSED / "player_embeddings.csv")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    rows = []
    latest = df.sort_values("season_year").groupby("name_key", as_index=False).tail(1)
    for q in QUERIES:
        cand = df[df["short_name"].astype(str).str.contains(q.split()[0], case=False, na=False)]
        if cand.empty:
            continue
        query = cand.sort_values("overall", ascending=False).iloc[0]
        sims = cosine_similarity(query[emb_cols].to_numpy(float).reshape(1,-1), latest[emb_cols].to_numpy(float)).ravel()
        tmp = latest.copy()
        tmp["query_player"] = query["short_name"]
        tmp["query_season"] = query["season_label"]
        tmp["similarity"] = sims
        tmp = tmp[tmp["player_season_id"] != query["player_season_id"]].sort_values("similarity", ascending=False).head(10)
        rows.append(tmp)
    if rows:
        pd.concat(rows).to_csv(PROCESSED / "similarity_examples.csv", index=False)
    else:
        pd.DataFrame().to_csv(PROCESSED / "similarity_examples.csv", index=False)
    print("Built similarity_examples.csv")

if __name__ == "__main__":
    main()
