from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

N_CLUSTERS = 24

TRAIT_COLS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

TRAIT_ADJECTIVES = {
    "pace": "Pacey",
    "shooting": "Clinical",
    "passing": "Creative",
    "dribbling": "Ball-Carrying",
    "defending": "Disciplined",
    "physic": "Physical",
}

POSITION_LABELS = [
    (["GK"], "Keeper"),
    (["CB"], "Centre Back"),
    (["RB", "LB", "RWB", "LWB"], "Fullback"),
    (["CDM"], "Defensive Midfielder"),
    (["CM"], "Midfielder"),
    (["CAM", "CF"], "Attacking Midfielder"),
    (["RW", "LW", "RM", "LM"], "Winger"),
    (["ST"], "Striker"),
]


def position_label(position_string: object) -> str:
    positions = str(position_string or "").upper()

    for codes, label in POSITION_LABELS:
        if any(code in positions for code in codes):
            return label

    return "Utility Player"


def name_cluster(cluster_df: pd.DataFrame, trait_z: pd.Series, used_names: set[str]) -> str:
    dominant_position = cluster_df["position_label"].mode().iloc[0]

    top_traits = trait_z.sort_values(ascending=False).index.tolist()

    for n_traits in range(1, len(top_traits) + 1):
        adjectives = " ".join(TRAIT_ADJECTIVES[t] for t in top_traits[:n_traits])
        candidate = f"{adjectives} {dominant_position}"

        if candidate not in used_names:
            return candidate

    # Extremely unlikely fallback: every combination up to 3 traits collided.
    suffix = 2

    while f"{candidate} {suffix}" in used_names:
        suffix += 1

    return f"{candidate} {suffix}"


def main():
    df = pd.read_csv(PROCESSED / "player_embeddings.csv", low_memory=False)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    if not emb_cols:
        raise ValueError("No embedding columns found in player_embeddings.csv")

    X = StandardScaler().fit_transform(df[emb_cols].to_numpy(float))
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["archetype_id"] = kmeans.fit_predict(X)

    df["position_label"] = df["player_positions"].map(position_label)

    population_means = df[TRAIT_COLS].apply(pd.to_numeric, errors="coerce").mean()
    population_stds = df[TRAIT_COLS].apply(pd.to_numeric, errors="coerce").std().replace(0, 1)

    names_by_id: dict[int, str] = {}
    used_names: set[str] = set()

    for archetype_id, cluster_df in df.groupby("archetype_id"):
        cluster_means = cluster_df[TRAIT_COLS].apply(pd.to_numeric, errors="coerce").mean()
        trait_z = (cluster_means - population_means) / population_stds
        name = name_cluster(cluster_df, trait_z, used_names)
        used_names.add(name)
        names_by_id[archetype_id] = name

    df["archetype_name"] = df["archetype_id"].map(names_by_id)
    df = df.drop(columns=["position_label"])
    df.to_csv(PROCESSED / "player_embeddings.csv", index=False)

    summary = []

    for archetype_id, sub in df.groupby("archetype_id"):
        examples = sub.sort_values("overall", ascending=False).head(8)["short_name"].tolist()
        summary.append({
            "archetype_id": archetype_id,
            "archetype_name": names_by_id[archetype_id],
            "player_count": len(sub),
            "avg_overall": round(pd.to_numeric(sub["overall"], errors="coerce").mean(), 2),
            "example_players": ", ".join(map(str, examples)),
        })

    pd.DataFrame(summary).sort_values("archetype_id").to_csv(PROCESSED / "archetypes.csv", index=False)
    print(f"Built {len(summary)} archetypes (KMeans over {len(emb_cols)}-dim embedding)")


if __name__ == "__main__":
    main()
