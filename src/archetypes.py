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

TRAIT_DESCRIPTIONS = {
    "pace": "electric pace",
    "shooting": "clinical finishing",
    "passing": "elite passing range",
    "dribbling": "close control and ball-carrying",
    "defending": "defensive discipline",
    "physic": "physical dominance",
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


def name_cluster(dominant_position: str, top_traits: list[str], used_names: set[str]) -> str:
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


def define_cluster(dominant_position: str, top_traits: list[str]) -> str:
    descriptions = [TRAIT_DESCRIPTIONS[t] for t in top_traits[:2]]
    article = "An" if dominant_position[0] in "AEIOU" else "A"
    return f"{article} {dominant_position} defined by {' and '.join(descriptions)}."


def overall_tier(overall_z: float) -> str:
    if overall_z > 0.4:
        return "Elite"
    if overall_z < -0.4:
        return "Development"
    return "Reliable"


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

    # Goalkeepers have no outfield trait ratings (pace/shooting/etc are all NaN),
    # so clusters made up mostly of keepers need a different differentiator: overall tier.
    overall_numeric = df["overall"].apply(pd.to_numeric, errors="coerce")
    is_keeper = df["position_label"].eq("Keeper")
    keeper_overall_mean = overall_numeric[is_keeper].mean()
    keeper_overall_std = overall_numeric[is_keeper].std() or 1

    names_by_id: dict[int, str] = {}
    used_names: set[str] = set()
    summary = []

    for archetype_id, cluster_df in df.groupby("archetype_id"):
        cluster_means = cluster_df[TRAIT_COLS].apply(pd.to_numeric, errors="coerce").mean()
        trait_z = (cluster_means - population_means) / population_stds
        dominant_position = cluster_df["position_label"].mode().iloc[0]

        if dominant_position == "Keeper":
            # Most players in this cluster have no outfield trait ratings, so a
            # handful of rows with stray pace/shooting data would otherwise
            # produce a trait-based name representing a small minority of the
            # cluster. Differentiate by overall rating tier instead.
            cluster_overall = pd.to_numeric(cluster_df["overall"], errors="coerce").mean()
            overall_z = (cluster_overall - keeper_overall_mean) / keeper_overall_std
            tier = overall_tier(overall_z)
            top_traits = []
            name = f"{tier} {dominant_position}"

            if name in used_names:
                suffix = 2
                while f"{name} {suffix}" in used_names:
                    suffix += 1
                name = f"{name} {suffix}"

            article = "An" if dominant_position[0] in "AEIOU" else "A"
            definition = (
                f"{article} {dominant_position} in the {tier.lower()} tier by overall "
                f"rating. Outfield attributes like pace and shooting don't apply to goalkeepers."
            )
            key_traits = "Overall, Potential, Age"
        else:
            top_traits = trait_z.sort_values(ascending=False).index.tolist()
            name = name_cluster(dominant_position, top_traits, used_names)
            definition = define_cluster(dominant_position, top_traits)
            key_traits = ", ".join(t.title() for t in top_traits[:3])

        used_names.add(name)
        names_by_id[archetype_id] = name

        examples = cluster_df.sort_values("overall", ascending=False).head(8)["short_name"].tolist()

        summary.append({
            "archetype_id": archetype_id,
            "archetype_name": name,
            "definition": definition,
            "key_traits": key_traits,
            "player_count": len(cluster_df),
            "avg_overall": round(pd.to_numeric(cluster_df["overall"], errors="coerce").mean(), 2),
            "example_players": ", ".join(map(str, examples)),
            **{f"avg_{trait}": round(cluster_means[trait], 1) for trait in TRAIT_COLS},
        })

    df["archetype_name"] = df["archetype_id"].map(names_by_id)
    df = df.drop(columns=["position_label"])
    df.to_csv(PROCESSED / "player_embeddings.csv", index=False)

    pd.DataFrame(summary).sort_values("archetype_id").to_csv(PROCESSED / "archetypes.csv", index=False)
    print(f"Built {len(summary)} archetypes (KMeans over {len(emb_cols)}-dim embedding)")


if __name__ == "__main__":
    main()
