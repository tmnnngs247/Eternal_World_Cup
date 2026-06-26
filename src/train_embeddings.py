from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

BASE_FEATURES = [
    "overall","potential","age","height_cm","weight_kg","weak_foot","skill_moves","international_reputation",
    "pace","shooting","passing","dribbling","defending","physic","ball_control","skill_dribbling","crossing",
    "finishing","short_passing","volleys","long_passing","acceleration","sprint_speed","agility","reactions",
    "balance","shot_power","jumping","stamina","strength","long_shots","aggression","interceptions","positioning",
    "vision","composure","marking","standing_tackle","sliding_tackle",
]


def main():
    players = pd.read_csv(PROCESSED / "players_master.csv")
    perf_cols = [c for c in players.columns if c.startswith("perf_")]
    feature_cols = [c for c in BASE_FEATURES + perf_cols if c in players.columns]
    numeric = players[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    # Drop completely empty features
    numeric = numeric.loc[:, numeric.notna().sum() > 100]
    feature_cols = list(numeric.columns)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(numeric))
    n_components = min(32, X.shape[1], max(2, X.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=42)
    emb = pca.fit_transform(X)
    out = players.copy()
    for i in range(n_components):
        out[f"emb_{i:02d}"] = emb[:, i]

    # Keep the deployed app lightweight: top player-seasons by season plus all 2026 top players and World Cup roster matches.
    out["overall_sort"] = pd.to_numeric(out.get("overall"), errors="coerce").fillna(0)
    top_by_season = out.sort_values("overall_sort", ascending=False).groupby("season_year", group_keys=False).head(4500)
    roster_matches = out[out.get("roster_id", pd.Series(index=out.index)).notna()] if "roster_id" in out.columns else out.iloc[0:0]
    deployed = pd.concat([top_by_season, roster_matches], ignore_index=True).drop_duplicates("player_season_id", keep="first")
    keep_cols = [
        "player_season_id", "name_key", "short_name", "long_name", "season_year", "season_label",
        "club_name", "league_name", "nationality_name", "flag", "player_positions", "age", "overall", "potential",
        "height_cm", "weight_kg", "preferred_foot", "weak_foot", "skill_moves", "international_reputation",
        "image_url", "roster_id", "roster_pos", "roster_no", "roster_club", "roster_caps", "roster_goals",
        "wc_apps", "wc_goals",
    ]
    keep_cols = [c for c in keep_cols if c in deployed.columns] + [c for c in BASE_FEATURES if c in deployed.columns] + [f"emb_{i:02d}" for i in range(n_components)]
    deployed = deployed.loc[:, list(dict.fromkeys(keep_cols))]
    deployed.to_csv(PROCESSED / "player_embeddings.csv", index=False)
    meta = {"method":"PCA baseline over standardised FIFA/FBRef-style features", "n_components": int(n_components), "feature_cols": feature_cols, "explained_variance": pca.explained_variance_ratio_.tolist(), "deployed_rows": int(len(deployed))}
    (MODELS / "embedding_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Built embeddings: {out.shape}, components={n_components}")

if __name__ == "__main__":
    main()
