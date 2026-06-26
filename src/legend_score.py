from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main():
    df = pd.read_csv(PROCESSED / "player_embeddings.csv")
    latest = df.sort_values("season_year").groupby("name_key", as_index=False).tail(1).copy()
    for c in ["overall", "potential", "age", "international_reputation", "wc_apps", "wc_goals"]:
        if c not in latest.columns:
            latest[c] = 0
        latest[c] = pd.to_numeric(latest[c], errors="coerce").fillna(0)
    # Prototype narrative score: current quality + ceiling + youth + World Cup signal.
    age_bonus = np.clip((30 - latest["age"]) / 12, 0, 1) * 100
    raw = (
        latest["overall"].fillna(0) * 0.45 +
        latest["potential"].fillna(latest["overall"]) * 0.25 +
        age_bonus * 0.15 +
        latest["international_reputation"].fillna(0) * 6 +
        np.log1p(latest["wc_apps"].fillna(0)) * 6 +
        np.log1p(latest["wc_goals"].fillna(0)) * 7
    )
    scaler = MinMaxScaler(feature_range=(0, 100))
    latest["legend_style_score"] = scaler.fit_transform(raw.to_numpy().reshape(-1,1)).ravel().round(1)
    latest.sort_values("legend_style_score", ascending=False).to_csv(PROCESSED / "legend_scores.csv", index=False)
    print("Built legend_scores.csv")

if __name__ == "__main__":
    main()
