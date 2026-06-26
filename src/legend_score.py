"""Prototype Legend Style Score.

This is deliberately labelled exploratory. It combines peak rating, potential, reputation,
current age curve and similarity-space prominence. Replace with a supervised model once a
proper historical legend label is defined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import PROCESSED_DIR


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "player_embeddings.csv", low_memory=False)
    current = df[df["season_year"].eq(df["season_year"].max())].copy() if "season_year" in df.columns else df.copy()
    current["age_curve_bonus"] = np.select(
        [current["age"].between(16, 23, inclusive="both"), current["age"].between(24, 29, inclusive="both")],
        [1.0, 0.75], default=0.35
    ) if "age" in current.columns else 0.5
    score = (
        0.40 * minmax(current.get("overall", 0))
        + 0.25 * minmax(current.get("potential", current.get("overall", 0)))
        + 0.15 * minmax(current.get("international_reputation", 1))
        + 0.10 * current["age_curve_bonus"]
        + 0.10 * minmax(current.get("value_eur", 0))
    )
    current["legend_style_score"] = (100 * score).round(1)
    keep = [c for c in ["short_name", "long_name", "age", "club_name", "nationality_name", "player_positions", "overall", "potential", "legend_style_score", "player_face_url"] if c in current.columns]
    out = current[keep].sort_values("legend_style_score", ascending=False).head(500)
    out.to_csv(PROCESSED_DIR / "legend_scores.csv", index=False)
    print(f"Wrote legend scores: {out.shape}")

if __name__ == "__main__":
    main()
