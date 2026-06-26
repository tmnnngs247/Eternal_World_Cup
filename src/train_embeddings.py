"""
Train compressed player representations.

This script uses sklearn PCA as a lightweight default so the pipeline is portable.
For a full neural autoencoder, add PyTorch and replace `fit_embedding_model` with an encoder/decoder model.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR, OUTPUTS_DIR, FIFA_CORE_FEATURES, FBREF_FEATURE_PATTERNS


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in FIFA_CORE_FEATURES + FBREF_FEATURE_PATTERNS if c in df.columns]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def main(n_components: int = 32) -> None:
    df = pd.read_csv(PROCESSED_DIR / "players_master.csv", low_memory=False)
    cols = feature_columns(df)
    if not cols:
        raise ValueError("No numeric feature columns found")
    X = df[cols]
    n_components = min(n_components, len(cols), max(2, len(df) - 1))
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])
    Z = pipe.fit_transform(X)
    emb_cols = [f"dna_{i+1:02d}" for i in range(Z.shape[1])]
    id_cols = [c for c in ["player_season_id", "sofifa_id", "short_name", "long_name", "player_positions", "overall", "age", "club_name", "nationality_name", "season_year", "player_face_url"] if c in df.columns]
    out = pd.concat([df[id_cols].reset_index(drop=True), pd.DataFrame(Z, columns=emb_cols)], axis=1)
    out.to_csv(PROCESSED_DIR / "player_embeddings.csv", index=False)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": cols}, OUTPUTS_DIR / "embedding_model.joblib")
    print(f"Wrote embeddings: {out.shape}")

if __name__ == "__main__":
    main()
