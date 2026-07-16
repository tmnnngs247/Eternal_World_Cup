# The Eternal World Cup

Using player attributes, performance-style data and neural-style football DNA embeddings to explore player similarity, successors, archetypes and World Cup storytelling.

## App entry point

Streamlit Community Cloud should run:

```text
app.py
```

## Project structure

```text
app.py                  # Streamlit entry point
app/                    # UI helpers and styles
src/                    # Reproducible data/model pipeline
data/processed/         # App-ready outputs
models/                 # Saved model metadata
tests/                  # Sanity checks over data/processed outputs
```

## Current build

The football-DNA embedding is a trained autoencoder (a bottlenecked neural network, not PCA) over standardised FIFA attribute ratings and, where available, FBRef per-90 performance stats. Players are grouped into 24 archetypes via KMeans clustering directly on that embedding, similarity is calculated with cosine distance in the same space, and player cards are enriched with flags, photos, and World Cup roster data (apps/goals) sourced from fbref.com for the 2018, 2022 and 2026 tournaments.

Known open items: FBRef performance data covers roughly 12% of player-seasons (name-matching the rest would raise this), and this is a similarity model, not a predictive one.

## Running the pipeline

```text
python src/run_pipeline.py
```

Rebuilds everything in `data/processed/` and `models/` from `data/raw/`. World Cup roster data is sourced from `data/raw/world_cup_<year>_player_stats.csv`, which isn't committed (same `data/raw/*` policy as the rest of the raw data) -- re-sourcing it requires a browser session against fbref.com, which sits behind bot protection that a plain scraper won't get past.

## Running tests

```text
pip install -r requirements-dev.txt
pytest
```

Tests validate the committed `data/processed/` outputs (schema, coverage, no corrupted text, no dead scoring components) rather than re-running the full pipeline.
