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
```

## Current build

This version is a stable foundation. It uses a reproducible PCA embedding pipeline over FIFA/FBRef-style numerical player features, clusters players into 24 archetypes, calculates nearest-neighbour similarity in embedding space, and enriches player cards with World Cup roster metadata and flags where available.

The next upgrade is to replace/extend the PCA embedding with a trained autoencoder once the data pipeline is fully stable.
