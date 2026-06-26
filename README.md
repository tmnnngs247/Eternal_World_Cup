# The Eternal World Cup

Using neural-network-style football DNA to identify modern successors to football's greats.

This repository is now structured as a small ML/data product rather than a single Streamlit file.
The Streamlit app reads clean files from `data/processed/`; reproducible data and modelling scripts live in `src/`.

## Repository structure

```text
Eternal_World_Cup/
  app.py                    # Streamlit entrypoint
  app/                      # UI components and CSS
  src/                      # data/model pipeline
  data/raw/                 # raw datasets kept locally
  data/processed/           # app-ready outputs
  outputs/                  # fitted models / artifacts
  requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuilding the data pipeline

Place the raw CSVs in `data/raw/`, then run:

```bash
python src/data_build.py
python src/train_embeddings.py
python src/similarity.py
python src/archetypes.py
python src/legend_score.py
```

The app will use the generated files in `data/processed/`.

## Caveat

The current Legend Style Score is exploratory. It is intended as a storytelling layer, not a validated prediction model.
