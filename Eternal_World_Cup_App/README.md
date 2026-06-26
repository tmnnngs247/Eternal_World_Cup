# The Eternal World Cup

Using neural-network football DNA to identify the modern successors to football's greats.

## What this app does

This Streamlit app explores player similarity using neural embeddings built from FIFA-style player attributes and FBRef-style performance data.

Core features:

- Find modern successors for any player-season
- Compare two players by football DNA similarity
- Rank current players by a prototype Legend Style Score
- Explore a 2D Football DNA map
- Review archetype clusters and model caveats

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy

1. Upload this whole folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Choose this repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Data caveats

The model is exploratory. Similarity means profile similarity, not equal quality. FIFA 26 records are useful for current-player comparison, but true 25/26 FBRef production should be added when available.
