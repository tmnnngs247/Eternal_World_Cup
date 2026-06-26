from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

KEEP_COLS = [
    "player_season_id", "name_key", "short_name", "long_name",
    "season_year", "season_label", "club_name", "league_name",
    "nationality_name", "flag", "player_positions", "age",
    "overall", "potential", "archetype_id", "archetype_name",
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
]

def main() -> None:
    players = pd.read_csv(PROCESSED / "player_embeddings.csv", low_memory=False)

    emb_cols = [c for c in players.columns if c.startswith("emb_")]
    keep = [c for c in KEEP_COLS if c in players.columns] + emb_cols

    app_players = players[keep].copy()

    # keep all latest player-seasons + top historical player-seasons
    latest = app_players.sort_values("season_year").groupby("name_key", as_index=False).tail(1)
    historical_top = app_players.sort_values("overall", ascending=False).head(25000)

    app_players = (
        pd.concat([latest, historical_top], ignore_index=True)
        .drop_duplicates("player_season_id")
    )

    app_players.to_csv(PROCESSED / "app_players.csv", index=False)

    print("Built app_players.csv")
    print(f"Rows: {len(app_players):,}")
    print(f"Columns: {len(app_players.columns):,}")
    print(f"Output: {PROCESSED / 'app_players.csv'}")

if __name__ == "__main__":
    main()