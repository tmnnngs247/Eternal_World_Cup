from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

COUNTRY_FLAGS = {
    "Argentina": "🇦🇷",
    "Brazil": "🇧🇷",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Spain": "🇪🇸",
    "Portugal": "🇵🇹",
    "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪",
    "Italy": "🇮🇹",
    "Croatia": "🇭🇷",
    "Uruguay": "🇺🇾",
    "Norway": "🇳🇴",
    "Poland": "🇵🇱",
    "Morocco": "🇲🇦",
    "Japan": "🇯🇵",
    "South Korea": "🇰🇷",
    "United States": "🇺🇸",
    "Mexico": "🇲🇽",
}

KEEP_COLS = [
    "player_season_id",
    "name_key",
    "short_name",
    "long_name",
    "sofifa_id",
    "image_url",
    "player_url",
    "season_year",
    "season_label",
    "club_name",
    "league_name",
    "nationality_name",
    "flag",
    "player_positions",
    "age",
    "overall",
    "potential",
    "archetype_id",
    "archetype_name",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]


def clean_optional_text(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .replace({
            "nan": "",
            "None": "",
            "<NA>": "",
        })
        .str.strip()
    )


def main() -> None:
    source_path = PROCESSED / "player_embeddings.csv"

    if not source_path.exists():
        raise FileNotFoundError(
            "player_embeddings.csv not found. Run the earlier pipeline stages first."
        )

    players = pd.read_csv(
        source_path,
        low_memory=False,
    )

    emb_cols = [
        column
        for column in players.columns
        if column.startswith("emb_")
    ]

    keep = (
        [column for column in KEEP_COLS if column in players.columns]
        + emb_cols
    )

    app_players = players[keep].copy()

    latest = (
        app_players
        .sort_values("season_year")
        .groupby("name_key", as_index=False)
        .tail(1)
    )

    historical_top = (
        app_players
        .sort_values("overall", ascending=False)
        .head(25000)
    )

    app_players = (
        pd.concat(
            [latest, historical_top],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["player_season_id"],
            keep="first",
        )
    )

    if "flag" not in app_players.columns:
        app_players["flag"] = ""

    app_players["flag"] = clean_optional_text(
        app_players["flag"]
    )

    missing_flag = app_players["flag"].eq("")

    app_players.loc[missing_flag, "flag"] = (
        app_players
        .loc[missing_flag, "nationality_name"]
        .map(COUNTRY_FLAGS)
        .fillna("")
    )

    for column in ["image_url", "player_url"]:
        if column not in app_players.columns:
            app_players[column] = ""

        app_players[column] = clean_optional_text(
            app_players[column]
        )

    if "sofifa_id" not in app_players.columns:
        app_players["sofifa_id"] = pd.NA

    app_players["sofifa_id"] = pd.to_numeric(
        app_players["sofifa_id"],
        errors="coerce",
    ).astype("Int64")

    output_path = PROCESSED / "app_players.csv"

    app_players.to_csv(
        output_path,
        index=False,
    )

    valid_image = (
        app_players["image_url"]
        .fillna("")
        .str.startswith(("http://", "https://"))
    )

    image_coverage_by_season = (
        app_players
        .assign(has_image=valid_image)
        .groupby("season_label")["has_image"]
        .agg(["sum", "count", "mean"])
        .sort_index()
    )

    print("Built app_players.csv")
    print(f"Rows: {len(app_players):,}")
    print(f"Columns: {len(app_players.columns):,}")
    print(f"Rows with image URLs: {valid_image.sum():,}")
    print(f"Image coverage: {valid_image.mean():.1%}")
    print(f"Output: {output_path}")

    print("\nImage coverage by season:")
    print(image_coverage_by_season.tail(12))


if __name__ == "__main__":
    main()