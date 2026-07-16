from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from config import FBREF_FEATURE_PATTERNS


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def infer_season(path: Path) -> int | None:
    """Infer FIFA season year from filenames such as players_25.csv."""
    match = re.search(r"players?_(\d{2})\.csv", path.name)

    if not match:
        return None

    return 2000 + int(match.group(1))


def first_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first available column, or a blank Series if none exist."""
    for column in columns:
        if column in df.columns:
            return df[column]

    return pd.Series(
        [None] * len(df),
        index=df.index,
        dtype="object",
    )


def clean_key(series: pd.Series) -> pd.Series:
    """Create a normalised lowercase key for joining and deduplication."""
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def clean_optional_text(series: pd.Series) -> pd.Series:
    """Clean optional text fields such as image and profile URLs."""
    return (
        series
        .fillna("")
        .astype(str)
        .replace(
            {
                "nan": "",
                "None": "",
                "<NA>": "",
            }
        )
        .str.strip()
    )


def standardise_fifa_file(path: Path) -> pd.DataFrame:
    """Load and standardise one FIFA player file."""
    season = infer_season(path)

    if season is None:
        raise ValueError(f"Could not infer season from {path.name}")

    print(f"Reading {path.name}")

    df = pd.read_csv(
        path,
        low_memory=False,
        encoding="latin1",
    )

    out = pd.DataFrame(index=df.index)

    out["source_file"] = path.name
    out["season_year"] = season
    out["season_label"] = f"FIFA {str(season)[-2:]}"

    out["short_name"] = first_existing(
        df,
        ["short_name", "Name", "name", "player_name"],
    )

    out["long_name"] = first_existing(
        df,
        ["long_name", "full_name", "short_name", "Name", "name"],
    )

    out["club_name"] = first_existing(
        df,
        ["club_name", "club", "Club"],
    )

    out["league_name"] = first_existing(
        df,
        ["league_name", "league"],
    )

    out["nationality_name"] = first_existing(
        df,
        ["nationality_name", "nationality", "country"],
    )

    out["player_positions"] = first_existing(
        df,
        ["player_positions", "positions", "position"],
    )

    # Optional identity and image fields.
    # Older FIFA files may not contain these, so blank values are expected.
    out["sofifa_id"] = pd.to_numeric(
        first_existing(
            df,
            ["sofifa_id", "player_id", "id"],
        ),
        errors="coerce",
    )

    out["image_url"] = clean_optional_text(
        first_existing(
            df,
            ["image_url", "player_face_url", "face_url"],
        )
    )

    out["player_url"] = clean_optional_text(
        first_existing(
            df,
            ["player_url", "sofifa_url", "url"],
        )
    )

    numeric_columns = {
        "age": ["age"],
        "overall": ["overall", "OVR", "ovr"],
        "potential": ["potential", "POT", "pot"],
        "value_eur": ["value_eur"],
        "wage_eur": ["wage_eur"],
        "pace": ["pace", "PAC", "pac"],
        "shooting": ["shooting", "SHO", "sho"],
        "passing": ["passing", "PAS", "pas"],
        "dribbling": ["dribbling", "DRI", "dri"],
        "defending": ["defending", "DEF", "def"],
        "physic": ["physic", "PHY", "phy"],
        "acceleration": [
            "movement_acceleration",
            "acceleration",
        ],
        "sprint_speed": [
            "movement_sprint_speed",
            "sprint_speed",
        ],
        "finishing": [
            "attacking_finishing",
            "finishing",
        ],
        "short_passing": [
            "attacking_short_passing",
            "short_passing",
        ],
        "long_passing": [
            "skill_long_passing",
            "long_passing",
        ],
        "ball_control": [
            "skill_ball_control",
            "ball_control",
        ],
        "agility": [
            "movement_agility",
            "agility",
        ],
        "reactions": [
            "movement_reactions",
            "reactions",
        ],
        "balance": [
            "movement_balance",
            "balance",
        ],
        "shot_power": [
            "power_shot_power",
            "shot_power",
        ],
        "stamina": [
            "power_stamina",
            "stamina",
        ],
        "strength": [
            "power_strength",
            "strength",
        ],
        "vision": [
            "mentality_vision",
            "vision",
        ],
        "composure": [
            "mentality_composure",
            "composure",
        ],
        "interceptions": [
            "mentality_interceptions",
            "interceptions",
        ],
        "standing_tackle": [
            "defending_standing_tackle",
            "standing_tackle",
        ],
        "sliding_tackle": [
            "defending_sliding_tackle",
            "sliding_tackle",
        ],
    }

    for new_column, candidates in numeric_columns.items():
        out[new_column] = pd.to_numeric(
            first_existing(df, candidates),
            errors="coerce",
        )

    out["name_key"] = clean_key(out["short_name"])
    club_key = clean_key(out["club_name"])

    out["player_season_id"] = (
        out["name_key"].fillna("unknown")
        + "_"
        + out["season_year"].astype(str)
        + "_"
        + club_key.fillna("")
    )

    return out


def load_fbref_performance() -> pd.DataFrame | None:
    """Load per-90 FBRef performance stats, keyed to match players_master rows."""
    path = RAW / "fifa_fbref_merged.csv"

    if not path.exists():
        print(f"Missing: {path.name} (no FBRef performance data)")
        return None

    fbref = pd.read_csv(path, low_memory=False)

    fbref["name_key"] = clean_key(fbref["short_name"])
    fbref["season_year"] = 2000 + pd.to_numeric(fbref["fifa_version"], errors="coerce")

    feature_cols = [c for c in FBREF_FEATURE_PATTERNS if c in fbref.columns]

    fbref = fbref[["name_key", "season_year"] + feature_cols].dropna(subset=["season_year"])
    fbref = fbref.drop_duplicates(subset=["name_key", "season_year"], keep="first")
    fbref["season_year"] = fbref["season_year"].astype(int)

    return fbref.rename(columns={c: f"perf_{c}" for c in feature_cols})


def main() -> None:
    fifa_files = sorted(
        list(RAW.glob("players_*.csv"))
        + list(RAW.glob("player_*.csv"))
    )

    fifa_files = [
        file
        for file in fifa_files
        if infer_season(file) is not None
    ]

    if not fifa_files:
        raise FileNotFoundError(
            "No FIFA player files found in data/raw"
        )

    print("Found FIFA files:")

    for file in fifa_files:
        print(f"  - {file.name}")

    frames = [
        standardise_fifa_file(file)
        for file in fifa_files
    ]

    players = pd.concat(
        frames,
        ignore_index=True,
    )

    players = players.dropna(
        subset=["short_name"]
    )

    players = players.drop_duplicates(
        subset=["player_season_id"],
        keep="first",
    )

    fbref = load_fbref_performance()
    has_fbref = None

    if fbref is not None:
        players = players.merge(fbref, on=["name_key", "season_year"], how="left")
        perf_cols = [c for c in players.columns if c.startswith("perf_")]
        has_fbref = players[perf_cols].notna().any(axis=1) if perf_cols else None

    output_path = PROCESSED / "players_master.csv"

    players.to_csv(
        output_path,
        index=False,
    )

    valid_image_urls = (
        players["image_url"]
        .fillna("")
        .astype(str)
        .str.startswith(("http://", "https://"))
    )

    print("\nBuilt players_master.csv")
    print(f"Rows: {len(players):,}")
    print(f"Columns: {len(players.columns):,}")
    print(f"Rows with image URLs: {valid_image_urls.sum():,}")
    print(f"Image coverage: {valid_image_urls.mean():.1%}")

    if has_fbref is not None:
        print(f"Rows with FBRef performance data: {has_fbref.sum():,}")
        print(f"FBRef coverage: {has_fbref.mean():.1%}")

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()