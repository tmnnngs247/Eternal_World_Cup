from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def infer_season(path: Path) -> int | None:
    match = re.search(r"players?_(\d{2})\.csv", path.name)
    if not match:
        return None
    return 2000 + int(match.group(1))


def first_existing(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    for col in cols:
        if col in df.columns:
            return df[col]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def clean_key(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def standardise_fifa_file(path: Path) -> pd.DataFrame:
    season = infer_season(path)
    if season is None:
        raise ValueError(f"Could not infer season from {path.name}")

    print(f"Reading {path.name}")
    df = pd.read_csv(path, low_memory=False, encoding="latin1")

    out = pd.DataFrame(index=df.index)
    out["source_file"] = path.name
    out["season_year"] = season
    out["season_label"] = f"FIFA {str(season)[-2:]}"
    out["short_name"] = first_existing(df, ["short_name", "Name", "name", "player_name"])
    out["long_name"] = first_existing(df, ["long_name", "full_name", "short_name", "Name", "name"])
    out["club_name"] = first_existing(df, ["club_name", "club", "Club"])
    out["league_name"] = first_existing(df, ["league_name", "league"])
    out["nationality_name"] = first_existing(df, ["nationality_name", "nationality", "country"])
    out["player_positions"] = first_existing(df, ["player_positions", "positions", "position"])

    numeric_cols = {
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
        "acceleration": ["movement_acceleration", "acceleration"],
        "sprint_speed": ["movement_sprint_speed", "sprint_speed"],
        "finishing": ["attacking_finishing", "finishing"],
        "short_passing": ["attacking_short_passing", "short_passing"],
        "long_passing": ["skill_long_passing", "long_passing"],
        "ball_control": ["skill_ball_control", "ball_control"],
        "agility": ["movement_agility", "agility"],
        "reactions": ["movement_reactions", "reactions"],
        "balance": ["movement_balance", "balance"],
        "shot_power": ["power_shot_power", "shot_power"],
        "stamina": ["power_stamina", "stamina"],
        "strength": ["power_strength", "strength"],
        "vision": ["mentality_vision", "vision"],
        "composure": ["mentality_composure", "composure"],
        "interceptions": ["mentality_interceptions", "interceptions"],
        "standing_tackle": ["defending_standing_tackle", "standing_tackle"],
        "sliding_tackle": ["defending_sliding_tackle", "sliding_tackle"],
    }

    for new_col, candidates in numeric_cols.items():
        out[new_col] = pd.to_numeric(first_existing(df, candidates), errors="coerce")

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


def main() -> None:
    fifa_files = sorted(
        list(RAW.glob("players_*.csv")) +
        list(RAW.glob("player_*.csv"))
    )

    fifa_files = [f for f in fifa_files if infer_season(f) is not None]

    if not fifa_files:
        raise FileNotFoundError("No FIFA player files found in data/raw")

    print("Found FIFA files:")
    for f in fifa_files:
        print(f"  - {f.name}")

    frames = [standardise_fifa_file(f) for f in fifa_files]

    players = pd.concat(frames, ignore_index=True)
    players = players.dropna(subset=["short_name"])
    players = players.drop_duplicates(subset=["player_season_id"], keep="first")

    out_csv = PROCESSED / "players_master.csv"
    players.to_csv(out_csv, index=False)

    print("\nBuilt players_master.csv")
    print(f"Rows: {len(players):,}")
    print(f"Columns: {len(players.columns):,}")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()