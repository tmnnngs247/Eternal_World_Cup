"""
Build clean app/model-ready datasets from raw FIFA, FBRef and World Cup files.

Run from repo root:
    python src/data_build.py
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR, FIFA_CORE_FEATURES, FBREF_FEATURE_PATTERNS


def normalise_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def season_from_filename(path: Path) -> int | None:
    m = re.search(r"(\d{2})", path.stem)
    return 2000 + int(m.group(1)) if m else None


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Missing: {path.name}")
        return None
    return pd.read_csv(path, low_memory=False)


def standardise_sofifa(df: pd.DataFrame, season: int, source_file: str) -> pd.DataFrame:
    out = df.copy()
    if "player_id" in out.columns and "sofifa_id" not in out.columns:
        out = out.rename(columns={"player_id": "sofifa_id"})
    out["fifa_version"] = season - 2000
    out["season_year"] = season
    out["source_file"] = source_file
    keep = [
        "sofifa_id", "short_name", "long_name", "player_positions", "overall", "potential", "age", "dob",
        "height_cm", "weight_kg", "club_name", "league_name", "nationality_name", "preferred_foot",
        "weak_foot", "skill_moves", "international_reputation", "player_face_url", "club_logo_url", "nation_flag_url",
        "fifa_version", "season_year", "source_file",
    ]
    features = [c for c in FIFA_CORE_FEATURES if c in out.columns]
    keep = [c for c in keep if c in out.columns] + features
    out = out[keep].copy()
    if "short_name" not in out.columns and "player" in out.columns:
        out["short_name"] = out["player"]
    out["player_key"] = out.get("long_name", out.get("short_name", "")).map(normalise_name)
    return out


def standardise_simple_fifa(df: pd.DataFrame, season: int, source_file: str) -> pd.DataFrame:
    rename = {
        "player": "short_name", "player_name": "short_name", "country": "nationality_name", "nationality": "nationality_name",
        "club": "club_name", "height": "height_cm", "height_(in cm)": "height_cm", "weight": "weight_kg", "weight_(in kg)": "weight_kg",
        "ovr": "overall", "pac": "pace", "sho": "shooting", "pas": "passing", "dri": "dribbling", "def": "defending", "phy": "physic",
        "image_url": "player_face_url", "position": "player_positions", "preffered_foot": "preferred_foot",
        "ball_control": "skill_ball_control", "short_pass": "attacking_short_passing", "long_pass": "skill_long_passing",
        "acceleration": "movement_acceleration", "sprint_speed": "movement_sprint_speed", "agility": "movement_agility", "balance": "movement_balance",
        "reactions": "movement_reactions", "stamina": "power_stamina", "strength": "power_strength", "jumping": "power_jumping",
        "shot_power": "power_shot_power", "long_shots": "power_long_shots", "finishing": "attacking_finishing", "heading": "attacking_heading_accuracy",
        "crossing": "attacking_crossing", "curve": "skill_curve", "fk_acc": "skill_fk_accuracy", "aggression": "mentality_aggression",
        "interceptions": "mentality_interceptions", "att_position": "mentality_positioning", "vision": "mentality_vision", "penalties": "mentality_penalties",
        "composure": "mentality_composure", "stand_tackle": "defending_standing_tackle", "slide_tackle": "defending_sliding_tackle", "marking": "defending_marking_awareness",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    out["fifa_version"] = season - 2000
    out["season_year"] = season
    out["source_file"] = source_file
    keep = [
        "short_name", "long_name", "player_positions", "overall", "potential", "age", "height_cm", "weight_kg",
        "club_name", "league_name", "nationality_name", "preferred_foot", "weak_foot", "skill_moves", "international_reputation",
        "player_face_url", "fifa_version", "season_year", "source_file",
    ]
    features = [c for c in FIFA_CORE_FEATURES if c in out.columns]
    out = out[[c for c in keep if c in out.columns] + features].copy()
    if "long_name" not in out.columns:
        out["long_name"] = out["short_name"]
    out["player_key"] = out["long_name"].map(normalise_name)
    return out


def build_players_master() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # Rich SoFIFA seasons 15-22 and 26
    for year in list(range(2015, 2023)) + [2026]:
        fname = f"players_{str(year)[-2:]}.csv"
        df = read_csv_if_exists(RAW_DIR / fname)
        if df is not None:
            frames.append(standardise_sofifa(df, year, fname))

    # Simpler 23/24 files
    for year, fname in [(2023, "player_23.csv"), (2024, "player_24.csv"), (2025, "players_25.csv")]:
        df = read_csv_if_exists(RAW_DIR / fname)
        if df is not None:
            frames.append(standardise_simple_fifa(df, year, fname))

    if not frames:
        raise FileNotFoundError("No raw FIFA files found in data/raw")

    master = pd.concat(frames, ignore_index=True, sort=False)
    master["player_season_id"] = (
        master["player_key"].astype(str) + "_" + master["season_year"].astype(str) + "_" + master.get("club_name", "").fillna("").map(normalise_name)
    )
    return master


def build_fbref_enriched(master: pd.DataFrame) -> pd.DataFrame:
    fb = read_csv_if_exists(RAW_DIR / "fifa_fbref_merged.csv")
    if fb is None:
        master["has_fbref"] = False
        return master
    fb = fb.copy()
    fb["player_key"] = fb.get("long_name", fb.get("player", fb.get("short_name", ""))).map(normalise_name)
    fb["season_year"] = np.where(fb["fifa_version"].notna(), 2000 + fb["fifa_version"].astype(int), np.nan)
    fb_cols = ["player_key", "season_year", "season", "team", "league", "nation", "pos", "Playing Time_90s"]
    fb_cols += [c for c in FBREF_FEATURE_PATTERNS if c in fb.columns]
    fb_small = fb[[c for c in fb_cols if c in fb.columns]].drop_duplicates(["player_key", "season_year"])
    merged = master.merge(fb_small, on=["player_key", "season_year"], how="left")
    merged["has_fbref"] = merged.get("Playing Time_90s", pd.Series(index=merged.index)).notna()
    return merged


def build_roster_metadata() -> pd.DataFrame | None:
    path = RAW_DIR / "world_cup_rosters_2026.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for team_code, team in data.items():
        for p in team.get("players", []):
            rows.append({
                "player_slug": p.get("id"), "short_name": p.get("name"), "long_name": p.get("name"),
                "player_key": normalise_name(p.get("name")), "team_code": team_code, "position_roster": p.get("pos"),
                "shirt_number": p.get("no"), "date_of_birth": p.get("dob"), "caps": p.get("caps"),
                "international_goals": p.get("goals"), "club_roster": p.get("club"), "club_country_code": p.get("clubNat"),
                "wiki_url": p.get("wiki"), "world_cup_apps_2026": p.get("wcApps"), "world_cup_goals_2026": p.get("wcGoals"),
                "world_cup_yellow_2026": p.get("wcYellow"), "world_cup_red_2026": p.get("wcRed"), "coach": team.get("coach"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master = build_players_master()
    master = build_fbref_enriched(master)
    master.to_csv(PROCESSED_DIR / "players_master.csv", index=False)
    print(f"Wrote {PROCESSED_DIR / 'players_master.csv'}: {master.shape}")

    roster = build_roster_metadata()
    if roster is not None:
        roster.to_csv(PROCESSED_DIR / "player_roster_metadata_2026.csv", index=False)
        print(f"Wrote roster metadata: {roster.shape}")

if __name__ == "__main__":
    main()
