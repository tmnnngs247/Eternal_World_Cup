from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from utils import normalize_name, country_to_flag, ABBR_TO_COUNTRY

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

FEATURE_MAP = {
    "pace": ["pace", "pac"],
    "shooting": ["shooting", "sho"],
    "passing": ["passing", "pas"],
    "dribbling": ["dribbling", "dri"],
    "defending": ["defending", "def"],
    "physic": ["physic", "phy"],
    "ball_control": ["skill_ball_control", "ball_control"],
    "skill_dribbling": ["skill_dribbling", "dribbling"],
    "crossing": ["attacking_crossing", "crossing"],
    "finishing": ["attacking_finishing", "finishing"],
    "short_passing": ["attacking_short_passing", "short_pass"],
    "volleys": ["attacking_volleys", "volleys"],
    "long_passing": ["skill_long_passing", "long_pass"],
    "acceleration": ["movement_acceleration", "acceleration"],
    "sprint_speed": ["movement_sprint_speed", "sprint_speed"],
    "agility": ["movement_agility", "agility"],
    "reactions": ["movement_reactions", "reactions"],
    "balance": ["movement_balance", "balance"],
    "shot_power": ["power_shot_power", "shot_power"],
    "jumping": ["power_jumping", "jumping"],
    "stamina": ["power_stamina", "stamina"],
    "strength": ["power_strength", "strength"],
    "long_shots": ["power_long_shots", "long_shots"],
    "aggression": ["mentality_aggression", "aggression"],
    "interceptions": ["mentality_interceptions", "interceptions"],
    "positioning": ["mentality_positioning", "att_position"],
    "vision": ["mentality_vision", "vision"],
    "composure": ["mentality_composure", "composure"],
    "marking": ["defending_marking_awareness", "marking"],
    "standing_tackle": ["defending_standing_tackle", "stand_tackle"],
    "sliding_tackle": ["defending_sliding_tackle", "slide_tackle"],
}


def first_existing(df: pd.DataFrame, names: list[str], default=np.nan):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)


def standardise_frame(df: pd.DataFrame, season: int, source: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["season_year"] = season
    out["source_file"] = source
    out["short_name"] = first_existing(df, ["short_name", "player", "player_name", "Name"])
    out["long_name"] = first_existing(df, ["long_name", "short_name", "player", "player_name"])
    out["club_name"] = first_existing(df, ["club_name", "club"])
    out["league_name"] = first_existing(df, ["league_name", "league"])
    out["nationality_name"] = first_existing(df, ["nationality_name", "nationality", "country"])
    out["player_positions"] = first_existing(df, ["player_positions", "position", "positions"])
    out["age"] = pd.to_numeric(first_existing(df, ["age", "age_fifa"]), errors="coerce")
    out["overall"] = pd.to_numeric(first_existing(df, ["overall", "ovr"]), errors="coerce")
    out["potential"] = pd.to_numeric(first_existing(df, ["potential"]), errors="coerce")
    out["height_cm"] = pd.to_numeric(first_existing(df, ["height_cm", "height", "height_(in cm)"]), errors="coerce")
    out["weight_kg"] = pd.to_numeric(first_existing(df, ["weight_kg", "weight", "weight_(in kg)"]), errors="coerce")
    out["preferred_foot"] = first_existing(df, ["preferred_foot", "preffered_foot"])
    out["weak_foot"] = pd.to_numeric(first_existing(df, ["weak_foot"]), errors="coerce")
    out["skill_moves"] = pd.to_numeric(first_existing(df, ["skill_moves"]), errors="coerce")
    out["international_reputation"] = pd.to_numeric(first_existing(df, ["international_reputation"]), errors="coerce")
    out["image_url"] = first_existing(df, ["image_url", "player_face_url"], default="")
    for feat, candidates in FEATURE_MAP.items():
        out[feat] = pd.to_numeric(first_existing(df, candidates), errors="coerce")
    # FBRef-like extras if available
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["xg", "xag", "sca", "gca", "prgc", "prgp", "progressive", "90s"]):
            if c not in out.columns:
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().sum() > 50:
                    out[f"perf_{c}"] = vals
    out["name_key"] = out["short_name"].map(normalize_name)
    out["season_label"] = "FIFA " + out["season_year"].astype(str).str[-2:]
    out["player_season_id"] = out["name_key"] + "_" + out["season_year"].astype(str)
    out["flag"] = out["nationality_name"].map(country_to_flag)
    return out


def load_roster_metadata() -> pd.DataFrame:
    candidates = [RAW / "worldcup_rosters.json", RAW / "Pasted text(3).txt"]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            for nat, team in data.items():
                for p in team.get("players", []):
                    rows.append({
                        "roster_nat_abbr": nat,
                        "roster_country": ABBR_TO_COUNTRY.get(nat, nat),
                        "roster_id": p.get("id"),
                        "roster_name": p.get("name"),
                        "name_key": normalize_name(p.get("name")),
                        "roster_pos": p.get("pos"),
                        "roster_no": p.get("no"),
                        "roster_club": p.get("club"),
                        "roster_caps": p.get("caps"),
                        "roster_goals": p.get("goals"),
                        "roster_wiki": p.get("wiki"),
                        "wc_apps": p.get("wcApps"),
                        "wc_goals": p.get("wcGoals"),
                        "roster_flag": country_to_flag(ABBR_TO_COUNTRY.get(nat, nat)),
                    })
            return pd.DataFrame(rows)
    return pd.DataFrame()


def build() -> None:
    files = {
        2015: "players_15.csv", 2016: "players_16.csv", 2017: "players_17.csv", 2018: "players_18.csv",
        2019: "players_19.csv", 2020: "players_20.csv", 2021: "players_21.csv", 2022: "players_22.csv",
        2023: "player_23.csv", 2024: "player_24.csv", 2025: "players_25.csv", 2026: "players_26.csv",
    }
    frames = []
    for season, fname in files.items():
        p = RAW / fname
        if p.exists():
            frames.append(standardise_frame(pd.read_csv(p, encoding="latin1"), season, fname))
    fbref = RAW / "fifa_fbref_merged.csv"
    if fbref.exists():
        df = pd.read_csv(fbref, encoding="latin1")
        if "fifa_version" in df.columns:
            for season, sub in df.groupby("fifa_version"):
                frames.append(standardise_frame(sub, int(season), "fifa_fbref_merged.csv"))
    if not frames:
        raise FileNotFoundError("No raw player files found in data/raw")
    players = pd.concat(frames, ignore_index=True)
    players = players.dropna(subset=["short_name"]).copy()
    players = players.drop_duplicates(subset=["player_season_id", "club_name", "overall"], keep="first")
    roster = load_roster_metadata()
    if not roster.empty:
        roster.to_csv(PROCESSED / "player_metadata.csv", index=False)
        players = players.merge(roster.drop_duplicates("name_key"), on="name_key", how="left")
        players["flag"] = players["roster_flag"].fillna(players["flag"])
    else:
        pd.DataFrame().to_csv(PROCESSED / "player_metadata.csv", index=False)
    # Keep columns that are mostly useful for the app
    players.to_csv(PROCESSED / "players_master.csv", index=False)
    print(f"Built players_master: {players.shape}")

if __name__ == "__main__":
    build()
