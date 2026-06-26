from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

ARCHETYPE_NAMES = {
    0: "Goal Threat Forward", 1: "Ball-Carrying Creator", 2: "Tempo Controller", 3: "Defensive Midfield Anchor",
    4: "Aggressive Centre Back", 5: "Modern Fullback", 6: "Direct Wide Runner", 7: "Pressing Forward",
    8: "Creative 10", 9: "Box Striker", 10: "Deep Playmaker", 11: "Inverted Fullback",
    12: "Sweeper Keeper", 13: "Shot Stopper", 14: "Progressive Centre Back", 15: "Ball-Winning 8",
    16: "Hybrid Wide Creator", 17: "Physical Target Forward", 18: "High-Volume Progressor", 19: "Defensive Utility Player",
    20: "Elite Attacking Midfielder", 21: "Recovery Defender", 22: "Two-Way Fullback", 23: "Development Prospect",
}

def val(row, col):
    try:
        x = row.get(col, 0)
        return 0 if pd.isna(x) else float(x)
    except Exception:
        return 0


def assign(row) -> int:
    pos = str(row.get("player_positions", "")).upper()
    age = val(row, "age")
    overall = val(row, "overall")
    pace = val(row, "pace") or (val(row, "acceleration") + val(row, "sprint_speed")) / 2
    sho = val(row, "shooting") or (val(row, "finishing") + val(row, "shot_power")) / 2
    pas = val(row, "passing") or (val(row, "short_passing") + val(row, "long_passing") + val(row, "vision")) / 3
    dri = val(row, "dribbling") or (val(row, "ball_control") + val(row, "skill_dribbling")) / 2
    deff = val(row, "defending") or (val(row, "standing_tackle") + val(row, "interceptions") + val(row, "marking")) / 3
    phy = val(row, "physic") or (val(row, "strength") + val(row, "stamina") + val(row, "aggression")) / 3

    if age and age <= 20 and overall >= 72:
        return 23
    if "GK" in pos:
        return 12 if pas >= 55 or pace >= 55 else 13
    if any(p in pos for p in ["CB"]):
        if pas >= 65 and dri >= 55: return 14
        if pace >= 72: return 21
        return 4
    if any(p in pos for p in ["RB", "LB", "RWB", "LWB"]):
        if pas >= 70 and dri >= 70: return 11
        if pace >= 78 and deff >= 65: return 22
        return 5
    if any(p in pos for p in ["CDM"]):
        return 10 if pas >= deff else 3
    if any(p in pos for p in ["CM"]):
        if pas >= 78 and dri >= 76: return 2
        if deff >= 70 and phy >= 70: return 15
        if pace >= 72 and pas >= 72: return 18
        return 19
    if any(p in pos for p in ["CAM", "CF"]):
        if overall >= 84: return 20
        return 8
    if any(p in pos for p in ["RW", "LW", "RM", "LM"]):
        if dri >= 82 and pas >= 75: return 16
        if pace >= 84 and sho >= 70: return 6
        return 1
    if any(p in pos for p in ["ST"]):
        if phy >= 78 and sho >= 75: return 17
        if pace >= 78 and phy >= 70: return 7
        if sho >= 78: return 9
        return 0
    return 18


def main():
    df = pd.read_csv(PROCESSED / "player_embeddings.csv", low_memory=False)
    df["archetype_id"] = df.apply(assign, axis=1)
    df["archetype_name"] = df["archetype_id"].map(ARCHETYPE_NAMES)
    df.to_csv(PROCESSED / "player_embeddings.csv", index=False)
    summary = []
    for aid, sub in df.groupby("archetype_id"):
        examples = sub.sort_values("overall", ascending=False).head(8)["short_name"].tolist()
        summary.append({
            "archetype_id": aid,
            "archetype_name": ARCHETYPE_NAMES.get(int(aid), f"Archetype {aid}"),
            "player_count": len(sub),
            "avg_overall": round(pd.to_numeric(sub["overall"], errors="coerce").mean(), 2),
            "example_players": ", ".join(map(str, examples)),
        })
    pd.DataFrame(summary).sort_values("archetype_id").to_csv(PROCESSED / "archetypes.csv", index=False)
    print(f"Built {len(summary)} archetypes")

if __name__ == "__main__":
    main()
