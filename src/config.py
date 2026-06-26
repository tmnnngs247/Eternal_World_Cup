from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIFA_CORE_FEATURES = [
    "overall", "potential", "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_reactions", "movement_balance",
    "power_shot_power", "power_jumping", "power_stamina", "power_strength", "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
]

FBREF_FEATURE_PATTERNS = [
    "Per 90 Minutes_Gls", "Per 90 Minutes_Ast", "Per 90 Minutes_xG", "Per 90 Minutes_xAG",
    "Per 90 Minutes_PrgC", "Per 90 Minutes_PrgP", "Per 90 Minutes_PrgR",
    "Standard_Sh/90", "Standard_SoT/90", "Per 90 Minutes_KP", "SCA_SCA90", "GCA_GCA90",
    "Per 90 Minutes_Tackles_Tkl", "Per 90 Minutes_Int", "Per 90 Minutes_Blocks_Blocks", "Per 90 Minutes_Clr",
]
