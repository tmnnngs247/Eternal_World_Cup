"""Sanity checks over the committed data/processed outputs.

These validate the pipeline's *output artifacts* rather than re-running the
(slow, ~60s) pipeline itself. Run with `pytest` from the repo root after
`python src/run_pipeline.py`. They exist to catch the kind of regression that
has actually shipped silently before: corrupted accented names, a World Cup
score component that was always zero, and archetype naming collisions.
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

MOJIBAKE_PATTERN = re.compile(r"Ã.|â€")


@pytest.fixture(scope="module")
def players_master() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "players_master.csv", low_memory=False)


@pytest.fixture(scope="module")
def player_embeddings() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "player_embeddings.csv", low_memory=False)


@pytest.fixture(scope="module")
def app_players() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "app_players.csv", low_memory=False)


@pytest.fixture(scope="module")
def archetypes() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "archetypes.csv")


@pytest.fixture(scope="module")
def legend_scores() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "legend_scores.csv", low_memory=False)


def assert_no_mojibake(series: pd.Series, column_name: str) -> None:
    text = series.dropna().astype(str)
    hits = text[text.str.contains(MOJIBAKE_PATTERN, regex=True)]
    assert hits.empty, (
        f"Found {len(hits)} mojibake-looking values in {column_name}, "
        f"e.g. {hits.head(3).tolist()}. A raw CSV may be mis-decoded "
        f"(see build_dataset.py's UTF-8/latin1 fallback)."
    )


class TestPlayersMaster:
    def test_required_columns_present(self, players_master):
        required = {"player_season_id", "short_name", "season_year", "player_positions", "overall"}
        missing = required - set(players_master.columns)
        assert not missing, f"players_master.csv is missing columns: {missing}"

    def test_player_season_id_is_unique(self, players_master):
        duplicates = players_master["player_season_id"].duplicated().sum()
        assert duplicates == 0, f"{duplicates} duplicate player_season_id values"

    def test_no_mojibake_in_names(self, players_master):
        assert_no_mojibake(players_master["short_name"], "short_name")
        assert_no_mojibake(players_master["nationality_name"], "nationality_name")

    def test_overall_and_age_in_sane_ranges(self, players_master):
        overall = pd.to_numeric(players_master["overall"], errors="coerce").dropna()
        age = pd.to_numeric(players_master["age"], errors="coerce").dropna()
        assert overall.between(30, 99).all(), "overall rating outside the plausible 30-99 range"
        # Upper bound is generous: Kazuyoshi Miura was still playing professionally at 54.
        assert age.between(14, 60).all(), "age outside the plausible 14-60 range"


class TestPlayerEmbeddings:
    def test_embedding_columns_present_and_complete(self, player_embeddings):
        emb_cols = [c for c in player_embeddings.columns if c.startswith("emb_")]
        assert len(emb_cols) >= 2, "No emb_ columns found"
        assert player_embeddings[emb_cols].isna().sum().sum() == 0, "NaNs found in embedding columns"

    def test_player_season_id_is_unique(self, player_embeddings):
        duplicates = player_embeddings["player_season_id"].duplicated().sum()
        assert duplicates == 0, f"{duplicates} duplicate player_season_id values"


class TestAppPlayers:
    def test_required_columns_present(self, app_players):
        required = {
            "player_season_id", "short_name", "season_year", "flag",
            "archetype_id", "archetype_name", "has_fbref",
        }
        missing = required - set(app_players.columns)
        assert not missing, f"app_players.csv is missing columns: {missing}"

    def test_player_season_id_is_unique(self, app_players):
        duplicates = app_players["player_season_id"].duplicated().sum()
        assert duplicates == 0, f"{duplicates} duplicate player_season_id values"

    def test_flag_coverage_is_complete(self, app_players):
        missing_flags = app_players["flag"].isna() | app_players["flag"].astype(str).eq("")
        coverage = 1 - missing_flags.mean()
        assert coverage > 0.99, f"Flag coverage dropped to {coverage:.1%} (expected ~100%)"

    def test_image_url_coverage_is_complete(self, app_players):
        valid_image = app_players["image_url"].fillna("").astype(str).str.startswith(("http://", "https://"))
        assert valid_image.mean() > 0.99, f"Image URL coverage dropped to {valid_image.mean():.1%}"

    def test_has_fbref_is_boolean_with_some_true(self, app_players):
        assert app_players["has_fbref"].dtype == bool
        assert app_players["has_fbref"].sum() > 0, "No rows flagged has_fbref=True"

    def test_no_mojibake_in_names(self, app_players):
        assert_no_mojibake(app_players["short_name"], "short_name")
        assert_no_mojibake(app_players["nationality_name"], "nationality_name")


class TestArchetypes:
    def test_archetype_names_are_unique(self, archetypes):
        duplicates = archetypes["archetype_name"].duplicated().sum()
        assert duplicates == 0, (
            f"{duplicates} duplicate archetype names -- the naming heuristic in "
            f"archetypes.py collided and fell back to a numbered suffix"
        )

    def test_every_cluster_has_players(self, archetypes):
        assert (archetypes["player_count"] > 0).all(), "An archetype cluster has zero players"


class TestLegendScore:
    def test_world_cup_component_is_not_dead(self, legend_scores):
        """Regression guard: wc_apps/wc_goals silently defaulted to 0 for every
        player for a long time because no roster data was ever wired in."""
        has_wc_data = pd.to_numeric(legend_scores.get("wc_apps"), errors="coerce").fillna(0).gt(0)
        assert has_wc_data.sum() > 0, "No player has wc_apps > 0 -- World Cup roster data may be missing"

    def test_scores_in_valid_range(self, legend_scores):
        scores = pd.to_numeric(legend_scores["legend_style_score"], errors="coerce").dropna()
        assert scores.between(0, 100).all(), "legend_style_score outside the expected 0-100 range"
