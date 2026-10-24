"""Pins for the player-grain FantasyPoints expansion recipes.

The ``ps_*`` QB context splits and the expected-TD / RYOE / red-zone share
rates read the new API's ``__raw`` counts under their unprefixed names, so
each source name must be in the collector's per-tool vocabulary and must
pool as ``sum(num) / sum(den)`` over the window, not as a mean of per-game
rates.
"""

import json
from pathlib import Path

import pandas as pd

from sportstradamus.stats import nfl_fp_weekly
from sportstradamus.stats.nfl_fp_aggregation import PLAYER_GROUP_COL
from sportstradamus.stats.nfl_fp_weekly_aggregate import _AGGREGATE_RECIPES, _apply_recipes

_TOOL_COLUMNS = Path(__file__).parent / "fixtures" / "fantasypoints_tool_columns.json"

# file_kind -> the collector tool whose column vocabulary the fixture records.
_TOOL_BY_KIND = {
    "passing_situation": "passing-situation",
    "passing_basic": "passing",
    "rushing_basic": "rushing",
    "receiving_basic": "receiving",
}

_EXPANSION_OUTPUTS = frozenset(
    {
        "ps_blitz_share",
        "ps_man_share",
        "ps_two_high_share",
        "ps_pa_share",
        "ps_motion_share",
        "ps_pressured_ypa",
        "ps_clean_ypa",
        "ps_blitz_ypa",
        "ps_man_ypa",
        "ps_zone_ypa",
        "ps_two_high_ypa",
        "ps_pa_ypa",
        "ps_pressured_epa_per_db",
        "ps_clean_epa_per_db",
        "ps_blitz_int_rate",
        "ps_pressured_int_rate",
        "ps_scramble_yards_per_db",
        "pass_xtd_per_att",
        "pass_endzone_att_rate",
        "rush_xtd_per_att",
        "rush_ryoe_per_att",
        "rush_rz10_carry_share",
        "rush_rz5_carry_share",
        "rec_xtd_per_target",
        "rec_xyards_per_target",
    }
)


def _recipe(output_col):
    (recipe,) = [r for r in _AGGREGATE_RECIPES if r.output_col == output_col]
    return recipe


def test_rate_pools_counts_across_games_before_dividing():
    per_game = pd.DataFrame(
        {
            PLAYER_GROUP_COL: ["qb1", "qb1"],
            "pressured_yards": [100, 50],
            "pressured_attempts": [10, 20],
        }
    )
    pooled = _apply_recipes(per_game, [_recipe("ps_pressured_ypa")])
    # 150 / 30, not mean(10.0, 2.5) = 6.25
    assert pooled.loc["qb1", "ps_pressured_ypa"] == 5.0


def test_expansion_recipe_sources_are_in_the_tool_vocabulary():
    vocabulary = json.loads(_TOOL_COLUMNS.read_text())
    recipes = [r for r in _AGGREGATE_RECIPES if r.output_col in _EXPANSION_OUTPUTS]
    assert {r.output_col for r in recipes} == _EXPANSION_OUTPUTS
    missing = {
        (r.output_col, arg)
        for r in recipes
        for arg in r.args
        if arg not in vocabulary[_TOOL_BY_KIND[r.file_kind]]
    }
    assert not missing


def test_recipe_with_absent_source_column_is_skipped():
    per_game = pd.DataFrame({PLAYER_GROUP_COL: ["rb1"], "attempts": [12]})
    assert list(_apply_recipes(per_game, [_recipe("rush_ryoe_per_att")]).columns) == []


def test_passing_situation_is_a_registered_file_kind():
    assert nfl_fp_weekly.FILE_KINDS["passing_situation"] == "passing_situation.parquet"
