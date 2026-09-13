"""Behavior guards for ``persist.write_current_game_corr``'s rho / modifier split.

``find_correlation`` fills one flat ``corr_sink`` list with both the leg-pair
correlation and the DFS-app pair payout modifier prophecize priced with. The
writer fans that out into two on-disk snapshots with different dedup keys: rho
is platform- and line-independent, but a pair modifier is platform-specific
(banned_combos.json is keyed per platform).
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.helpers.io import PAIR_MODIFIER_COLS
from sportstradamus.prediction import persist


def _row(platform, league, game, leg_a, leg_b, rho, modifier):
    return {
        "Platform": platform,
        "League": league,
        "Game": game,
        "leg_a": leg_a,
        "leg_b": leg_b,
        "rho": rho,
        "modifier": modifier,
    }


@pytest.fixture
def redirected_paths(tmp_path, monkeypatch):
    corr_path = tmp_path / "current_game_corr.parquet"
    mod_path = tmp_path / "current_pair_modifiers.parquet"
    monkeypatch.setattr(persist, "CURRENT_GAME_CORR_PATH", corr_path)
    monkeypatch.setattr(persist, "CURRENT_PAIR_MODIFIERS_PATH", mod_path)
    return corr_path, mod_path


def test_rho_file_has_five_cols_and_dedups_across_platforms_and_lines(redirected_paths):
    """Same pair, two platforms x two alt lines -> one rho row (line-independent)."""
    corr_path, _ = redirected_paths
    rows = [
        _row("Underdog", "WNBA", "LVA/NYL", "A|PTS|Over", "B|PTS|Over", 0.3, 0.83),
        _row("Underdog", "WNBA", "LVA/NYL", "A|PTS|Over", "B|PTS|Over", 0.3, 0.83),
        _row("Sleeper", "WNBA", "LVA/NYL", "A|PTS|Over", "B|PTS|Over", 0.3, 0.0),
        _row("Sleeper", "WNBA", "LVA/NYL", "A|PTS|Over", "B|PTS|Over", 0.3, 0.0),
    ]

    persist.write_current_game_corr(rows)

    corr_df = pd.read_parquet(corr_path)
    assert list(corr_df.columns) == ["League", "Game", "leg_a", "leg_b", "rho"]
    assert len(corr_df) == 1
    assert corr_df.iloc[0]["rho"] == pytest.approx(0.3)


def test_modifier_file_drops_neutral_rows_and_keeps_both_platforms(redirected_paths):
    """1.0 (no-op) modifiers are dropped; distinct per-platform modifiers survive."""
    _, mod_path = redirected_paths
    rows = [
        _row("Underdog", "WNBA", "LVA/NYL", "A|PTS|Over", "B|PTS|Over", 0.3, 0.83),
        _row("Sleeper", "WNBA", "LVA/NYL", "A|PTS|Over", "B|PTS|Over", 0.3, 0.0),
        _row("Underdog", "WNBA", "LVA/NYL", "A|PTS|Over", "C|AST|Over", 0.1, 1.0),
    ]

    persist.write_current_game_corr(rows)

    mod_df = pd.read_parquet(mod_path)
    assert list(mod_df.columns) == PAIR_MODIFIER_COLS
    assert len(mod_df) == 2  # the 1.0 row is dropped
    by_platform = dict(zip(mod_df["Platform"], mod_df["modifier"], strict=True))
    assert by_platform == {"Underdog": pytest.approx(0.83), "Sleeper": pytest.approx(0.0)}


def test_empty_corr_rows_still_writes_both_files(redirected_paths):
    corr_path, mod_path = redirected_paths

    persist.write_current_game_corr([])

    corr_df = pd.read_parquet(corr_path)
    mod_df = pd.read_parquet(mod_path)
    assert list(corr_df.columns) == ["League", "Game", "leg_a", "leg_b", "rho"]
    assert corr_df.empty
    assert list(mod_df.columns) == PAIR_MODIFIER_COLS
    assert mod_df.empty
