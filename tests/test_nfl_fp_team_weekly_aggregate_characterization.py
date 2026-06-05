"""Characterization pins for the over-CC functions in ``nfl_fp_team_weekly_aggregate``:
``_dispatch`` (the recipe-pattern router + per-pattern missing-column skip) and
``_aggregate_pattern_a`` (the per-file_kind pool-windows / apply-recipes / re-key-to-
abbreviation orchestration).

``_dispatch`` is pinned directly on a synthetic per-game frame, one assertion per
pattern (weighted_rate / weighted_mean / proe_formula / rpr_bucket_pass_rate) plus the
missing-column -> None and unknown-pattern -> None skips that the ``_missing_cols`` DRY
extraction must preserve. ``_aggregate_pattern_a`` is pinned with ``_ALL_RECIPES`` and
``_load_kind_window`` monkeypatched to known synthetic inputs, freezing the grain
filter, the window pooling (start>end skip), the combine_first across file_kinds, and
the teamTeamId->abbreviation re-key (unmapped ids dropped).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import sportstradamus.stats.nfl_fp_team_weekly_aggregate as m

TGC = m.TEAM_GROUP_COL


def _dispatch_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TGC: [1, 1, 2],
            "num": [10.0, 20.0, 5.0],
            "den": [2.0, 2.0, 1.0],
            "val": [3.0, 9.0, 4.0],
            "wt": [10.0, 30.0, 5.0],
            "act": [8.0, 2.0, 6.0],
            "exp": [4.0, 4.0, 3.0],
            "bucket": [
                json.dumps(
                    {"Inside10": {"teamStatsSnapsOffensePass": 6, "teamStatsSnapsOffenseRush": 4}}
                ),
                json.dumps(
                    {"Inside10": {"teamStatsSnapsOffensePass": 4, "teamStatsSnapsOffenseRush": 6}}
                ),
                json.dumps(
                    {"Inside10": {"teamStatsSnapsOffensePass": 3, "teamStatsSnapsOffenseRush": 1}}
                ),
            ],
        }
    )


def _recipe(pattern, args):
    return m._TeamRecipe("out_col", "kind", pattern, args, "team")


def test_dispatch_weighted_rate():
    out = m._dispatch(_dispatch_df(), _recipe("weighted_rate", ("num", "den")))
    assert out.loc[1] == pytest.approx(7.5)  # (10+20)/(2+2)
    assert out.loc[2] == pytest.approx(5.0)  # 5/1


def test_dispatch_weighted_mean():
    out = m._dispatch(_dispatch_df(), _recipe("weighted_mean", ("val", "wt")))
    assert out.loc[1] == pytest.approx(7.5)  # (3*10+9*30)/(10+30)
    assert out.loc[2] == pytest.approx(4.0)  # (4*5)/5


def test_dispatch_proe_formula():
    out = m._dispatch(_dispatch_df(), _recipe("proe_formula", ("act", "exp")))
    assert out.loc[1] == pytest.approx(0.25)  # ((8+2)-(4+4))/(4+4)
    assert out.loc[2] == pytest.approx(1.0)  # (6-3)/3
    assert out.name == "out_col"


def test_dispatch_rpr_bucket_pass_rate():
    out = m._dispatch(_dispatch_df(), _recipe("rpr_bucket_pass_rate", ("Inside10",)))
    assert out.loc[1] == pytest.approx(0.5)  # (6+4)/((6+4)+(4+6))
    assert out.loc[2] == pytest.approx(0.75)  # 3/(3+1)


def test_dispatch_missing_column_returns_none():
    assert m._dispatch(_dispatch_df(), _recipe("weighted_rate", ("num", "ABSENT"))) is None
    assert m._dispatch(_dispatch_df(), _recipe("weighted_mean", ("ABSENT", "wt"))) is None
    assert m._dispatch(_dispatch_df(), _recipe("proe_formula", ("act", "ABSENT"))) is None


def test_dispatch_unknown_pattern_returns_none():
    assert m._dispatch(_dispatch_df(), _recipe("not_a_pattern", ("a", "b"))) is None


def _patch_pattern_a(monkeypatch):
    recipes = (
        m._TeamRecipe("tm_rate", "kindA", "weighted_rate", ("num", "den"), "team"),
        m._TeamRecipe("def_mean", "kindB", "weighted_mean", ("val", "wt"), "defense"),
    )
    monkeypatch.setattr(m, "_ALL_RECIPES", recipes)
    frames = {
        "kindA": pd.DataFrame(
            {TGC: [1, 1, 2, 3], "num": [10.0, 20.0, 5.0, 9.0], "den": [2.0, 2.0, 1.0, 3.0]}
        ),
        "kindB": pd.DataFrame({TGC: [1, 2], "val": [3.0, 4.0], "wt": [10.0, 5.0]}),
    }
    monkeypatch.setattr(
        m, "_load_kind_window", lambda season, start, end, kind: frames.get(kind, pd.DataFrame())
    )


def test_aggregate_pattern_a_team_grain(monkeypatch):
    _patch_pattern_a(monkeypatch)
    abbr_map = pd.Series({1: "AAA", 2: "BBB"})  # team 3 deliberately unmapped
    out = m._aggregate_pattern_a([(2024, 1, 5)], grain="team", abbr_map=abbr_map)

    assert list(out.columns) == ["tm_rate"]
    assert set(out.index) == {"AAA", "BBB"}  # team 3 (id=3) dropped: no abbr
    assert out.loc["AAA", "tm_rate"] == pytest.approx(7.5)  # (10+20)/(2+2)
    assert out.loc["BBB", "tm_rate"] == pytest.approx(5.0)  # 5/1


def test_aggregate_pattern_a_defense_grain(monkeypatch):
    _patch_pattern_a(monkeypatch)
    abbr_map = pd.Series({1: "AAA", 2: "BBB"})
    out = m._aggregate_pattern_a([(2024, 1, 5)], grain="defense", abbr_map=abbr_map)

    assert list(out.columns) == ["def_mean"]
    assert out.loc["AAA", "def_mean"] == pytest.approx(3.0)
    assert out.loc["BBB", "def_mean"] == pytest.approx(4.0)


def test_aggregate_pattern_a_empty_windows():
    assert m._aggregate_pattern_a([], grain="team", abbr_map=pd.Series(dtype=object)).empty


def test_aggregate_pattern_a_start_after_end_skipped(monkeypatch):
    _patch_pattern_a(monkeypatch)
    abbr_map = pd.Series({1: "AAA", 2: "BBB"})
    # start_week > end_week -> window skipped -> no frames -> empty result
    assert m._aggregate_pattern_a([(2024, 5, 1)], grain="team", abbr_map=abbr_map).empty
