"""Unit tests for MLB structural plate-appearance volume normalization.

The projector is structural (no trained model), so every path is testable
without training: slot->curve mapping, home/away selection, the unresolved-slot
fallback, and the bounded team offense adjustment.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import mlb


def test_slot_constants_are_well_formed():
    for curve in (mlb.SLOT_PA_HOME, mlb.SLOT_PA_AWAY, mlb.SLOT_STD):
        assert len(curve) == 9
        assert all(v > 0 for v in curve)
    # Leadoff bats more than the nine-hole; away teams bat a full ninth so out-PA them.
    assert mlb.SLOT_PA_HOME[0] > mlb.SLOT_PA_HOME[-1]
    assert mlb.SLOT_PA_AWAY[0] > mlb.SLOT_PA_AWAY[-1]
    assert all(a >= h for a, h in zip(mlb.SLOT_PA_AWAY, mlb.SLOT_PA_HOME, strict=True))
    expected_all = tuple(
        (h + a) / 2 for h, a in zip(mlb.SLOT_PA_HOME, mlb.SLOT_PA_AWAY, strict=True)
    )
    assert expected_all == mlb.SLOT_PA_ALL
    lo, hi = mlb.OFFENSE_ADJ_CLIP
    assert 0 < lo < 1 < hi
    assert 0 < mlb.LG_AVG_OBP < 0.5
    total_weight = mlb.OBP_ADJ_WEIGHT + mlb.MARKET_ADJ_WEIGHT
    assert total_weight == pytest.approx(1.0)


def _bare_mlb():
    stats = mlb.StatsMLB.__new__(mlb.StatsMLB)
    stats.league = "MLB"
    stats.log_strings = {
        "team": "team", "opponent": "opponent", "player": "playerName",
        "date": "gameDate", "home": "home",
    }
    return stats


def test_projector_maps_slot_to_home_away_curve_and_fallback(monkeypatch):
    stats = _bare_mlb()
    game_day = date(2024, 5, 1)
    # get_depth output: A leadoff, B nine-hole, C unresolved (bench/no history).
    stats.playerProfile = pd.DataFrame(
        {"team": ["NYY", "NYY", "BOS"], "depth": [1.0, 9.0, 0.0]},
        index=["A", "B", "C"],
    )
    # gamelog supplies the home/away flag for the settled game day.
    stats.gamelog = pd.DataFrame(
        {
            "playerName": ["A", "B", "C"],
            "gameDate": ["2024-05-01"] * 3,
            "home": [True, False, True],
        }
    )
    monkeypatch.setattr(stats, "get_depth", lambda offers, d: None)
    monkeypatch.setattr(stats, "_mlb_offense_adjustment", lambda teams, offers, d, hm: {})

    offers = [
        {"Player": "A", "Team": "NYY", "Opponent": "BOS"},
        {"Player": "B", "Team": "NYY", "Opponent": "BOS"},
        {"Player": "C", "Team": "BOS", "Opponent": "NYY"},
    ]
    stats._project_plate_appearances(offers, game_day)
    pp = stats.playerProfile

    assert pp.at["A", "proj plateAppearances mean"] == pytest.approx(mlb.SLOT_PA_HOME[0])
    assert pp.at["B", "proj plateAppearances mean"] == pytest.approx(mlb.SLOT_PA_AWAY[8])
    assert pp.at["C", "proj plateAppearances mean"] == pytest.approx(mlb.SLOT_PA_LEAGUE_AVG)
    assert pp.at["A", "proj plateAppearances std"] == pytest.approx(mlb.SLOT_STD[0])
    assert pp.at["C", "proj plateAppearances std"] == pytest.approx(mlb.SLOT_STD_UNKNOWN)


def test_projector_applies_team_offense_multiplier(monkeypatch):
    stats = _bare_mlb()
    stats.playerProfile = pd.DataFrame({"team": ["NYY"], "depth": [1.0]}, index=["A"])
    stats.gamelog = pd.DataFrame(
        {"playerName": ["A"], "gameDate": ["2024-05-01"], "home": [True]}
    )
    monkeypatch.setattr(stats, "get_depth", lambda offers, d: None)
    monkeypatch.setattr(stats, "_mlb_offense_adjustment", lambda *a, **k: {"NYY": 1.05})
    stats._project_plate_appearances(
        [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}], date(2024, 5, 1)
    )
    assert stats.playerProfile.at["A", "proj plateAppearances mean"] == pytest.approx(
        mlb.SLOT_PA_HOME[0] * 1.05
    )
