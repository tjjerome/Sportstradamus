"""Unit tests for MLB structural plate-appearance volume normalization.

The projector is structural (no trained model), so every path is testable
without training: slot->curve mapping, home/away selection, the unresolved-slot
fallback, and the bounded team offense adjustment.

For a game still to come, each read of the schedule is of the offer's own game
date, and a team's one multiplier describes the earliest game it is offered on.
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
        "team": "team",
        "opponent": "opponent",
        "player": "playerName",
        "date": "gameDate",
        "home": "home",
    }
    return stats


def test_projector_maps_slot_to_home_away_curve_and_fallback(monkeypatch):
    stats = _bare_mlb()
    game_day = date(2024, 5, 1)
    # get_depth output: A leadoff, B nine-hole, C unresolved (bench/no history).
    # "team" is the real fillna(0) sentinel -- the live team must come from offers.
    stats.playerProfile = pd.DataFrame(
        {"team": [0, 0, 0], "depth": [1.0, 9.0, 0.0]},
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
    # "team" is the fillna(0) sentinel; the NYY multiplier must be keyed from the
    # offer's Team, so reading profile["team"] (== 0) would drop it and fail here.
    stats.playerProfile = pd.DataFrame({"team": [0], "depth": [1.0]}, index=["A"])
    stats.gamelog = pd.DataFrame({"playerName": ["A"], "gameDate": ["2024-05-01"], "home": [True]})
    monkeypatch.setattr(stats, "get_depth", lambda offers, d: None)
    monkeypatch.setattr(stats, "_mlb_offense_adjustment", lambda *a, **k: {"NYY": 1.05})
    stats._project_plate_appearances(
        [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}], date(2024, 5, 1)
    )
    assert stats.playerProfile.at["A", "proj plateAppearances mean"] == pytest.approx(
        mlb.SLOT_PA_HOME[0] * 1.05
    )


class _FakeArchive:
    """Stand-in for the module-level ``archive`` singleton. The real one is a
    LazyArchive proxy (``__slots__ = ()``) that opens a DuckDB connection on any
    attribute access, so tests swap the module name rather than patch a method."""

    def __init__(self, total):
        self._total = total

    def get_total(self, league, date, team):
        return self._total


def test_offense_adjustment_blends_obp_and_market(monkeypatch):
    stats = _bare_mlb()
    stats.park_factors = {"NYY": {"OBP": 1.00}, "BOS": {"OBP": 1.00}}
    stats.teamProfile = pd.DataFrame({"OBP": [0.320]}, index=["NYY"])
    stats.gamelog = pd.DataFrame(
        {
            "playerName": ["Ace", "Ace", "A"],
            "team": ["BOS", "BOS", "NYY"],
            "gameDate": ["2024-04-01", "2024-04-08", "2024-05-01"],
            "starting pitcher": [True, True, False],
            "hits allowed": [5, 5, 0],
            "walks allowed": [3, 3, 0],
            "batters faced": [25, 25, 0],
            "opponent pitcher": ["", "", "Ace"],
        }
    )
    monkeypatch.setattr(mlb, "archive", _FakeArchive(4.90))

    offers = [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}]
    home_map = {"A": True}
    adj = stats._mlb_offense_adjustment({"NYY"}, offers, date(2024, 5, 1), home_map)

    team_obp = 0.320
    park_obp = 1.00  # NYY home -> NYY park
    starter_obp = (5 + 3 + 5 + 3) / (25 + 25)  # pooled (H+BB)/BF over recent starts
    obp_exp = team_obp * park_obp * (starter_obp / mlb.LG_AVG_OBP)
    obp_factor = (1 - mlb.LG_AVG_OBP) / (1 - obp_exp)
    market_factor = 4.90 / mlb.LG_AVG_TEAM_TOTAL
    expected = float(
        np.clip(
            mlb.OBP_ADJ_WEIGHT * obp_factor + mlb.MARKET_ADJ_WEIGHT * market_factor,
            *mlb.OFFENSE_ADJ_CLIP,
        )
    )
    assert adj["NYY"] == pytest.approx(expected)
    lo, hi = mlb.OFFENSE_ADJ_CLIP
    assert lo < adj["NYY"] < hi


def test_offense_adjustment_degrades_without_obp_history(monkeypatch):
    stats = _bare_mlb()
    stats.park_factors = {}
    stats.teamProfile = pd.DataFrame({"OBP": []}, index=pd.Index([], name="team"))
    stats.gamelog = pd.DataFrame(
        {
            "playerName": [],
            "team": [],
            "gameDate": [],
            "starting pitcher": [],
            "hits allowed": [],
            "walks allowed": [],
            "batters faced": [],
            "opponent pitcher": [],
        }
    )
    monkeypatch.setattr(mlb, "archive", _FakeArchive(mlb.LG_AVG_TEAM_TOTAL))
    adj = stats._mlb_offense_adjustment(
        {"NYY"}, [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}], date(2024, 5, 1), {}
    )
    assert adj["NYY"] == pytest.approx(1.0)


def test_get_volume_stats_routes_hitter_to_structural(monkeypatch):
    stats = _bare_mlb()
    calls = {}
    monkeypatch.setattr(
        stats,
        "_project_plate_appearances",
        lambda offers, d: calls.__setitem__("hitter", (offers, d)),
    )
    monkeypatch.setattr(
        stats,
        "load_volume_model_params",
        lambda *a, **k: calls.__setitem__("pitcher", (a, k)),
    )
    offers = [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}]

    stats.get_volume_stats(offers, date(2024, 5, 1), pitcher=False)
    assert "hitter" in calls and "pitcher" not in calls

    calls.clear()
    stats.get_volume_stats(offers, date(2024, 5, 1), pitcher=True)
    assert "pitcher" in calls and "hitter" not in calls
    # pitcher track still loads the "pitches thrown" model
    assert calls["pitcher"][0][1] == "pitches thrown"


def test_plate_appearances_retired_but_pitches_thrown_kept():
    from sportstradamus.training.markets import ALL_MARKETS

    assert "plateAppearances" not in ALL_MARKETS["MLB"]
    assert "pitches thrown" in ALL_MARKETS["MLB"]


def test_dispatch_routes_markets_correctly(monkeypatch):
    stats = _bare_mlb()
    stats.volume_stats = ["pitches thrown"]
    seen = {}
    monkeypatch.setattr(stats, "get_depth", lambda offers, d: seen.__setitem__("depth", True))
    monkeypatch.setattr(
        stats,
        "get_volume_stats",
        lambda offers, d, pitcher: seen.__setitem__("volume", pitcher),
    )
    offers = [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}]

    # a trained volume market -> get_depth only
    stats._dispatch_volume_stats(offers, date(2024, 5, 1), "pitches thrown")
    assert seen == {"depth": True}

    # a hitter market -> get_volume_stats(pitcher=False)
    seen.clear()
    stats._dispatch_volume_stats(offers, date(2024, 5, 1), "total bases")
    assert seen == {"volume": False}

    # a pitcher market -> get_volume_stats(pitcher=True)
    seen.clear()
    stats._dispatch_volume_stats(offers, date(2024, 5, 1), "hits allowed")
    assert seen == {"volume": True}


_TODAY, _TOMORROW = "2099-01-01", "2099-01-02"


def _mlb_on_a_two_day_schedule():
    """NYY hosts BOS today and visits TOR tomorrow; SEA is off today and hosts TEX tomorrow."""
    stats = _bare_mlb()
    nyy_today = {"Home": True, "Opponent Pitcher": "Bos Arm", "Batting Order": ["A", "B"]}
    stats.upcoming_games = {"NYY": nyy_today}
    stats.upcoming_games_by_date = {
        ("NYY", _TODAY): nyy_today,
        ("NYY", _TOMORROW): {"Home": False, "Opponent Pitcher": "Tor Arm", "Batting Order": []},
        ("SEA", _TOMORROW): {"Home": True, "Opponent Pitcher": "Tex Arm", "Batting Order": []},
    }
    return stats


def _offer(player, team, opponent, game_date):
    return {"Player": player, "Team": team, "Opponent": opponent, "Date": game_date}


def test_projector_reads_home_and_away_from_the_offers_own_game_day(monkeypatch):
    """A's team is home today and away tomorrow; C's team is off today."""
    stats = _mlb_on_a_two_day_schedule()
    stats.playerProfile = pd.DataFrame({"team": [0, 0], "depth": [1.0, 1.0]}, index=["A", "C"])
    monkeypatch.setattr(stats, "get_depth", lambda offers, d: None)
    monkeypatch.setattr(stats, "_mlb_offense_adjustment", lambda teams, offers, d, hm: {})

    stats._project_plate_appearances(
        [_offer("A", "NYY", "TOR", _TOMORROW), _offer("C", "SEA", "TEX", _TOMORROW)],
        date(2099, 1, 1),
    )

    pp = stats.playerProfile
    assert pp.at["A", "proj plateAppearances mean"] == pytest.approx(mlb.SLOT_PA_AWAY[0])
    assert pp.at["C", "proj plateAppearances mean"] == pytest.approx(mlb.SLOT_PA_HOME[0])


def test_offense_adjustment_describes_the_earliest_game_a_team_is_offered_on(monkeypatch):
    """One multiplier a team, so its opponent, park side and opposing starter are one game's."""
    stats = _mlb_on_a_two_day_schedule()
    games = {}
    monkeypatch.setattr(
        stats, "_obp_factor", lambda team, *game: games.__setitem__(team, game) or 1.0
    )
    monkeypatch.setattr(stats, "_market_factor", lambda team, d: 1.0)
    run_date = date(2099, 1, 1)
    offers = [
        _offer("A", "NYY", "BOS", _TODAY),
        _offer("B", "NYY", "TOR", _TOMORROW),
        _offer("C", "SEA", "TEX", _TOMORROW),
    ]

    stats._mlb_offense_adjustment(
        {"NYY", "SEA"}, offers, run_date, {"A": True, "B": False, "C": True}
    )

    assert games == {
        "NYY": ("BOS", True, "Bos Arm", run_date),
        "SEA": ("TEX", True, "Tex Arm", run_date),
    }


def test_depth_reads_the_posted_order_of_the_offers_own_game_day(monkeypatch):
    """Today's posted order places nobody whose offer is for tomorrow's game."""
    stats = _mlb_on_a_two_day_schedule()
    stats.short_gamelog = pd.DataFrame({"playerName": ["A", "B"], "battingOrder": [7, 4]})
    monkeypatch.setattr(stats, "base_profile", lambda d: None)

    stats.playerProfile = pd.DataFrame(index=["A", "B"])
    stats.get_depth(
        [_offer("A", "NYY", "TOR", _TOMORROW), _offer("B", "NYY", "BOS", _TODAY)], date(2099, 1, 1)
    )
    assert stats.playerProfile["depth"].to_dict() == {"A": 7, "B": 2}

    # The parlay search asks again with no date: that reads the next game, as before.
    stats.playerProfile = pd.DataFrame(index=["A", "B"])
    stats.get_depth([{"Player": "A", "Team": "NYY"}, {"Player": "B", "Team": "NYY"}])
    assert stats.playerProfile["depth"].to_dict() == {"A": 1, "B": 2}
