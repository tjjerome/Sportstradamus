"""Characterization pin for ``StatsNFL.parse_pbp``.

``parse_pbp`` is a CC-19, ~490-line per-(week, team) play-by-play aggregator with
ZERO coverage and three distinct return shapes: a ~48-key team-stats dict when
``playerName`` is empty, a 35-key all-zero dict when the player id is unknown, a
35-key full player-stats dict otherwise, and a bare ``0`` when no plays match.

This pins all four shapes so the decomposition (extracting the fetch/team/empty/
player branches + collapsing the ``num/den if den>0 else nan`` ratios into a shared
``_ratio`` helper) is proven behavior-preserving. ``need_pbp`` is set False and the
post-setup ``pbp``/``ngs``/``pfr``/``ids``/``gamelog`` frames are injected directly,
so the network fetch block is bypassed; the synthetic plays drive both KC offense
(``posteam == "KC"``) and KC defense (``posteam == "BUF"``) aggregates plus the
versatile receiver "Test WR" (3 targets, 1 carry) that exercises the ratio paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats.nfl import StatsNFL

_PBP_NUM = ["xpass", "down", "ydstogo", "epa", "yards_gained", "cpoe", "complete_pass",
            "n_blitzers", "play_time", "home_score", "away_score", "air_yards",
            "passing_yards", "rushing_yards", "receiving_yards"]
_PBP_BOOL = ["pass", "rush", "qb_dropback", "pass_attempt", "redzone", "sack"]


def _play(**kw):
    r = {"week": 1, "home_team": "KC", "away_team": "BUF", "posteam": "KC",
         "home_score": 21, "away_score": 17, "read_thrown": "0", "pass_location": "",
         "passer_player_id": None, "rusher_player_id": None, "receiver_player_id": None}
    for c in _PBP_NUM:
        r.setdefault(c, 0.0)
    for c in _PBP_BOOL:
        r.setdefault(c, False)
    r.update(kw)
    return r


def _pbp():
    rows = [
        _play(**{"pass": True, "qb_dropback": True, "pass_attempt": True, "complete_pass": 1,
                 "pass_location": "middle", "read_thrown": "1", "receiver_player_id": "WR1",
                 "passer_player_id": "QB1", "air_yards": 10.0, "yards_gained": 12.0,
                 "receiving_yards": 12.0, "passing_yards": 12.0, "epa": 0.5, "xpass": 0.6,
                 "cpoe": 5.0, "down": 1, "ydstogo": 10, "play_time": 30.0, "n_blitzers": 1}),
        _play(**{"pass": True, "qb_dropback": True, "pass_attempt": True, "complete_pass": 0,
                 "pass_location": "left", "receiver_player_id": "WR1", "passer_player_id": "QB1",
                 "air_yards": 20.0, "yards_gained": 0.0, "epa": -0.3, "xpass": 0.7, "cpoe": -2.0,
                 "down": 2, "ydstogo": 8, "play_time": 6.0}),
        _play(rush=True, rusher_player_id="RB1", yards_gained=5.0, epa=0.2, down=1, ydstogo=10, play_time=28.0),
        _play(rush=True, rusher_player_id="WR1", yards_gained=3.0, epa=0.1, redzone=True, play_time=25.0),
        _play(**{"pass": True, "qb_dropback": True, "pass_attempt": True, "complete_pass": 1,
                 "pass_location": "middle", "redzone": True, "receiver_player_id": "WR1",
                 "passer_player_id": "QB1", "air_yards": 5.0, "yards_gained": 6.0,
                 "receiving_yards": 6.0, "passing_yards": 6.0, "epa": 0.8, "xpass": 0.5,
                 "down": 1, "ydstogo": 6, "play_time": 20.0}),
        _play(**{"posteam": "BUF", "pass": True, "qb_dropback": True, "pass_attempt": True,
                 "complete_pass": 1, "pass_location": "middle", "read_thrown": "1", "epa": 0.4,
                 "xpass": 0.55, "yards_gained": 9.0, "cpoe": 3.0, "down": 1, "ydstogo": 10,
                 "play_time": 27.0, "n_blitzers": 1}),
        _play(posteam="BUF", rush=True, yards_gained=2.0, epa=-0.1, down=2, ydstogo=5, play_time=26.0),
        _play(**{"posteam": "BUF", "pass": True, "qb_dropback": True, "pass_attempt": True,
                 "complete_pass": 0, "pass_location": "right", "epa": -0.5, "xpass": 0.65,
                 "yards_gained": 0.0, "sack": True, "down": 3, "ydstogo": 8, "play_time": 5.0}),
    ]
    return pd.DataFrame(rows)


def _ngs():
    return pd.DataFrame([
        {"player_display_name": "Test WR", "week": 1, "player_position": "WR", "team_abbr": "KC",
         "expected_rush_yards": 0.0, "rush_attempts": 0.0,
         "completion_percentage_above_expectation": 0.0, "completion_percentage": 0.0,
         "passer_rating": 0.0, "avg_intended_air_yards": 0.0, "avg_air_yards_differential": 0.0,
         "avg_time_to_throw": 0.0, "aggressiveness": 0.0, "rush_yards_over_expected": 0.0,
         "rush_pct_over_expected": 0.0, "avg_yac_above_expectation": 2.5, "avg_separation": 3.0,
         "avg_cushion": 5.0},
        {"player_display_name": "Test RB", "week": 1, "player_position": "RB", "team_abbr": "KC",
         "expected_rush_yards": 40.0, "rush_attempts": 10.0,
         "completion_percentage_above_expectation": 0.0, "completion_percentage": 0.0,
         "passer_rating": 0.0, "avg_intended_air_yards": 0.0, "avg_air_yards_differential": 0.0,
         "avg_time_to_throw": 0.0, "aggressiveness": 0.0, "rush_yards_over_expected": 5.0,
         "rush_pct_over_expected": 0.1, "avg_yac_above_expectation": 0.0, "avg_separation": 0.0,
         "avg_cushion": 0.0},
    ])


def _pfr():
    return pd.DataFrame([
        {"pfr_player_name": "Test WR", "week": 1, "opponent": "BUF", "team": "KC",
         "times_pressured": 0, "rushing_broken_tackles": 1, "receiving_broken_tackles": 2,
         "rushing_yards_after_contact_avg": 3.5, "receiving_drop_pct": 5.0, "passing_drop_pct": 0.0},
        {"pfr_player_name": "Opp QB", "week": 1, "opponent": "KC", "team": "BUF",
         "times_pressured": 4, "rushing_broken_tackles": 0, "receiving_broken_tackles": 0,
         "rushing_yards_after_contact_avg": 0.0, "receiving_drop_pct": 0.0, "passing_drop_pct": 0.0},
        {"pfr_player_name": "Home QB", "week": 1, "opponent": "BUF", "team": "KC",
         "times_pressured": 3, "rushing_broken_tackles": 0, "receiving_broken_tackles": 0,
         "rushing_yards_after_contact_avg": 0.0, "receiving_drop_pct": 0.0, "passing_drop_pct": 0.0},
    ])


def _seed():
    s = StatsNFL()
    s.need_pbp = False
    s.pbp = _pbp()
    s.ngs = _ngs()
    s.pfr = _pfr()
    s.ids = {"Test QB": "QB1", "Test RB": "RB1", "Test WR": "WR1"}
    s.gamelog = pd.DataFrame([{"season": 2024, "week": 1, "player display name": "Test WR",
                               "snap pct": 0.8}])
    return s


def test_parse_pbp_empty_returns_zero():
    assert _seed().parse_pbp(5, "KC", 2024) == 0


def test_parse_pbp_unknown_player_all_zero():
    out = _seed().parse_pbp(1, "KC", 2024, "Nobody Here")
    assert len(out) == 35
    assert all(v == 0 for v in out.values())
    assert "passing_first_downs" in out and "carry_share" in out


def test_parse_pbp_team_stats():
    t = _seed().parse_pbp(1, "KC", 2024)
    assert len(t) == 48
    assert t["WL"] == "W"
    assert t["points"] == 21
    assert t["plays_per_game"] == 5
    assert t["pass_rate"] == pytest.approx(0.6)
    assert t["pass_rate_over_expected"] == pytest.approx(0.24)
    assert t["pass_rate_over_expected_110"] == pytest.approx(0.2)
    assert t["pass_rate_against"] == pytest.approx(0.666667, rel=1e-5)
    assert t["epa_per_pass"] == pytest.approx(0.333333, rel=1e-5)
    assert t["epa_per_rush"] == pytest.approx(0.15)
    assert t["midfield_epa"] == pytest.approx(0.65)
    assert t["yards_per_pass"] == pytest.approx(6.0)
    assert t["yards_allowed_per_pass"] == pytest.approx(4.5)
    assert t["expected_yards_per_rush"] == pytest.approx(4.0)
    assert t["blitz_rate"] == pytest.approx(0.5)
    assert t["pressure_per_pass"] == pytest.approx(2.0)
    assert t["pressure_allowed_per_pass"] == pytest.approx(1.0)
    assert t["time_of_possession"] == pytest.approx(0.652695, rel=1e-5)
    assert t["time_per_play"] == pytest.approx(21.8)
    assert t["completion_percentage_allowed"] == pytest.approx(0.5)
    assert t["cpoe_allowed"] == pytest.approx(0.015)
    assert np.isnan(t["redzone_epa_allowed"])
    assert np.isnan(t["redzone_success_rate_allowed"])


def test_parse_pbp_player_stats():
    p = _seed().parse_pbp(1, "KC", 2024, "Test WR")
    assert len(p) == 35
    # ratio paths (the `num/den if den>0 else nan` consolidation)
    assert p["targets_per_route_run"] == pytest.approx(1.5)
    assert p["first_read_targets_per_route_run"] == pytest.approx(0.5)
    assert p["first_read_target_share"] == pytest.approx(1.0)
    assert p["route_participation"] == pytest.approx(0.666667, rel=1e-5)
    assert p["yards_per_route_run"] == pytest.approx(9.0)
    assert p["midfield_target_rate"] == pytest.approx(0.0)
    assert p["midfield_tprr"] == pytest.approx(1.0)
    assert p["redzone_target_share"] == pytest.approx(1.0)
    assert p["redzone_carry_share"] == pytest.approx(1.0)
    assert p["carry_share"] == pytest.approx(0.5)
    # straight aggregates
    assert p["average_depth_of_target"] == pytest.approx(11.666667, rel=1e-5)
    assert p["receiver_cp_over_expected"] == pytest.approx(1.0)
    assert p["yards_per_carry"] == pytest.approx(3.0)
    assert p["longest_reception"] == pytest.approx(12.0)
    assert p["longest_rush"] == pytest.approx(0.0)
    assert p["first_downs"] == 2
    assert p["passing_first_downs"] == 0
    assert p["sacks_taken"] == 0
    assert p["broken_tackles"] == pytest.approx(3.0)
    assert p["breakaway_yards"] == pytest.approx(3.5)
    assert p["drop_rate"] == pytest.approx(5.0)
    assert p["yac_over_expected"] == pytest.approx(2.5)
    assert p["separation_created"] == pytest.approx(-2.0)
    assert np.isnan(p["longest_completion"])
    assert np.isnan(p["pass_yards_per_attempt"])
