"""Pins for ``prediction.stories.details`` — the P6a offer-detail prerender.

These pure helpers run at ``prophecize`` time so the dashboard reads precomputed
detail rows instead of recomputing comps / volume / SHAP context live. Each helper
is fed plain frames (extracted from the ``Stats`` object by the cli adapter) so it
stays import-safe and unit-testable with synthetic fixtures.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from sportstradamus.prediction.stories.details import (
    build_offer_details,
    comps_vs_opponent,
    position_percentile,
    rank_other_stats,
    volume_trend,
)

_COLS = {"player": "PLAYER", "opponent": "OPP", "date": "DATE"}
_TODAY = dt.date(2026, 6, 15)


def _gamelog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # CompA: two recent games vs PHI (avg 25) + one vs BOS (10) ⇒ 300d avg 20.
            {"PLAYER": "CompA", "OPP": "PHI", "DATE": "2026-06-01", "PTS": 20.0},
            {"PLAYER": "CompA", "OPP": "PHI", "DATE": "2026-05-01", "PTS": 30.0},
            {"PLAYER": "CompA", "OPP": "BOS", "DATE": "2026-04-01", "PTS": 10.0},
            # An old CompA-vs-PHI game outside the 300-day window — excluded both ways.
            {"PLAYER": "CompA", "OPP": "PHI", "DATE": "2025-01-01", "PTS": 100.0},
            # CompB: one recent vs PHI (10) + one vs NYK (30) ⇒ 300d avg 20.
            {"PLAYER": "CompB", "OPP": "PHI", "DATE": "2026-06-02", "PTS": 10.0},
            {"PLAYER": "CompB", "OPP": "NYK", "DATE": "2026-05-02", "PTS": 30.0},
            # CompC: never faced PHI ⇒ excluded entirely.
            {"PLAYER": "CompC", "OPP": "MIA", "DATE": "2026-06-03", "PTS": 15.0},
        ]
    )


def _comp_pairs() -> pd.DataFrame:
    # Flattened ``_comp_pairs`` shape, closest-first per the BallTree order.
    return pd.DataFrame(
        [
            {"player": "Star", "comp": "CompA", "dist": 0.1},
            {"player": "Star", "comp": "CompB", "dist": 0.2},
            {"player": "Star", "comp": "CompC", "dist": 0.3},
            {"player": "Other", "comp": "CompZ", "dist": 0.1},
        ]
    )


def test_comps_vs_opponent_avg_and_pct_diff():
    out = comps_vs_opponent(
        "Star", "PHI", "PTS", _comp_pairs(), _gamelog(), _COLS, today=_TODAY
    )
    # CompC never faced PHI ⇒ dropped; order follows the closest-first comp list.
    assert [r["comp"] for r in out] == ["CompA", "CompB"]
    a, b = out
    assert a["n_games"] == 2
    assert a["avg_vs_opp"] == 25.0
    assert round(a["pct_diff"], 4) == 0.25  # 25 / 20 - 1
    assert b["n_games"] == 1
    assert b["avg_vs_opp"] == 10.0
    assert round(b["pct_diff"], 4) == -0.5  # 10 / 20 - 1


def test_comps_vs_opponent_none_faced_is_empty():
    out = comps_vs_opponent(
        "Star", "DEN", "PTS", _comp_pairs(), _gamelog(), _COLS, today=_TODAY
    )
    assert out == []


def test_comps_vs_opponent_unknown_player_is_empty():
    out = comps_vs_opponent(
        "Ghost", "PHI", "PTS", _comp_pairs(), _gamelog(), _COLS, today=_TODAY
    )
    assert out == []


def _minutes_log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PLAYER": "Star", "DATE": "2026-04-01", "MIN": 32.0},
            {"PLAYER": "Star", "DATE": "2026-05-01", "MIN": 28.0},
            {"PLAYER": "Star", "DATE": "2026-06-01", "MIN": 30.0},
            {"PLAYER": "Other", "DATE": "2026-06-01", "MIN": 12.0},
        ]
    )


def test_volume_trend_chronological():
    # Oldest-first so a sparkline reads left-to-right like build_recent_history.
    out = volume_trend("Star", _minutes_log(), _COLS, "MIN")
    assert out == [32.0, 28.0, 30.0]


def test_volume_trend_caps_to_n():
    out = volume_trend("Star", _minutes_log(), _COLS, "MIN", n=2)
    assert out == [28.0, 30.0]


def test_volume_trend_missing_stat_is_empty():
    assert volume_trend("Star", _minutes_log(), _COLS, "SNAPS") == []
    assert volume_trend("Ghost", _minutes_log(), _COLS, "MIN") == []


def _importances() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "NBA_PTS": {
                "Player comps z": 8.0,
                "Mean10_Ratio": 7.0,
                "MIN": 4.4,
                "Avg5": 3.0,
                "FGA": 2.5,
                "REB": 2.0,
                "DaysIntoSeason": 2.7,
            }
        }
    )


def test_rank_other_stats_excludes_shown_and_keeps_gamelog_backed():
    # Sorted by |SHAP|: comps z (excluded), Mean10_Ratio (not a gamelog col),
    # MIN/Avg5 (excluded), FGA + REB (gamelog-backed, kept), DaysIntoSeason (not a col).
    out = rank_other_stats(
        "NBA_PTS",
        _importances(),
        exclude={"Avg5", "MIN", "Player comps z"},
        available={"FGA", "REB", "MIN", "Avg5"},
    )
    assert out == ["FGA", "REB"]


def test_rank_other_stats_caps_to_k():
    out = rank_other_stats(
        "NBA_PTS",
        _importances(),
        exclude=set(),
        available={"FGA", "REB", "MIN", "Avg5"},
        k=2,
    )
    assert out == ["MIN", "Avg5"]  # top two by |SHAP| among available


def test_rank_other_stats_unknown_market_is_empty():
    assert rank_other_stats("NBA_AST", _importances(), exclude=set(), available={"FGA"}) == []


def test_rank_other_stats_strips_player_feature_decorations():
    # Real SHAP features are engineered ("Player TS_PCT short", "Player MIN growth"); the
    # underlying gamelog stat is recovered by stripping the "Player " prefix + the
    # short/growth/long suffix, so efficiency stats surface as buildable "other stats".
    imp = pd.DataFrame(
        {"WNBA_PTS": {"Player TS_PCT short": 5.0, "Player MIN growth": 4.0, "Mean10": 9.0}}
    )
    out = rank_other_stats("WNBA_PTS", imp, exclude={"MIN"}, available={"TS_PCT", "MIN"})
    # Mean10 (top |SHAP|) is not a gamelog col ⇒ dropped; "Player MIN growth" ⇒ MIN, the
    # excluded volume stat; "Player TS_PCT short" ⇒ TS_PCT, kept.
    assert out == ["TS_PCT"]


def test_position_percentile():
    cohort = [10.0, 20.0, 30.0, 40.0]
    assert position_percentile(30.0, cohort) == 0.75  # 3 of 4 values <= 30
    assert position_percentile(40.0, cohort) == 1.0
    assert position_percentile(5.0, cohort) == 0.0
    assert position_percentile(25.0, []) is None  # no cohort ⇒ no percentile


def _slate_offers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"League": "NBA", "Date": "2026-06-15", "Player": "Star", "Market": "PTS",
             "Opponent": "PHI", "Position": "G1", "Stat": "PTS"},
            {"League": "NBA", "Date": "2026-06-15", "Player": "Nova", "Market": "PTS",
             "Opponent": "BOS", "Position": "G1", "Stat": "PTS"},
        ]
    )


def _slate_gamelog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PLAYER": "Star", "OPP": "PHI", "DATE": "2026-06-01", "PTS": 25.0, "MIN": 30.0, "FGA": 10.0, "REB": 5.0},
            {"PLAYER": "Star", "OPP": "BOS", "DATE": "2026-05-01", "PTS": 20.0, "MIN": 28.0, "FGA": 14.0, "REB": 7.0},
            {"PLAYER": "Nova", "OPP": "BOS", "DATE": "2026-06-01", "PTS": 18.0, "MIN": 25.0, "FGA": 8.0, "REB": 6.0},
            {"PLAYER": "Nova", "OPP": "PHI", "DATE": "2026-05-01", "PTS": 22.0, "MIN": 27.0, "FGA": 8.0, "REB": 4.0},
            {"PLAYER": "CompA", "OPP": "PHI", "DATE": "2026-06-01", "PTS": 20.0, "MIN": 0.0, "FGA": 0.0, "REB": 0.0},
            {"PLAYER": "CompA", "OPP": "PHI", "DATE": "2026-05-01", "PTS": 30.0, "MIN": 0.0, "FGA": 0.0, "REB": 0.0},
            {"PLAYER": "CompA", "OPP": "BOS", "DATE": "2026-04-01", "PTS": 10.0, "MIN": 0.0, "FGA": 0.0, "REB": 0.0},
            {"PLAYER": "CompB", "OPP": "PHI", "DATE": "2026-06-02", "PTS": 10.0, "MIN": 0.0, "FGA": 0.0, "REB": 0.0},
            {"PLAYER": "CompB", "OPP": "NYK", "DATE": "2026-05-02", "PTS": 30.0, "MIN": 0.0, "FGA": 0.0, "REB": 0.0},
        ]
    )


def _league_data() -> dict:
    return {
        "NBA": {
            "comp_pairs": _comp_pairs(),
            "gamelog": _slate_gamelog(),
            "cols": _COLS,
            "volume_stat": "MIN",
        }
    }


def test_build_offer_details_assembles_per_offer_row():
    out = build_offer_details(
        _slate_offers(),
        _league_data(),
        _importances(),
        exclude={"Avg5", "AvgH2H", "Player comps z"},
        today=_TODAY,
    ).set_index("Player")

    assert sorted(out.index) == ["Nova", "Star"]
    star = out.loc["Star"]

    comps = json.loads(star["comps_vs_opp"])
    assert [c["comp"] for c in comps] == ["CompA", "CompB"]
    assert comps[0]["avg_vs_opp"] == 25.0 and round(comps[0]["pct_diff"], 2) == 0.25

    assert json.loads(star["volume_trend"]) == [28.0, 30.0]  # MIN, chronological

    other = json.loads(star["other_stats"])
    assert [o["stat"] for o in other] == ["FGA", "REB"]  # PTS (market) + MIN (volume) excluded
    fga = other[0]
    assert fga["value"] == 12.0 and fga["series"] == [14.0, 10.0]
    assert fga["percentile"] == 1.0  # 12 is top of the G1 FGA cohort {Star 12, Nova 8}

    # Nova has no comps in the pair table ⇒ empty comps panel, no crash.
    assert json.loads(out.loc["Nova"]["comps_vs_opp"]) == []


def test_build_offer_details_empty_safe():
    out = build_offer_details(
        pd.DataFrame(), {}, _importances(), exclude=set(), today=_TODAY
    )
    assert list(out.columns) == [
        "League", "Date", "Player", "Market", "Opponent",
        "comps_vs_opp", "volume_trend", "other_stats",
    ]
    assert out.empty
