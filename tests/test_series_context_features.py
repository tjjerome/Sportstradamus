"""Derivation pins for the playoff series-context feature (model track).

``Stats._series_context`` tallies the within-series W-L record from prior playoff
teamlog games against the same opponent (inside ``_SERIES_LOOKBACK_DAYS``) and derives
``FacingElimination`` / ``CanClinch`` from a per-league wins-to-clinch:

  * NBA / NHL -- best-of-7 every round (base ``_series_games_to_win``);
  * WNBA -- best-of-3 first round, best-of-5 later rounds (round inferred from the
    count of distinct prior-postseason opponents);
  * MLB -- by round via ``game type`` (Wild Card bo3 / LDS bo5 / LCS+WS bo7);
  * NFL -- single elimination, so the flags collapse onto the Playoff flag.

These pin the record tally, the series window, each league's clinch arithmetic, the
non-playoff short-circuit, and the MLB serve branch (round from ``upcoming_games``).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA

D = date(2025, 5, 20)
FUTURE = date.today() + timedelta(days=3)
NBA_PO = "0042400001"  # GAME_ID[2] == "4" -> playoffs
NHL_PO = "2024030001"  # gameId[4:6] == "03" -> playoffs
WNBA_PO = "1042400001"  # GAME_ID[2] == "4" -> playoffs


def _playoff(index) -> pd.DataFrame:
    stats = pd.DataFrame(index=list(index))
    stats["Playoff"] = 1
    return stats


def _gameid_teamlog(cls, gameid, rows, ref=D):
    """rows: (team, opp, days_before_ref, WL)."""
    s = cls()
    ls = s.log_strings
    s.teamlog = pd.DataFrame(
        [
            {
                ls["team"]: t,
                ls["opponent"]: o,
                ls["date"]: (ref - timedelta(days=d)).isoformat(),
                "WL": wl,
                ls["game"]: gameid,
            }
            for (t, o, d, wl) in rows
        ]
    )
    return s


def test_nba_series_record_and_bo7_flags():
    # BOS leads MIA 3-1 within the series window.
    rows = [
        ("BOS", "MIA", 8, "W"), ("MIA", "BOS", 8, "L"),
        ("BOS", "MIA", 6, "W"), ("MIA", "BOS", 6, "L"),
        ("BOS", "MIA", 4, "L"), ("MIA", "BOS", 4, "W"),
        ("BOS", "MIA", 2, "W"), ("MIA", "BOS", 2, "L"),
    ]
    s = _gameid_teamlog(StatsNBA, NBA_PO, rows)
    stats = _playoff(["pb", "pm"])
    s._series_context(stats, D, {"pb": "BOS", "pm": "MIA"}, {"pb": "MIA", "pm": "BOS"})
    assert stats.loc["pb", ["SeriesWins", "SeriesLosses"]].tolist() == [3, 1]
    assert stats.loc["pb", "CanClinch"] == 1  # 3 wins == 4 - 1
    assert stats.loc["pb", "FacingElimination"] == 0
    assert stats.loc["pm", ["SeriesWins", "SeriesLosses"]].tolist() == [1, 3]
    assert stats.loc["pm", "FacingElimination"] == 1  # 3 losses == 4 - 1
    assert stats.loc["pm", "CanClinch"] == 0


def test_series_window_excludes_prior_season_meeting():
    rows = [
        ("BOS", "MIA", 2, "W"),  # this series
        ("BOS", "MIA", 60, "W"),  # a meeting > _SERIES_LOOKBACK_DAYS ago -> excluded
    ]
    s = _gameid_teamlog(StatsNBA, NBA_PO, rows)
    stats = _playoff(["pb"])
    s._series_context(stats, D, {"pb": "BOS"}, {"pb": "MIA"})
    assert stats.loc["pb", "SeriesWins"] == 1


def test_non_playoff_short_circuits_to_zero():
    s = StatsNBA()
    s.teamlog = pd.DataFrame()  # never consulted on the short-circuit
    stats = pd.DataFrame(index=["p"])
    stats["Playoff"] = 0
    s._series_context(stats, D, {"p": "BOS"}, {"p": "MIA"})
    assert stats.loc["p", list(("SeriesWins", "SeriesLosses", "FacingElimination", "CanClinch"))].tolist() == [0, 0, 0, 0]


def test_nhl_bo7_facing_elimination_at_three_losses():
    rows = [("PIT", "TB", d, "L") for d in (2, 4, 6)] + [("TB", "PIT", d, "W") for d in (2, 4, 6)]
    s = _gameid_teamlog(StatsNHL, NHL_PO, rows)
    stats = _playoff(["p"])
    s._series_context(stats, D, {"p": "PIT"}, {"p": "TB"})
    assert stats.loc["p", "SeriesLosses"] == 3
    assert stats.loc["p", "FacingElimination"] == 1  # bo7: 3 == 4 - 1


def test_wnba_first_round_is_best_of_three():
    # Only one opponent this postseason -> round 1 -> bo3 -> one loss is elimination.
    rows = [("BOS", "MIA", 2, "L"), ("MIA", "BOS", 2, "W")]
    s = _gameid_teamlog(StatsWNBA, WNBA_PO, rows)
    stats = _playoff(["pb"])
    s._series_context(stats, D, {"pb": "BOS"}, {"pb": "MIA"})
    assert stats.loc["pb", "SeriesLosses"] == 1
    assert stats.loc["pb", "FacingElimination"] == 1  # bo3: 1 == 2 - 1


def test_wnba_later_round_is_best_of_five():
    # A prior-round opponent (ATL) this postseason -> semis -> bo5 -> one loss is NOT
    # elimination, two losses is.
    base = [("BOS", "ATL", 20, "W")]  # round 1, outside the MIA series window
    one_loss = _gameid_teamlog(StatsWNBA, WNBA_PO, base + [("BOS", "MIA", 3, "L")])
    stats = _playoff(["pb"])
    one_loss._series_context(stats, D, {"pb": "BOS"}, {"pb": "MIA"})
    assert stats.loc["pb", "SeriesLosses"] == 1
    assert stats.loc["pb", "FacingElimination"] == 0  # bo5 needs 2

    two_loss = _gameid_teamlog(StatsWNBA, WNBA_PO, base + [("BOS", "MIA", 5, "L"), ("BOS", "MIA", 3, "L")])
    stats2 = _playoff(["pb"])
    two_loss._series_context(stats2, D, {"pb": "BOS"}, {"pb": "MIA"})
    assert stats2.loc["pb", "SeriesLosses"] == 2
    assert stats2.loc["pb", "FacingElimination"] == 1  # bo5: 2 == 3 - 1


def _mlb_teamlog(rows, ref=D):
    """rows: (team, opp, days_before_ref, WL, game_type)."""
    s = StatsMLB()
    s.teamlog = pd.DataFrame(
        [
            {
                "team": t,
                "opponent": o,
                "gameDate": (ref - timedelta(days=d)).isoformat(),
                "WL": wl,
                "gameId": i,
                "game type": gt,
            }
            for i, (t, o, d, wl, gt) in enumerate(rows)
        ]
    )
    return s


def test_mlb_wildcard_is_best_of_three():
    rows = [
        ("NYY", "BOS", 0, "L", "F"),  # current game (date D) -> game_type lookup, excluded from tally
        ("NYY", "BOS", 2, "L", "F"),
        ("BOS", "NYY", 2, "W", "F"),
    ]
    s = _mlb_teamlog(rows)
    stats = _playoff(["p"])
    s._series_context(stats, D, {"p": "NYY"}, {"p": "BOS"})
    assert stats.loc["p", "SeriesLosses"] == 1
    assert stats.loc["p", "FacingElimination"] == 1  # Wild Card bo3: 1 == 2 - 1


def test_mlb_lcs_is_best_of_seven():
    rows = [
        ("NYY", "BOS", 0, "L", "L"),  # current game, LCS
        ("NYY", "BOS", 2, "L", "L"),
        ("NYY", "BOS", 4, "L", "L"),
    ]
    s = _mlb_teamlog(rows)
    stats = _playoff(["p"])
    s._series_context(stats, D, {"p": "NYY"}, {"p": "BOS"})
    assert stats.loc["p", "SeriesLosses"] == 2
    assert stats.loc["p", "FacingElimination"] == 0  # LCS bo7 needs 3


def test_mlb_serve_reads_round_from_upcoming_games():
    rows = [("NYY", "BOS", 2, "L", "F"), ("BOS", "NYY", 2, "W", "F")]
    s = _mlb_teamlog(rows, ref=FUTURE)
    s.upcoming_games = {"NYY": {"game type": "F"}}  # no current-game teamlog row at serve time
    stats = _playoff(["p"])
    s._series_context(stats, FUTURE, {"p": "NYY"}, {"p": "BOS"})
    assert stats.loc["p", "SeriesLosses"] == 1
    assert stats.loc["p", "FacingElimination"] == 1  # bo3 from the upcoming game type


def test_nfl_series_collapses_onto_playoff_flag():
    s = StatsNFL()
    playoff = pd.DataFrame(index=["p"])
    playoff["Playoff"] = 1
    s._series_context(playoff, D, {"p": "KC"}, {"p": "BUF"})
    assert playoff.loc["p", ["FacingElimination", "CanClinch"]].tolist() == [1, 1]
    assert playoff.loc["p", ["SeriesWins", "SeriesLosses"]].tolist() == [0, 0]

    regular = pd.DataFrame(index=["p"])
    regular["Playoff"] = 0
    s._series_context(regular, D, {"p": "KC"}, {"p": "BUF"})
    assert regular.loc["p", ["FacingElimination", "CanClinch"]].tolist() == [0, 0]
