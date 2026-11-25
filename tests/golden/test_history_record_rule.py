"""History records the board a bettor could act on: a row scored too late is not stored.

A scoring run publishes once, at its end. A row whose game starts inside the betting
margin of that instant, or has started, leaves the stored row of the last run that landed
in time alone, and a key first seen that late is never stored.
"""

import numpy as np
import pandas as pd

from sportstradamus.prediction.cli import _BET_MARGIN, _recordable, _upsert_history

PUBLISHED = pd.Timestamp("2026-11-24T00:46:00")
_KICKOFF = "2026-11-24T01:10:00Z"
_STARTED = "2026-11-24T00:10:00Z"


def _row(player: str, team: str, platform: str, commence, **extra) -> dict:
    return {
        "Player": player,
        "League": "NBA",
        "Date": "2026-11-23",
        "Market": "points",
        "Team": team,
        "Line": 24.5,
        "Platform": platform,
        "Commence": commence,
        "Win Prob": 0.6,
        "Close Market Prob": np.nan,
        "First Scored At": pd.NaT,
    } | extra


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_row("", "", "", "")))


def test_a_row_whose_game_is_still_ahead_is_recorded():
    scored = pd.DataFrame([_row("A", "BOS", "Underdog", _KICKOFF)])

    assert _recordable(_empty(), scored, PUBLISHED)["Player"].tolist() == ["A"]


def test_a_row_whose_game_has_started_is_not_recorded():
    # The started game sits between two live ones, so a filter that reads position
    # instead of each row's own kickoff keeps the wrong rows.
    scored = pd.DataFrame(
        [
            _row("A", "BOS", "Underdog", _KICKOFF),
            _row("B", "LAL", "Underdog", _STARTED),
            _row("C", "MIA", "Underdog", _KICKOFF),
        ]
    )

    assert _recordable(_empty(), scored, PUBLISHED)["Player"].tolist() == ["A", "C"]


def test_the_margin_is_what_a_bettor_needs_to_place_the_entry():
    on_the_margin = (PUBLISHED + _BET_MARGIN).strftime("%Y-%m-%dT%H:%M:%SZ")
    inside_it = (PUBLISHED + _BET_MARGIN - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    scored = pd.DataFrame(
        [_row("A", "BOS", "Underdog", inside_it), _row("B", "LAL", "Underdog", on_the_margin)]
    )

    assert _recordable(_empty(), scored, PUBLISHED)["Player"].tolist() == ["B"]


def test_a_late_rescore_leaves_the_stored_row_of_the_last_run_in_time():
    in_time = pd.Timestamp("2026-11-23T23:46:00")
    stored = pd.DataFrame([_row("B", "LAL", "Underdog", _STARTED, **{"First Scored At": in_time})])
    rescored = pd.DataFrame([_row("B", "LAL", "Underdog", _STARTED, **{"Win Prob": 0.9})])

    history = _upsert_history(stored, _recordable(stored, rescored, PUBLISHED))

    assert history[["Win Prob", "First Scored At"]].to_dict("records") == [
        {"Win Prob": 0.6, "First Scored At": in_time}
    ]


def test_a_key_first_seen_too_late_is_never_stored():
    stored = pd.DataFrame([_row("B", "LAL", "Underdog", _STARTED)])
    # A rung the app posted after the game began: same player, a line history never held.
    late_rung = pd.DataFrame([_row("B", "LAL", "Underdog", _STARTED, Line=27.5)])

    history = _upsert_history(stored, _recordable(stored, late_rung, PUBLISHED))

    assert history["Line"].tolist() == [24.5]


def test_a_row_with_no_start_time_borrows_its_teams_from_a_stored_row():
    """Sleeper posts no start time, and Underdog drops a game's offers once it locks: on
    the late run the only row that still knows the kickoff is one stored earlier."""
    stored = pd.DataFrame([_row("B", "LAL", "Underdog", _STARTED)])
    scored = pd.DataFrame(
        [
            _row("D", "LAL", "Sleeper", ""),
            _row("A", "BOS", "Sleeper", ""),
            _row("E", "BOS", "Underdog", _KICKOFF),
        ]
    )

    assert _recordable(stored, scored, PUBLISHED)["Player"].tolist() == ["A", "E"]


def test_a_row_with_no_kickoff_anywhere_is_recorded():
    stored = pd.DataFrame([_row("B", "LAL", "Underdog", _STARTED)])
    scored = pd.DataFrame([_row("F", "DEN", "Sleeper", ""), _row("G", "PHX", "Sleeper", np.nan)])

    assert _recordable(stored, scored, PUBLISHED)["Player"].tolist() == ["F", "G"]


def test_a_run_that_scored_nothing_records_nothing():
    stored = pd.DataFrame([_row("B", "LAL", "Underdog", _STARTED)])

    assert _recordable(stored, pd.DataFrame(), PUBLISHED).empty
