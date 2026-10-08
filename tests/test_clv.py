"""Unit tests for the CLV fill / summary helpers in ``sportstradamus.clv``."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sportstradamus import clv
from sportstradamus.helpers.distributions import get_odds


class _StubArchive:
    """Minimal stub that mirrors ``Archive.get_ev``'s lookup contract.

    ``get_ev`` returns a book stat-mean keyed by ``ev_table``; ``get_composite_under_prob``
    returns a devigged under-probability keyed by ``under_table``, used only by the
    NaN-Dist/CV fallback path. ``asked`` records every lookup key, so a test can assert
    a group was never read; ``asked_at`` and ``movement_until`` record the player and
    time-series cutoff of each closing and movement read. The cutoff does not change
    what the stub answers.
    """

    def __init__(self, ev_table, under_table=None):
        self._ev_table = ev_table
        self._under_table = under_table or {}
        self.asked = []
        self.asked_at = []
        self.movement_until = []

    def get_ev(self, league, market, date, player, *, at=None):
        self.asked.append((league, market, date, player))
        self.asked_at.append((player, at))
        return self._ev_table.get((league, market, date, player), float("nan"))

    def get_composite_under_prob(self, league, market, date, player, *, at=None):
        self.asked.append((league, market, date, player))
        self.asked_at.append((player, at))
        return self._under_table.get((league, market, date, player), float("nan"))

    def get_movement(self, league, market, date, player, *, until=None):
        del league, market, date
        self.movement_until.append((player, until))


# Player A: NBA points, a Gamma cell. Archive close-EV is a realistic points stat mean
# (~11), not a probability — the whole point of this fixture is to prove the conversion
# actually inverts the mean through the distribution rather than passing it through raw.
_PLAYER_A_CLOSE_EV = 11.2
_PLAYER_A_CV = 0.35


def _build_history():
    """One Over offer (Gamma, resolvable) and one Under offer without archive coverage."""
    return pd.DataFrame(
        [
            {
                "Player": "Player A",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 10.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Over",
                "Win Prob": 0.60,
                "Market Prob": 0.55,
                "Dist": "Gamma",
                "CV": _PLAYER_A_CV,
                "Gate": np.nan,
                "Step": 0.5,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            },
            {
                "Player": "Player B",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 12.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Under",
                "Win Prob": 0.48,
                "Market Prob": 0.50,
                "Dist": "Gamma",
                "CV": 0.35,
                "Gate": np.nan,
                "Step": 0.5,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            },
        ]
    )


def _player_a_expected_close_p():
    """Hand oracle: Player A is Over, so close_p = 1 - P(under line | close_ev)."""
    close_under = get_odds(10.5, _PLAYER_A_CLOSE_EV, "Gamma", cv=_PLAYER_A_CV, gate=None, step=0.5)
    return 1.0 - close_under


def test_fill_from_archive_writes_close_and_clv_for_resolved_leg():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): _PLAYER_A_CLOSE_EV})
    df = clv.fill_from_archive(_build_history(), archive)

    expected_close_p = _player_a_expected_close_p()
    row = df.loc[0]
    assert 0.0 <= expected_close_p <= 1.0
    assert row["Close Market Prob"] == pytest.approx(expected_close_p)
    assert row["Market CLV"] == pytest.approx(expected_close_p - 0.55)
    assert row["Model CLV"] == pytest.approx(expected_close_p - 0.60)


def test_fill_from_archive_leaves_unresolved_leg_nan():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): _PLAYER_A_CLOSE_EV})
    df = clv.fill_from_archive(_build_history(), archive)

    row = df.loc[1]
    assert math.isnan(row["Close Market Prob"])
    assert math.isnan(row["Market CLV"])
    assert math.isnan(row["Model CLV"])


def test_fill_from_archive_skips_rows_already_closed():
    """A row with a non-NaN Close Market Prob is not re-queried against archive."""
    history = _build_history()
    history.loc[0, "Close Market Prob"] = 0.99
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): _PLAYER_A_CLOSE_EV})
    df = clv.fill_from_archive(history, archive)
    assert df.loc[0, "Close Market Prob"] == pytest.approx(0.99)


def test_fill_from_archive_repairs_inverted_clv_on_existing_close():
    """A row already carrying Close Market Prob, stamped under the pre-fix,
    sign-flipped convention, gets its Market/Model CLV corrected on the next
    run — it isn't "pending" so the archive is never consulted for it."""
    history = pd.DataFrame(
        [
            {
                "Player": "Player C",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "blocks",
                "Line": 3.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Under",
                "Win Prob": 0.55,
                "Market Prob": 0.50,
                "Dist": "ZINB",
                "CV": 0.9,
                "Gate": 0.12,
                "Step": 1.0,
                "Close Market Prob": 0.65,
                "Market CLV": -(0.65 - 0.50),
                "Model CLV": -(0.65 - 0.55),
            }
        ]
    )
    df = clv.fill_from_archive(history, _StubArchive({}))

    row = df.loc[0]
    assert row["Close Market Prob"] == pytest.approx(0.65)
    assert row["Market CLV"] == pytest.approx(0.65 - 0.50)
    assert row["Model CLV"] == pytest.approx(0.65 - 0.55)


def test_fill_from_archive_under_bet_hand_computed():
    """An Under bet's close_p is the raw P(under line), not 1 minus it, and its
    CLV is close_p minus the bet-side probability with no extra sign flip —
    Market Prob/Win Prob are already expressed on the bet side."""
    history = pd.DataFrame(
        [
            {
                "Player": "Player C",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "blocks",
                "Line": 3.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Under",
                "Win Prob": 0.55,
                "Market Prob": 0.50,
                "Dist": "ZINB",
                "CV": 0.9,
                "Gate": 0.12,
                "Step": 1.0,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            }
        ]
    )
    close_ev = 2.8
    archive = _StubArchive({("NBA", "blocks", "2026-05-04", "Player C"): close_ev})
    df = clv.fill_from_archive(history, archive)

    expected_close_p = get_odds(3.5, close_ev, "ZINB", cv=0.9, gate=0.12, step=1.0)
    row = df.loc[0]
    assert 0.0 <= expected_close_p <= 1.0
    assert row["Close Market Prob"] == pytest.approx(expected_close_p)
    assert row["Market CLV"] == pytest.approx(expected_close_p - 0.50)
    assert row["Model CLV"] == pytest.approx(expected_close_p - 0.55)


def test_fill_from_archive_zinb_gate_changes_close_p():
    """The stored Gate must actually reach get_odds — omitting it silently mis-prices
    zero-inflated cells (NBA carries several live ZINB markets)."""
    history = pd.DataFrame(
        [
            {
                "Player": "Player D",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "blocks",
                "Line": 3.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Under",
                "Win Prob": 0.55,
                "Market Prob": 0.50,
                "Dist": "ZINB",
                "CV": 0.9,
                "Gate": 0.12,
                "Step": 1.0,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            }
        ]
    )
    close_ev = 2.8
    archive = _StubArchive({("NBA", "blocks", "2026-05-04", "Player D"): close_ev})
    df = clv.fill_from_archive(history, archive)

    gated = get_odds(3.5, close_ev, "ZINB", cv=0.9, gate=0.12, step=1.0)
    ungated = get_odds(3.5, close_ev, "ZINB", cv=0.9, gate=None, step=1.0)
    assert gated != pytest.approx(ungated)
    assert df.loc[0, "Close Market Prob"] == pytest.approx(gated)


def test_fill_from_archive_skewnormal_uses_book_shape(monkeypatch):
    """A fitted book_shape cell must feed (sigma, skew_alpha) into get_odds, not the plain
    symmetric mean*cv approximation — mirrors the fidelity model_prob._book_over_prob
    already applies for the live book read."""
    from sportstradamus.helpers import config

    league, market = "TESTLG", "TESTMK"
    coeffs = {"a": 1.3, "b": 1.1, "skew_c": 0.6, "skew_d": -0.01, "n_bins": 9}
    monkeypatch.setitem(config.stat_meta, league, {market: {"cv": 0.4, "book_shape": coeffs}})

    history = pd.DataFrame(
        [
            {
                "Player": "Player E",
                "League": league,
                "Date": "2026-05-04",
                "Market": market,
                "Line": 22.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Over",
                "Win Prob": 0.50,
                "Market Prob": 0.50,
                "Dist": "SkewNormal",
                "CV": 0.4,
                "Gate": np.nan,
                "Step": 0.5,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            }
        ]
    )
    close_ev = 21.0
    archive = _StubArchive({(league, market, "2026-05-04", "Player E"): close_ev})
    df = clv.fill_from_archive(history, archive)

    sigma, skew = config.book_skewnormal_shape(league, market, close_ev, 0.4)
    expected_under = get_odds(
        22.5, close_ev, "SkewNormal", cv=0.4, step=0.5, sigma=float(sigma), skew_alpha=float(skew)
    )
    expected_close_p = 1.0 - expected_under
    symmetric_under = get_odds(22.5, close_ev, "SkewNormal", cv=0.4, step=0.5)
    assert expected_under != pytest.approx(symmetric_under)
    assert df.loc[0, "Close Market Prob"] == pytest.approx(expected_close_p)


def test_fill_from_archive_falls_back_to_composite_under_prob_when_dist_cv_nan():
    """Book-fallback leagues/markets have no trained model, so Dist/CV are NaN; the group
    must fall back to the line-inexact composite under-prob rather than crash or NaN out."""
    history = pd.DataFrame(
        [
            {
                "Player": "Player F",
                "League": "MLB",
                "Date": "2026-05-04",
                "Market": "hits",
                "Line": 1.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Over",
                "Win Prob": 0.52,
                "Market Prob": 0.50,
                "Dist": np.nan,
                "CV": np.nan,
                "Gate": np.nan,
                "Step": np.nan,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            }
        ]
    )
    composite_under = 0.58
    archive = _StubArchive(
        ev_table={},
        under_table={("MLB", "hits", "2026-05-04", "Player F"): composite_under},
    )
    df = clv.fill_from_archive(history, archive)

    expected_close_p = 1.0 - composite_under
    row = df.loc[0]
    assert 0.0 <= expected_close_p <= 1.0
    assert row["Close Market Prob"] == pytest.approx(expected_close_p)
    assert row["Market CLV"] == pytest.approx(expected_close_p - 0.50)


def test_fill_from_archive_resolves_each_distinct_line_independently():
    """Two offer rows share one PREDICTION_KEY group (same Player/League/Date/Market)
    but quote different Lines — e.g. two books, or a Alt Line sitting beside the
    consensus offer. Each row's Close Market Prob must be computed at its OWN Line,
    not copy-pasted from whichever row happens to be first in the group."""
    close_ev = 11.2
    cv = 0.35
    history = pd.DataFrame(
        [
            {
                "Player": "Player G",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 10.5,
                "Boost": 1.0,
                "Platform": "BookOne",
                "Bet": "Over",
                "Win Prob": 0.60,
                "Market Prob": 0.55,
                "Dist": "Gamma",
                "CV": cv,
                "Gate": np.nan,
                "Step": 0.5,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            },
            {
                "Player": "Player G",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 12.5,
                "Boost": 1.0,
                "Platform": "BookTwo",
                "Bet": "Over",
                "Win Prob": 0.60,
                "Market Prob": 0.45,
                "Dist": "Gamma",
                "CV": cv,
                "Gate": np.nan,
                "Step": 0.5,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            },
        ]
    )
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player G"): close_ev})
    df = clv.fill_from_archive(history, archive)

    expected_under_row0 = get_odds(10.5, close_ev, "Gamma", cv=cv, gate=None, step=0.5)
    expected_under_row1 = get_odds(12.5, close_ev, "Gamma", cv=cv, gate=None, step=0.5)
    expected_close_p_row0 = 1.0 - expected_under_row0
    expected_close_p_row1 = 1.0 - expected_under_row1

    assert expected_close_p_row0 != pytest.approx(expected_close_p_row1)
    assert df.loc[0, "Close Market Prob"] == pytest.approx(expected_close_p_row0)
    assert df.loc[1, "Close Market Prob"] == pytest.approx(expected_close_p_row1)
    assert df.loc[0, "Market CLV"] == pytest.approx(expected_close_p_row0 - 0.55)
    assert df.loc[1, "Market CLV"] == pytest.approx(expected_close_p_row1 - 0.45)


def test_fill_from_archive_quarantines_out_of_range_close_p(monkeypatch):
    """A conversion result outside [0, 1] must never be written — clamp to NaN instead,
    consistent with migrate_leg_schema.py's existing Close Market Prob > 1 quarantine."""

    def _bogus_get_odds(*args, **kwargs):
        del args, kwargs
        return -0.3  # engineered out-of-range "probability"

    monkeypatch.setattr(clv, "get_odds", _bogus_get_odds)
    history = _build_history()
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): _PLAYER_A_CLOSE_EV})
    df = clv.fill_from_archive(history, archive)

    row = df.loc[0]
    assert math.isnan(row["Close Market Prob"])
    assert math.isnan(row["Market CLV"])
    assert math.isnan(row["Model CLV"])


# The fixture date's stand-in kickoff: `_build_history` rows carry no `Commence`, so
# this is the instant they close at and are read as of.
_CUT = datetime(2026, 5, 4, 20)
_EARLY_KICKOFF = datetime(2026, 5, 4, 17)
_EVENING_KICKOFF = datetime(2026, 5, 4, 23, 30)
_PLAYER_A_KEY = ("NBA", "points", "2026-05-04", "Player A")
_CLOSING_TRIO = ["Close Market Prob", "Market CLV", "Model CLV"]


def test_fill_from_archive_leaves_group_unfilled_while_cut_is_ahead():
    """Before the cut an ``at=cut`` read returns the newest quote so far, not the close,
    so the group is skipped without the archive being asked; a group past its own cut
    fills in the same pass."""
    history = _build_history()
    history.loc[1, "Date"] = "2026-05-03"
    player_b_key = ("NBA", "points", "2026-05-03", "Player B")
    archive = _StubArchive({_PLAYER_A_KEY: _PLAYER_A_CLOSE_EV, player_b_key: 12.0})
    df = clv.fill_from_archive(history, archive, now=_CUT - timedelta(seconds=1))

    assert archive.asked == [player_b_key]
    assert df.loc[0, _CLOSING_TRIO].isna().all()
    assert df.loc[1, "Close Market Prob"] == pytest.approx(
        get_odds(12.5, 12.0, "Gamma", cv=0.35, gate=None, step=0.5)
    )


def test_fill_from_archive_clears_close_stamped_before_cut():
    """A row closed ahead of its cut loses the closing trio (a closed row beats every
    later scoring in ``prediction.cli._upsert_history``, so it would stop updating);
    a row whose cut has passed keeps its close."""
    history = _build_history()
    history.loc[0, _CLOSING_TRIO] = [0.62, 0.07, 0.02]
    history.loc[1, "Date"] = "2026-05-03"
    history.loc[1, _CLOSING_TRIO] = [0.57, 0.07, 0.09]
    df = clv.fill_from_archive(history, _StubArchive({}), now=_CUT - timedelta(seconds=1))

    assert df.loc[0, _CLOSING_TRIO].isna().all()
    assert df.loc[1, "Close Market Prob"] == pytest.approx(0.57)
    assert df.loc[1, "Market CLV"] == pytest.approx(0.57 - 0.50)
    assert df.loc[1, "Model CLV"] == pytest.approx(0.57 - 0.48)


def test_fill_from_archive_fills_once_cut_has_passed_and_rerun_changes_nothing():
    archive = _StubArchive({_PLAYER_A_KEY: _PLAYER_A_CLOSE_EV})
    df = clv.fill_from_archive(_build_history(), archive, now=_CUT)

    expected_close_p = _player_a_expected_close_p()
    assert df.loc[0, "Close Market Prob"] == pytest.approx(expected_close_p)
    assert df.loc[0, "Market CLV"] == pytest.approx(expected_close_p - 0.55)
    assert df.loc[0, "Model CLV"] == pytest.approx(expected_close_p - 0.60)

    first = df.copy()
    rerun = clv.fill_from_archive(df, archive, now=_CUT)
    pd.testing.assert_frame_equal(rerun, first)
    assert archive.asked.count(_PLAYER_A_KEY) == 1


def test_fill_from_archive_row_without_parseable_date_has_no_cut():
    """A date that does not parse gives no cut to wait for: the row fills when pending
    and keeps a close it already carries, whatever ``now`` is."""
    history = _build_history()
    history["Date"] = "TBD"
    history.loc[1, "Close Market Prob"] = 0.57
    archive = _StubArchive({("NBA", "points", "TBD", "Player A"): _PLAYER_A_CLOSE_EV})
    df = clv.fill_from_archive(history, archive, now=datetime(2000, 1, 1))

    assert df.loc[0, "Close Market Prob"] == pytest.approx(_player_a_expected_close_p())
    assert df.loc[1, "Close Market Prob"] == pytest.approx(0.57)


def _kickoff_history(*rows):
    """Player A's resolvable offer, once per ``(player, platform, team, commence)``."""
    offer = _build_history().iloc[0].to_dict()
    return pd.DataFrame(
        [
            {**offer, "Player": player, "Platform": platform, "Team": team, "Commence": commence}
            for player, platform, team, commence in rows
        ]
    )


def _closing_archive(*players):
    return _StubArchive(
        {("NBA", "points", "2026-05-04", player): _PLAYER_A_CLOSE_EV for player in players}
    )


def test_fill_from_archive_evening_game_closes_at_its_kickoff_not_the_stand_in():
    """A 23:30 UTC kickoff is still ahead at 21:00 although the 20:00 stand-in has
    passed; once it has passed, the close is read as of the kickoff itself."""
    history = _kickoff_history(("Player A", "Underdog", "BOS", "2026-05-04T23:30:00Z"))
    archive = _closing_archive("Player A")

    df = clv.fill_from_archive(history, archive, now=datetime(2026, 5, 4, 21))
    assert archive.asked_at == []
    assert df.loc[0, _CLOSING_TRIO].isna().all()

    df = clv.fill_from_archive(df, archive, now=_EVENING_KICKOFF)
    assert archive.asked_at == [("Player A", _EVENING_KICKOFF)]
    assert df.loc[0, "Close Market Prob"] == pytest.approx(_player_a_expected_close_p())


def test_fill_from_archive_early_game_closes_at_its_kickoff_before_the_stand_in():
    history = _kickoff_history(("Player A", "Underdog", "BOS", "2026-05-04T17:00:00Z"))
    archive = _closing_archive("Player A")
    df = clv.fill_from_archive(history, archive, now=datetime(2026, 5, 4, 18))

    assert archive.asked_at == [("Player A", _EARLY_KICKOFF)]
    assert df.loc[0, "Close Market Prob"] == pytest.approx(_player_a_expected_close_p())


def test_fill_from_archive_sleeper_row_borrows_its_teams_kickoff():
    """Sleeper posts no kickoff: its row takes the one an Underdog row of the same team
    and day carries, in its own prediction or in another, and falls back to the
    stand-in when its team has none."""
    history = _kickoff_history(
        ("Player A", "Underdog", "BOS", "2026-05-04T23:30:00Z"),
        ("Player A", "Sleeper", "BOS", ""),
        ("Player B", "Sleeper", "BOS", ""),
        ("Player C", "Sleeper", "LAL", ""),
    )
    archive = _closing_archive("Player A", "Player B", "Player C")

    df = clv.fill_from_archive(history, archive, now=datetime(2026, 5, 4, 21))
    assert archive.asked_at == [("Player C", _CUT)]
    assert df.loc[:2, "Close Market Prob"].isna().all()

    df = clv.fill_from_archive(df, archive, now=_EVENING_KICKOFF)
    assert archive.asked_at[1:] == [
        ("Player A", _EVENING_KICKOFF),
        ("Player B", _EVENING_KICKOFF),
    ]
    assert df["Close Market Prob"].notna().all()


def test_fill_from_archive_prediction_is_read_once_at_its_latest_kickoff():
    """Rows of one prediction can carry different kickoffs (a doubleheader shares a
    PREDICTION_KEY): none closes before the latest, and one read serves them all."""
    history = _kickoff_history(
        ("Player A", "Underdog", "BOS", "2026-05-04T17:00:00Z"),
        ("Player A", "Underdog", "BOS", "2026-05-04T23:30:00Z"),
    )
    archive = _closing_archive("Player A")

    df = clv.fill_from_archive(history, archive, now=datetime(2026, 5, 4, 18))
    assert archive.asked_at == []

    df = clv.fill_from_archive(df, archive, now=_EVENING_KICKOFF)
    assert archive.asked_at == [("Player A", _EVENING_KICKOFF)]
    assert df["Close Market Prob"].notna().all()


def test_fill_from_archive_row_filed_under_another_team_shares_its_predictions_kickoff():
    """A row with no kickoff to borrow by team still closes with the rest of its
    prediction, at the real kickoff rather than at the later stand-in."""
    history = _kickoff_history(
        ("Player A", "Underdog", "BOS", "2026-05-04T17:00:00Z"),
        ("Player A", "Sleeper", "NYK", ""),
    )
    archive = _closing_archive("Player A")
    df = clv.fill_from_archive(history, archive, now=datetime(2026, 5, 4, 18))

    assert archive.asked_at == [("Player A", _EARLY_KICKOFF)]
    assert df["Close Market Prob"].notna().all()


@pytest.mark.parametrize("commence", [None, ""], ids=["scored_before_the_column", "sleeper"])
def test_fill_from_archive_row_with_no_kickoff_anywhere_uses_the_stand_in(commence):
    history = _kickoff_history(("Player A", "Sleeper", "BOS", commence))
    archive = _closing_archive("Player A")

    clv.fill_from_archive(history, archive, now=_CUT - timedelta(seconds=1))
    assert archive.asked_at == []

    clv.fill_from_archive(history, archive, now=_CUT)
    assert archive.asked_at == [("Player A", _CUT)]


def test_fill_from_archive_frame_without_commence_column_reads_at_the_stand_in():
    archive = _StubArchive({_PLAYER_A_KEY: _PLAYER_A_CLOSE_EV})
    clv.fill_from_archive(_build_history(), archive, now=_CUT)

    assert archive.asked_at == [("Player A", _CUT), ("Player B", _CUT)]


def test_row_without_parseable_date_is_read_with_no_cutoff_rather_than_nat():
    """NaT must not reach the archive: DuckDB binds it as NULL, which matches no quote."""
    history = _build_history()
    history["Date"] = "TBD"
    archive = _StubArchive({})
    clv.fill_from_archive(history, archive, now=datetime(2000, 1, 1))
    clv.summarize(history, archive=archive)

    no_cutoff = [("Player A", None), ("Player B", None)]
    assert archive.asked_at == no_cutoff
    assert archive.movement_until == no_cutoff


def test_summarize_reads_line_movement_up_to_the_instant_the_close_was_read_at():
    history = _kickoff_history(("Player A", "Underdog", "BOS", "2026-05-04T23:30:00Z"))
    archive = _closing_archive("Player A")
    df = clv.fill_from_archive(history, archive, now=_EVENING_KICKOFF)
    clv.summarize(df, archive=archive)

    assert archive.movement_until == [("Player A", _EVENING_KICKOFF)]


def test_summarize_drops_unresolved_legs():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): _PLAYER_A_CLOSE_EV})
    df = clv.fill_from_archive(_build_history(), archive)

    expected_close_p = _player_a_expected_close_p()
    summary = clv.summarize(df)
    assert summary["n"] == 1
    assert summary["market_clv_mean"] == pytest.approx(expected_close_p - 0.55)
    assert summary["model_clv_mean"] == pytest.approx(expected_close_p - 0.60)
    assert summary["frac_beat_close"] == pytest.approx(1.0 if expected_close_p > 0.55 else 0.0)


def test_summarize_returns_zero_n_when_no_legs():
    summary = clv.summarize(
        pd.DataFrame(
            {
                "League": pd.Series(dtype=str),
                "Market": pd.Series(dtype=str),
                "Platform": pd.Series(dtype=str),
                "Bet": pd.Series(dtype=str),
                "Win Prob": pd.Series(dtype=float),
                "Close Market Prob": pd.Series(dtype=float),
                "Market CLV": pd.Series(dtype=float),
                "Model CLV": pd.Series(dtype=float),
                "Date": pd.Series(dtype=str),
                "Player": pd.Series(dtype=str),
            }
        )
    )
    assert summary["n"] == 0
    assert math.isnan(summary["market_clv_mean"])
