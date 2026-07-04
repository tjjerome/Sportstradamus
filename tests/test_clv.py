"""Unit tests for the CLV fill / summary helpers in ``sportstradamus.clv``."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus import clv
from sportstradamus.helpers.distributions import get_odds


class _StubArchive:
    """Minimal stub that mirrors ``Archive.get_ev``'s lookup contract.

    ``get_ev`` returns a book stat-mean keyed by ``ev_table``; ``get_composite_under_prob``
    returns a devigged under-probability keyed by ``under_table``, used only by the
    NaN-Dist/CV fallback path.
    """

    def __init__(self, ev_table, under_table=None):
        self._ev_table = ev_table
        self._under_table = under_table or {}

    def get_ev(self, league, market, date, player, *, at=None):
        del at  # the stub ignores the time-series cutoff
        return self._ev_table.get((league, market, date, player), float("nan"))

    def get_composite_under_prob(self, league, market, date, player, *, at=None):
        del at
        return self._under_table.get((league, market, date, player), float("nan"))


def test_signed_clv_over_uses_positive_sign():
    assert clv._signed_clv(0.50, 0.55, "Over") == pytest.approx(0.05)


def test_signed_clv_under_flips_sign():
    assert clv._signed_clv(0.50, 0.55, "Under") == pytest.approx(-0.05)


def test_signed_clv_higher_is_over_synonym():
    assert clv._signed_clv(0.40, 0.45, "Higher") == pytest.approx(0.05)


def test_signed_clv_returns_nan_when_open_is_nan():
    assert math.isnan(clv._signed_clv(float("nan"), 0.55, "Over"))


def test_signed_clv_returns_nan_when_close_is_nan():
    assert math.isnan(clv._signed_clv(0.55, float("nan"), "Under"))


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


def test_fill_from_archive_under_bet_hand_computed():
    """An Under bet's close_p is the raw P(under line), not 1 minus it."""
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
    assert row["Market CLV"] == pytest.approx(-(expected_close_p - 0.50))
    assert row["Model CLV"] == pytest.approx(-(expected_close_p - 0.55))


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
