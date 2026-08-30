"""Modal-line cohort book fallback + payout-implied unquoted pricing.

``book_fallback_prob`` must price a model-less leg from the same modal-line cohort
quote the training matrix resolves (``resolve_training_quote`` over
``Archive.get_training_quote_inputs``) — never from a cross-line EV average that a
DFS platform's reposted or discounted line can poison — and must price every
offered line with the exact shape used to invert the quote, so
``decode(invert(p)) == p`` at the quote line. Rows nothing real priced never
serve. ``_finalize_records`` fills a missing ``Market EV`` with the payout-implied
probability of the chosen side (never a flat 0.5) and drops unquoted
single-player rows whose model probability disagrees with that only-available
price beyond ``_UNQUOTED_BOOK_DISAGREEMENT_MAX``.
"""

from __future__ import annotations

import contextlib
import datetime
import importlib

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import get_odds
from sportstradamus.helpers.archive import Archive
from sportstradamus.helpers.distributions import (
    UNDERDOG_BOOST_BASELINE,
    dfs_boost_probs,
)

mp = importlib.import_module("sportstradamus.prediction.model_prob")

_LEAGUE = "NBA"
_MARKET = "TESTMKT"
_DATE = "2026-06-03"
_TS = datetime.datetime(2026, 6, 3, 12, 0, 0)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.duckdb"
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(db_path))
    if Archive._instance is not None:
        with contextlib.suppress(Exception):
            Archive._instance._connection.close()
        Archive._instance._initialized = False
    a = Archive()
    monkeypatch.setattr(mp, "archive", a)
    yield a
    with contextlib.suppress(Exception):
        a._connection.close()
    Archive._instance._initialized = False


def _insert_odds(a, entity, book, under, line, ev=None):
    a._connection.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, under_prob, line, "
        "observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [_LEAGUE, _MARKET, _DATE, entity, book, ev, under, line, _TS],
    )


def _patch_cell(monkeypatch, dist, cv):
    monkeypatch.setattr(mp, "stat_dist", {_LEAGUE: {_MARKET: dist}})
    monkeypatch.setattr(mp, "stat_cv", {_LEAGUE: {_MARKET: cv}})


class _StubStats:
    league = _LEAGUE

    def __init__(self, combo_ev=float("nan")):
        self._combo_ev = combo_ev

    def get_stats(self, market, offers):
        return pd.DataFrame()

    def check_combo_markets(self, market, player, date):
        return self._combo_ev


def _offer(player, line):
    return {
        "Player": player,
        "League": _LEAGUE,
        "Team": "LAL",
        "Opponent": "BOS",
        "Date": _DATE,
        "Market": _MARKET,
        "Line": line,
        "Boost_Over": 1.0,
        "Boost_Under": 1.0,
    }


def test_modal_cohort_prices_all_lines_shape_consistently(archive, monkeypatch):
    """The 2-book 33.5 cohort wins; Sleeper's discounted 24.5 line cannot poison it."""
    _patch_cell(monkeypatch, "SkewNormal", 0.2)
    _insert_odds(archive, "Star Guard", "Sleeper", 0.5, 24.5)
    _insert_odds(archive, "Star Guard", "fanduel", 0.60, 33.5)
    _insert_odds(archive, "Star Guard", "draftkings", 0.70, 33.5)

    records = mp.book_fallback_prob(
        [_offer("Star Guard", 33.5), _offer("Star Guard", 24.5)],
        _LEAGUE,
        _MARKET,
        "Underdog",
        _StubStats(),
    )

    by_line = {r["Line"]: r for r in records}
    assert set(by_line) == {33.5, 24.5}
    # Round-trip identity at the modal quote line: the served probability IS the
    # cohort's consensus under-probability, decode(invert(p)) == p.
    at_quote = by_line[33.5]
    assert at_quote["Bet"] == "Under"
    assert at_quote["Win Prob"] == pytest.approx(0.65, abs=1e-6)
    assert at_quote["Market Prob"] == pytest.approx(0.65, abs=1e-6)
    # The discounted alternate line prices off the SAME implied mean and lands far
    # over 50/50 — never at the platform's fake coin flip.
    discounted = by_line[24.5]
    assert discounted["Bet"] == "Over"
    assert discounted["Win Prob"] > 0.7


def test_all_dfs_cohort_produces_no_served_row(archive, monkeypatch):
    _patch_cell(monkeypatch, "SkewNormal", 0.2)
    _insert_odds(archive, "Star Guard", "Sleeper", 0.62, 24.5)
    _insert_odds(archive, "Star Guard", "Underdog", 0.58, 24.5)

    records = mp.book_fallback_prob(
        [_offer("Star Guard", 24.5)], _LEAGUE, _MARKET, "Underdog", _StubStats()
    )

    assert records == []


def test_one_real_book_beside_dfs_serves(archive, monkeypatch):
    """One real sportsbook in the cohort satisfies _MIN_FALLBACK_REAL_BOOKS."""
    _patch_cell(monkeypatch, "SkewNormal", 0.2)
    _insert_odds(archive, "Star Guard", "Sleeper", 0.60, 24.5)
    _insert_odds(archive, "Star Guard", "fanduel", 0.60, 24.5)

    records = mp.book_fallback_prob(
        [_offer("Star Guard", 24.5)], _LEAGUE, _MARKET, "Underdog", _StubStats()
    )

    assert len(records) == 1
    assert records[0]["Win Prob"] == pytest.approx(0.60, abs=1e-6)


def test_combo_ev_inversion_still_serves(archive, monkeypatch):
    """A combo_props market's combo consensus serves without any archived rows."""
    _patch_cell(monkeypatch, "NegBin", 0.5)
    monkeypatch.setitem(mp.combo_props, _MARKET, ["A", "B"])

    records = mp.book_fallback_prob(
        [_offer("Combo Guy", 25.5)], _LEAGUE, _MARKET, "Underdog", _StubStats(combo_ev=22.0)
    )

    assert len(records) == 1
    rec = records[0]
    assert rec["Projection"] == pytest.approx(22.0, abs=1e-4)
    assert rec["Model Version"] == mp._BOOK_FALLBACK_VERSION


def test_fantasy_market_never_serves_combo_fallback(archive, monkeypatch):
    """A market outside combo_props (fantasy scores) gets no combo second pass.

    The 8-component weighted mean plus generic-cv tail clean-graded 0.40 at
    claimed >= 0.85 -- those quotes must not reach the board.
    """
    _patch_cell(monkeypatch, "NegBin", 0.5)
    assert _MARKET not in mp.combo_props

    records = mp.book_fallback_prob(
        [_offer("Bench Guy", 5.5)], _LEAGUE, _MARKET, "Underdog", _StubStats(combo_ev=3.0)
    )

    assert records == []


class _TotalsOnlyArchive:
    default_totals = {_LEAGUE: 220.0}


def _finalize_input(rows):
    df = pd.DataFrame(rows)
    df["League"] = _LEAGUE
    df["Date"] = _DATE
    df["Team"] = "LAL"
    df["Opponent"] = "BOS"
    df["Market"] = _MARKET
    df["Projection"] = 1.2
    df["Market Projection"] = 1.2
    df["Push Prob"] = 0.0
    return df


def test_unquoted_rows_price_payout_implied_and_phantoms_drop(monkeypatch):
    monkeypatch.setattr(mp, "archive", _TotalsOnlyArchive())
    decimal = 1.55 * UNDERDOG_BOOST_BASELINE
    rows = [
        # Unquoted boosted over at the payout-implied breakeven; model gap 0.1375
        # sits inside the 0.15 tolerance, so the row survives.
        {
            "Player": "Near Miss",
            "Line": 1.5,
            "Model Over": 0.50,
            "Model Under": 0.50,
            "Market EV": np.nan,
            "Boost_Over": 1.55,
            "Boost_Under": np.nan,
        },
        # Same boost, model 0.55: the gap to ~0.3625 exceeds 0.15 — the +51%-edge
        # phantom class — so the row is dropped.
        {
            "Player": "Phantom",
            "Line": 1.5,
            "Model Over": 0.55,
            "Model Under": 0.45,
            "Market EV": np.nan,
            "Boost_Over": 1.55,
            "Boost_Under": np.nan,
        },
        # Quoted twin of the phantom: a real book probability always keeps the row.
        {
            "Player": "Quoted Twin",
            "Line": 1.5,
            "Model Over": 0.55,
            "Model Under": 0.45,
            "Market EV": 0.62,
            "Boost_Over": 1.55,
            "Boost_Under": np.nan,
        },
        # Combo/H2H entities are exempt from the unquoted-disagreement gate.
        {
            "Player": "A vs. B",
            "Line": 1.5,
            "Model Over": 0.60,
            "Model Under": 0.40,
            "Market EV": np.nan,
            "Boost_Over": 1.55,
            "Boost_Under": np.nan,
        },
    ]

    records = mp._finalize_records(
        _finalize_input(rows),
        _LEAGUE,
        "Underdog",
        "NegBin",
        0.5,
        1.0,
        None,
        1.0,
        mp._BOOK_FALLBACK_VERSION,
    )

    by_player = {r["Player"]: r for r in records}
    assert set(by_player) == {"Near Miss", "Quoted Twin", "A vs. B"}
    implied = dfs_boost_probs(decimal, 0)[0]
    assert implied == pytest.approx(1 / decimal)
    assert by_player["Near Miss"]["Market Prob"] == pytest.approx(implied)
    assert by_player["Near Miss"]["Market EV"] == pytest.approx(implied * decimal)
    assert by_player["A vs. B"]["Market Prob"] == pytest.approx(implied)
    # A quoted probability is never overwritten by the payout-implied one.
    assert by_player["Quoted Twin"]["Market Prob"] == pytest.approx(0.62)


def _model_book_leg(offer_df):
    return mp._book_evs_for_players(
        offer_df,
        _LEAGUE,
        _MARKET,
        "NegBin",
        0.5,
        0.0,
        dict.fromkeys(offer_df.index, _DATE),
        _StubStats(),
        offer_df.index,
    )


def test_model_book_leg_dfs_only_is_nan(archive, monkeypatch):
    """The model path's book leg refuses a DFS platform's self-quote.

    The NaN mean makes the blend run model-only and sends the row to
    ``_finalize_records``' payout-implied pricing plus the disagreement drop —
    never a cross-line inversion of the platform's own price (the Baez 2.47-mean
    phantom).
    """
    _patch_cell(monkeypatch, "NegBin", 0.5)
    _insert_odds(archive, "Star Guard", "Underdog", 0.675, 1.5, ev=1.17)

    offer_df = pd.DataFrame([_offer("Star Guard", 1.5)])
    offer_df.index = offer_df.Player
    evs = _model_book_leg(offer_df)

    assert len(evs) == 1 and np.isnan(evs[0])


def test_model_book_leg_prices_modal_cohort(archive, monkeypatch):
    """A real-book cohort inverts to one mean that reproduces the cohort prob.

    Replaces the cross-line average of stored per-row means: the discounted
    Sleeper 24.5 row cannot drag the 33.5 consensus."""
    _patch_cell(monkeypatch, "NegBin", 0.5)
    _insert_odds(archive, "Star Guard", "Sleeper", 0.5, 24.5)
    _insert_odds(archive, "Star Guard", "fanduel", 0.60, 33.5)
    _insert_odds(archive, "Star Guard", "draftkings", 0.70, 33.5)

    offer_df = pd.DataFrame([_offer("Star Guard", 33.5)])
    offer_df.index = offer_df.Player
    evs = _model_book_leg(offer_df)

    assert get_odds(33.5, evs[0], "NegBin", cv=0.5) == pytest.approx(0.65, abs=1e-6)
