"""Unit tests for the missing-model book-odds fallback in
:mod:`sportstradamus.prediction.model_prob`.

``book_fallback_prob`` is routed to by ``process_offers`` whenever a cell would
otherwise score empty (no trained model pickle, or a model that matched no
players). It resolves the same modal-line cohort quote the training matrix uses
(``resolve_training_quote``) and mirrors it into the model slot — ``Model``
mirroring ``Books`` — with full feature parity when a feature matrix is
available and neutral fills (never a NaN ``Player position``, which would crash
correlation) when it is not. Rows with no servable quote (nothing a real
sportsbook or combo consensus priced) are dropped, never served at a guess.
"""

from __future__ import annotations

import datetime
import importlib

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.distributions import get_ev
from sportstradamus.helpers.training_quotes import ArchivedBookQuote
from sportstradamus.stats import base

# ``sportstradamus.prediction`` re-exports the ``model_prob`` function, which
# shadows the submodule under attribute access — fetch the module explicitly so
# monkeypatching its module-level ``archive`` / ``stat_*`` globals works.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

_LEAGUE = "NBA"
_RAW_MARKET = "points"
_PLATFORM = "Underdog"
_BOOK_UNDER = 0.42
_LINE = 25.5
_CV = 0.5
_TS = datetime.datetime(2026, 6, 3, 12, 0, 0)


class _StubArchive:
    """Minimal archive surface read by book_fallback_prob / _finalize_records."""

    default_totals = {_LEAGUE: 220.0}

    def __init__(self, rows_by_player):
        self._rows = rows_by_player

    def get_training_quote_inputs(self, league, market, date, entities, at=None):
        return {e: (self._rows.get(e, []), None) for e in entities}


class _StubStats:
    """Stand-in Stats object exposing only what the fallback reads."""

    league = _LEAGUE

    def __init__(self, feature_frame, combo_ev=float("nan")):
        self._features = feature_frame
        self._combo_ev = combo_ev

    def get_stats(self, market, offers):
        return self._features

    def check_combo_markets(self, market, player, date):
        return self._combo_ev

    # Bound rather than stubbed: the serving second pass prices through the real
    # component-sum kernel, so a stand-in would stop tracking its admission rules.
    combo_quote = base.Stats.combo_quote


def _offer(player="Test Player", line=_LINE):
    return {
        "Player": player,
        "League": _LEAGUE,
        "Team": "LAL",
        "Opponent": "BOS",
        "Date": "2026-06-03",
        "Market": _RAW_MARKET,
        "Line": line,
        "Boost_Over": 1.0,
        "Boost_Under": 1.0,
        "Boost": 1.0,
    }


def _quote_rows(player="Test Player", book="fanduel", under=_BOOK_UNDER, line=_LINE, ev=None):
    return {player: [ArchivedBookQuote(book, ev, under, line, _TS)]}


def _patch_cell(monkeypatch, dist="NegBin", cv=_CV):
    """Point the config dicts at one synthetic cell, keyed on the normalized
    market so the lookup inside the fallback resolves."""
    norm = mp.normalize_market(_LEAGUE, _RAW_MARKET, _PLATFORM)
    monkeypatch.setattr(mp, "stat_dist", {_LEAGUE: {norm: dist}})
    monkeypatch.setattr(mp, "stat_cv", {_LEAGUE: {norm: cv}})
    monkeypatch.setattr(mp, "stat_zi", {})
    return norm


def _feature_frame(players, player_position=1):
    return pd.DataFrame(
        {
            "Avg5": [20.0] * len(players),
            "AvgH2H": [21.0] * len(players),
            "H2HPlayed": [3] * len(players),
            "Total": [220.0] * len(players),
            "Defense position": [0.1] * len(players),
            "Player position": [player_position] * len(players),
            "Moneyline": [0.5] * len(players),
            "Home": [True] * len(players),
        },
        index=players,
    )


def test_full_parity_model_mirrors_book(monkeypatch):
    _patch_cell(monkeypatch)
    monkeypatch.setattr(mp, "archive", _StubArchive(_quote_rows()))
    stats = _StubStats(_feature_frame(["Test Player"], player_position=1))

    records = mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats)

    assert len(records) == 1
    rec = records[0]
    # The model slot is the book: identical probability and EV-weighted value.
    assert rec["Win Prob"] == pytest.approx(rec["Market Prob"])
    assert rec["Model EV"] == pytest.approx(rec["Market EV"])
    # An offer at the quote line round-trips the cohort probability exactly.
    assert rec["Bet"] == "Over"
    assert rec["Win Prob"] == pytest.approx(1 - _BOOK_UNDER, abs=1e-6)
    # The projection is the mean implied by the quote under the cell's shape.
    assert rec["Projection"] == pytest.approx(
        get_ev(_LINE, _BOOK_UNDER, _CV, dist="NegBin"), abs=1e-6
    )
    assert rec["Dist"] == "NegBin"
    # A model-less devigged leg attributes to the book-fallback sentinel.
    assert rec["Model Version"] == mp._BOOK_FALLBACK_VERSION
    # Feature-derived parity columns are populated; position mapped to an int.
    assert isinstance(rec["Player position"], (int, np.integer))
    assert rec["Player position"] == 1
    assert rec["Avg 5"] == pytest.approx(20.0 - _LINE)
    assert rec["Home"] is True


def test_neutral_fill_when_no_feature_matrix(monkeypatch):
    _patch_cell(monkeypatch)
    monkeypatch.setattr(mp, "archive", _StubArchive(_quote_rows()))
    stats = _StubStats(pd.DataFrame())  # no players matched -> no features

    records = mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats)

    assert len(records) == 1
    rec = records[0]
    assert rec["Model EV"] == pytest.approx(rec["Market EV"])
    # Player position must be the int sentinel -1, never NaN, or correlation breaks.
    assert rec["Player position"] == -1
    assert isinstance(rec["Player position"], (int, np.integer))
    assert np.isnan(rec["Avg 5"])
    # Home defaults to False, not NaN (NaN is truthy and would render the row as host).
    assert rec["Home"] is False


def test_returns_empty_when_market_unknown(monkeypatch):
    monkeypatch.setattr(mp, "stat_dist", {})  # no distribution to devig with
    monkeypatch.setattr(mp, "archive", _StubArchive(_quote_rows()))
    stats = _StubStats(pd.DataFrame())

    assert mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats) == []


def test_returns_empty_when_no_book_quote(monkeypatch):
    _patch_cell(monkeypatch)
    monkeypatch.setattr(mp, "archive", _StubArchive({}))  # no archived rows at all
    stats = _StubStats(pd.DataFrame())  # check_combo_markets also returns NaN

    assert mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats) == []


def test_dfs_only_cohort_never_serves(monkeypatch):
    """A cohort of DFS platforms alone is the platform quoting itself — no serve."""
    _patch_cell(monkeypatch)
    monkeypatch.setattr(mp, "archive", _StubArchive(_quote_rows(book="Sleeper", under=0.5)))
    stats = _StubStats(pd.DataFrame())

    assert mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats) == []


def test_pure_ev_inversion_never_serves(monkeypatch):
    """A legacy ev-only row (no native under-prob) is derived, not independent support."""
    _patch_cell(monkeypatch)
    monkeypatch.setattr(mp, "archive", _StubArchive(_quote_rows(under=None, line=None, ev=22.0)))
    stats = _StubStats(pd.DataFrame())

    assert mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats) == []


def test_legacy_combo_scalar_no_longer_serves(monkeypatch):
    """A combo market with unquoted components serves nothing at all.

    The second pass now prices the component sum rather than inverting
    ``check_combo_markets``' scalar mean under a generic cv, so a market whose
    components nothing priced has no honest quote left to fall back to. The stub
    still offers the scalar; the board must refuse it.
    """
    _patch_cell(monkeypatch)
    stub = _StubArchive({})
    monkeypatch.setattr(mp, "archive", stub)
    monkeypatch.setattr(base, "archive", stub)
    monkeypatch.setitem(mp.combo_props, _RAW_MARKET, ["A", "B"])
    stats = _StubStats(pd.DataFrame(), combo_ev=22.0)

    assert mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats) == []
