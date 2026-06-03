"""Unit tests for the missing-model book-odds fallback in
:mod:`sportstradamus.prediction.model_prob`.

``book_fallback_prob`` is routed to by ``process_offers`` whenever a cell would
otherwise score empty (no trained model pickle, or a model that matched no
players). It must devig the composite book odds into the model slot so the leg
still scores — ``Model`` mirroring ``Books`` — with full feature parity when a
feature matrix is available and neutral fills (never a NaN ``Player position``,
which would crash correlation) when it is not.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import get_odds

# ``sportstradamus.prediction`` re-exports the ``model_prob`` function, which
# shadows the submodule under attribute access — fetch the module explicitly so
# monkeypatching its module-level ``archive`` / ``stat_*`` globals works.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

_LEAGUE = "NBA"
_RAW_MARKET = "points"
_PLATFORM = "Underdog"
_BOOK_EV = 22.0
_LINE = 25.5
_CV = 0.5


class _StubArchive:
    """Minimal archive surface read by book_fallback_prob / _finalize_records."""

    default_totals = {_LEAGUE: 220.0}

    def __init__(self, ev):
        self._ev = ev

    def get_ev(self, league, market, date, player):
        return self._ev


class _StubStats:
    """Stand-in Stats object exposing only what the fallback reads."""

    league = _LEAGUE

    def __init__(self, feature_frame):
        self._features = feature_frame

    def get_stats(self, market, offers):
        return self._features

    def check_combo_markets(self, market, player, date):
        return float("nan")


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
        },
        index=players,
    )


def test_full_parity_model_mirrors_book(monkeypatch):
    _patch_cell(monkeypatch)
    monkeypatch.setattr(mp, "archive", _StubArchive(_BOOK_EV))
    stats = _StubStats(_feature_frame(["Test Player"], player_position=1))

    records = mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats)

    assert len(records) == 1
    rec = records[0]
    # The model slot is the book: identical probability and EV-weighted value.
    assert rec["Model P"] == pytest.approx(rec["Books P"])
    assert rec["Model"] == pytest.approx(rec["Books"])
    assert rec["Model EV"] == pytest.approx(_BOOK_EV)
    assert rec["Dist"] == "NegBin"
    # Feature-derived parity columns are populated; position mapped to an int.
    assert isinstance(rec["Player position"], (int, np.integer))
    assert rec["Player position"] == 1
    assert rec["Avg 5"] == pytest.approx(20.0 - _LINE)
    # Probability equals the devigged book over/under at the line.
    over = 1 - get_odds(_LINE, _BOOK_EV, "NegBin", _CV, step=1.0, gate=None)
    assert rec["Model P"] == pytest.approx(max(over, 1 - over))


def test_neutral_fill_when_no_feature_matrix(monkeypatch):
    _patch_cell(monkeypatch)
    monkeypatch.setattr(mp, "archive", _StubArchive(_BOOK_EV))
    stats = _StubStats(pd.DataFrame())  # no players matched -> no features

    records = mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats)

    assert len(records) == 1
    rec = records[0]
    assert rec["Model"] == pytest.approx(rec["Books"])
    # Player position must be the int sentinel -1, never NaN, or correlation breaks.
    assert rec["Player position"] == -1
    assert isinstance(rec["Player position"], (int, np.integer))
    assert np.isnan(rec["Avg 5"])


def test_returns_empty_when_market_unknown(monkeypatch):
    monkeypatch.setattr(mp, "stat_dist", {})  # no distribution to devig with
    monkeypatch.setattr(mp, "archive", _StubArchive(_BOOK_EV))
    stats = _StubStats(pd.DataFrame())

    assert mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats) == []


def test_returns_empty_when_no_book_odds(monkeypatch):
    _patch_cell(monkeypatch)
    monkeypatch.setattr(mp, "archive", _StubArchive(float("nan")))  # no archive price
    stats = _StubStats(pd.DataFrame())  # check_combo_markets also returns NaN

    assert mp.book_fallback_prob([_offer()], _LEAGUE, _RAW_MARKET, _PLATFORM, stats) == []
