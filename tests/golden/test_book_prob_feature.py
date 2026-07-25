"""WS2 2d: the training-matrix book over-prob feature reads the stored consensus
``under_prob`` directly (shape-free) and falls back to the symmetric ``get_odds`` read
only when no book quoted a shape-free under-prob.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import config
from sportstradamus.helpers.distributions import get_odds
from sportstradamus.helpers.training_quotes import ArchivedBookQuote
from sportstradamus.stats import base as base_mod


class _FakeArchive:
    def __init__(self, ev, line, under):
        self._ev, self._line, self._under = ev, line, under

    def get_training_book_quotes(self, *a, **k):
        return [ArchivedBookQuote("book", self._ev, self._under, self._line, None)]

    def get_line(self, *a, **k):
        return self._line

    def get_training_quote_inputs(self, _league, _market, _date, entities, **kwargs):
        rows = [ArchivedBookQuote("book", self._ev, self._under, self._line, None)]
        return dict.fromkeys(entities, (rows, self._line))


class _Stub:
    league = "WNBA"

    def check_combo_markets(self, *a, **k):
        return np.nan

    _resolve_player_market_odds = base_mod.Stats._resolve_player_market_odds


def _run(monkeypatch, ev, line, under):
    monkeypatch.setattr(base_mod, "archive", _FakeArchive(ev, line, under))
    stats = pd.DataFrame({"Avg10": [2.0]}, index=["P"])
    return _Stub()._resolve_player_market_odds(stats, "AST", "2026-05-08", None)


def test_feature_uses_stored_under_prob(monkeypatch):
    """A real consensus under_prob feeds 1 - p_under directly, not the get_odds reconstruction."""
    quote = _run(monkeypatch, ev=2.4, line=1.5, under=0.55).iloc[0]
    assert quote["Odds"] == pytest.approx(1 - 0.55)
    assert quote["QuoteAuthenticity"] == "authentic"


def test_feature_falls_back_to_get_odds_when_absent(monkeypatch):
    """No shape-free quote (NaN) -> the legacy symmetric get_odds(line, ev) feature."""
    quote = _run(monkeypatch, ev=2.4, line=1.5, under=np.nan).iloc[0]
    cv = config.stat_cv["WNBA"].get("AST", 1)
    dist = config.stat_dist.get("WNBA", {}).get("AST", "Gamma")
    assert quote["Odds"] == pytest.approx(1 - get_odds(1.5, 2.4, dist, cv=cv))
    assert quote["QuoteAuthenticity"] == "derived"
    assert quote["Odds_synthetic"]
