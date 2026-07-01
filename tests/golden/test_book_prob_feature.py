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
from sportstradamus.stats import base as base_mod


class _FakeArchive:
    def __init__(self, ev, line, under):
        self._ev, self._line, self._under = ev, line, under

    def get_ev(self, *a, **k):
        return self._ev

    def get_line(self, *a, **k):
        return self._line

    def get_composite_under_prob(self, *a, **k):
        return self._under


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
    _, odds, _, _ = _run(monkeypatch, ev=2.4, line=1.5, under=0.55)
    assert odds[0] == pytest.approx(1 - 0.55)


def test_feature_falls_back_to_get_odds_when_absent(monkeypatch):
    """No shape-free quote (NaN) -> the legacy symmetric get_odds(line, ev) feature."""
    _, odds, _, _ = _run(monkeypatch, ev=2.4, line=1.5, under=np.nan)
    cv = config.stat_cv["WNBA"].get("AST", 1)
    dist = config.stat_dist.get("WNBA", {}).get("AST", "Gamma")
    assert odds[0] == pytest.approx(1 - get_odds(1.5, 2.4, dist, cv=cv))
