"""Guard test for the canonical Gate-1 decisions file (data/gate1_decisions.json)."""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

from sportstradamus import data
from sportstradamus.training.baselines import STRATEGY_SLUGS
from sportstradamus.training.markets import ALL_MARKETS

# Gate-1 ships under the 5-gate strict offline lifecycle (NBA 7, NFL 5, WNBA 10) after
# the 2026-05-22 post-refresh audit on the v2 gate (35db7d5 on the research branch).
_EXPECTED_DECISION_COUNT = 22


def _load_decisions() -> dict[str, dict[str, str]]:
    path = pkg_resources.files(data) / "gate1_decisions.json"
    with open(str(path)) as fh:
        return json.load(fh)


def test_decisions_has_expected_cell_count():
    decisions = _load_decisions()
    n_cells = sum(len(markets) for markets in decisions.values())
    assert n_cells == _EXPECTED_DECISION_COUNT


def test_every_decision_is_a_real_strategy():
    decisions = _load_decisions()
    for league, markets in decisions.items():
        for market, strategy in markets.items():
            assert strategy in STRATEGY_SLUGS, f"{league}/{market}={strategy!r}"


def test_every_decision_cell_in_all_markets():
    decisions = _load_decisions()
    for league, markets in decisions.items():
        assert league in ALL_MARKETS, league
        for market in markets:
            assert market in ALL_MARKETS[league], f"{league}/{market} not in ALL_MARKETS"
