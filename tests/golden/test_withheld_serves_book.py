"""A withheld cell serves the book consensus: never its model, never nothing.

``shipped: "withheld"`` in ``stat_meta`` keeps a cell's model off the board, but its
offers still belong on the dashboard at the book's price (owner, 2026-10-23). So
``match_offers`` hands a withheld cell no feature matrix, before any feature work,
and ``_score_market`` routes it to ``book_fallback_prob`` even when a missed
``meditate`` prune left the pickle on disk (two weeks of WNBA PRA, 2026-08). That
stale pickle draws a warning, unless the cell is a volume denominator, which keeps
its pickle on purpose.
"""

import logging
import pickle
from datetime import date

import pandas as pd
import pytest

from sportstradamus.helpers import stat_meta
from sportstradamus.prediction import scoring
from sportstradamus.training.ship_config import WITHHELD

_DAY = date(2026, 10, 23)
_BOOK_ROW = {"Player": "A", "Model Weight": 0.0}
_MODEL_ROW = {"Player": "A", "Model Weight": 0.6}


@pytest.fixture
def scorers(monkeypatch, tmp_path):
    """Models load from ``tmp_path``; each scorer ``_score_market`` reaches is logged in order."""
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(scoring.pkg_resources, "files", lambda package: tmp_path)
    reached = []

    def book_fallback_prob(offers, league, market, platform, stat_data, day):
        reached.append(("book", market))
        return [_BOOK_ROW]

    def model_prob(offers, league, market, platform, stat_data, player_stats):
        reached.append(("model", market))
        return [_MODEL_ROW]

    monkeypatch.setattr(scoring, "book_fallback_prob", book_fallback_prob)
    monkeypatch.setattr(scoring, "model_prob", model_prob)
    return reached


class _Stats:
    """The ``Stats`` surface ``match_offers`` reads, logging every feature build."""

    def __init__(self):
        self.gamelog = pd.DataFrame(columns=["PRA", "MIN"])
        self.built = []

    def get_stats(self, market, offers, day):
        self.built.append(market)
        return pd.DataFrame({"MeanYr": [25.0]}, index=["A"])


def _write_pickle(tmp_path, market):
    with open(tmp_path / "models" / f"WNBA_{market}.mdl", "wb") as outfile:
        pickle.dump({"expected_columns": ["MeanYr"]}, outfile)


def _score(market, stats):
    offers = [{"Player": "A", "Market": market, "Line": 20.5, "Date": _DAY.isoformat()}]
    return scoring._score_market(offers, "WNBA", market, "Underdog", stats, _DAY)


@pytest.mark.parametrize(
    ("market", "pickled", "warned"),
    [
        ("PRA", True, True),  # a missed meditate prune left the model on disk
        ("PRA", False, False),  # pruned, as meditate does
        ("MIN", True, False),  # a volume denominator keeps its pickle on purpose
    ],
)
def test_a_withheld_cell_serves_the_book(
    monkeypatch, tmp_path, scorers, caplog, market, pickled, warned
):
    monkeypatch.setitem(stat_meta["WNBA"][market], "shipped", WITHHELD)
    if pickled:
        _write_pickle(tmp_path, market)
    stats = _Stats()

    with caplog.at_level(logging.WARNING, logger="log"):
        served = _score(market, stats)

    assert served == [_BOOK_ROW]
    assert scorers == [("book", market)]
    assert stats.built == []
    warning = f"WNBA_{market} withheld but pickle on disk — serving the book consensus"
    assert (warning in caplog.text) is warned


def test_a_shipped_cell_with_a_pickle_serves_its_model(monkeypatch, tmp_path, scorers):
    monkeypatch.setitem(stat_meta["WNBA"]["PRA"], "shipped", "devel")
    _write_pickle(tmp_path, "PRA")
    stats = _Stats()

    assert _score("PRA", stats) == [_MODEL_ROW]
    assert scorers == [("model", "PRA")]
    assert stats.built == ["PRA"]
