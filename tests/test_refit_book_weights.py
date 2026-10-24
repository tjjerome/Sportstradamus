"""``scripts.refit_book_weights`` (the post-repair weights driver) and the fit it runs.

The driver's job is orchestration only: run the existing ``fit_book_weights``
over every player market of one league and rewrite ``book_weights.json`` with
every other entry preserved, so its test monkeypatches the fit. The one rule
pinned on the fit itself is which books it weighs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
from click.testing import CliRunner

from sportstradamus.scripts import refit_book_weights as refit
from sportstradamus.training.calibration import fit_book_weights
from sportstradamus.training.markets import ALL_MARKETS


class _StubStats:
    def load(self):
        self.loaded = True


def test_refit_calls_fitter_per_market_and_preserves_other_entries(tmp_path, monkeypatch):
    path = tmp_path / "book_weights.json"
    path.write_text(
        json.dumps(
            {
                "NBA": {"PTS": {"fanduel": 0.9}},
                "WNBA": {"Moneyline": {"fanduel": 1.0}, "PTS": {"Sleeper": 0.63}},
            }
        )
    )

    calls = []

    def fake_fit(league, market, stat_data, archive, book_weights):
        calls.append((league, market))
        assert isinstance(stat_data, _StubStats) and stat_data.loaded
        return {"fanduel": 0.5}

    monkeypatch.setattr(refit, "_BOOK_WEIGHTS_PATH", path)
    monkeypatch.setattr(refit, "_LEAGUE_CLASSES", {"WNBA": _StubStats})
    monkeypatch.setattr(refit, "fit_book_weights", fake_fit)

    result = CliRunner().invoke(refit.main, ["--league", "WNBA"])

    assert result.exit_code == 0, result.output
    assert calls == [("WNBA", market) for market in ALL_MARKETS["WNBA"]]

    written = json.loads(path.read_text())
    assert written["NBA"] == {"PTS": {"fanduel": 0.9}}
    assert written["WNBA"]["Moneyline"] == {"fanduel": 1.0}
    for market in ALL_MARKETS["WNBA"]:
        assert written["WNBA"][market] == {"fanduel": 0.5}


def test_fit_book_weights_gives_a_pickem_platform_no_weight():
    """A platform's column is the entry being graded, not a book to weigh."""
    dates = pd.date_range("2026-05-01", periods=12).strftime("%Y-%m-%d")
    index = pd.MultiIndex.from_product([dates, ["P"]], names=["date", "player"])
    wide = pd.DataFrame({"fanduel": 4.0, "draftkings": 5.0, "Underdog": 9.0}, index=index)
    stats = SimpleNamespace(
        gamelog=pd.DataFrame({"player": "P", "date": dates, "XMKT": 4.0}),
        log_strings={"player": "player", "date": "date"},
    )
    archive = SimpleNamespace(to_pandas=lambda league, market: wide)

    weights = fit_book_weights("WNBA", "XMKT", stats, archive, {})

    assert set(weights) == {"fanduel", "draftkings"}
