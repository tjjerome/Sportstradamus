"""Smoke test for ``scripts.refit_book_weights`` — the post-repair weights driver.

The driver's job is orchestration only: run the existing ``fit_book_weights``
over every player market of one league and rewrite ``book_weights.json`` with
every other entry preserved. The fit itself is pinned elsewhere, so it is
monkeypatched here.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from sportstradamus.scripts import refit_book_weights as refit
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
