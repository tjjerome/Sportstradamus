"""Pins for the ``Stats.update`` season gate and season-opener adoption.

``base.Stats.update`` wraps every league's ``_update`` behind
``odds_budget.update_window_open`` and, when the activity feed has observed
a newer opening night, adopts it as ``season_start`` before fetching (unless
``SPORTSTRADAMUS_FORCE_UPDATE`` pins manual control for replay tools). The
``_set_season_start`` hook keeps NBA's ``"2025-26"`` season string and
WNBA's season year derived from whatever date is adopted.
"""

from __future__ import annotations

from datetime import date

import pytest

from sportstradamus.helpers import odds_budget
from sportstradamus.stats import StatsNBA, StatsWNBA
from sportstradamus.stats.base import Stats

_CONSTANT_START = date(2025, 10, 1)
_FEED_OPENER = date(2026, 10, 20)


class _FakeLeague(Stats):
    def __init__(self):
        super().__init__()
        self.league = "TST"
        self.season_start = _CONSTANT_START
        self.update_ran = False

    def _update(self):
        self.update_ran = True


@pytest.fixture
def gates(monkeypatch):
    """Window open, no observed opener — tests override per case."""
    monkeypatch.setattr(odds_budget, "update_window_open", lambda league, season_start: True)
    monkeypatch.setattr(odds_budget, "season_opener", lambda league: None)
    return monkeypatch


def test_update_skipped_outside_season_window(gates) -> None:
    gates.setattr(odds_budget, "update_window_open", lambda league, season_start: False)
    s = _FakeLeague()
    s.update()
    assert not s.update_ran
    assert s.season_start == _CONSTANT_START


def test_update_adopts_newer_feed_opener(gates) -> None:
    gates.setattr(odds_budget, "season_opener", lambda league: _FEED_OPENER)
    s = _FakeLeague()
    s.update()
    assert s.update_ran
    assert s.season_start == _FEED_OPENER


def test_update_keeps_constant_when_opener_older_or_absent(gates) -> None:
    s = _FakeLeague()
    s.update()
    assert s.update_ran
    assert s.season_start == _CONSTANT_START

    gates.setattr(odds_budget, "season_opener", lambda league: _CONSTANT_START)
    s2 = _FakeLeague()
    s2.update()
    assert s2.season_start == _CONSTANT_START


def test_force_env_pins_manual_season_start(gates) -> None:
    # reset_stat_structs replays historical seasons by overriding
    # season_start; the bypass must not let the feed clobber that.
    gates.setattr(odds_budget, "season_opener", lambda league: _FEED_OPENER)
    gates.setenv("SPORTSTRADAMUS_FORCE_UPDATE", "1")
    s = _FakeLeague()
    s.season_start = date(2021, 9, 9)
    s.update()
    assert s.update_ran
    assert s.season_start == date(2021, 9, 9)


def test_nba_season_string_derives_from_season_start() -> None:
    nba = StatsNBA()
    assert nba.season == "2025-26"
    nba._set_season_start(date(2026, 10, 20))
    assert nba.season_start == date(2026, 10, 20)
    assert nba.season == "2026-27"


def test_wnba_season_year_derives_from_season_start() -> None:
    wnba = StatsWNBA()
    assert wnba.season == 2026
    wnba._set_season_start(date(2027, 5, 14))
    assert wnba.season == 2027
