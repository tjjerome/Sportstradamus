"""Pins for the ``Stats.update`` season gate and season-opener adoption.

``base.Stats.update`` wraps every league's ``_update`` behind
``odds_budget.update_window_open`` and, when the activity feed has observed
an opening night that starts a new season, adopts it as ``season_start``
before fetching (unless ``SPORTSTRADAMUS_FORCE_UPDATE`` pins manual control
for replay tools). A feed opener inside the season already held was captured
mid-season and is ignored. The ``_set_season_start`` hook keeps NBA's
``"2025-26"`` season string and WNBA's season year derived from whatever date
is adopted. ``_full_clean_due`` is the one rule the five leagues' updates read
for when the clean pass over the whole gamelog runs.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from sportstradamus.helpers import odds_budget
from sportstradamus.stats import StatsNBA, StatsWNBA, base
from sportstradamus.stats.base import SEASON_MAX_DAYS, Stats

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


# Production's feed on 2026-10-10. MLB's opener was captured again when the league came
# back from the All-Star break, 113 days into its season; NHL's is next season's opening
# night, 357 days after the start the class holds.
@pytest.mark.parametrize(
    ("held_start", "feed_opener", "adopted"),
    [
        (date(2026, 3, 25), date(2026, 7, 16), False),
        (date(2025, 10, 7), date(2026, 9, 29), True),
    ],
    ids=["mid-season capture", "next season"],
)
def test_update_adopts_a_feed_opener_only_when_it_starts_a_new_season(
    gates, held_start, feed_opener, adopted
) -> None:
    gates.setattr(odds_budget, "season_opener", lambda league: feed_opener)
    s = _FakeLeague()
    s.season_start = held_start
    s.update()
    assert s.update_ran
    assert s.season_start == (feed_opener if adopted else held_start)


def test_update_keeps_constant_when_opener_older_or_absent(gates) -> None:
    s = _FakeLeague()
    s.update()
    assert s.update_ran
    assert s.season_start == _CONSTANT_START

    gates.setattr(odds_budget, "season_opener", lambda league: _CONSTANT_START)
    s2 = _FakeLeague()
    s2.update()
    assert s2.season_start == _CONSTANT_START


@pytest.mark.parametrize(
    ("days_since_start", "forced", "due"),
    [(SEASON_MAX_DAYS + 5, False, True), (SEASON_MAX_DAYS - 5, False, False), (30, True, True)],
    ids=["season over", "season live", "forced"],
)
def test_full_clean_is_due_once_the_season_held_is_over_or_when_forced(
    monkeypatch, days_since_start, forced, due
) -> None:
    monkeypatch.setattr(base, "clean_data", forced)
    s = _FakeLeague()
    s.season_start = date.today() - timedelta(days=days_since_start)
    assert s._full_clean_due() is due


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
