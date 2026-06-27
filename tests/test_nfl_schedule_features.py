"""Derivation + parity pin for the QW-4 NFL ``Weekday`` schedule feature (model track §6.3).

``StatsNFL._schedule_context`` emits ``Weekday`` (the game date's day-of-week) from the
realized gamelog row (historical) or ``upcoming_games`` (upcoming). Both branches derive it
from the game date, so they must agree. This pins that derivation and the historical-vs-
upcoming parity the main harness can't reach (its synthetic logs are NBA, no schedule fields).

PrimeTime / RestDiff were reverted 2026-06-20 (dead features -- zero importance, and the
gametime/own rest/opp rest inputs were NaN on all historical rows; see the §6.3 ledger).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from sportstradamus.stats import StatsNFL
from sportstradamus.stats import nfl as nfl_module

GAMEDAY = date(2025, 1, 5)  # a Sunday
TEAMS = {"P1": "AAA", "P2": "BBB"}


class _FrozenDatetime(datetime):
    @classmethod
    def today(cls):
        return datetime(GAMEDAY.year, GAMEDAY.month, GAMEDAY.day)


def _nfl_with_schedule() -> StatsNFL:
    s = StatsNFL()
    s.gamelog = pd.DataFrame(
        [{"player display name": p, "gameday": GAMEDAY.isoformat()} for p in TEAMS]
    )
    s.upcoming_games = {
        t: {"Opponent": "BBB" if t == "AAA" else "AAA", "Home": True, "gameday": GAMEDAY.isoformat()}
        for t in set(TEAMS.values())
    }
    return s


def _context(s: StatsNFL) -> pd.DataFrame:
    stats = pd.DataFrame(index=list(TEAMS))
    s._schedule_context(stats, GAMEDAY, dict(TEAMS))
    return stats


def test_schedule_context_weekday_derivation():
    out = _context(_nfl_with_schedule())
    assert (out["Weekday"] == GAMEDAY.weekday()).all()  # 2025-01-05 is a Sunday -> 6


def test_schedule_context_historical_and_upcoming_agree(monkeypatch):
    s = _nfl_with_schedule()
    historical = _context(s)  # real today >> GAMEDAY -> historical branch

    monkeypatch.setattr(nfl_module, "datetime", _FrozenDatetime)  # today == GAMEDAY
    upcoming = _context(s)  # -> upcoming branch (reads upcoming_games)

    assert historical["Weekday"].tolist() == upcoming["Weekday"].tolist()
