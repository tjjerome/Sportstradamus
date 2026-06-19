"""Derivation + parity pins for the QW-4 NFL schedule context (model track §6.3).

``StatsNFL._schedule_context`` emits ``Weekday`` / ``PrimeTime`` / ``RestDiff`` from
the realized gamelog row (historical) or ``upcoming_games`` (upcoming). ``Weekday``
is derived from the game date's day-of-week in both branches (not the ragged
``weekday`` string ``_compute_upcoming_games`` builds), so the two must agree. This
pins the historical arithmetic and the historical-vs-upcoming parity that the main
parity harness can't reach (its synthetic logs are NBA and carry no schedule fields).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from sportstradamus.stats import StatsNFL
from sportstradamus.stats import nfl as nfl_module

GAMEDAY = date(2025, 1, 5)  # a Sunday
TEAMS = {"P1": "AAA", "P2": "BBB"}
# AAA: 20:20 kickoff (prime time), +3 rest edge. BBB: 13:00, -3 rest edge.
_GAMETIME = {"AAA": "20:20", "BBB": "13:00"}
_OWN_REST = {"AAA": 7, "BBB": 4}
_OPP_REST = {"AAA": 4, "BBB": 7}


class _FrozenDatetime(datetime):
    @classmethod
    def today(cls):
        return datetime(GAMEDAY.year, GAMEDAY.month, GAMEDAY.day)


def _nfl_with_schedule() -> StatsNFL:
    s = StatsNFL()
    s.gamelog = pd.DataFrame(
        [
            {
                "player display name": p,
                "gameday": GAMEDAY.isoformat(),
                "gametime": _GAMETIME[t],
                "own rest": _OWN_REST[t],
                "opp rest": _OPP_REST[t],
            }
            for p, t in TEAMS.items()
        ]
    )
    s.upcoming_games = {
        t: {
            "Opponent": "BBB" if t == "AAA" else "AAA",
            "Home": True,
            "gameday": GAMEDAY.isoformat(),
            # upcoming gametime carries the weekday prefix _compute_upcoming_games builds.
            "gametime": f"Sun {_GAMETIME[t]}",
            "rest_diff": _OWN_REST[t] - _OPP_REST[t],
        }
        for t in set(TEAMS.values())
    }
    return s


def _context(s: StatsNFL) -> pd.DataFrame:
    stats = pd.DataFrame(index=list(TEAMS))
    s._schedule_context(stats, GAMEDAY, dict(TEAMS))
    return stats


def test_schedule_context_historical_derivations():
    out = _context(_nfl_with_schedule())
    assert (out["Weekday"] == GAMEDAY.weekday()).all()  # Sunday -> 6
    assert out.loc["P1", "PrimeTime"] == 1  # 20:20 kickoff
    assert out.loc["P2", "PrimeTime"] == 0  # 13:00 kickoff
    assert out.loc["P1", "RestDiff"] == 3  # 7 - 4
    assert out.loc["P2", "RestDiff"] == -3  # 4 - 7


def test_schedule_context_historical_and_upcoming_agree(monkeypatch):
    s = _nfl_with_schedule()
    historical = _context(s)  # real today >> GAMEDAY -> historical branch

    monkeypatch.setattr(nfl_module, "datetime", _FrozenDatetime)  # today == GAMEDAY
    upcoming = _context(s)  # -> upcoming branch (reads upcoming_games)

    for col in ("Weekday", "PrimeTime", "RestDiff"):
        assert historical[col].tolist() == upcoming[col].tolist(), (
            f"_schedule_context branch divergence on {col}"
        )
