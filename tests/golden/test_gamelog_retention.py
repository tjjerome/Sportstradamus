"""The NHL keeps four years of gamelog, as the NBA and MLB do.

Its own constant read 1,431 days where theirs read 1,461: one digit off, so its gamelog
was trimmed a month short of theirs. The three leagues now read one constant in
``stats.base``.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from sportstradamus.stats import StatsNHL, nhl
from sportstradamus.stats.base import GAMELOG_RETENTION_DAYS


class _SilentSites:
    """A scraper every site answers with nothing: no new game and no roster."""

    def get(self, url):
        return {}

    def get_csv(self, url):
        return pd.DataFrame()


def test_nhl_update_keeps_four_years_of_gamelog(monkeypatch):
    monkeypatch.setattr(nhl, "scraper", _SilentSites())
    monkeypatch.setattr(nhl, "write_gamelog", lambda *bundle: None)
    stats = StatsNHL()
    # Before opening night, so the empty roster pull is not an error.
    stats.season_start = date.today() + timedelta(days=5)
    cut, kept, recent = (
        str(date.today() - timedelta(days=days))
        for days in (GAMELOG_RETENTION_DAYS + 10, GAMELOG_RETENTION_DAYS - 10, 1)
    )
    stats.gamelog = pd.DataFrame(
        {
            "gameId": [1, 2, 3],
            "gameDate": [cut, kept, recent],
            "playerName": ["A", "B", "C"],
            "team": ["TOR", "TOR", "TOR"],
        }
    )
    stats.teamlog = pd.DataFrame(columns=["gameId", "gameDate", "team"])

    stats._update()

    assert GAMELOG_RETENTION_DAYS == 1461
    assert list(stats.gamelog["gameDate"]) == [kept, recent]
