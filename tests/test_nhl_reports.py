"""The NHL stats service's per-game reports fill the gamelog's report columns.

A published game's rows take the reports' values, page by page, with ice time in minutes; its
goalies, whom the skater reports leave out, and a share of a power play that never was read 0.
An unpublished game's rows keep NaN for a later run, and a rerun asks only for the games still
lacking.
"""

import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import StatsNHL, nhl, nhl_reports
from sportstradamus.stats.nhl_reports import REPORT_COLUMNS, fill_report_columns

PUBLISHED, UNPUBLISHED, LATER = 2025020001, 2025020002, 2025020100
CENTER, WINGER, GOALIE, OTHER = 8470001, 8470002, 8470003, 8470004
REPORTS = {
    "faceoffwins": [
        {"gameId": PUBLISHED, "playerId": CENTER, "totalFaceoffs": 18},
        {"gameId": PUBLISHED, "playerId": WINGER, "totalFaceoffs": 2},
        {"gameId": LATER, "playerId": OTHER, "totalFaceoffs": 9},
    ],
    "timeonice": [
        {"gameId": PUBLISHED, "playerId": CENTER, "ppTimeOnIce": 150, "shTimeOnIce": 90},
        {"gameId": PUBLISHED, "playerId": WINGER, "ppTimeOnIce": 0, "shTimeOnIce": 0},
        {"gameId": LATER, "playerId": OTHER, "ppTimeOnIce": 0, "shTimeOnIce": 30},
    ],
    "powerplay": [
        {"gameId": PUBLISHED, "playerId": CENTER, "ppShots": 2, "ppTimeOnIcePctPerGame": 0.6},
        {"gameId": PUBLISHED, "playerId": WINGER, "ppShots": 0, "ppTimeOnIcePctPerGame": 0.0},
        {"gameId": LATER, "playerId": OTHER, "ppShots": 0, "ppTimeOnIcePctPerGame": None},
    ],
}


class _StatsService:
    """Serves each report's rows for the games a request names, ``page`` rows at a time."""

    def __init__(self, page=1, reports=REPORTS):
        self.page = page
        self.reports = reports
        self.requests = []

    def get(self, url, max_attempts, params):
        report = url.rsplit("/", 1)[1]
        games = re.search(r"gameId in \(([^)]*)\)", params["cayenneExp"]).group(1)
        self.requests.append((report, games, params["start"]))
        rows = [row for row in self.reports[report] if str(row["gameId"]) in games.split(",")]
        start = params["start"]
        return {"data": rows[start : start + self.page], "total": len(rows)}


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(nhl_reports, "sleep", lambda seconds: None)
    service = _StatsService()
    monkeypatch.setattr(nhl_reports, "scraper", service)
    return service


def _gamelog():
    return pd.DataFrame(
        {
            "gameId": [PUBLISHED] * 3 + [UNPUBLISHED, LATER],
            "gameDate": ["2025-10-07"] * 3 + ["2025-10-08", "2025-10-30"],
            "playerId": [str(CENTER), str(WINGER), str(GOALIE), str(OTHER), str(OTHER)],
            "position": ["C", "W", "G", "C", "C"],
        }
    )


def test_a_published_game_takes_its_reports_and_an_unpublished_one_waits(service):
    filled = fill_report_columns(_gamelog())

    # faceOffsTaken, powerPlayTimeOnIce, shortHandedTimeOnIce, powerPlayShots, powerPlayShare
    np.testing.assert_array_equal(
        filled[REPORT_COLUMNS].to_numpy(),
        [[18, 2.5, 1.5, 2, 0.6], [2, 0, 0, 0, 0], [0] * 5, [np.nan] * 5, [9, 0, 0.5, 0, 0]],
    )
    # Two windows, the later game's alone; a page holds one row, so a published game's two
    # skaters take two pages of each report.
    first_window = f"{PUBLISHED},{UNPUBLISHED}"
    assert service.requests == [
        (report, games, start)
        for games, starts in [(first_window, (0, 1)), (str(LATER), (0,))]
        for report in ("faceoffwins", "timeonice", "powerplay")
        for start in starts
    ]


def test_a_rerun_asks_only_for_the_games_still_lacking(service):
    filled = fill_report_columns(_gamelog())
    service.requests.clear()

    fill_report_columns(filled)
    asked = {games for _, games, _ in service.requests}
    fill_report_columns(filled, since=date(2025, 10, 9))

    assert asked == {str(UNPUBLISHED)}
    assert len(service.requests) == 3


def test_a_game_one_report_lists_before_another_waits(service):
    service.reports = REPORTS | {
        "timeonice": [row for row in REPORTS["timeonice"] if row["gameId"] != LATER]
    }

    filled = fill_report_columns(_gamelog())

    assert filled[REPORT_COLUMNS].isna().any(axis=1).tolist() == [False] * 3 + [True, True]


def test_the_rows_wait_when_the_service_stops_answering(service, monkeypatch):
    monkeypatch.setattr(service, "get", lambda url, max_attempts, params: {})

    filled = fill_report_columns(_gamelog())

    assert filled[REPORT_COLUMNS].isna().all().all()


def test_the_update_fills_the_report_columns_two_weeks_back(monkeypatch):
    class _SilentSites:
        def get(self, url):
            return {}

        def get_csv(self, url):
            return pd.DataFrame()

    bounds = []
    monkeypatch.setattr(nhl, "scraper", _SilentSites())
    monkeypatch.setattr(nhl, "write_gamelog", lambda *bundle: None)
    monkeypatch.setattr(StatsNHL, "dump_goalie_list", lambda self: None)
    monkeypatch.setattr(StatsNHL, "backfill_season_sums", lambda self, seasons=None: 0)
    monkeypatch.setattr(
        nhl,
        "fill_report_columns",
        lambda gamelog, since, max_attempts: bounds.append((since, max_attempts)) or gamelog,
    )
    stats = StatsNHL()
    stats.season_start = date.today() + timedelta(days=5)
    stats.gamelog = pd.DataFrame(columns=["gameId", "gameDate", "playerId", "playerName", "team"])
    stats.teamlog = pd.DataFrame(columns=["gameId", "gameDate", "team"])

    stats._update()

    # One try per request: the update runs inside prophecize.
    assert bounds == [(date.today() - timedelta(days=14), 1)]
