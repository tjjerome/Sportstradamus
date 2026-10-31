"""NBA daily lineups: the rows and the due window."""

from __future__ import annotations

from datetime import UTC, datetime

from sportstradamus.collectors import nba_lineups as lu

_PAYLOAD = {
    "games": [
        {
            "gameId": "0022600132",
            "gameStatus": 1,
            "gameStatusText": "7:00 pm ET",
            "homeTeam": {
                "teamId": 1,
                "teamAbbreviation": "ATL",
                "players": [
                    {
                        "personId": 1629638,
                        "playerName": "Nickeil Alexander-Walker",
                        "lineupStatus": "Confirmed",
                        "position": "SG",
                        "rosterStatus": "Active",
                        "timestamp": "2026-10-28T23:22:00",
                    }
                ],
            },
            "awayTeam": {
                "teamId": 2,
                "teamAbbreviation": "BOS",
                "players": [
                    {
                        "personId": 2,
                        "playerName": "Jayson Tatum",
                        "lineupStatus": "",
                        "position": "",
                        "rosterStatus": "Inactive",
                        "timestamp": None,
                    }
                ],
            },
        }
    ]
}


def test_rows_flatten_both_teams():
    rows = lu.lineup_rows(_PAYLOAD, datetime(2026, 10, 28, 23, 30, tzinfo=UTC))
    assert list(rows["team"]) == ["ATL", "BOS"]
    assert list(rows["roster_status"]) == ["Active", "Inactive"]
    assert list(rows["lineup_status"]) == ["Confirmed", ""]
    assert rows["polled_at"].dt.tz is None
    assert rows["source_timestamp"].dtype == "string" and rows["person_id"].dtype == "Int64"


def test_due_only_within_two_hours_of_an_nba_tip():
    now = datetime(2026, 10, 28, 22, 0, tzinfo=UTC)
    nba = {"league": "NBA", "commence_time": "2026-10-28T23:30:00Z"}
    nhl = {"league": "NHL", "commence_time": "2026-10-28T23:00:00+00:00"}
    late = {"league": "NBA", "commence_time": "2026-10-29T02:30:00+00:00"}
    assert lu.due([nba], now) and not lu.due([nhl], now) and not lu.due([late], now)
