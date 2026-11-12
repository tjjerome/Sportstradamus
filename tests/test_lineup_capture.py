"""Per-run lineup capture: each live league's starters for its next games, and the NBA's news."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import requests

from sportstradamus.collectors import lineup_capture as lc

# The run's Scored At, naive UTC, with the microseconds a real stamp carries.
_SCORED_AT = pd.Timestamp("2026-07-04 23:50:07.123456")
_ORDER = ["Paul Goldschmidt", "Aaron Judge", "Cody Bellinger", "Ben Rice", "Jazz Chisholm Jr."]


def _mlb(pitcher: str, home: bool, opponent: str, order: list[str]) -> dict:
    """An ``upcoming_games`` entry as ``StatsMLB.update`` builds it."""
    return {
        "Pitcher": pitcher,
        "Home": home,
        "Opponent": opponent,
        "Opponent Pitcher": "",
        "Batting Order": order,
        "Playoff": False,
        "game type": "R",
    }


# A posted order, one not posted yet, and a probable pitcher still to be announced.
_MLB_GAMES = {
    "NYY": _mlb("Gerrit Cole", True, "BOS", _ORDER),
    "BOS": _mlb("Brayan Bello", False, "NYY", []),
    "SEA": _mlb("", True, "HOU", []),
}
_NHL_GAMES = {
    "EDM": {"Opponent": "CGY", "Home": True, "Goalie": "Stuart Skinner"},
    "CGY": {"Opponent": "EDM", "Home": False, "Goalie": ""},
}
# 9:30 PM Eastern on the 28th, already the 29th in UTC.
_NBA_SCORED_AT = pd.Timestamp("2026-10-29 01:30:07.123456")
_NBA_PAYLOAD = {
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


def test_mlb_rows_hold_the_pitcher_and_the_order_posted_so_far():
    rows = lc.lineup_rows("MLB", _MLB_GAMES, _SCORED_AT)

    assert rows.dtypes.astype(str).to_dict() == lc.LINEUP_DTYPES
    assert (rows["polled_at"] == _SCORED_AT).all()
    nyy = rows[rows["team"] == "NYY"]
    assert nyy["player"].tolist() == ["Gerrit Cole", *_ORDER]
    assert nyy["role"].tolist() == ["pitcher"] + ["batter"] * len(_ORDER)
    assert nyy["slot"].tolist() == [pd.NA, *range(1, len(_ORDER) + 1)]
    # No order posted: the pitcher row alone. No pitcher named: the row stays, unnamed.
    others = rows[rows["team"] != "NYY"]
    assert others[["team", "opponent", "role"]].values.tolist() == [
        ["BOS", "NYY", "pitcher"],
        ["SEA", "HOU", "pitcher"],
    ]
    assert others["player"].tolist() == ["Brayan Bello", pd.NA]


def test_nhl_rows_keep_a_goalie_not_yet_named():
    rows = lc.lineup_rows("NHL", _NHL_GAMES, _SCORED_AT)

    assert rows[["team", "opponent", "home", "role"]].values.tolist() == [
        ["EDM", "CGY", True, "goalie"],
        ["CGY", "EDM", False, "goalie"],
    ]
    assert rows["player"].tolist() == ["Stuart Skinner", pd.NA]
    assert rows["slot"].isna().all()


def test_capture_writes_each_live_league_named_by_the_run_stamp(monkeypatch, tmp_path, caplog):
    """A league out of season (absent) or without a game ahead writes nothing, quietly."""
    monkeypatch.setattr(lc, "LINEUP_CAPTURE_DIR", tmp_path)

    with caplog.at_level(logging.WARNING, logger="log"):
        lc.capture_lineups({"MLB": SimpleNamespace(upcoming_games=_MLB_GAMES)}, _SCORED_AT)
        lc.capture_lineups({"NHL": SimpleNamespace(upcoming_games={})}, _SCORED_AT)

    (written,) = tmp_path.rglob("*.parquet")
    assert written.relative_to(tmp_path).as_posix() == "MLB/date=2026-07-04/235007.parquet"
    assert (pd.read_parquet(written)["polled_at"] == _SCORED_AT).all()
    assert not caplog.records


def test_a_league_whose_capture_raises_does_not_stop_the_next(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(lc, "LINEUP_CAPTURE_DIR", tmp_path)
    no_pitcher = {"NYY": {"Home": True, "Opponent": "BOS"}}
    stats = {
        "MLB": SimpleNamespace(upcoming_games=no_pitcher),
        "NHL": SimpleNamespace(upcoming_games=_NHL_GAMES),
    }

    with caplog.at_level(logging.ERROR, logger="log"):
        lc.capture_lineups(stats, _SCORED_AT)

    assert [path.relative_to(tmp_path).parts[0] for path in tmp_path.rglob("*.parquet")] == ["NHL"]
    assert "MLB lineup capture not written" in caplog.text


def test_nba_rows_flatten_both_teams():
    rows = lc.nba_lineup_rows(_NBA_PAYLOAD, _NBA_SCORED_AT)

    assert rows.dtypes.astype(str).to_dict() == lc.NBA_LINEUP_DTYPES
    assert list(rows["team"]) == ["ATL", "BOS"]
    assert list(rows["roster_status"]) == ["Active", "Inactive"]
    assert list(rows["lineup_status"]) == ["Confirmed", ""]
    assert (rows["polled_at"] == _NBA_SCORED_AT).all()


def test_due_only_within_two_hours_of_an_nba_tip():
    now = datetime(2026, 10, 28, 22, 0, tzinfo=UTC)
    nba = {"league": "NBA", "commence_time": "2026-10-28T23:30:00Z"}
    nhl = {"league": "NHL", "commence_time": "2026-10-28T23:00:00+00:00"}
    late = {"league": "NBA", "commence_time": "2026-10-29T02:30:00+00:00"}
    assert lc.due([nba], now) and not lc.due([nhl], now) and not lc.due([late], now)


def _nba_run(monkeypatch, tmp_path, tip: str, status: int = 200) -> list:
    """Stub the ledger and stats.nba.com; return the URLs requested."""
    requested = []

    def fake_get(url, **_):
        requested.append(url)
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(_NBA_PAYLOAD).encode()
        return response

    monkeypatch.setattr(lc, "LINEUP_CAPTURE_DIR", tmp_path)
    monkeypatch.setattr(
        lc, "read_upcoming_events", lambda: [{"league": "NBA", "commence_time": tip}]
    )
    monkeypatch.setattr(lc.requests, "get", fake_get)
    return requested


def test_nba_capture_reads_the_eastern_day_while_a_tip_is_due(monkeypatch, tmp_path):
    requested = _nba_run(monkeypatch, tmp_path, tip="2026-10-29T02:30:00Z")

    lc.capture_lineups({"NBA": None}, _NBA_SCORED_AT)

    assert requested == [lc.LINEUPS_URL.format(day="20261028")]
    (written,) = tmp_path.rglob("*.parquet")
    assert written.relative_to(tmp_path).as_posix() == "NBA/date=2026-10-29/013007.parquet"
    assert (pd.read_parquet(written)["polled_at"] == _NBA_SCORED_AT).all()


def test_nba_capture_without_a_tip_due_or_the_nba_live_makes_no_request(monkeypatch, tmp_path):
    """No lineup request outside the window, nor with a tip due but the NBA not live."""
    requested = _nba_run(monkeypatch, tmp_path, tip="2026-10-29T04:30:00Z")
    lc.capture_lineups({"NBA": None}, _NBA_SCORED_AT)
    assert not requested

    requested = _nba_run(monkeypatch, tmp_path, tip="2026-10-29T02:30:00Z")
    lc.capture_lineups({"MLB": SimpleNamespace(upcoming_games={})}, _NBA_SCORED_AT)
    assert not requested and not any(tmp_path.rglob("*.parquet"))


def test_nba_lineup_failure_is_logged_not_raised(monkeypatch, tmp_path, caplog):
    """A refused lineup fetch is an ERROR in the log, never the run's exception."""
    _nba_run(monkeypatch, tmp_path, tip="2026-10-29T02:30:00Z", status=500)

    with caplog.at_level(logging.ERROR, logger="log"):
        lc.capture_lineups({"NBA": None}, _NBA_SCORED_AT)

    assert not any(tmp_path.rglob("*.parquet"))
    assert "NBA lineup capture not written" in caplog.text
