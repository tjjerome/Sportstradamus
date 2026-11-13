"""Per-run injury capture: every league's ESPN injury report."""

from __future__ import annotations

import json
import logging

import pandas as pd
import requests

from sportstradamus.collectors import injury_capture as ic
from sportstradamus.helpers.scraping import CAPTURE_TIMEOUT_S

# The run's Scored At, naive UTC, with the microseconds a real stamp carries.
_SCORED_AT = pd.Timestamp("2026-11-12 18:50:07.123456")
# Trimmed from the 2026-11-11 NFL and MLB feeds, which share one shape: an Active row that is a
# news item, a row with no details, one with every field and one with no side. As in the feeds,
# each athlete's own status reads Active.
_PAYLOAD = {
    "injuries": [
        {
            "displayName": "Chicago Bears",
            "injuries": [
                {
                    "status": "Active",
                    "date": "2026-11-11T02:33Z",
                    "shortComment": "Thomas hauled in one of his two targets for 13 yards.",
                    "athlete": {
                        "displayName": "Zavion Thomas",
                        "links": [
                            {"href": "https://www.espn.com/nfl/player/_/id/4869748/zavion-thomas"}
                        ],
                        "position": {"abbreviation": "WR"},
                        "team": {"abbreviation": "CHI"},
                        "status": {"name": "Active"},
                    },
                    "type": {"name": "INJURY_STATUS_ACTIVE"},
                },
                {
                    "status": "Out",
                    "date": "2026-11-09T00:04Z",
                    "shortComment": "Keenum (coach's decision) is inactive.",
                    "athlete": {
                        "displayName": "Case Keenum",
                        # The app link put first: it carries the id as a:<digits>.
                        "links": [
                            {
                                "href": "sportscenter://x-callback-url/showClubhouse"
                                "?uid=s:20~l:28~a:15168&section=stats"
                            },
                            {"href": "https://www.espn.com/nfl/player/_/id/15168/case-keenum"},
                        ],
                        "position": {"abbreviation": "QB"},
                        "team": {"abbreviation": "CHI"},
                        "status": {"name": "Active"},
                    },
                    "type": {"name": "INJURY_STATUS_OUT"},
                },
            ],
        },
        {
            "displayName": "Boston Red Sox",
            "injuries": [
                {
                    "status": "Out",
                    "date": "2026-11-10T17:11Z",
                    "shortComment": "Anthony (finger) will be re-evaluated in a couple weeks.",
                    "athlete": {
                        "displayName": "Roman Anthony",
                        "links": [{"href": "https://www.espn.com/mlb/player/_/id/5080767"}],
                        "position": {"abbreviation": "LF"},
                        "team": {"abbreviation": "BOS"},
                        "status": {"name": "Active"},
                    },
                    "type": {"name": "INJURY_STATUS_OUT"},
                    "details": {
                        "type": "Hand",
                        "side": "Right",
                        "returnDate": "2027-02-01",
                        "fantasyStatus": {"abbreviation": "OUT"},
                    },
                },
                {
                    "status": "Out",
                    "date": "2026-11-03T21:44Z",
                    "shortComment": "Gonzalez is still recovering from a hyperextended elbow.",
                    "athlete": {
                        "displayName": "Romy Gonzalez",
                        "links": [{"href": "https://www.espn.com/mlb/player/_/id/42826"}],
                        "position": {"abbreviation": "DH"},
                        "team": {"abbreviation": "BOS"},
                        "status": {"name": "Active"},
                    },
                    "type": {"name": "INJURY_STATUS_OUT"},
                    "details": {
                        "type": "Elbow",
                        "returnDate": "2027-02-01",
                        "fantasyStatus": {"abbreviation": "OUT"},
                    },
                },
            ],
        },
    ]
}


def test_rows_drop_the_news_items_by_the_rows_own_status():
    """Keenum's athlete status reads Active too; his row's Out keeps him."""
    rows = ic.injury_rows(_PAYLOAD, _SCORED_AT)

    assert rows.dtypes.astype(str).to_dict() == ic.INJURY_DTYPES
    assert rows[["team", "name", "position", "status", "type"]].values.tolist() == [
        ["CHI", "Case Keenum", "QB", "Out", "INJURY_STATUS_OUT"],
        ["BOS", "Roman Anthony", "LF", "Out", "INJURY_STATUS_OUT"],
        ["BOS", "Romy Gonzalez", "DH", "Out", "INJURY_STATUS_OUT"],
    ]
    assert (rows["polled_at"] == _SCORED_AT).all()
    assert rows["updated"].tolist() == [
        pd.Timestamp("2026-11-09 00:04"),
        pd.Timestamp("2026-11-10 17:11"),
        pd.Timestamp("2026-11-03 21:44"),
    ]


def test_athlete_id_comes_from_the_first_espn_player_link():
    """Not the app link's a:<digits>; and an MLB or NHL player card ends at the id."""
    rows = ic.injury_rows(_PAYLOAD, _SCORED_AT)

    assert rows["athlete_id"].tolist() == [15168, 5080767, 42826]


def test_a_field_the_feed_leaves_out_is_na():
    """Most NFL rows carry no details, and a side is missing more often still."""
    rows = ic.injury_rows(_PAYLOAD, _SCORED_AT).set_index("name")
    details = ["body_part", "side", "return_date", "fantasy_status"]

    assert rows.loc["Roman Anthony", details].tolist() == ["Hand", "Right", "2027-02-01", "OUT"]
    assert rows.loc["Romy Gonzalez", details].tolist() == ["Elbow", pd.NA, "2027-02-01", "OUT"]
    assert rows.loc["Case Keenum", details].isna().all()


def _espn(monkeypatch, tmp_path, **responses: tuple[int, dict]) -> list[tuple[str, float]]:
    """Serve the fixture to every feed, or a league's ``(status, payload)``; return the requests."""
    league_of = {
        ic.INJURIES_URL.format(feed=feed): league for league, feed in ic.INJURY_FEEDS.items()
    }
    requested = []

    def fake_get(url, **kwargs):
        requested.append((url, kwargs.get("timeout")))
        status, payload = responses.get(league_of[url], (200, _PAYLOAD))
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(ic, "INJURY_CAPTURE_DIR", tmp_path)
    monkeypatch.setattr(ic.requests, "get", fake_get)
    return requested


def _written(tmp_path) -> list[str]:
    return sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.parquet"))


def test_capture_writes_each_league_named_by_the_run_stamp(monkeypatch, tmp_path, caplog):
    """Every league's feed is read each run; one listing nobody writes nothing, quietly."""
    requested = _espn(monkeypatch, tmp_path, WNBA=(200, {"injuries": []}))

    with caplog.at_level(logging.WARNING, logger="log"):
        ic.capture_injuries(_SCORED_AT)

    feeds = ["football/nfl", "basketball/nba", "basketball/wnba", "hockey/nhl", "baseball/mlb"]
    assert requested == [
        (f"https://site.api.espn.com/apis/site/v2/sports/{feed}/injuries", CAPTURE_TIMEOUT_S)
        for feed in feeds
    ]
    assert _written(tmp_path) == [
        f"{league}/date=2026-11-12/185007.parquet" for league in ("MLB", "NBA", "NFL", "NHL")
    ]
    nfl = pd.read_parquet(tmp_path / "NFL/date=2026-11-12/185007.parquet")
    assert nfl["name"].tolist() == ["Case Keenum", "Roman Anthony", "Romy Gonzalez"]
    assert (nfl["polled_at"] == _SCORED_AT).all()
    assert not caplog.records


def test_a_league_whose_feed_fails_does_not_stop_the_others(monkeypatch, tmp_path, caplog):
    """A refused fetch is an ERROR in the log, never the run's exception."""
    _espn(monkeypatch, tmp_path, NBA=(500, {}))

    with caplog.at_level(logging.ERROR, logger="log"):
        ic.capture_injuries(_SCORED_AT)

    assert [path.split("/")[0] for path in _written(tmp_path)] == ["MLB", "NFL", "NHL", "WNBA"]
    assert [record.getMessage() for record in caplog.records] == ["NBA injury capture not written"]
