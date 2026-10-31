"""Per-poll Underdog capture: every line with its state, the lineup status id, the match entry counts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from sportstradamus.collectors import underdog_capture as cap
from sportstradamus.collectors.underdog_lobby import feed_pills

_APPEARANCE = "7e076ff5"
_FEED = {
    "players": [{"id": "p1", "first_name": "Kyren", "last_name": "Williams", "sport_id": "NFL"}],
    "appearances": [
        {
            "id": _APPEARANCE,
            "player_id": "p1",
            "team_id": "t1",
            "match_id": 177184,
            "type": "Player",
            "lineup_status_id": "2d7ccaac",
        }
    ],
    "games": [{"id": 177184, "sport_id": "NFL", "scheduled_at": "2026-09-11T00:35:00Z"}],
    "over_under_lines": [
        {
            "id": "l1",
            "over_under_id": "ou1",
            "status": "active",
            "live_event": False,
            "stat_value": "54.5",
            "updated_at": "2026-09-10T23:37:40Z",
            "expires_at": None,
            "provider_id": "manual",
            "over_under": {
                "category": "player_prop",
                "has_alternates": True,
                "appearance_stat": {
                    "appearance_id": _APPEARANCE,
                    "pickem_stat_id": "s1",
                    "stat": "rushing_yds",
                    "display_stat": "Rush Yards",
                },
            },
            "options": [
                {"choice": "higher", "payout_multiplier": "1.0", "status": "active"},
                {"choice": "lower", "payout_multiplier": "1.0", "status": "active"},
            ],
        },
        {
            "id": "l2",
            "over_under_id": "ou2",
            "status": "suspended",
            "live_event": False,
            "stat_value": "0.5",
            "updated_at": "2026-09-10T23:40:00Z",
            "expires_at": None,
            "provider_id": "manual",
            "over_under": {
                "category": "player_prop",
                "has_alternates": False,
                "appearance_stat": {
                    "appearance_id": _APPEARANCE,
                    "pickem_stat_id": "s2",
                    "stat": "rushing_tds",
                    "display_stat": "Rush TDs",
                },
            },
            "options": [
                {"choice": "higher", "payout_multiplier": "1.6", "status": "suspended"},
                {"choice": "lower", "payout_multiplier": "0.0", "status": "suspended"},
            ],
        },
    ],
}
_LOBBY = {
    "over_under_lines": {},
    "appearances": {},
    "players": {},
    "games": {},
    "match_groups": [{"id": 177184, "entry_count": "75.2K"}],
}
_POLLED = datetime(2026, 9, 10, 23, 50, tzinfo=UTC)


def test_poll_rows_keep_suspended_lines_and_the_lineup_status():
    rows = cap.poll_rows(_FEED, "feed", _POLLED)
    assert list(rows.columns) == list(cap.CAPTURE_DTYPES)
    assert list(rows["status"]) == ["active", "suspended"]
    assert list(rows["lineup_status_id"]) == ["2d7ccaac", "2d7ccaac"]
    assert list(rows["player"]) == ["Kyren Williams"] * 2
    assert list(rows["stat_value"]) == [54.5, 0.5]
    assert list(rows["over_status"]) == ["active", "suspended"]
    assert list(rows["match_id"]) == [177184, 177184]
    assert rows["polled_at"].dt.tz is None


def test_write_poll_capture_lands_lines_and_matches(tmp_path):
    cap.write_poll_capture(_FEED, {"NFL": _LOBBY}, tmp_path, _POLLED)
    lines = pd.read_parquet(tmp_path / "lines" / "date=2026-09-10" / "235000.parquet")
    matches = pd.read_parquet(tmp_path / "matches" / "date=2026-09-10" / "235000.parquet")
    assert len(lines) == 2 and list(lines["source"]) == ["feed", "feed"]
    assert matches.to_dict("records")[0]["entry_count"] == "75.2K"


def test_polls_share_one_schema_across_a_date(tmp_path):
    """A full poll, one whose lines lack their appearance, and an empty one read back as one table."""
    cap.write_poll_capture(_FEED, {"NFL": _LOBBY}, tmp_path, _POLLED)
    cap.write_poll_capture(
        {**_FEED, "appearances": []}, {}, tmp_path, _POLLED + timedelta(minutes=5)
    )
    cap.write_poll_capture({"over_under_lines": []}, {}, tmp_path, _POLLED + timedelta(minutes=8))
    lines = pd.read_parquet(tmp_path / "lines" / "date=2026-09-10")
    matches = pd.read_parquet(tmp_path / "matches" / "date=2026-09-10")
    assert len(lines) == 4 and lines.dtypes.astype(str).to_dict() == cap.CAPTURE_DTYPES
    assert lines["lineup_status_id"].tolist() == ["2d7ccaac", "2d7ccaac", pd.NA, pd.NA]
    assert len(matches) == 1 and matches.dtypes.astype(str).to_dict() == cap.MATCH_DTYPES


def test_feed_pills_are_the_modeled_leagues_stats():
    assert feed_pills(_FEED) == {("NFL", "s1"), ("NFL", "s2")}
