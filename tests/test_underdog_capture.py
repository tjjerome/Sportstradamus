"""Per-poll Underdog capture: every line with its state, the lineup status id, the entry counts."""

from __future__ import annotations

import json
import logging
from datetime import timedelta

import pandas as pd

from sportstradamus.collectors import underdog_capture as cap

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
# The run's Scored At, naive UTC, with the microseconds a real stamp carries.
_POLLED = pd.Timestamp("2026-09-10 23:50:07.123456")


def test_poll_rows_keep_suspended_lines_and_the_lineup_status():
    rows = cap.poll_rows(_FEED, "feed", _POLLED)
    assert list(rows.columns) == list(cap.CAPTURE_DTYPES)
    assert list(rows["status"]) == ["active", "suspended"]
    assert list(rows["lineup_status_id"]) == ["2d7ccaac", "2d7ccaac"]
    assert list(rows["player"]) == ["Kyren Williams"] * 2
    assert list(rows["stat_value"]) == [54.5, 0.5]
    assert list(rows["over_status"]) == ["active", "suspended"]
    assert list(rows["match_id"]) == [177184, 177184]
    assert (rows["polled_at"] == _POLLED).all()


def test_write_poll_capture_lands_lines_and_matches(tmp_path):
    cap.write_poll_capture(_FEED, {"NFL": _LOBBY}, tmp_path, _POLLED)
    lines = pd.read_parquet(tmp_path / "lines" / "date=2026-09-10" / "235007.parquet")
    matches = pd.read_parquet(tmp_path / "matches" / "date=2026-09-10" / "235007.parquet")
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


_PILLS = {
    ("NFL", "s1"): {
        "lines": [
            {
                "over_under_line_id": "l1",
                "appearance_id": _APPEARANCE,
                "entry_count": "12.3K",
                "header": "Rush Yards",
            }
        ]
    },
    ("NFL", "s2"): {},
}
_STATUSES = {"lineup_statuses": [{"id": "2d7ccaac", "abbr": "Q", "display": "QUESTIONABLE"}]}


def test_write_entry_capture_lands_the_answered_pills_under_the_run_stamp(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="log"):
        cap.write_entry_capture(_PILLS, _STATUSES, tmp_path, _POLLED)

    (written,) = (tmp_path / "entry_counts").rglob("*.parquet")
    assert written.relative_to(tmp_path).as_posix() == "entry_counts/date=2026-09-10/235007.parquet"
    entries = pd.read_parquet(written)
    assert entries.dtypes.astype(str).to_dict() == cap.ENTRY_DTYPES
    assert entries[["pickem_stat_id", "over_under_line_id", "entry_count"]].values.tolist() == [
        ["s1", "l1", "12.3K"]
    ]
    assert (entries["polled_at"] == _POLLED).all()
    assert json.loads((tmp_path / "lineup_statuses.json").read_text()) == _STATUSES
    assert "1 Underdog pills answered empty: ['NFL/s2']" in caplog.text


def test_a_capture_whose_reads_all_answered_empty(tmp_path):
    """Its entry-count file still lands, empty under the pinned schema; the last decode stays."""
    (tmp_path / "lineup_statuses.json").write_text(json.dumps(_STATUSES))
    cap.write_entry_capture({("NFL", "s2"): {}}, {}, tmp_path, _POLLED)

    entries = pd.read_parquet(tmp_path / "entry_counts" / "date=2026-09-10")
    assert entries.empty and entries.dtypes.astype(str).to_dict() == cap.ENTRY_DTYPES
    assert json.loads((tmp_path / "lineup_statuses.json").read_text()) == _STATUSES
