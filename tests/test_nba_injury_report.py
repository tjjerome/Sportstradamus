"""The official NBA injury report: the quarter-hour probe and the row parser."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from sportstradamus.collectors import nba_injury_report as inj

# The run's Scored At, naive UTC: the report's fetched_at.
_FETCHED_AT = pd.Timestamp("2026-10-28 21:50:07.123456")


def test_candidate_stamps_walk_back_by_quarter_hour_in_eastern_time():
    stamps = inj.candidate_stamps(datetime(2026, 10, 28, 21, 7, tzinfo=UTC))
    assert stamps[:3] == ["2026-10-28_05_00PM", "2026-10-28_04_45PM", "2026-10-28_04_30PM"]
    assert len(stamps) == inj.PROBE_SLOTS
    midnight = inj.candidate_stamps(datetime(2026, 10, 29, 4, 7, tzinfo=UTC))
    assert midnight[:2] == ["2026-10-29_12_00AM", "2026-10-28_11_45PM"]


def test_parse_lines_carries_game_and_team_forward():
    lines = [
        "Injury Report: 10/28/26 05:00 PM",
        "Game Date Game Time Matchup Team Player Name Current Status Reason",
        "10/28/2026 07:00 (ET) CHA@WAS Charlotte Hornets Carey Jr., Vernon Out Health and Safety Protocols",
        "Washington, P.J. Questionable Injury/Illness - Right Knee; Soreness",
        "Washington Wizards Bryant, Thomas Out Injury/Illness - Left Knee; Left ACL injury recovery",
        "07:30 (ET) MEM@BKN Brooklyn Nets Aldridge, LaMarcus Available Return to Competition Reconditioning",
        "SAS@TOR San Antonio Spurs NOT YET SUBMITTED",
        "Toronto Raptors Dragic, Goran Out Not With Team",
        "ORL@LAC Orlando Magic da Silva, Tristan Out Injury/Illness - Right Knee; Soreness",
        "Suggs, Jalen Available Injury/Illness - Left Knee; Soreness",
        "LA Clippers Konan Niederhauser, Yanic Out G League - Two-Way",
        "Harden, James Probable Injury/Illness - Right Ankle; Sprain",
    ]
    rows = inj.parse_lines(lines)
    assert [r["player"] for r in rows] == [
        "Carey Jr., Vernon",
        "Washington, P.J.",
        "Bryant, Thomas",
        "Aldridge, LaMarcus",
        "Dragic, Goran",
        "da Silva, Tristan",
        "Suggs, Jalen",
        "Konan Niederhauser, Yanic",
        "Harden, James",
    ]
    assert [r["status"] for r in rows] == [
        "Out",
        "Questionable",
        "Out",
        "Available",
        "Out",
        "Out",
        "Available",
        "Out",
        "Probable",
    ]
    assert [r["team"] for r in rows] == [
        "Charlotte Hornets",
        "Charlotte Hornets",
        "Washington Wizards",
        "Brooklyn Nets",
        "Toronto Raptors",
        "Orlando Magic",
        "Orlando Magic",
        "LA Clippers",
        "LA Clippers",
    ]
    assert [r["matchup"] for r in rows] == ["CHA@WAS"] * 3 + ["MEM@BKN", "SAS@TOR"] + [
        "ORL@LAC"
    ] * 4
    assert rows[3]["game_time"] == "07:30 (ET)" and rows[3]["game_date"] == "10/28/2026"
    assert rows[2]["reason"] == "Injury/Illness - Left Knee; Left ACL injury recovery"


def test_probe_loop_skips_403_stores_the_first_200_and_raises_on_anything_else(
    monkeypatch, tmp_path, caplog
):
    stamps = ["2026-10-28_05_00PM", "2026-10-28_04_45PM", "2026-10-28_04_30PM"]
    answers = {stamps[0]: 403, stamps[1]: 200}
    requested = []

    def fake_get(url, **_):
        stamp = url.split("Injury-Report_")[1].removesuffix(".pdf")
        requested.append(stamp)
        response = requests.Response()
        response.status_code = answers[stamp]
        response._content = b"%PDF"
        return response

    monkeypatch.setattr(inj, "NBA_INJURY_DIR", tmp_path)
    monkeypatch.setattr(inj.requests, "get", fake_get)
    monkeypatch.setattr(
        inj, "pdf_lines", lambda raw: ["Toronto Raptors Dragic, Goran Out Not With Team"]
    )
    monkeypatch.setattr(inj, "candidate_stamps", lambda now: stamps)
    inj.capture_injury_report(_FETCHED_AT)
    out = tmp_path / "reports" / "date=2026-10-28" / f"{stamps[1]}.parquet"
    assert (
        requested == stamps[:2] and (tmp_path / "pdf" / f"{stamps[1]}.pdf").read_bytes() == b"%PDF"
    )
    assert pq.read_schema(out).field("last_modified").type == pa.string()
    assert (pd.read_parquet(out)["fetched_at"] == _FETCHED_AT).all()
    monkeypatch.setattr(inj, "candidate_stamps", lambda now: stamps[1:])
    inj.capture_injury_report(_FETCHED_AT)
    assert requested == stamps[:2]
    answers[stamps[0]] = 500
    monkeypatch.setattr(inj, "candidate_stamps", lambda now: stamps)
    with pytest.raises(requests.HTTPError):
        inj.capture_injury_report(_FETCHED_AT)
    unposted = ["2026-10-29_05_00PM", "2026-10-29_04_45PM"]
    answers.update(dict.fromkeys(unposted, 403))
    monkeypatch.setattr(inj, "candidate_stamps", lambda now: unposted)
    monkeypatch.setattr(inj, "read_upcoming_events", list)
    with caplog.at_level(logging.INFO, logger="log"):
        inj.capture_injury_report(_FETCHED_AT)
    assert (
        caplog.messages[-1]
        == f"NBA injury report: nothing posted from {unposted[1]} to {unposted[0]}"
    )
    monkeypatch.setattr(inj, "read_upcoming_events", lambda: [{"league": "NBA"}])
    with pytest.raises(RuntimeError, match="with an NBA game on the ledger"):
        inj.capture_injury_report(_FETCHED_AT)
    assert requested[-4:] == unposted * 2
