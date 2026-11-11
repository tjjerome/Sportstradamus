"""Per-run lineup capture: the starters each live league's update names for its next games."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

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
