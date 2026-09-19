"""Pins for the team-grain FantasyPoints expansion recipes.

``pace`` / ``pace_opp`` carry the new API's ``__raw`` counts and must pool as
``sum(num) / sum(den)`` over the window, not as a mean of per-game rates.
``ol_combos`` carries one row per OL five-man unit per team-week with only
displayed rates, so its recipes snap-weight across units, and the continuity
score is the share of a team's snaps its single most-used unit played.
``pace_opp`` is a defense-grain kind: its columns land in the defense frame,
never the team frame.
"""

import pandas as pd
import pytest

from sportstradamus.stats import (
    nfl_fp_team_weekly,
    nfl_fp_team_weekly_aggregate,
    nfl_fp_weekly,
)
from sportstradamus.stats.nfl_fp_aggregation import TEAM_GROUP_COL, top_unit_share

SEASON = 2026
WINDOWS = [(SEASON, 1, 2)]


def _team_rows(team: str, *rows: dict) -> pd.DataFrame:
    """Current-era rows: the collector writes the abbreviation into both team id columns."""
    return pd.DataFrame([{TEAM_GROUP_COL: team, "teamAbbreviation": team, **row} for row in rows])


def _serve(monkeypatch, frames: dict[str, pd.DataFrame]) -> None:
    """Serve ``frames`` (file_kind -> pooled window rows) as the only snapshots on disk."""
    # The abbreviation map is read off coverage_matrix, and a season without one
    # contributes no rows at all; current-era frames are self-describing, so hand
    # back the team columns of whatever is being served.
    abbr_source = pd.concat(frames.values())[[TEAM_GROUP_COL, "teamAbbreviation"]]

    monkeypatch.setattr(
        nfl_fp_team_weekly,
        "load_window_or_empty",
        lambda _season, _start, _end, file_kind: frames.get(file_kind, pd.DataFrame()).copy(),
    )
    monkeypatch.setattr(
        nfl_fp_team_weekly,
        "load_snapshot",
        lambda _season, _week, file_kind: (
            abbr_source.copy() if file_kind == "coverage_matrix" else None
        ),
    )
    monkeypatch.setattr(nfl_fp_team_weekly, "available_snapshots", lambda _season: [1])
    monkeypatch.setattr(nfl_fp_weekly, "available_snapshots", lambda _season: [])
    monkeypatch.setattr(nfl_fp_team_weekly_aggregate, "_ABBR_MAP_CACHE", {})


def test_top_unit_share_pools_a_units_snaps_across_weeks():
    units = pd.DataFrame(
        [
            {TEAM_GROUP_COL: "KC", "row_id": "KC|x", "snaps": 60},
            {TEAM_GROUP_COL: "KC", "row_id": "KC|y", "snaps": 40},
            {TEAM_GROUP_COL: "KC", "row_id": "KC|x", "snaps": 60},
            {TEAM_GROUP_COL: "PHI", "row_id": "PHI|z", "snaps": 70},
        ]
    )
    share = top_unit_share(units, "row_id", "snaps")
    assert share["KC"] == pytest.approx(120 / 160)
    assert share["PHI"] == pytest.approx(1.0)
    assert top_unit_share(units.drop(columns="snaps"), "row_id", "snaps").empty


def test_pace_pools_seconds_and_plays_before_dividing(monkeypatch):
    # Two games at 20 and 40 s/play: the pooled rate is 3000 / 120 = 25; a mean
    # of the per-game rates would read 30.
    _serve(
        monkeypatch,
        {
            "pace": _team_rows(
                "KC",
                {"top_seconds_sum": 1800, "plays_total": 90, "games": 1},
                {"top_seconds_sum": 1200, "plays_total": 30, "games": 1},
            )
        },
    )
    team, _defense = nfl_fp_team_weekly_aggregate.load_team_and_defense_features(WINDOWS)
    assert team.loc["KC", "pace_sec_per_play"] == pytest.approx(25.0)
    assert team.loc["KC", "pace_plays_per_game"] == pytest.approx(60.0)


def test_pace_opp_lands_in_the_defense_frame_only(monkeypatch):
    _serve(
        monkeypatch,
        {
            "pace": _team_rows("KC", {"top_seconds_sum": 1800, "plays_total": 60}),
            "pace_opp": _team_rows("KC", {"top_seconds_sum": 1200, "plays_total": 60}),
        },
    )
    team, defense = nfl_fp_team_weekly_aggregate.load_team_and_defense_features(WINDOWS)
    assert team.loc["KC", "pace_sec_per_play"] == pytest.approx(30.0)
    assert "def_pace_sec_per_play_forced" not in team.columns
    assert defense.loc["KC", "def_pace_sec_per_play_forced"] == pytest.approx(20.0)
    assert "pace_sec_per_play" not in defense.columns


def test_ol_pressure_rate_is_snap_weighted_across_units(monkeypatch):
    _serve(
        monkeypatch,
        {
            "ol_combos": _team_rows(
                "KC",
                {"row_id": "KC|x", "snaps": 60, "pressure_rate": 40.0},
                {"row_id": "KC|y", "snaps": 40, "pressure_rate": 20.0},
            )
        },
    )
    team, _defense = nfl_fp_team_weekly_aggregate.load_team_and_defense_features(WINDOWS)
    # (60 * 40 + 40 * 20) / 100, not the unit mean of 30.
    assert team.loc["KC", "ol_pressure_rate"] == pytest.approx(32.0)
    assert team.loc["KC", "ol_top_unit_share"] == pytest.approx(0.6)
