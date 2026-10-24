"""Pin that FP ``line_matchups`` never re-enters the NFL feature path.

The kind is filed under the week whose games it describes, and its
``teamStats*`` columns are that week's *realized* totals -- 2025 week 1
records PIT 20 / NYJ 39 / GB 25 rushing attempts, exactly what
``player_data/NFL/2025/week_01/rushing_basic.parquet`` sums to per team.
Reading the target week's row as a feature is same-game target leakage.
It shipped in six NFL models (``carries``, ``rushing yards``,
``receiving yards``, ``passing yards``, ``tds``, ``interceptions``) as
the ``lm_*`` columns and was removed in 2026-09 after a season rollover
made the never-available snapshot crash serving.

The kind is still loaded -- it is the only team-grain parquet carrying
``teamAbbreviation``, so it backs the ``teamTeamId`` -> abbreviation map.
That is week-agnostic and leakage-free. Both halves of the contract are
pinned here: the feature path must not read it, the abbreviation map must.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from sportstradamus.stats import nfl_fp_team_weekly, nfl_fp_team_weekly_aggregate

SEASON = 2025
TEAM_IDS = {27: "PIT", 25: "NYJ"}

# The realized week-1 team rushing attempts that proved the leak. A sentinel
# only in the sense that no aggregation could invent it.
REALIZED_RUSH_ATTEMPTS = {27: 20, 25: 39}


@pytest.fixture
def line_matchups_only(monkeypatch):
    """Make ``line_matchups`` the only snapshot on disk for ``SEASON``."""
    frame = pd.DataFrame(
        [
            {
                "teamTeamId": team_id,
                "teamAbbreviation": abbr,
                "teamStatsRushingAttemptsTotal": REALIZED_RUSH_ATTEMPTS[team_id],
                "teamStatsGamesPlayed": 1,
            }
            for team_id, abbr in TEAM_IDS.items()
        ]
    )

    def load_snapshot(season, week, file_kind):
        if season == SEASON and file_kind == "line_matchups":
            return frame.copy()
        return None

    monkeypatch.setattr(nfl_fp_team_weekly, "load_snapshot", load_snapshot)
    monkeypatch.setattr(nfl_fp_team_weekly, "available_snapshots", lambda season: [1])
    monkeypatch.setattr(
        nfl_fp_team_weekly, "load_window_or_empty", lambda *_args, **_kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(nfl_fp_team_weekly_aggregate, "_ABBR_MAP_CACHE", {})
    return frame


def test_feature_frames_carry_no_line_matchup_columns(line_matchups_only):
    team, defense = nfl_fp_team_weekly_aggregate.load_team_and_defense_features([(SEASON, 1, 2)])

    for frame in (team, defense):
        assert [c for c in frame.columns if c.startswith("lm_")] == []
        assert "teamStatsRushingAttemptsTotal" not in frame.columns


def test_abbreviation_map_still_reads_line_matchups(line_matchups_only):
    mapping = nfl_fp_team_weekly_aggregate._build_team_abbreviation_map(SEASON)

    assert mapping.to_dict() == TEAM_IDS


def test_only_the_abbreviation_map_loads_line_matchups():
    """Static backstop: a second ``load_snapshot`` call for the kind fails here.

    Matches the quoted literal, which is how the kind is named in code; the
    docstrings refer to it in prose and are not the thing being pinned.
    """
    module = inspect.getsource(nfl_fp_team_weekly_aggregate)
    abbr_reader = inspect.getsource(nfl_fp_team_weekly_aggregate._abbr_from_team_grain)

    assert module.count('"line_matchups"') == 1
    assert '"line_matchups"' in abbr_reader
