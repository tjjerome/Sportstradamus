"""Pin that the FP team aggregate pools across the two snapshot id spaces.

The archived legacy pulls key teams by a numeric ``teamTeamId`` and carry no
``teamAbbreviation`` on the stat parquets; the 2026 API pulls key them by the
abbreviation itself. A pre-week-5 lookback blends one window from each era, so
the aggregate has to reduce both onto the one key they agree on -- the
abbreviation -- before the per-team groupby. Re-keying the pooled frame
afterwards instead dropped whichever era the single map didn't cover, and when
that map came from a season whose snapshots were empty it returned no features
at all: 2026-09-15's week 1 -> week 2 rollover took NFL out of prophecize for
four days that way.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.stats import (
    nfl_fp_team_weekly,
    nfl_fp_team_weekly_aggregate,
    nfl_fp_weekly,
)

LEGACY_SEASON, CURRENT_SEASON = 2025, 2026
WINDOWS = [(LEGACY_SEASON, 11, 18), (CURRENT_SEASON, 1, 1)]
TEAM_IDS = {27: "PIT", 25: "NYJ"}

# Man-coverage dropbacks / total dropbacks, chosen so each era alone reads as an
# extreme and the pooled sum lands exactly halfway: one era surviving is visible
# in the value, not just the row count.
LEGACY_MAN, CURRENT_MAN = 10, 90
DROPBACKS = 100
POOLED_MAN_PCT = (LEGACY_MAN + CURRENT_MAN) / (2 * DROPBACKS)


def _coverage_rows(team_key_by_abbr: dict, man_dropbacks: int, *, with_abbr: bool) -> pd.DataFrame:
    rows = []
    for abbr, team_key in team_key_by_abbr.items():
        row = {
            "teamTeamId": team_key,
            "teamStatsCoverageSchemeManPassingDropbacksTotal": man_dropbacks,
            "teamStatsPassingDropbacksTotal": DROPBACKS,
        }
        if with_abbr:
            row["teamAbbreviation"] = abbr
        rows.append(row)
    return pd.DataFrame(rows)


def _legacy_ids() -> dict:
    return {abbr: team_id for team_id, abbr in TEAM_IDS.items()}


def _current_ids() -> dict:
    return {abbr: abbr for abbr in TEAM_IDS.values()}


@pytest.fixture
def two_eras(monkeypatch):
    """Serve a legacy (numeric-keyed) and a current (abbreviation-keyed) season."""
    windows = {
        LEGACY_SEASON: _coverage_rows(_legacy_ids(), LEGACY_MAN, with_abbr=False),
        CURRENT_SEASON: _coverage_rows(_current_ids(), CURRENT_MAN, with_abbr=True),
    }
    # The legacy stat parquets carry no abbreviation, so that season's map comes
    # from line_matchups; the current ones are self-describing.
    line_matchups = pd.DataFrame(
        [{"teamTeamId": team_id, "teamAbbreviation": abbr} for team_id, abbr in TEAM_IDS.items()]
    )

    def load_window_or_empty(season, _start, _end, file_kind):
        if file_kind != "coverage_matrix":
            return pd.DataFrame()
        return windows.get(season, pd.DataFrame()).copy()

    def load_snapshot(season, _week, file_kind):
        if season == LEGACY_SEASON and file_kind == "line_matchups":
            return line_matchups.copy()
        if season == CURRENT_SEASON and file_kind == "coverage_matrix":
            return windows[CURRENT_SEASON].copy()
        return None

    monkeypatch.setattr(nfl_fp_team_weekly, "load_window_or_empty", load_window_or_empty)
    monkeypatch.setattr(nfl_fp_team_weekly, "load_snapshot", load_snapshot)
    monkeypatch.setattr(nfl_fp_team_weekly, "available_snapshots", lambda _season: [1])
    monkeypatch.setattr(nfl_fp_weekly, "available_snapshots", lambda _season: [])
    monkeypatch.setattr(nfl_fp_team_weekly_aggregate, "_ABBR_MAP_CACHE", {})
    return windows


def test_both_eras_pool_into_one_row_per_team(two_eras):
    team, _defense = nfl_fp_team_weekly_aggregate.load_team_and_defense_features(WINDOWS)

    assert sorted(team.index) == sorted(TEAM_IDS.values())
    for abbr in TEAM_IDS.values():
        assert team.loc[abbr, "off_faced_man_pct"] == pytest.approx(POOLED_MAN_PCT)


def test_an_empty_season_does_not_blank_the_other_window(two_eras, monkeypatch):
    """The 2026-09-15 outage: the current season had only empty snapshots."""
    monkeypatch.setitem(two_eras, CURRENT_SEASON, pd.DataFrame())
    monkeypatch.setattr(
        nfl_fp_team_weekly,
        "load_snapshot",
        lambda season, _week, file_kind: (
            pd.DataFrame(
                [
                    {"teamTeamId": team_id, "teamAbbreviation": abbr}
                    for team_id, abbr in TEAM_IDS.items()
                ]
            )
            if season == LEGACY_SEASON and file_kind == "line_matchups"
            else None
        ),
    )
    monkeypatch.setattr(nfl_fp_team_weekly_aggregate, "_ABBR_MAP_CACHE", {})

    team, _defense = nfl_fp_team_weekly_aggregate.load_team_and_defense_features(WINDOWS)

    assert sorted(team.index) == sorted(TEAM_IDS.values())
    for abbr in TEAM_IDS.values():
        assert team.loc[abbr, "off_faced_man_pct"] == pytest.approx(LEGACY_MAN / DROPBACKS)
