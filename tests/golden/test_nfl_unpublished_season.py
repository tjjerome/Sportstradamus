"""Opening-week NFL backfill must survive nflverse season files that don't exist yet.

nflverse publishes a season's first-game pbp, player stats and snap counts before it
creates that season's FTN charting and weekly PFR advstats files, so
``StatsNFL._ensure_pbp_loaded`` sees a 404 for them. Prophecize crashed on exactly that
on 2026-09-10 (``ftn_charting_2026.parquet``). Inside the publish grace window a 404
backfills without the missing source; any other HTTP error, or a 404 past the window,
still raises.
"""

from datetime import date, timedelta
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import StatsNFL
from sportstradamus.stats import nfl as nfl_module

TEAM = "SEA"
RECEIVER = "Test Receiver"
NGS_TEXT_COLUMNS = ["player_display_name", "player_position", "team_abbr"]
NGS_NUMERIC_COLUMNS = [
    "week",
    "expected_rush_yards",
    "rush_attempts",
    "completion_percentage_above_expectation",
    "completion_percentage",
    "passer_rating",
    "avg_intended_air_yards",
    "avg_air_yards_differential",
    "avg_time_to_throw",
    "aggressiveness",
    "rush_yards_over_expected",
    "rush_pct_over_expected",
    "avg_yac_above_expectation",
    "avg_separation",
    "avg_cushion",
]


def _pbp() -> pd.DataFrame:
    """One pass and one run per side, so every per-team denominator is non-zero."""
    return pd.DataFrame(
        {
            "game_id": "2026_01_NE_SEA",
            "play_id": [1, 2, 3, 4],
            "week": 1,
            "home_team": "SEA",
            "away_team": "NE",
            "posteam": ["SEA", "SEA", "NE", "NE"],
            "play_type": ["pass", "run", "pass", "run"],
            "desc": "",
            "game_seconds_remaining": [3600, 3500, 3400, 3300],
            "pass": [1, 0, 1, 0],
            "rush": [0, 1, 0, 1],
            "pass_attempt": [1, 0, 1, 0],
            "qb_dropback": [1, 0, 1, 0],
            "complete_pass": [1, 0, 0, 0],
            "qb_hit": 0,
            "sack": 0,
            "down": 1,
            "ydstogo": 10,
            "yardline_100": [60, 15, 55, 8],
            "yards_gained": [12, 4, 0, 3],
            "air_yards": [9, np.nan, 7, np.nan],
            "passing_yards": [12, np.nan, np.nan, np.nan],
            "rushing_yards": [np.nan, 4, np.nan, 3],
            "receiving_yards": [12, np.nan, np.nan, np.nan],
            "epa": [0.8, -0.1, -0.5, 0.2],
            "xpass": 0.6,
            "cpoe": [20.0, np.nan, -30.0, np.nan],
            "pass_location": ["middle", None, "left", None],
            "passer_player_id": ["QB1", None, "QB2", None],
            "rusher_player_id": [None, "RB1", None, "RB2"],
            "receiver_player_id": ["WR1", None, "WR2", None],
            "home_score": [0, 0, 7, 7],
            "away_score": 0,
        }
    )


def _ftn(years) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nflverse_game_id": "2026_01_NE_SEA",
            "nflverse_play_id": [1, 3],
            "week": 1,
            "season": 2026,
            "is_qb_out_of_pocket": False,
            "is_throw_away": False,
            "read_thrown": ["1", "2"],
            "n_blitzers": [2, 0],
        }
    )


def _http_error(code: int):
    def fetch(*args, **kwargs):
        raise HTTPError("https://github.com/nflverse/nflverse-data", code, "stub", None, None)

    return fetch


def _stats(monkeypatch, ftn_fetch, season_age_days: int = 1) -> StatsNFL:
    """StatsNFL mid-backfill whose season opened ``season_age_days`` ago; PFR unpublished."""
    ngs = pd.DataFrame(
        {c: pd.Series(dtype=object) for c in NGS_TEXT_COLUMNS}
        | {c: pd.Series(dtype=float) for c in NGS_NUMERIC_COLUMNS}
    )
    monkeypatch.setattr(nfl_module.nflr, "load_pbp", lambda year: SimpleNamespace(to_pandas=_pbp))
    monkeypatch.setattr(nfl_module.nfl, "import_ftn_data", ftn_fetch)
    monkeypatch.setattr(nfl_module.nfl, "import_ngs_data", lambda stat_type, years: ngs.copy())
    monkeypatch.setattr(nfl_module.nfl, "import_weekly_pfr", _http_error(404))
    s = StatsNFL()
    s.season_start = date.today() - timedelta(days=season_age_days)
    s.need_pbp = True
    s.ids = {RECEIVER: "WR1"}
    s.gamelog = pd.DataFrame(
        {
            "season": [s.season_start.year],
            "week": [1],
            "player display name": [RECEIVER],
            "snap pct": [1.0],
        }
    )
    return s


@pytest.mark.parametrize("ftn_published", [False, True])
def test_backfill_through_unpublished_season_files(monkeypatch, ftn_published):
    s = _stats(monkeypatch, _ftn if ftn_published else _http_error(404))
    season = s.season_start.year

    team = s.parse_pbp(1, TEAM, season)
    player = s.parse_pbp(1, TEAM, season, RECEIVER)

    assert isinstance(team, dict)
    assert isinstance(player, dict)
    assert s.need_pbp is False
    assert s.pfr.empty
    assert team["pressure_per_pass"] == 0
    assert np.isnan(player["drop_rate"])
    assert np.isnan(player["broken_tackles"])
    # The receiver's one target was the first read in the FTN stub; the unpublished
    # defaults mark no read at all.
    assert player["first_read_targets_per_route_run"] == (1.0 if ftn_published else 0.0)


@pytest.mark.parametrize(("code", "past_grace"), [(503, False), (404, True)])
def test_other_http_failures_still_raise(monkeypatch, code, past_grace):
    age = nfl_module._NFLVERSE_PUBLISH_GRACE_DAYS + 1 if past_grace else 1
    s = _stats(monkeypatch, _http_error(code), age)

    with pytest.raises(HTTPError):
        s.parse_pbp(1, TEAM, s.season_start.year)
