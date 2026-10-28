"""How ``StatsNFL._update`` dates a new gamelog row.

``_update`` writes ``moneyline`` / ``totals`` once, at ingestion, from the odds archive
under ``(gameday, team)``. The schedule join that dates a new row had been running on the
week alone, so every row was read under its week's first game day: of the 96 team-games
of 2026 weeks 1-3, 90 were stored at the no-quote default, 84 of them with their line in
the archive.

A new row takes its opponent and game from the schedule, fetched once a run. Production's
fetch failed on 2026-09-07 and 10-27 while new rows were arriving; the schedule lookup
crashed the NFL update both times, and NFL sat out that board.
"""

from datetime import date, timedelta
from types import SimpleNamespace
from urllib.error import URLError

import pandas as pd

from sportstradamus.stats import StatsNFL, base
from sportstradamus.stats import nfl as nfl_module

# The last finished Sunday and the Thursday before it, dated off today: _update keeps only
# games already played, and re-reads every stored line once the season is 300 days old.
SUNDAY = date.today() - timedelta(days=date.today().isoweekday())
THURSDAY = SUNDAY - timedelta(days=3)
WEEK = 2
MONEYLINE = {THURSDAY: 0.25, SUNDAY: 0.75}
TOTALS = {THURSDAY: 17.5, SUNDAY: 27.5}
# nflverse spells the Rams "LA" in the schedule and the player stats alike; the gamelog
# and the archive know them as "LAR".
PLAYER_TEAM = {"Thu Home": "BUF", "Thu Away": "DET", "Sun Home": "LA", "Sun Away": "KC"}
GAMEDAY = {"BUF": THURSDAY, "DET": THURSDAY, "LAR": SUNDAY, "KC": SUNDAY}
STAT_COLUMNS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "sack_fumbles",
    "sack_fumbles_lost",
    "passing_2pt_conversions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles",
    "receiving_fumbles_lost",
    "receiving_2pt_conversions",
    "target_share",
    "air_yards_share",
    "wopr",
]


def _player_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [f"00-000000{i}" for i in range(len(PLAYER_TEAM))],
            "player_display_name": list(PLAYER_TEAM),
            "position_group": "WR",
            "team": list(PLAYER_TEAM.values()),
            "season": THURSDAY.year,
            "week": WEEK,
            "season_type": "REG",
        }
        | dict.fromkeys(STAT_COLUMNS, 0.0)
    )


def _snap_counts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player": list(PLAYER_TEAM),
            "position": "WR",
            "season": THURSDAY.year,
            "week": WEEK,
            "offense_pct": 1.0,
        }
    )


def _schedule(years) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["thursday", "sunday"],
            "week": WEEK,
            "gameday": [str(THURSDAY), str(SUNDAY)],
            "home_team": ["BUF", "LA"],
            "away_team": ["DET", "KC"],
        }
    )


def _schedule_unreachable(years):
    raise URLError("[Errno -3] Temporary failure in name resolution")


class _LinePerDate:
    """Archive quoting every team on both days, so a stored line names the day it was read under."""

    default_totals = {}

    def get_team_market_map(self, league, market, *, cutoff, dates):
        line = MONEYLINE if market == "Moneyline" else TOTALS
        return {(str(day), team): line[day] for day in line for team in GAMEDAY}


def _stubbed_stats(monkeypatch, player_stats) -> tuple[StatsNFL, dict]:
    """A StatsNFL whose fetches return ``player_stats()`` and the fakes above.

    Returns it with the dict each ``write_gamelog`` call stores its gamelog in.
    """
    written = {}
    monkeypatch.setattr(base, "archive", _LinePerDate())
    monkeypatch.setattr(
        nfl_module.nflr, "load_player_stats", lambda: SimpleNamespace(to_pandas=player_stats)
    )
    monkeypatch.setattr(
        nfl_module.nflr, "load_snap_counts", lambda: SimpleNamespace(to_pandas=_snap_counts)
    )
    monkeypatch.setattr(nfl_module.nfl, "import_schedules", _schedule)
    monkeypatch.setattr(
        nfl_module,
        "write_gamelog",
        lambda league, gamelog, teamlog, players: written.update(gamelog=gamelog),
    )
    stats = StatsNFL()
    stats.season_start = THURSDAY
    monkeypatch.setattr(stats, "_load_player_ids", lambda: None)
    # The week's play-by-play is not out yet.
    monkeypatch.setattr(stats, "parse_pbp", lambda *args: 0)
    return stats, written


def _stored_games(gamelog: pd.DataFrame) -> dict:
    return {row.team: (row.gameday, row.moneyline, row.totals) for row in gamelog.itertuples()}


def test_new_rows_take_the_line_of_their_own_gameday(monkeypatch):
    stats, written = _stubbed_stats(monkeypatch, _player_stats)

    stats._update()

    assert _stored_games(written["gamelog"]) == {
        team: (str(day), MONEYLINE[day], TOTALS[day]) for team, day in GAMEDAY.items()
    }


def test_new_rows_wait_out_a_run_without_the_schedule(monkeypatch):
    stats, written = _stubbed_stats(monkeypatch, lambda: _player_stats().iloc[:2])
    stats._update()
    # The stored Thursday team rows read zero, so the reopen asks FTN and PFR: neither
    # holds a game yet.
    monkeypatch.setattr(stats, "_late_source_games", lambda season: (set(), set()))
    # The Sunday game's players arrive on a run whose schedule fetch fails.
    monkeypatch.setattr(
        nfl_module.nflr, "load_player_stats", lambda: SimpleNamespace(to_pandas=_player_stats)
    )
    monkeypatch.setattr(nfl_module.nfl, "import_schedules", _schedule_unreachable)

    stats._update()

    assert set(written["gamelog"]["team"]) == {"BUF", "DET"}

    monkeypatch.setattr(nfl_module.nfl, "import_schedules", _schedule)

    stats._update()

    assert _stored_games(written["gamelog"]) == {
        team: (str(day), MONEYLINE[day], TOTALS[day]) for team, day in GAMEDAY.items()
    }
