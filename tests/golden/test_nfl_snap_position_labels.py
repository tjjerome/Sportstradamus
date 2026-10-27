"""PFR's snap counts label the Bengals' running backs ``HB``.

``StatsNFL._assemble_gamelog_frame`` keeps a player only where the snap counts hold a row
under his position, so the ``RB``-only filter dropped Chase Brown and Samaje Perine from
every game after 2024 (2025: 49 HB rows against 2 RB; no other team uses the label).
Fullbacks stay out.
"""

import pandas as pd

from sportstradamus.stats import StatsNFL

SEASON = 2026
WEEK = 1
# The player-stats columns the frame sums and rates, as _fetch_player_stats names them.
STAT_COLUMNS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
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
# Both are RB to nflverse's position_group; only the snap counts tell them apart.
SNAP_POSITION = {"Chase Brown": "HB", "Lead Blocker": "FB"}


def test_snap_counts_keep_the_hb_label_and_not_fullbacks():
    nfl_data = pd.DataFrame(
        {
            "player_id": [f"00-000000{i}" for i in range(len(SNAP_POSITION))],
            "player_display_name": list(SNAP_POSITION),
            "position_group": "RB",
            "team": "CIN",
            "season": SEASON,
            "week": WEEK,
            "season_type": "REG",
        }
        | dict.fromkeys(STAT_COLUMNS, 1.0)
    )
    sched = pd.DataFrame(
        {
            "game_id": [f"{SEASON}_0{WEEK}_CLE_CIN"],
            "week": WEEK,
            "gameday": f"{SEASON}-09-13",
            "home_team": "CIN",
            "away_team": "CLE",
        }
    )
    snaps = pd.DataFrame(
        {
            "player": list(SNAP_POSITION),
            "season": SEASON,
            "week": WEEK,
            "position": list(SNAP_POSITION.values()),
            "offense_pct": 1.0,
        }
    )

    frame = StatsNFL()._assemble_gamelog_frame(nfl_data, sched, snaps)

    assert frame["player display name"].tolist() == ["Chase Brown"]
