"""Pins for the constellation hover card's last-five form sparklines.

``form_sparks`` is the only thing standing between the gamelog and the card's markup:
the frontend drops what it returns straight into ``innerHTML`` without validating it,
and a leg the map can't build a series for has to go missing rather than draw an empty
box. The window and its ordering are pinned here because the chart reads left to right.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.components import form_spark
from sportstradamus.dashboard.components.form_spark import form_sparks
from sportstradamus.dashboard.components.spark_svg import form_svg

# Deliberately out of date order: a correct read sorts on GAME_DATE, not row position.
_NBA_GAMELOG = pd.DataFrame(
    {
        "PLAYER_NAME": ["Star"] * 7 + ["Bench"],
        "GAME_DATE": [
            "2026-01-05",
            "2026-01-01",
            "2026-01-07",
            "2026-01-02",
            "2026-01-06",
            "2026-01-03",
            "2026-01-04",
            "2026-01-04",
        ],
        "PTS": [24.0, 10.0, 31.0, 12.0, 27.0, 18.0, 22.0, 4.0],
    }
)
# Star's games oldest-first: 10, 12, 18, 22, 24, 27, 31 -> the last five are these.
_STAR_LAST_5 = [18.0, 22.0, 24.0, 27.0, 31.0]


def _pool(*rows: dict) -> pd.DataFrame:
    """A focus game's offer rows. Empty keeps its columns, the way an ``offers`` slice does."""
    filled = [
        {"League": "NBA", "Market": "Points", "Bet": "Over", "Stat": "PTS", "Line": 20.5, **row}
        for row in rows
    ]
    return pd.DataFrame(filled, columns=["League", "Player", "Market", "Bet", "Stat", "Line"])


def _sparks(pool: pd.DataFrame, monkeypatch, gamelog=_NBA_GAMELOG) -> dict:
    monkeypatch.setattr(form_spark, "load_gamelog", lambda _league: gamelog)
    return form_sparks(pool)


def test_the_window_is_the_last_five_games_oldest_first(monkeypatch):
    sparks = _sparks(_pool({"Player": "Star"}), monkeypatch)
    assert sparks["Star|Points|Over"].startswith(form_svg(_STAR_LAST_5, 20.5))


def test_the_caption_counts_the_games_that_cleared_the_line(monkeypatch):
    sparks = _sparks(_pool({"Player": "Star"}), monkeypatch)
    assert "last 5 vs line · 4/5 over" in sparks["Star|Points|Over"]


def test_a_short_gamelog_captions_the_games_it_has(monkeypatch):
    sparks = _sparks(_pool({"Player": "Bench", "Line": 3.5}), monkeypatch)
    assert "last 1 vs line · 1/1 over" in sparks["Bench|Points|Over"]


def test_a_player_the_gamelog_never_carries_is_absent(monkeypatch):
    """Absent, not empty: the frontend draws its "no last 5 for this leg" scar for a
    missing key, which is the honest read for a player we have no games for."""
    sparks = _sparks(_pool({"Player": "Star"}, {"Player": "Rookie"}), monkeypatch)
    assert set(sparks) == {"Star|Points|Over"}


def test_a_stat_that_names_no_gamelog_column_is_absent(monkeypatch):
    pool = _pool({"Player": "Star"}, {"Player": "Star", "Market": "Blocks", "Stat": "BLK"})
    assert set(_sparks(pool, monkeypatch)) == {"Star|Points|Over"}


def test_an_offer_carrying_no_stat_column_falls_back_to_its_market(monkeypatch):
    """The same coalesce the History tab reads the gamelog through: ``current_offers``
    always ships ``Stat``, but the dashboard's own render fixtures do not."""
    pool = pd.DataFrame(
        [{"League": "NBA", "Player": "Star", "Market": "PTS", "Bet": "Over", "Line": 20.5}]
    )
    assert _sparks(pool, monkeypatch)["Star|PTS|Over"].startswith(form_svg(_STAR_LAST_5, 20.5))


def test_a_mixed_league_pool_reads_each_league_against_its_own_gamelog(monkeypatch):
    """The "look wider" lens puts other games' stars on the map, and those games need not
    share the focus game's league — so the pool it sparks is not single-league."""
    nfl = pd.DataFrame({"player display name": ["Wide"], "receptions": [7.0]})
    monkeypatch.setattr(
        form_spark, "load_gamelog", lambda league: _NBA_GAMELOG if league == "NBA" else nfl
    )
    pool = _pool(
        {"Player": "Star"},
        {
            "League": "NFL",
            "Player": "Wide",
            "Market": "Receptions",
            "Stat": "receptions",
            "Line": 4.5,
        },
    )
    assert set(form_sparks(pool)) == {"Star|Points|Over", "Wide|Receptions|Over"}


def test_an_empty_pool_asks_nothing_of_the_gamelog(monkeypatch):
    assert _sparks(_pool(), monkeypatch) == {}


def test_an_nfl_pool_reads_the_frame_order_it_has_no_date_column(monkeypatch):
    """``GAMELOG_SCHEMA["NFL"]["date"]`` is None — the parquet is written in
    (season, week) order, so row order is the chronology and must be left alone."""
    gamelog = pd.DataFrame(
        {
            "player display name": ["Star"] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "receptions": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    pool = _pool(
        {
            "League": "NFL",
            "Player": "Star",
            "Market": "Receptions",
            "Stat": "receptions",
            "Line": 4.5,
        }
    )
    sparks = _sparks(pool, monkeypatch, gamelog=gamelog)
    assert sparks["Star|Receptions|Over"].startswith(form_svg([2.0, 3.0, 4.0, 5.0, 6.0], 4.5))
