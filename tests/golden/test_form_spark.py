"""Pins for the constellation hover card's chart rows: last-five form and line movement.

``form_sparks`` and ``move_sparks`` are all that stand between the data and the card's
markup: the frontend drops what they return straight into ``innerHTML`` without
validating it, and a leg they can't build a row for has to go missing rather than draw
an empty box. The form window and its ordering are pinned here because the chart reads
left to right.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.components import form_spark
from sportstradamus.dashboard.components.form_spark import form_sparks, move_sparks
from sportstradamus.dashboard.components.spark_svg import form_svg, movement_summary, movement_svg
from sportstradamus.helpers.io import LINE_MOVEMENT_COLS

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


# The line-movement snapshot, column-stable the way its loader returns it. Star's posted
# line fell a point and its fair line with it; Wide, an NFL TD prop, held its line while
# its price moved. No row for anyone else.
_MOVEMENT = pd.DataFrame(
    [
        {
            "League": "NBA",
            "Platform": "Underdog",
            "Market": "Points",
            "Player": "Star",
            "Date": "2026-01-08",
            "n_moves": 1,
            "n_price_moves": 0,
            "series": "[20.5, 19.5]",
            "fair_series": "[20.5, 19.8]",
        },
        {
            "League": "NFL",
            "Platform": "Underdog",
            "Market": "tds",
            "Player": "Wide",
            "Date": "2026-01-09",
            "n_moves": 0,
            "n_price_moves": 1,
            "series": "[0.5, 0.5]",
            "fair_series": "[0.5, 0.66]",
        },
    ]
).reindex(columns=LINE_MOVEMENT_COLS)


def _pool(*rows: dict) -> pd.DataFrame:
    """A focus game's offer rows. Empty keeps its columns, the way an ``offers`` slice does."""
    filled = [
        {
            "League": "NBA",
            "Platform": "Underdog",
            "Date": "2026-01-08",
            "Market": "Points",
            "Bet": "Over",
            "Stat": "PTS",
            "Line": 20.5,
            **row,
        }
        for row in rows
    ]
    return pd.DataFrame(
        filled, columns=["League", "Platform", "Date", "Player", "Market", "Bet", "Stat", "Line"]
    )


def _sparks(pool: pd.DataFrame, monkeypatch, gamelog=_NBA_GAMELOG) -> dict:
    monkeypatch.setattr(form_spark, "load_gamelog", lambda _league: gamelog)
    return form_sparks(pool)


def _moves(pool: pd.DataFrame, monkeypatch) -> dict:
    monkeypatch.setattr(form_spark, "load_current_line_movement", lambda: _MOVEMENT)
    return move_sparks(pool)


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


def test_an_offer_with_a_movement_row_gets_its_trace_beside_its_summary(monkeypatch):
    summary = movement_summary([20.5, 19.8], [20.5, 19.5], n_moves=1, n_price_moves=0)
    trace = movement_svg([20.5, 19.8], bet="Over", n_changes=1, title=summary)
    moves = _moves(_pool({"Player": "Star"}), monkeypatch)
    assert moves == {"Star|Points|Over": f"{trace}<span>{summary}</span>"}


def test_an_offer_the_snapshot_has_no_row_for_is_absent(monkeypatch):
    """Absent rather than blank: the frontend draws no movement row for a missing key,
    since an offer the ladder never saw has no movement to be missing."""
    moves = _moves(_pool({"Player": "Star"}, {"Player": "Rookie"}), monkeypatch)
    assert set(moves) == {"Star|Points|Over"}


def test_a_wider_lens_star_from_another_league_reads_its_own_row(monkeypatch):
    """``_render_constellation`` concatenates the lens's other-game records onto the focus
    pool, so the frame it hands over mixes leagues and repeats index labels — each offer
    must still read the movement row its own keys name, and the focus star's is unchanged."""
    wide = {
        "League": "NFL",
        "Date": "2026-01-09",
        "Player": "Wide",
        "Market": "tds",
        "Bet": "Under",
        "Stat": "tds",
        "Line": 0.5,
    }
    summary = movement_summary([0.5, 0.66], [0.5, 0.5], n_moves=0, n_price_moves=1)
    trace = movement_svg([0.5, 0.66], bet="Under", n_changes=1, title=summary)
    alone = _moves(_pool({"Player": "Star"}), monkeypatch)
    moves = _moves(pd.concat([_pool({"Player": "Star"}), _pool(wide)]), monkeypatch)
    assert moves == {**alone, "Wide|tds|Under": f"{trace}<span>{summary}</span>"}
