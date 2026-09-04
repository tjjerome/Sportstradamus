"""Pins for the constellation's readability pass: the capped star set and the
minimum-distance spacing lattice.

A busy game offers far more model-liked legs than a 980x360 box can hold, so the
map draws a bounded default set — the strongest by Kelly edge, both teams
represented, no player owning it — and everything else waits behind *look
deeper*. What is drawn then goes through ``settle``: the biggest stars keep their
wanted position to the float and only a star that would collide moves, to the
nearest free lattice cell on its own team's half.
"""

from __future__ import annotations

import math

import pandas as pd

from sportstradamus.dashboard.components.constellation_spacing import (
    _CELL_PX,
    _FRAME_INSET,
    _STAR_GAP_PX,
    DEFAULT_STARS,
    X_RANGE,
    Y_RANGE,
    default_stars,
    settle,
)

_PX = (100.0, 100.0)  # round px-per-unit so the primitive's distances are hand-checkable
_TEAMS = ("NYK", "SAS")


def _row(player: str, team: str, kelly: float, market: str = "PTS") -> dict:
    return {
        "Player": player,
        "Market": market,
        "Bet": "Over",
        "Line": 10.5,
        "Game": "NYK/SAS",
        "League": "NBA",
        "Team": team,
        "Kelly": kelly,
        "Win Prob": 0.6,
        "Boost": 1.5,
    }


def _ladder(n: int, *, top: float = 0.9, step: float = 0.05) -> pd.DataFrame:
    """``n`` one-leg players, teams alternating, Kelly strictly descending."""
    return pd.DataFrame([_row(f"P{i:02d}", _TEAMS[i % 2], top - i * step) for i in range(n)])


def _key(i: int) -> str:
    return f"P{i:02d}|PTS|Over"


def test_a_clear_anchor_is_returned_exactly():
    anchors = {"a": (-0.4, 0.25), "b": (0.7, -0.3)}
    placed = settle(anchors, dict.fromkeys(anchors, 20.0), _PX)
    assert placed == anchors  # no quantisation for a star nobody crowds


def test_a_colliding_star_moves_to_the_nearest_clear_cell():
    anchors = {"first": (0.1, 0.1), "second": (0.1, 0.1)}
    sizes = dict.fromkeys(anchors, 20.0)
    placed = settle(anchors, sizes, _PX)
    assert placed["first"] == (0.1, 0.1)  # priority is iteration order — the first stays put
    moved = placed["second"]
    shift = math.hypot((moved[0] - 0.1) * _PX[0], (moved[1] - 0.1) * _PX[1])
    assert shift >= 20.0 + _STAR_GAP_PX  # (20 + 20) / 2 + gap of clear air
    assert shift <= 20.0 + _STAR_GAP_PX + _CELL_PX  # and no further than a cell past it


def test_a_moved_star_never_crosses_the_team_axis():
    anchors = {"left": (-0.02, 0.0)}
    sizes = {"left": 20.0, "blocker": 20.0}
    placed = settle(anchors, sizes, _PX, fixed={"blocker": (-0.02, 0.0)}, side={"left": -1})
    assert placed["left"][0] <= 0.0


def test_fixed_stars_are_neither_moved_nor_returned():
    fixed = {"pinned": (0.0, 0.0)}
    sizes = {"pinned": 20.0, "newcomer": 20.0}
    placed = settle({"newcomer": (0.0, 0.0)}, sizes, _PX, fixed=fixed)
    assert set(placed) == {"newcomer"}
    assert placed["newcomer"] != (0.0, 0.0)


def test_no_star_lands_inside_the_exclusion_box():
    box = (-30.0, -30.0, 30.0, 30.0)
    anchors = {"a": (0.0, 0.0), "b": (0.1, 0.05), "c": (-0.2, 0.1)}
    placed = settle(anchors, dict.fromkeys(anchors, 20.0), _PX, exclude=box)
    for x, y in placed.values():
        assert not (box[0] <= x * _PX[0] <= box[2] and box[1] <= y * _PX[1] <= box[3])


def test_settle_is_deterministic():
    anchors = {"a": (0.0, 0.0), "b": (0.0, 0.0), "c": (0.05, 0.0)}
    sizes = dict.fromkeys(anchors, 24.0)
    assert settle(anchors, sizes, _PX) == settle(anchors, sizes, _PX)
    reversed_order = dict(reversed(list(anchors.items())))
    assert settle(reversed_order, sizes, _PX) == settle(reversed_order, sizes, _PX)
    assert settle(reversed_order, sizes, _PX)["c"] == (0.05, 0.0)  # first in, first served


def test_the_lattice_stays_inside_the_inset_frame():
    anchors = {"edge": (X_RANGE, Y_RANGE)}
    sizes = {"edge": 20.0, "blocker": 20.0}
    placed = settle(anchors, sizes, _PX, fixed={"blocker": (X_RANGE, Y_RANGE)})
    x, y = placed["edge"]
    assert abs(x) <= X_RANGE * _FRAME_INSET
    assert abs(y) <= Y_RANGE * _FRAME_INSET


def test_default_set_is_top_n_by_kelly():
    universe = {_key(i): row for i, row in enumerate(_ladder(15).to_dict("records"))}
    chosen = default_stars(universe, list(_TEAMS))
    assert len(chosen) == DEFAULT_STARS
    assert set(chosen) == {_key(i) for i in range(DEFAULT_STARS)}  # the three weakest are out


def test_default_set_guarantees_both_teams():
    rows = [_row(f"N{i:02d}", "NYK", 0.9 - i * 0.05) for i in range(14)]
    rows += [_row(f"S{i:02d}", "SAS", 0.05) for i in range(4)]
    universe = {f"{row['Player']}|PTS|Over": row for row in rows}
    chosen = default_stars(universe, list(_TEAMS))
    by_team = [universe[key]["Team"] for key in chosen]
    assert len(chosen) == DEFAULT_STARS
    assert by_team.count("SAS") == 4  # the both-teams floor, over a lopsided Kelly ranking
    assert by_team.count("NYK") == 8


def test_default_set_caps_legs_per_player():
    markets = ("PTS", "REB", "AST", "STL", "BLK", "TOV")
    rows = [_row("Hot", "NYK", 0.9 - i * 0.04, market) for i, market in enumerate(markets)]
    rows += [_row(f"N{i}", "NYK", 0.3) for i in range(5)]
    rows += [_row(f"S{i}", "SAS", 0.3) for i in range(5)]
    universe = {f"{row['Player']}|{row['Market']}|Over": row for row in rows}
    chosen = default_stars(universe, list(_TEAMS))
    teams = [universe[key]["Team"] for key in chosen]
    assert len(chosen) == DEFAULT_STARS
    assert len([key for key in chosen if key.startswith("Hot|")]) == 2
    assert teams.count("NYK") >= 4 and teams.count("SAS") >= 4
