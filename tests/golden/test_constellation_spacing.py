"""Pins for the constellation's readability pass: the capped star set, the
minimum-distance spacing lattice, and sparse captions.

A busy game offers far more model-liked legs than a 980x360 box can hold, so the
map draws a bounded default set — the strongest by Kelly edge, both teams
represented, no player owning it — and everything else waits behind *look
deeper*. What is drawn then goes through ``settle``: the biggest stars keep their
template vertices to the float and only a star that would collide moves, to the
nearest free lattice cell on its own team's half. Captions stop being universal:
the slip's stars plus the biggest few candidates carry one, and only where its
box clears every glyph and every caption already accepted.
"""

from __future__ import annotations

import itertools
import math

import pandas as pd

from sportstradamus.dashboard.components.constellation import (
    _LABEL_FONT_SIZE,
    _LABEL_FONT_SIZE_MOBILE,
    constellation_figure,
)
from sportstradamus.dashboard.components.constellation_shapes import shape_catalog
from sportstradamus.dashboard.components.constellation_slate import (
    DECORATION,
    SHAPE_SCALE,
    SHAPE_SCALE_MOBILE,
    slate_shapes,
)
from sportstradamus.dashboard.components.constellation_spacing import (
    _CELL_PX,
    _CHAR_WIDTH_EM,
    _FRAME_INSET,
    _LINE_HEIGHT_EM,
    _STAR_GAP_PX,
    CAPTION_TOP_K,
    DEFAULT_STARS,
    PX_PER_UNIT,
    PX_PER_UNIT_MOBILE,
    X_RANGE,
    Y_RANGE,
    default_stars,
    settle,
)

_PX = (100.0, 100.0)  # round px-per-unit so the primitive's distances are hand-checkable
_TEAMS = ("NYK", "SAS")
_HOURGLASS = shape_catalog()["templates"]["the-hourglass"]


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


def _rect(cx: float, cy: float, width: float, height: float) -> tuple[float, ...]:
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


def _disjoint(one: tuple[float, ...], other: tuple[float, ...]) -> bool:
    return one[0] >= other[2] or one[2] <= other[0] or one[1] >= other[3] or one[3] <= other[1]


def _node_traces(fig) -> list:
    return [t for t in fig.data if t.mode and "markers" in t.mode and t.name != DECORATION]


def _stars(fig) -> dict:
    """key -> (position, marker px, caption text, textposition)."""
    stars = {}
    for trace in _node_traces(fig):
        rows = zip(
            trace.customdata,
            trace.x,
            trace.y,
            trace.marker.size,
            trace.text,
            trace.textposition,
            strict=True,
        )
        for card, x, y, size, text, place in rows:
            stars[card[0]] = ((float(x), float(y)), float(size), str(text), str(place))
    return stars


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
    # Two placed neighbours wall the star's own half, so the nearest free air is
    # across x=0. Left unmasked the star takes it — which is what the mask exists
    # to forbid, and what makes the second assertion mean something.
    anchors = {"left": (-0.02, 0.0)}
    sizes = {"left": 20.0, "twin": 20.0, "over": 38.0, "under": 38.0}
    fixed = {"twin": (-0.02, 0.0), "over": (-0.30, 0.30), "under": (-0.30, -0.30)}
    assert settle(anchors, sizes, _PX, fixed=fixed)["left"][0] > 0.0
    assert settle(anchors, sizes, _PX, fixed=fixed, side={"left": -1})["left"][0] <= 0.0


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


def test_settle_takes_its_priority_from_iteration_order():
    anchors = {"a": (0.0, 0.0), "b": (0.0, 0.0), "c": (0.05, 0.0)}
    sizes = dict.fromkeys(anchors, 24.0)
    first_in = settle(anchors, sizes, _PX)
    reversed_in = settle(dict(reversed(list(anchors.items()))), sizes, _PX)
    assert first_in["a"] == (0.0, 0.0)  # whoever is offered first keeps the anchor
    assert reversed_in["c"] == (0.05, 0.0)
    assert reversed_in["a"] != (0.0, 0.0)  # and whoever comes later gives way
    assert settle(anchors, sizes, _PX) == first_in


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


def test_in_slip_leg_beyond_the_cut_is_still_a_star():
    pool = _ladder(15)
    weakest = pool.to_dict("records")[14]
    fig = constellation_figure([weakest], None, pool)
    active = next(t for t in fig.data if t.name == "active")
    assert [card[0] for card in active.customdata] == [_key(14)]


def test_a_slip_with_only_a_passed_leg_still_renders():
    """Nothing in the pool is model-liked, so the default cut is empty and the
    promoted slip leg is the whole map — the one case that hands the spring solve
    an empty node set, which it cannot warm-start.
    """
    leg = _row("A", "NYK", -0.1)
    stars = _stars(constellation_figure([leg], None, pd.DataFrame([leg])))
    assert set(stars) == {"A|PTS|Over"}


def test_main_stars_keep_their_clearance_on_both_viewports():
    """Both viewports, lens off and on.

    Distances are the rendered ones — the figure's own coordinates already carry
    the "look wider" shrink, which is exactly what makes that lens the hard case:
    it pulls the stars together without shrinking a single marker.
    """
    pool = _ladder(15)
    for (mobile, px), wider in itertools.product(
        ((False, PX_PER_UNIT), (True, PX_PER_UNIT_MOBILE)), (None, [])
    ):
        stars = _stars(
            constellation_figure(
                [], None, pool, shape=_HOURGLASS, mobile=mobile, wider_groups=wider
            )
        )
        assert len(stars) == DEFAULT_STARS
        for one, other in itertools.combinations(sorted(stars), 2):
            (ax, ay), size_a, *_ = stars[one]
            (bx, by), size_b, *_ = stars[other]
            apart = math.hypot((ax - bx) * px[0], (ay - by) * px[1])
            assert apart >= (size_a + size_b) / 2 + _STAR_GAP_PX - 1e-9, (one, other, mobile, wider)


def test_spacing_never_moves_an_uncrowded_star():
    pool = pd.DataFrame(
        [
            _row("A", "NYK", 0.4),
            _row("B", "SAS", 0.3, "REB"),
            _row("C", "NYK", 0.2, "AST"),
            _row("D", "SAS", 0.1, "BLK"),
        ]
    )
    frame = {
        (round(v["x"] * scale[0], 5), round(v["y"] * scale[1], 5))
        for scale in (SHAPE_SCALE, SHAPE_SCALE_MOBILE)
        for v in _HOURGLASS["vertices"]
    }
    for mobile in (False, True):
        stars = _stars(constellation_figure([], None, pool, shape=_HOURGLASS, mobile=mobile))
        for (x, y), *_ in stars.values():
            assert (round(x, 5), round(y, 5)) in frame


def test_captions_are_the_slip_plus_the_biggest_candidates():
    pool = _ladder(15)
    picked = pool.to_dict("records")[11]  # the weakest leg the cut still draws
    stars = _stars(constellation_figure([picked], None, pool, shape=_HOURGLASS))
    captioned = {key for key, (_, _, text, _) in stars.items() if text}
    candidates = sorted((k for k in stars if k != _key(11)), key=lambda k: -stars[k][1])
    assert captioned <= {_key(11), *candidates[:CAPTION_TOP_K]}
    assert _key(11) in captioned  # the slip's own star is captioned whatever its size
    assert len(stars) > len(captioned)  # and the rest read from the hover card


def test_caption_boxes_never_overlap():
    """No caption lands on a glyph or on another caption, on any shape in the bank.

    Only caption-involving pairs: ``settle`` separates the stars' bounding
    *circles*, so two glyph squares may still share a corner — a real overlap of
    two star boxes is not what a caption has to clear, and pinning it here would
    pin the wrong guarantee.
    """
    pool = _ladder(15)
    viewports = (
        (False, PX_PER_UNIT, _LABEL_FONT_SIZE),
        (True, PX_PER_UNIT_MOBILE, _LABEL_FONT_SIZE_MOBILE),
    )
    for template, (mobile, px, font_px) in itertools.product(
        shape_catalog()["templates"].values(), viewports
    ):
        boxes = []
        for (x, y), size, text, place in _stars(
            constellation_figure([], None, pool, shape=template, mobile=mobile)
        ).values():
            cx, cy = x * px[0], y * px[1]
            boxes.append((False, _rect(cx, cy, size, size)))
            if not text:
                continue
            height = _LINE_HEIGHT_EM * font_px
            lift = (size / 2 + height / 2) * (1 if place == "top center" else -1)
            width = len(text) * _CHAR_WIDTH_EM * font_px
            boxes.append((True, _rect(cx, cy + lift, width, height)))
        for (one_is_caption, one), (other_is_caption, other) in itertools.combinations(boxes, 2):
            if one_is_caption or other_is_caption:
                assert _disjoint(one, other), (template["label"], mobile, one, other)


def test_slate_shapes_classify_the_capped_set():
    offers = pd.DataFrame([_row(f"P{i:02d}", _TEAMS[i % 2], 0.9 - i * 0.02) for i in range(30)])
    shape = slate_shapes(offers, None, "2026-09-04")["NYK/SAS"]
    assert shape.n_supernodes == DEFAULT_STARS  # uncorrelated, so every star is its own node
