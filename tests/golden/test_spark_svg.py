"""Pins for the shared inline-SVG sparklines the Board grid and the constellation card
both render.

These strings are handed to AG Grid's ``getGui`` and to the constellation frontend's
``innerHTML`` — no chart library validates them and no JS test runner reads them, so the
gates here are the only thing between the builder and the browser.
"""

from __future__ import annotations

import re
from itertools import pairwise

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.components.spark_svg import (
    _HEIGHT,
    _INSET,
    form_svg,
    movement_svg,
)

_HEXES = re.compile(r"#[0-9A-Fa-f]{6}")
_RECT = re.compile(
    r'<rect x="([\d.eE+-]+)" y="([\d.eE+-]+)" width="([\d.eE+-]+)" '
    r'height="([\d.eE+-]+)" fill="(#[0-9A-Fa-f]{6})"/>'
)


def _points(svg: str) -> list[tuple[float, float]]:
    match = re.search(r'points="([^"]+)"', svg)
    assert match, f"no polyline in {svg}"
    return [tuple(float(n) for n in pair.split(",")) for pair in match.group(1).split()]


def _rects(svg: str) -> list[dict]:
    """The form bars, left to right — i.e. oldest game first."""
    return [
        {"x": float(x), "y": float(y), "width": float(w), "height": float(h), "fill": fill}
        for x, y, w, h, fill in _RECT.findall(svg)
    ]


def _rule_y(svg: str) -> float:
    match = re.search(r'<line x1="0" y1="([\d.]+)"', svg)
    assert match, f"no rule in {svg}"
    return float(match.group(1))


def test_a_line_moving_toward_the_bet_reads_green_and_away_reads_red():
    rising = [209.5, 211.5, 213.5]
    assert theme.RED in movement_svg(rising, bet="Over", n_moves=1)
    assert theme.GREEN in movement_svg(rising, bet="Under", n_moves=1)
    assert theme.GREEN in movement_svg(list(reversed(rising)), bet="Over", n_moves=1)
    assert theme.RED in movement_svg(list(reversed(rising)), bet="Under", n_moves=1)


def test_an_unmoved_line_draws_a_flat_gray_rule_not_an_empty_cell():
    """The common case on a same-day slate — ~96% of MLB offers never move. "The app
    held" is an answer, so the cell has to say it rather than render blank."""
    svg = movement_svg([1.5, 1.5, 1.5], bet="Over", n_moves=0)
    assert theme.GRAY in svg
    assert "polyline" not in svg
    assert "Line held at 1.5" in svg


def test_a_single_observation_is_a_held_line():
    assert "Line held at 7.5" in movement_svg([7.5], bet="Under", n_moves=0)


def test_a_round_trip_draws_its_journey_in_gray_and_is_not_called_held():
    """Travis Etienne's receptions repriced seven times and landed back on 2.5. Deciding
    "held" from the net delta flattened that into a straight rule and hid every move; only
    ``n_moves`` separates the two, since a resampled series can miss a reverted move."""
    svg = movement_svg([2.5, 3.5, 3.5, 2.5], bet="Over", n_moves=7)
    assert "polyline" in svg
    assert theme.GRAY in svg
    assert theme.GREEN not in svg and theme.RED not in svg
    assert "came back" in svg
    assert "held" not in svg


def test_a_resample_that_flattened_a_reverted_move_still_reads_as_moved():
    """``series`` is a fixed-width resample, so a move that reverted between two ticks
    leaves it looking constant. ``n_moves`` is counted before resampling and still fires."""
    assert "held" not in movement_svg([2.5, 2.5, 2.5], bet="Over", n_moves=4)


def test_movement_title_carries_open_close_and_signed_delta():
    assert "Line 209.5 → 213.5 (+4)" in movement_svg([209.5, 213.5], bet="Over", n_moves=1)


def test_form_counts_hits_at_or_above_the_line():
    assert "2/5 over 13.5" in form_svg([12, 8, 15, 9, 20], 13.5)
    assert "1/1 over 3" in form_svg([3], 3)


def test_form_draws_a_bar_per_game_and_never_a_trace():
    """Five games are five discrete results, not a series. Every gamelog value is a whole
    number and every DFS line sits at a half, so a stroke connecting them would slope
    through counts that cannot occur — the reason this stopped being a sparkline."""
    svg = form_svg([12, 8, 15, 9, 20], 13.5)
    assert "polyline" not in svg
    assert len(_rects(svg)) == 5


def test_form_puts_clears_above_the_rule_and_misses_below():
    """SVG y grows downward, so a clearing bar's top edge sits above the centre rule and a
    missing bar starts at it. The whole read of the chart is this one relation."""
    mid = _HEIGHT / 2
    assert _rule_y(form_svg([10.0, 20.0], 15.0)) == mid

    (miss, clear) = _rects(form_svg([10.0, 20.0], 15.0))
    assert miss["y"] == mid and miss["y"] + miss["height"] > mid
    assert clear["y"] < mid and clear["y"] + clear["height"] == mid


def test_form_colors_the_bar_by_whether_the_game_cleared():
    """The regression that motivated the redesign: the old trace was unconditionally green
    whatever the games did, so its color carried no intent (DESIGN.md §2)."""
    mixed = form_svg([10.0, 20.0], 15.0)
    assert {r["fill"] for r in _rects(mixed)} == {theme.GREEN, theme.RED}

    assert {r["fill"] for r in _rects(form_svg([20.0, 30.0], 15.0))} == {theme.GREEN}
    assert {r["fill"] for r in _rects(form_svg([1.0, 2.0], 15.0))} == {theme.RED}


def test_form_rule_is_fixed_at_the_centre_whatever_the_games_did():
    """Unlike the old scaled rule, the line is the frame of reference and does not move —
    so two cards can be compared to each other, not just read one at a time."""
    assert _rule_y(form_svg([0, 0, 0, 0, 0], 0.5)) == _HEIGHT / 2
    assert _rule_y(form_svg([80, 92, 41, 103, 66], 41.5)) == _HEIGHT / 2


def test_form_scales_so_the_widest_margin_spans_the_half_box():
    reach = _HEIGHT / 2 - _INSET
    bars = _rects(form_svg([0, 4, 1, 3, 0], 1.5))
    assert max(r["height"] for r in bars) == reach

    # A quarter of real cards miss (or clear) by the same margin every game — five equal
    # bars is the honest picture of that, not a flat line.
    equal = _rects(form_svg([0, 0, 0, 0, 0], 0.5))
    assert {round(r["height"], 6) for r in equal} == {reach}


def test_form_bars_stay_inside_the_box():
    for svg in (form_svg([0, 4, 1, 3, 0], 1.5), form_svg([8, 70, 6, 36, 12], 41.5)):
        for rect in _rects(svg):
            assert 0 <= rect["y"] <= _HEIGHT
            assert 0 <= rect["y"] + rect["height"] <= _HEIGHT
            assert rect["height"] > 0


def test_form_games_all_on_the_line_keep_the_bare_rule():
    """No deviation to draw. Same answer shape as an unmoved line in ``movement_svg``."""
    svg = form_svg([3, 3], 3)
    assert not _rects(svg)
    assert "<line" in svg


def test_points_span_the_box_and_stay_inside_it():
    xs, ys = zip(*_points(movement_svg([1.0, 2.0, 3.0, 4.0], bet="Over", n_moves=1)), strict=True)
    assert (xs[0], xs[-1]) == (2.0, 76.0)
    # Coordinates are emitted at 4 significant figures, so equal gaps differ in the
    # second decimal; 1dp is well inside a 78px box and still catches real unevenness.
    gaps = {round(b - a, 1) for a, b in pairwise(xs)}
    assert len(gaps) == 1, f"x spacing is uneven: {xs}"
    assert (min(ys), max(ys)) == (2.0, 20.0)


def test_empty_series_render_nothing():
    assert movement_svg([], bet="Over", n_moves=1) == ""
    assert form_svg([], 1.0) == ""


def test_every_hex_is_a_theme_token():
    """DESIGN.md §2: semantic colors by intent, and gold is chrome — never a data mark."""
    allowed = {theme.GREEN, theme.RED, theme.GRAY, theme.BORDER}
    svgs = [
        movement_svg([1.0, 2.0], bet="Over", n_moves=1),
        movement_svg([2.0, 1.0], bet="Over", n_moves=1),
        movement_svg([1.0], bet="Over", n_moves=1),
        form_svg([1.0, 2.0], 1.5),
    ]
    for svg in svgs:
        assert set(_HEXES.findall(svg)) <= allowed
        assert theme.GOLD not in svg


def test_no_centering_leaks_into_a_grid_cell():
    """``test_grid_options`` bans the substring "center" across the whole dumped
    gridOptions to catch centered numerics; a renderer built from these strings must not
    be what trips it."""
    assert "center" not in form_svg([1.0, 2.0], 1.5)
    assert "center" not in movement_svg([1.0, 2.0], bet="Over", n_moves=1)
