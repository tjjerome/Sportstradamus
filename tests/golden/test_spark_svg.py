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
from sportstradamus.dashboard.components.spark_svg import form_svg, movement_svg

_HEXES = re.compile(r"#[0-9A-Fa-f]{6}")


def _points(svg: str) -> list[tuple[float, float]]:
    match = re.search(r'points="([^"]+)"', svg)
    assert match, f"no polyline in {svg}"
    return [tuple(float(n) for n in pair.split(",")) for pair in match.group(1).split()]


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


def test_form_scales_its_line_rule_with_the_trace():
    """The rule shares the trace's domain, so a value above the line has to plot above
    the rule — the whole read of the chart depends on it (SVG y grows downward)."""
    svg = form_svg([10.0, 20.0], 15.0)
    rule_y = float(re.search(r'<line x1="0" y1="([\d.]+)"', svg).group(1))
    (_, low_y), (_, high_y) = _points(svg)
    assert high_y < rule_y < low_y


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
