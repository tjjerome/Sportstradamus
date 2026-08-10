"""Pins for ``dashboard.components.glyphs`` — the unified game-shape glyph set.

Five bespoke celestial SVGs (comet/supernova/scales/lone-star/hourglass), one per
``current_game_context.shape`` value; unknown/empty shapes fall back to the lone
star. No Streamlit, no I/O — pure string construction, so it unit-tests directly.
"""

from __future__ import annotations

import re

from sportstradamus.dashboard.components.glyphs import _GLYPHS, game_shape_glyph

_SHAPES = ["shootout", "blowout", "coinflip", "even", "grind"]


def test_all_five_shapes_resolve_to_svg():
    for shape in _SHAPES:
        assert "<svg" in game_shape_glyph(shape)


def test_unknown_shape_falls_back_to_lone_star():
    assert game_shape_glyph("nonsense") == game_shape_glyph("even")


def test_empty_shape_falls_back_to_lone_star():
    assert game_shape_glyph("") == game_shape_glyph("even")


def test_gradient_ids_unique_across_all_glyphs():
    seen: list[str] = []
    for shape in _SHAPES:
        ids = re.findall(r'id="([^"]+)"', _GLYPHS[shape])
        seen.extend(ids)
    assert len(seen) == len(set(seen)), f"duplicate ids across glyph set: {seen}"


def test_size_overrides_width_and_height_but_not_viewbox():
    svg = game_shape_glyph("shootout", size=24)
    assert 'width="24" height="24"' in svg
    assert 'viewBox="0 0 60 60"' in svg


def test_default_size_is_40():
    svg = game_shape_glyph("shootout")
    assert 'width="40" height="40"' in svg
