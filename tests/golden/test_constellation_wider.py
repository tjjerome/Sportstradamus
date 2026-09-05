"""Pins for the constellation's *look wider* lens — other games, in the sky around it.

*Look wider* recedes the map slightly and scatters other games' best legs through
whatever open sky is left, clustered by game and team-coloured, never inside the
constellation's own footprint and never as the ring both lenses used to draw.
These assert that geometry: the sky's clustering, colouring, seeding, labels and cap.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import math

import pandas as pd

from sportstradamus.dashboard.components import constellation_deep, constellation_wider
from sportstradamus.dashboard.components.constellation import (
    _FIG_HEIGHT,
    _LABEL_FONT_SIZE,
    constellation_figure,
)
from sportstradamus.dashboard.components.constellation_spacing import (
    _FRAME_INSET,
    DEFAULT_STARS,
    PX_PER_UNIT,
    Y_RANGE,
)
from sportstradamus.dashboard.components.constellation_wider import (
    WIDER_GAMES,
    WIDER_STAR_SIZE,
)
from sportstradamus.dashboard.theme import team_colors

_GAME = "NYK/SAS"


def _row(player: str, team: str, kelly: float, *, market: str = "PTS", game: str = _GAME) -> dict:
    return {
        "Player": player,
        "Market": market,
        "Bet": "Over",
        "Line": 10.5,
        "Game": game,
        "League": "NBA",
        "Team": team,
        "Kelly": kelly,
        "Win Prob": 0.6,
        "Boost": 1.5,
    }


def _key(player: str, market: str = "PTS") -> str:
    return f"{player}|{market}|Over"


def _ladder(n: int, *, teams: tuple[str, str] = ("NYK", "SAS")) -> pd.DataFrame:
    return pd.DataFrame([_row(f"P{i:02d}", teams[i % 2], 0.9 - i * 0.03) for i in range(n)])


def _deep_pool(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [_row(f"Q{i:03d}", ("NYK", "SAS")[i % 2], -0.05 - i / 2000) for i in range(n)]
    )


def _wider_groups(n: int, *, per_game: int = 4) -> list[tuple[str, list[dict]]]:
    matchups = [
        ("MIA/ORL", "MIA"),
        ("BOS/PHI", "BOS"),
        ("LAL/GSW", "LAL"),
        ("DAL/HOU", "DAL"),
        ("DEN/PHX", "DEN"),
        ("MIL/CHI", "MIL"),
        ("ATL/CLE", "ATL"),
    ][:n]
    return [
        (
            game,
            [_row(f"{team}{i}", team, 0.3, market="3PM", game=game) for i in range(per_game)],
        )
        for game, team in matchups
    ]


def _trace(fig, name: str):
    return next((t for t in fig.data if getattr(t, "name", None) == name), None)


def _apart(one, other, px) -> float:
    return math.hypot((one[0] - other[0]) * px[0], (one[1] - other[1]) * px[1])


def test_wider_stars_cluster_by_game_and_wear_team_colors():
    groups = _wider_groups(2, per_game=3)
    fig = constellation_figure([], None, _ladder(13), wider_groups=groups)
    wider = _trace(fig, "wider")
    at = {
        card[0]: (float(x), float(y))
        for card, x, y in zip(wider.customdata, wider.x, wider.y, strict=True)
    }
    clusters = [[_key(row["Player"], "3PM") for row in rows] for _, rows in groups]
    intra = max(
        _apart(at[one], at[other], PX_PER_UNIT)
        for cluster in clusters
        for one, other in itertools.combinations(cluster, 2)
    )
    inter = min(
        _apart(at[one], at[other], PX_PER_UNIT) for one, other in itertools.product(*clusters)
    )
    assert intra < inter
    colors = dict(zip((card[0] for card in wider.customdata), wider.marker.color, strict=True))
    for game, rows in groups:
        for row in rows:
            assert colors[_key(row["Player"], "3PM")] == team_colors("NBA", row["Team"])[0], game


def test_wider_sky_is_seeded_by_md5_not_hash():
    """``hash()`` on a ``str`` is per-process randomized, which would unpin every
    position here between two test runs."""
    for module in (constellation_deep, constellation_wider):
        called = {
            node.func.id
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "hash" not in called, module.__name__
    groups = _wider_groups(3)
    one = constellation_figure([], None, _ladder(13), wider_groups=groups)
    two = constellation_figure([], None, _ladder(13), wider_groups=groups)
    assert one.to_json() == two.to_json()


def test_wider_keeps_its_game_labels():
    """A label sits under its cluster, or over it where the frame leaves no room
    below — a label clipped by the frame names nothing."""
    for mobile in (False, True):
        for count in (3, 5, WIDER_GAMES):
            groups = _wider_groups(count)
            fig = constellation_figure([], None, _ladder(13), wider_groups=groups, mobile=mobile)
            labels = _trace(fig, "wider_labels")
            assert set(labels.text) == {game for game, _ in groups}
            bound = float(fig.layout.yaxis.range[1]) * _FRAME_INSET
            assert max(abs(float(y)) for y in labels.y) <= bound, (mobile, count)


def test_a_deep_tier_that_closes_the_sky_grows_it_instead_of_drawing_nothing():
    """A tier deep enough to reach the frame leaves no band wide enough for a
    cluster, and the whole wider layer used to vanish with no feedback — on the
    phone first, whose map already spans the width, and on the desktop once the
    tier spreads past the side inset. Both grow in y instead. The owner wants
    every leg reachable under the deeper lens, so the tier itself is never capped."""
    groups = _wider_groups(3)
    for mobile, deep in ((True, 100), (False, 280)):
        sky_only = constellation_figure([], None, _ladder(13), wider_groups=groups, mobile=mobile)
        both = constellation_figure(
            [], None, _ladder(13), deep_pool=_deep_pool(deep), wider_groups=groups, mobile=mobile
        )
        assert len(_trace(both, "deep").x) > DEFAULT_STARS
        assert len(_trace(both, "wider").x) == len(_trace(sky_only, "wider").x), mobile
        assert set(_trace(both, "wider_labels").text) == {game for game, _ in groups}
        assert both.layout.height > sky_only.layout.height, mobile


def _sky_boxes(fig) -> tuple[list[tuple], list[tuple]]:
    """The sky's label ink boxes and star boxes in px, as ``(game, x0, y0, x1, y1)``.

    A plotly text label is centred on its point; 0.6 em a character is the usual
    estimate for a proportional face and 1.25 its line height.
    """
    text_trace, sky = _trace(fig, "wider_labels"), _trace(fig, "wider")
    labels, stars = [], []
    for text, x, y in zip(text_trace.text, text_trace.x, text_trace.y, strict=True):
        half_w, half_h = len(text) * 0.6 * _LABEL_FONT_SIZE / 2, 1.25 * _LABEL_FONT_SIZE / 2
        cx, cy = float(x) * PX_PER_UNIT[0], float(y) * PX_PER_UNIT[1]
        labels.append((text, cx - half_w, cy - half_h, cx + half_w, cy + half_h))
    for card, x, y in zip(sky.customdata, sky.x, sky.y, strict=True):
        cx, cy = float(x) * PX_PER_UNIT[0], float(y) * PX_PER_UNIT[1]
        half = WIDER_STAR_SIZE / 2
        stars.append((card[0].split("|")[0][:3], cx - half, cy - half, cx + half, cy + half))
    return labels, stars


def _clear(one: tuple, other: tuple) -> bool:
    return one[3] <= other[1] or other[3] <= one[1] or one[4] <= other[2] or other[4] <= one[2]


def test_sky_labels_never_land_on_another_games_group():
    """A vertical band stacks its games, and a label hangs below its own. With the
    slots spaced on the cluster alone the label fell into the group beneath it —
    measured on the desktop at two labels 10.6 px apart and a label over a
    neighbour's star by 8.7 px, both unreadable. The ordinary slate (two side
    bands) is pinned at the shipped six games; a single crammed band at three, the
    count where the reservation still leaves play — from four up the strip is
    consumed exactly and a settle nudge can still graze."""
    for names, deep, bands in (
        (["CIN/CLE", "LAA/LAD", "PHI/PIT", "ARI/ATH", "TOR/WSH", "MIN/NYM"], None, 2),
        (["CIN/CLE", "LAA/LAD", "PHI/PIT"], _deep_pool(190), 1),
    ):
        groups = [
            (
                game,
                [_row(f"{game[:3]}{i}", game[:3], 0.3, market="3PM", game=game) for i in range(4)],
            )
            for game in names
        ]
        fig = constellation_figure([], None, _ladder(13), deep_pool=deep, wider_groups=groups)
        labels, stars = _sky_boxes(fig)
        assert len({star[1] > 0 for star in stars}) == bands, "fixture no longer deals its bands"
        for one, other in itertools.combinations(labels, 2):
            assert _clear(one, other), (names, one[0], other[0])
        for label in labels:
            for star in stars:
                assert star[0] == label[0][:3] or _clear(label, star), (names, label[0], star[0])


def test_the_desktop_keeps_its_own_height_while_its_side_bands_hold():
    """The desktop's sky is the two side bands; growing in y is the last resort,
    not the default, so an ordinary slate must not reshape the figure."""
    for deep in (None, _deep_pool(40)):
        fig = constellation_figure(
            [], None, _ladder(13), deep_pool=deep, wider_groups=_wider_groups(3)
        )
        assert fig.layout.height == _FIG_HEIGHT
        assert tuple(fig.layout.yaxis.range) == (-Y_RANGE, Y_RANGE)


def test_only_the_best_wider_games_are_drawn():
    groups = _wider_groups(WIDER_GAMES + 1)
    fig = constellation_figure([], None, _ladder(13), wider_groups=groups)
    assert len(_trace(fig, "wider_labels").text) == WIDER_GAMES
    assert set(_trace(fig, "wider_labels").text) == {game for game, _ in groups[:WIDER_GAMES]}
