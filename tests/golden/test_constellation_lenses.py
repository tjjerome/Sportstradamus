"""Pins for the constellation's two lenses: deeper inside the map, wider around it.

*Look deeper* fades the game's remaining legs in as small stars **inside** the
constellation — each beside the main star it correlates with, or in its own
team's open space — with their ties drawn and no main star moved. *Look wider*
recedes the map slightly and scatters other games' best legs through whatever
open sky is left, clustered by game and team-coloured, never inside the
constellation's own footprint and never as the ring both lenses used to draw.
These assert that geometry: the ties a deep star follows, the half it may not
leave, the clearance and frame it keeps, the cut it makes visible, and the
sky's clustering, colouring, seeding, labels and cap.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import math

import pandas as pd

from sportstradamus.dashboard.components import constellation_lenses
from sportstradamus.dashboard.components.constellation import (
    _SIZE_MIN,
    constellation_figure,
)
from sportstradamus.dashboard.components.constellation_lenses import (
    DEEP_EDGES_PER_STAR,
    WIDER_GAMES,
)
from sportstradamus.dashboard.components.constellation_slate import DECORATION
from sportstradamus.dashboard.components.constellation_spacing import (
    _FRAME_INSET,
    _STAR_GAP_PX,
    DEFAULT_STARS,
    PX_PER_UNIT,
    X_RANGE,
    Y_RANGE,
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


def _corr(*triples: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"League": "NBA", "Game": _GAME, "leg_a": a, "leg_b": b, "rho": r} for a, b, r in triples]
    )


def _ladder(n: int, *, teams: tuple[str, str] = ("NYK", "SAS")) -> pd.DataFrame:
    return pd.DataFrame([_row(f"P{i:02d}", teams[i % 2], 0.9 - i * 0.05) for i in range(n)])


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


def _stars(fig, *names: str) -> dict[str, tuple[tuple[float, float], float]]:
    """key -> (position, marker px) over the named traces (all star traces by default)."""
    wanted = set(names) or None
    out = {}
    for trace in fig.data:
        if trace.customdata is None or trace.name == DECORATION:
            continue
        if wanted is not None and trace.name not in wanted:
            continue
        for card, x, y, size in zip(
            trace.customdata, trace.x, trace.y, trace.marker.size, strict=True
        ):
            out[card[0]] = ((float(x), float(y)), float(size))
    return out


def _apart(one, other, px) -> float:
    return math.hypot((one[0] - other[0]) * px[0], (one[1] - other[1]) * px[1])


# --- Look deeper ----------------------------------------------------------------


def test_a_correlated_deep_star_sits_beside_its_tie_not_the_other_main_star():
    """A tie is what places a deep star: it lands next to the leg it moves with."""
    pool = pd.DataFrame([_row("A", "NYK", 0.4), _row("B", "NYK", 0.3), _row("Z", "SAS", 0.35)])
    deep_pool = pd.DataFrame([_row("D", "NYK", -0.1)])
    fig = constellation_figure([], _corr((_key("A"), _key("D"), 0.6)), pool, deep_pool=deep_pool)
    stars = _stars(fig)
    deep = stars[_key("D")][0]
    assert _apart(deep, stars[_key("A")][0], PX_PER_UNIT) < _apart(
        deep, stars[_key("B")][0], PX_PER_UNIT
    )


def test_uncorrelated_deep_stars_stay_in_their_teams_half():
    """A star that drifted across the team axis would read as the other team's leg."""
    pool = pd.DataFrame([_row("A", "NYK", 0.4), _row("Z", "SAS", 0.35)])
    deep_pool = pd.DataFrame(
        [_row(f"N{i}", "NYK", -0.1 - i / 100) for i in range(3)]
        + [_row(f"S{i}", "SAS", -0.2 - i / 100) for i in range(3)]
    )
    stars = _stars(constellation_figure([], None, pool, deep_pool=deep_pool), "deep")
    for key, ((x, _), _) in stars.items():
        assert (x <= 0) if key.startswith("N") else (x >= 0), (key, x)


def test_deep_stars_clear_every_main_star():
    pool = pd.DataFrame([_row("A", "NYK", 0.4), _row("B", "SAS", 0.3), _row("C", "NYK", 0.2)])
    deep_pool = pd.DataFrame([_row(f"D{i}", ("NYK", "SAS")[i % 2], -0.1) for i in range(4)])
    stars = _stars(constellation_figure([], None, pool, deep_pool=deep_pool))
    for one, other in itertools.combinations(sorted(stars), 2):
        (pos_a, size_a), (pos_b, size_b) = stars[one], stars[other]
        gap = _apart(pos_a, pos_b, PX_PER_UNIT) - (size_a + size_b) / 2
        assert gap >= _STAR_GAP_PX - 1e-9, (one, other, gap)


def test_deep_stars_never_leave_the_frame():
    pool = pd.DataFrame([_row("A", "NYK", 0.4), _row("Z", "SAS", 0.35)])
    deep_pool = pd.DataFrame(
        [_row(f"D{i:02d}", ("NYK", "SAS")[i % 2], -0.1 - i / 100) for i in range(24)]
    )
    for (x, y), _ in _stars(constellation_figure([], None, pool, deep_pool=deep_pool)).values():
        assert abs(x) <= X_RANGE * _FRAME_INSET and abs(y) <= Y_RANGE * _FRAME_INSET


def test_liked_legs_beyond_the_cut_are_drawn_only_under_the_lens():
    """The cut is a display decision, not a verdict — the lens is where the rest live."""
    pool = _ladder(15)
    lens_off = _stars(constellation_figure([], None, pool))
    assert len(lens_off) == DEFAULT_STARS
    fig = constellation_figure([], None, pool, deep_pool=pool)
    assert len(_stars(fig)) == 15
    deep = _stars(fig, "deep")
    assert set(deep) == {_key(f"P{i}") for i in (12, 13, 14)}
    assert all(size < _SIZE_MIN for _, size in deep.values())


def test_untied_deep_stars_reach_every_main_star_on_their_side():
    """The owner's "don't let a small group dominate": an untied star takes its own
    side's mains in round-robin, so a full tier spreads over all of them rather
    than piling around the handful a global rank happens to land on."""
    pool = _ladder(12)
    deep_pool = pd.DataFrame(
        [_row(f"Q{i:02d}", ("NYK", "SAS")[i % 2], -0.1 - i / 100) for i in range(16)]
    )
    fig = constellation_figure([], None, pool, deep_pool=deep_pool)
    mains = {key: pos for key, (pos, _) in _stars(fig).items() if key.startswith("P")}
    nearest = {"NYK": set(), "SAS": set()}
    for key, (pos, _) in _stars(fig, "deep").items():
        side = ("NYK", "SAS")[int(key[1:3]) % 2]
        nearest[side].add(min(mains, key=lambda m: _apart(pos, mains[m], PX_PER_UNIT)))
    assert {len(hit) for hit in nearest.values()} == {DEFAULT_STARS // 2}


def test_a_deep_star_shows_only_its_strongest_ties():
    """Every tie a deep star has would fan a dozen lines off a 10 px star; the two
    that placed it are the reading."""
    mains = pd.DataFrame(
        [_row(c, ("NYK", "SAS")[i % 2], 0.5 - i / 100) for i, c in enumerate("ABCDE")]
    )
    corr = _corr(*[(_key(c), _key("Z"), 0.6 - i / 20) for i, c in enumerate("ABCDE")])
    fig = constellation_figure([], corr, mains, deep_pool=pd.DataFrame([_row("Z", "NYK", -0.1)]))
    drawn = [trace for trace in fig.data if trace.name == "deep_edge"]
    assert len(drawn) == DEEP_EDGES_PER_STAR
    assert {tuple(trace.meta[:2]) for trace in drawn} == {
        (_key("A"), _key("Z")),
        (_key("B"), _key("Z")),
    }


def test_a_promoted_star_holds_its_place_when_the_lens_opens():
    """A slip leg beyond the cut is already lit with the lens shut, so opening the
    lens may grow the tier it is ranked among but not move the star itself."""
    pool = pd.DataFrame([_row("A", "NYK", 0.4), _row("Z", "SAS", 0.35)])
    picked = _row("D", "NYK", -0.4)
    deep_pool = pd.DataFrame(
        [picked] + [_row(f"Q{i:02d}", ("NYK", "SAS")[i % 2], -0.1) for i in range(10)]
    )
    shut = _stars(constellation_figure([picked], None, pool))
    opened = _stars(constellation_figure([picked], None, pool, deep_pool=deep_pool))
    assert len(opened) == len(shut) + 10
    assert opened[_key("D")] == shut[_key("D")]


def test_adding_a_deep_star_moves_no_main_star():
    pool = _ladder(6)
    corr = _corr((_key("P00"), _key("P01"), 0.5))
    deep_pool = pd.DataFrame([_row("D", "NYK", -0.1), _row("E", "SAS", -0.3)])
    plain = _stars(constellation_figure([], corr, pool))
    lensed = _stars(constellation_figure([], corr, pool, deep_pool=deep_pool))
    for key, (pos, size) in plain.items():
        assert lensed[key] == (pos, size)


# --- Look wider -----------------------------------------------------------------


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
    called = {
        node.func.id
        for node in ast.walk(ast.parse(inspect.getsource(constellation_lenses)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "hash" not in called
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


def test_the_phone_grows_its_sky_past_a_deep_tier():
    """The phone has no side band, so a deep tier tall enough to squeeze its two
    y-bands under the cluster floor used to take the whole wider layer with it."""
    markets = ("PTS", "REB", "AST", "3PM", "STL")
    deep_pool = pd.DataFrame(
        [
            _row(f"Q{i:02d}", ("NYK", "SAS")[i % 2], -0.05 - i / 500, market=markets[i % 5])
            for i in range(50)
        ]
    )
    groups = _wider_groups(3)
    sky_only = constellation_figure([], None, _ladder(13), wider_groups=groups, mobile=True)
    both = constellation_figure(
        [], None, _ladder(13), deep_pool=deep_pool, wider_groups=groups, mobile=True
    )
    assert len(_trace(both, "deep").x) > DEFAULT_STARS
    assert len(_trace(both, "wider").x) == len(_trace(sky_only, "wider").x)
    assert set(_trace(both, "wider_labels").text) == {game for game, _ in groups}
    assert both.layout.height > sky_only.layout.height


def test_only_the_best_wider_games_are_drawn():
    groups = _wider_groups(WIDER_GAMES + 1)
    fig = constellation_figure([], None, _ladder(13), wider_groups=groups)
    assert len(_trace(fig, "wider_labels").text) == WIDER_GAMES
    assert set(_trace(fig, "wider_labels").text) == {game for game, _ in groups[:WIDER_GAMES]}
