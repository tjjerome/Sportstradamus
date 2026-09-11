"""Pins for the constellation's *look deeper* lens — the remaining legs, inside the map.

*Look deeper* fades the game's remaining legs in as small stars **inside** the
constellation — each scattered into the neighbourhood of the main star it
correlates with, or of one in its own team's open space — with their ties drawn
and no main star moved. These assert that geometry: the ties a deep star follows,
the half it may not leave, the clearance and frame it keeps, the scatter that
keeps the tier off ``settle``'s lattice, and the cut it makes visible.
"""

from __future__ import annotations

import itertools
import math

import pandas as pd

from sportstradamus.dashboard.components.constellation import constellation_figure
from sportstradamus.dashboard.components.constellation_deep import (
    DEEP_ALPHA_MIN,
    DEEP_EDGES_PER_STAR,
    DEEP_SIZE_MIN,
)
from sportstradamus.dashboard.components.constellation_slate import DECORATION
from sportstradamus.dashboard.components.constellation_spacing import (
    _CELL_PX,
    _FRAME_INSET,
    _STAR_GAP_PX,
    DEFAULT_STARS,
    PX_PER_UNIT,
    X_RANGE,
    Y_RANGE,
)
from sportstradamus.dashboard.components.constellation_traces import _SIZE_MAX, _SIZE_MIN

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
    return pd.DataFrame([_row(f"P{i:02d}", teams[i % 2], 0.9 - i * 0.03) for i in range(n)])


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


def test_a_deep_star_stays_nearer_its_main_than_any_other():
    """The scatter is bounded by the reading it must not break. A star that drifted
    past halfway to the next main would read as *that* star's satellite, so the throw
    band stops inside half the gap and the tie survives however the dart falls."""
    tied = dict(zip("012345", "ABCDEF", strict=True))
    pool = pd.DataFrame(
        [_row(main, ("NYK", "SAS")[i % 2], 0.5 - i / 100) for i, main in enumerate(tied.values())]
    )
    deep_pool = pd.DataFrame(
        [_row(f"D{i}", ("NYK", "SAS")[i % 2], -0.1 - i / 100) for i in range(len(tied))]
    )
    corr = _corr(*[(_key(main), _key(f"D{i}"), 0.6) for i, main in tied.items()])
    stars = _stars(constellation_figure([], corr, pool, deep_pool=deep_pool))
    mains = {_key(main): stars[_key(main)][0] for main in tied.values()}
    for i, main in tied.items():
        deep = stars[_key(f"D{i}")][0]
        assert min(mains, key=lambda m: _apart(deep, mains[m], PX_PER_UNIT)) == _key(main)


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


def _on_lattice(value: float, origin: float) -> bool:
    """Is ``value`` px on one of ``settle``'s cell lines, counting from ``origin``?"""
    offset = (value - origin) % _CELL_PX
    return min(offset, _CELL_PX - offset) < 1e-6


def test_deep_stars_do_not_sit_on_the_lattice():
    """The grid the owner saw: every deep star snapped to a 6 px cell drew concentric
    rings of identical dots around each main. A star with room now keeps its own
    seeded throw, and only the crowded remainder falls back to the lattice."""
    pool = pd.DataFrame([_row("A", "NYK", 0.4), _row("Z", "SAS", 0.35)])
    deep_pool = pd.DataFrame(
        [_row(f"D{i:02d}", ("NYK", "SAS")[i % 2], -0.1 - i / 100) for i in range(24)]
    )
    stars = _stars(constellation_figure([], None, pool, deep_pool=deep_pool), "deep")
    origin = (-X_RANGE * _FRAME_INSET * PX_PER_UNIT[0], -Y_RANGE * _FRAME_INSET * PX_PER_UNIT[1])
    snapped = [
        key
        for key, ((x, y), _) in stars.items()
        if _on_lattice(x * PX_PER_UNIT[0], origin[0]) and _on_lattice(y * PX_PER_UNIT[1], origin[1])
    ]
    assert len(stars) == 24
    assert len(snapped) < len(stars) / 2, snapped


def test_deep_star_size_and_opacity_follow_edge():
    """Edge reads twice on a lens star — a stronger leg is both bigger and brighter —
    so the tier carries its own ranking at a volume the main map still wins. Every
    model-passed leg sits at the floor of both, however far under zero its edge is."""
    pool = pd.DataFrame([_row("A", "NYK", 0.4), _row("Z", "SAS", 0.35)])
    weakest_first = [("X1", -5.0), ("X0", -0.1), ("L2", 0.1), ("L1", 0.2), ("L0", 0.3)]
    deep_pool = pd.DataFrame(
        [_row(name, ("NYK", "SAS")[i % 2], kelly) for i, (name, kelly) in enumerate(weakest_first)]
    )
    fig = constellation_figure([], None, pool, deep_pool=deep_pool)
    trace = next(t for t in fig.data if t.name == "deep")
    size = dict(zip((cd[0] for cd in trace.customdata), trace.marker.size, strict=True))
    alpha = dict(zip((cd[0] for cd in trace.customdata), trace.marker.opacity, strict=True))
    by_edge = [_key(name) for name, _ in weakest_first]
    assert [size[key] for key in by_edge] == sorted(size[key] for key in by_edge)
    assert [alpha[key] for key in by_edge] == sorted(alpha[key] for key in by_edge)
    assert size[_key("X1")] == size[_key("X0")] == DEEP_SIZE_MIN
    assert alpha[_key("X1")] == alpha[_key("X0")] == DEEP_ALPHA_MIN


def test_liked_legs_beyond_the_cut_are_drawn_only_under_the_lens():
    """The cut is a display decision, not a verdict — the lens is where the rest live."""
    pool = _ladder(DEFAULT_STARS + 3)
    lens_off = _stars(constellation_figure([], None, pool))
    assert len(lens_off) == DEFAULT_STARS
    fig = constellation_figure([], None, pool, deep_pool=pool)
    assert len(_stars(fig)) == DEFAULT_STARS + 3
    deep = _stars(fig, "deep")
    assert set(deep) == {_key(f"P{i}") for i in range(DEFAULT_STARS, DEFAULT_STARS + 3)}
    assert all(size < _SIZE_MIN for _, size in deep.values())


def _borrowed_mains(fig) -> dict[str, set[str]]:
    """Per team, the main stars the drawn deep stars ended up nearest to."""
    mains = {key: pos for key, (pos, _) in _stars(fig).items() if key.startswith("P")}
    borrowed: dict[str, set[str]] = {"NYK": set(), "SAS": set()}
    for key, (pos, _) in _stars(fig, "deep").items():
        borrowed[("NYK", "SAS")[int(key[1:3]) % 2]].add(
            min(mains, key=lambda main: _apart(pos, mains[main], PX_PER_UNIT))
        )
    return borrowed


def test_untied_deep_stars_spread_over_their_sides_mains():
    """The owner's "don't let a small group dominate": an untied star draws a main
    from its own half by key, so a tier spreads over most of the half rather than
    piling around the few a running rank happens to reach (which was three of six).
    An md5 draw collides, so this is a floor, not a sweep."""
    deep_pool = pd.DataFrame(
        [_row(f"Q{i:02d}", ("NYK", "SAS")[i % 2], -0.1 - i / 100) for i in range(24)]
    )
    per_side = DEFAULT_STARS // 2
    borrowed = _borrowed_mains(
        constellation_figure([], None, _ladder(DEFAULT_STARS), deep_pool=deep_pool)
    )
    assert min(len(hit) for hit in borrowed.values()) > per_side // 2, borrowed


def test_promoting_a_deep_star_does_not_re_deal_the_tier():
    """Nothing animates a click, so a star that shifts when a neighbour is picked
    teleports. The draw is key-intrinsic for exactly this reason: promoting a star
    takes it out of the tier's placement order, and a running rank would have
    re-dealt every star behind it (measured: a 301 px jump across the map).

    What is left is the promoted star's own glyph growing to full size and nudging
    whoever it touches — a local settle, never a new main to orbit."""
    pool = _ladder(12)
    rows = [_row(f"D{i:02d}", ("NYK", "SAS")[i % 2], -0.1 - i / 100) for i in range(16)]
    deep_pool = pd.DataFrame(rows)
    before = _stars(constellation_figure([], None, pool, deep_pool=deep_pool))
    mains = {key: pos for key, (pos, _) in before.items() if key.startswith("P")}
    nearest = {
        key: min(mains, key=lambda main: _apart(pos, mains[main], PX_PER_UNIT))
        for key, (pos, _) in before.items()
    }
    for leg in rows:
        after = _stars(constellation_figure([leg], None, pool, deep_pool=deep_pool))
        picked = _key(leg["Player"])
        for key, (pos, _) in before.items():
            if key == picked or after[key][0] == pos:
                continue
            moved = after[key][0]
            assert nearest[key] == min(
                mains, key=lambda main: _apart(moved, mains[main], PX_PER_UNIT)
            ), (picked, key)
            assert _apart(pos, after[picked][0], PX_PER_UNIT) <= 3 * _SIZE_MAX, (picked, key)


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
