"""Pins for the constellation's "won't pair" mark — an orange × over a refused star.

Once the slip has legs, a star whose leg the platform refuses to pair with one of them
wears an ×. Every channel the star itself carries is taken, so the mark is an overlay
trace of its own. These assert that overlay: where the × sits and how big it draws, that
it keeps its host star's opacity so a lens × never outshines a main star, that it stays
inert to the pointer and draws over every star, and that no star trace changes under it.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.components.constellation import constellation_figure
from sportstradamus.dashboard.components.constellation_bans import BAN_MARK_MIN, BAN_MARK_SCALE
from sportstradamus.dashboard.theme import ORANGE

_GAME = "NYK/SAS"
_WIDER_GAME = "MIA/ORL"
_STAR_TRACES = ("active", "candidate", "deep", "wider")


def _row(
    player: str,
    team: str,
    kelly: float,
    *,
    market: str = "PTS",
    game: str = _GAME,
    bet: str = "Over",
) -> dict:
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": 10.5,
        "Game": game,
        "League": "NBA",
        "Team": team,
        "Kelly": kelly,
        "Win Prob": 0.6,
        "Boost": 1.5,
    }


def _key(player: str, market: str = "PTS", bet: str = "Over") -> str:
    return f"{player}|{market}|{bet}"


def _banned(*keys: str) -> dict[str, str]:
    return dict.fromkeys(keys, "Sleeper won't pair this with A Points Over 10.5")


def _figure(**kwargs):
    """Slip leg ``A``, candidates ``B``/``C``, both lenses on — a star in every star trace.

    The deep tier holds one liked and one passed leg so its per-point opacities differ.
    """
    pool = pd.DataFrame([_row("A", "NYK", 0.4), _row("B", "SAS", 0.3), _row("C", "NYK", 0.2)])
    deep_pool = pd.DataFrame([_row("L", "NYK", 0.1), _row("D", "SAS", -0.1)])
    sky = [
        _row("W0", "MIA", 0.3, market="3PM", game=_WIDER_GAME),
        _row("W1", "MIA", 0.25, market="3PM", game=_WIDER_GAME, bet="Under"),
    ]
    return constellation_figure(
        [_row("A", "NYK", 0.4)],
        None,
        pool,
        deep_pool=deep_pool,
        wider_groups=[(_WIDER_GAME, sky)],
        **kwargs,
    )


def _trace(fig, name: str):
    return next((t for t in fig.data if t.name == name), None)


def _index(trace, key: str) -> int:
    return [card[0] for card in trace.customdata].index(key)


def test_no_bans_draw_no_mark_and_leave_the_figure_byte_identical():
    """The figure calls for marks unconditionally, so an empty slip must cost nothing."""
    fig = _figure(banned={})
    assert [t.name for t in fig.data if t.name.endswith("_banned")] == []
    assert fig.to_json() == _figure().to_json()


def test_a_banned_candidate_is_crossed_where_it_stands():
    """Only the refused star is crossed: at its own position, scaled off its own size, no
    brighter than the dimmed candidate beneath it. The × carries no customdata and skips
    hover, so ``main.js`` and plotly's closest-point pick both land on the star under it."""
    fig = _figure(banned=_banned(_key("B")))
    host, mark = _trace(fig, "candidate"), _trace(fig, "candidate_banned")
    at = _index(host, _key("B"))
    assert len(host.x) == 2
    assert (list(mark.x), list(mark.y)) == ([host.x[at]], [host.y[at]])
    assert list(mark.marker.size) == [max(BAN_MARK_SCALE * host.marker.size[at], BAN_MARK_MIN)]
    assert mark.marker.symbol == "x-thin"
    assert mark.marker.line.color == ORANGE
    assert mark.marker.opacity == host.marker.opacity
    assert mark.customdata is None
    assert mark.hoverinfo == "skip"


def test_a_banned_slip_leg_is_crossed_on_the_lit_map():
    fig = _figure(banned=_banned(_key("A")))
    host, mark = _trace(fig, "active"), _trace(fig, "active_banned")
    assert (list(mark.x), list(mark.y)) == (list(host.x), list(host.y))
    assert mark.marker.opacity == host.marker.opacity


def test_a_deep_mark_keeps_its_stars_own_dim():
    """DESIGN §4a holds every lens star under the main-star floor, so the × over one takes
    that star's own per-point opacity. The passed star sits at the lens's size floor, where
    a scaled × would be a speck, so the mark's floor is what sizes it."""
    fig = _figure(banned=_banned(_key("D")))
    host, mark = _trace(fig, "deep"), _trace(fig, "deep_banned")
    at = _index(host, _key("D"))
    assert len(set(host.marker.opacity)) == 2
    assert list(mark.marker.opacity) == [host.marker.opacity[at]]
    assert list(mark.marker.size) == [BAN_MARK_MIN]


def test_a_wider_mark_recedes_with_the_sky():
    """The sky dims at trace level, so its × takes the same trace opacity."""
    fig = _figure(banned=_banned(_key("W1", "3PM", "Under")))
    host, mark = _trace(fig, "wider"), _trace(fig, "wider_banned")
    assert host.opacity < 1
    assert mark.opacity == host.opacity


def test_marks_leave_every_star_trace_as_it_was_and_draw_over_them():
    """A ban adds traces and never edits one — fill, size, the Over/Under border and the
    card fields all read as they do without it — and every mark follows every star in
    draw order, so it lands on top."""
    keys = (_key("A"), _key("B"), _key("D"), _key("W1", "3PM", "Under"))
    plain, crossed = _figure(), _figure(banned=_banned(*keys))
    for name in _STAR_TRACES:
        assert _trace(crossed, name).to_plotly_json() == _trace(plain, name).to_plotly_json(), name
    names = [t.name for t in crossed.data]
    marks = [i for i, name in enumerate(names) if name.endswith("_banned")]
    assert len(marks) == len(_STAR_TRACES)
    assert max(names.index(name) for name in _STAR_TRACES) < min(marks)
