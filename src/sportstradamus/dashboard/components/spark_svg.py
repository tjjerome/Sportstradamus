"""Inline-SVG micro-charts for the Board grid and the constellation hover card.

Both consumers need a finished ``<svg>`` string rather than a chart object: AG Grid 34
only renders markup handed to it through a cellRenderer's ``getGui`` DOM node, and the
constellation's hand-authored frontend drops the card's body straight into ``innerHTML``.
Keeping every coordinate here means neither surface writes chart geometry in JavaScript,
where no gate we run would read it.

The two share a box and a reference rule but not an encoding, because they describe
different things: an offer's fair line is one quantity moving through time and reads as a
trace, while a player's last games are discrete results and read as bars off the line. The
movement's wording lives here too (:func:`movement_summary`), so the trace's tooltip and
every other surface that describes a move say it the same way.

Colors are semantic by intent (DESIGN.md §2) — never gold, which is chrome.
"""

from __future__ import annotations

from collections.abc import Sequence

from sportstradamus.dashboard import theme

# The mockup's slot geometry (docs/mockups/p8-celestial-refresh.html): a 78x22 box with
# the trace inset far enough that a 1.6px stroke at either extreme is not half-clipped.
_WIDTH = 78.0
_HEIGHT = 22.0
_INSET = 2.0
_STROKE = 1.6
_RULE_DASH = "2 3"
# How much of its slot a form bar fills. At five games a slot is ~15px, so this leaves a
# gap wide enough to read as a separator without starving the bar itself.
_BAR_SLOT_SHARE = 0.62
# The fair line is derived from prices rather than posted, so it can carry float noise past
# the hundredths the snapshot rounds ``fair_move`` to; the wording stops at the same place.
_FAIR_DECIMALS = 2


def _scale_y(value: float, lo: float, hi: float) -> float:
    """``value`` on the box's vertical axis. SVG y grows downward, so ``hi`` lands low."""
    if hi == lo:
        return _HEIGHT / 2
    return _HEIGHT - _INSET - (value - lo) / (hi - lo) * (_HEIGHT - 2 * _INSET)


def _plot_points(values: Sequence[float], lo: float, hi: float) -> str:
    """``x,y`` pairs for a polyline over ``values`` scaled into the box's inset area."""
    if len(values) == 1:
        return f"{_WIDTH / 2:.4g},{_scale_y(values[0], lo, hi):.4g}"
    step = (_WIDTH - 2 * _INSET) / (len(values) - 1)
    return " ".join(
        f"{_INSET + i * step:.4g},{_scale_y(v, lo, hi):.4g}" for i, v in enumerate(values)
    )


def _svg(body: str, title: str) -> str:
    return (
        f'<svg viewBox="0 0 {_WIDTH:.0f} {_HEIGHT:.0f}" width="{_WIDTH:.0f}" '
        f'height="{_HEIGHT:.0f}" preserveAspectRatio="none">'
        f"<title>{title}</title>{body}</svg>"
    )


def _rule(y: float, color: str) -> str:
    return (
        f'<line x1="0" y1="{y:.4g}" x2="{_WIDTH:.0f}" y2="{y:.4g}" '
        f'stroke="{color}" stroke-dasharray="{_RULE_DASH}"/>'
    )


def _trace(points: str, color: str) -> str:
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        f'stroke-width="{_STROKE}" stroke-linejoin="round"/>'
    )


def _bars(deviations: Sequence[float], reach: float) -> str:
    """One bar per game off the box's centre line, oldest first.

    ``reach`` is the largest deviation on the card, so the game that missed or cleared by
    the most spans the half-box and every other bar reads against it.
    """
    mid = _HEIGHT / 2
    slot = (_WIDTH - 2 * _INSET) / len(deviations)
    width = slot * _BAR_SLOT_SHARE
    bars = []
    for i, deviation in enumerate(deviations):
        length = abs(deviation) / reach * (mid - _INSET)
        x = _INSET + i * slot + (slot - width) / 2
        y = mid - length if deviation >= 0 else mid
        color = theme.GREEN if deviation >= 0 else theme.RED
        bars.append(
            f'<rect x="{x:.4g}" y="{y:.4g}" width="{width:.4g}" '
            f'height="{length:.4g}" fill="{color}"/>'
        )
    return "".join(bars)


def move_color(delta: float, bet: str) -> str:
    """The semantic color of a fair-line move, read from the bet's side.

    A falling fair line means the app now prices the Over more generously — a lower line to
    clear, or a richer multiplier at the same line — and a rising one the Under. So a move
    toward the bet's side is green and a move away red; a move that netted to zero is gray,
    whatever path it took.

    Args:
        delta: Fair close minus fair open, in the stat's own units.
        bet: The offer's side; anything starting with "o" (any case) is the Over.

    Returns:
        ``theme.GREEN``, ``theme.RED``, or ``theme.GRAY`` at zero.
    """
    if delta == 0:
        return theme.GRAY
    favorable = delta < 0 if bet.lower().startswith("o") else delta > 0
    return theme.GREEN if favorable else theme.RED


def movement_summary(
    fair: Sequence[float], posted: Sequence[float], *, n_moves: int, n_price_moves: int
) -> str:
    """One line of words for an offer's movement: the fair line's move, then the posted line's.

    The single source of this wording — the Board trace's tooltip and every other surface
    that describes a move read it from here. The fair line leads because it is what the
    trace draws and ``Move`` sorts by; the posted line follows because it is what the app
    shows. The counts, not the endpoints, decide held-or-moved: the series are fixed-width
    resamples, so a move that reverted between two ticks leaves no trace in them, while the
    counts are taken over every raw poll. That is why a net-zero move reads "back at",
    never "held".

    Args:
        fair: The fair line per resample tick, oldest first, in the stat's units.
        posted: The posted main line on the same ticks.
        n_moves: Polls where the posted line changed.
        n_price_moves: Polls where the fair line changed while the posted line held.

    Returns:
        ``"Held at 0.5"`` when neither moved, else the fair part and the line part, e.g.
        ``"Fair 0.5 → 0.66 (+0.16) · line held at 0.5"`` or
        ``"Fair 21.4 → 20.6 (-0.8) · line 21.5 → 20.5"``.
    """
    posted_close = f"{posted[-1]:.10g}"
    if n_moves == 0 and n_price_moves == 0:
        return f"Held at {posted_close}"
    fair_open, fair_close = (f"{round(v, _FAIR_DECIMALS):.10g}" for v in (fair[0], fair[-1]))
    fair_delta = round(fair[-1] - fair[0], _FAIR_DECIMALS)
    if fair_delta == 0:
        fair_part = f"Fair back at {fair_close}"
    else:
        fair_part = f"Fair {fair_open} → {fair_close} ({fair_delta:+.10g})"
    if n_moves == 0:
        line_part = f"line held at {posted_close}"
    elif posted[-1] != posted[0]:
        line_part = f"line {posted[0]:.10g} → {posted_close}"
    else:
        line_part = f"line back at {posted_close}"
    return f"{fair_part} · {line_part}"


def movement_svg(fair: Sequence[float], *, bet: str, n_changes: int, title: str) -> str:
    """An offer's fair-line trajectory, colored by :func:`move_color`.

    The fair line — the line at which the app's price would be even money — folds the
    posted line and its multiplier into one stat-unit trace, so a price-only move (a TD
    prop whose multiplier drifts while its line never does) draws like any other. Only an
    offer whose line and price both held draws the flat gray rule rather than an empty cell
    — "the app held" is itself the answer, and on a same-day slate it is the answer for
    most offers.

    ``n_changes``, not ``fair``, decides held-or-moved, for the reason
    :func:`movement_summary` gives; a move that came back draws a gray trace.

    Args:
        fair: The fair line per resample tick, oldest first.
        bet: The offer's side, read by :func:`move_color`.
        n_changes: Every recorded change — posted-line moves plus price-only moves.
        title: The tooltip; the Board passes :func:`movement_summary`.

    Returns:
        The ``<svg>`` markup, or ``""`` for an empty series.
    """
    values = [float(v) for v in fair]
    if not values:
        return ""
    if n_changes == 0:
        return _svg(_rule(_HEIGHT / 2, theme.GRAY), title)
    color = move_color(values[-1] - values[0], bet)
    return _svg(_trace(_plot_points(values, min(values), max(values)), color), title)


def form_svg(values: Sequence[float], line: float) -> str:
    """The player's recent games as their distance from this offer's line.

    Each game is its own bar off a fixed centre rule — green above the line, red below,
    length the margin. Bars and not a trace because these are discrete results rather than
    a series: every value in the gamelogs is a whole number and every DFS line sits at a
    half, so a connecting stroke would slope through counts that cannot happen. It also
    keeps the card reading the same way as the deep-dive History tab, which draws the same
    question at ten games (``deep_dive_charts.history_chart``).

    A card whose games all landed exactly on the line has no deviation to draw and keeps
    the bare rule, the way an unmoved line does in :func:`movement_svg`.
    """
    deviations = [float(v) - line for v in values]
    if not deviations:
        return ""
    hits = sum(1 for deviation in deviations if deviation >= 0)
    reach = max(abs(deviation) for deviation in deviations)
    body = _rule(_HEIGHT / 2, theme.BORDER)
    if reach:
        body += _bars(deviations, reach)
    return _svg(body, f"{hits}/{len(deviations)} over {line:.10g}")
