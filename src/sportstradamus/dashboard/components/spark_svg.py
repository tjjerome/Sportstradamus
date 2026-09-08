"""Inline-SVG micro-charts for the Board grid and the constellation hover card.

Both consumers need a finished ``<svg>`` string rather than a chart object: AG Grid 34
only renders markup handed to it through a cellRenderer's ``getGui`` DOM node, and the
constellation's hand-authored frontend drops the card's body straight into ``innerHTML``.
Keeping every coordinate here means neither surface writes chart geometry in JavaScript,
where no gate we run would read it.

The two share a box and a reference rule but not an encoding, because they describe
different things: a book's line is one quantity moving through time and reads as a trace,
while a player's last games are discrete results and read as bars off the line.

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


def movement_svg(series: Sequence[float], *, bet: str, n_moves: int) -> str:
    """The DFS book's own line trajectory for one offer, colored by who it favors.

    A line drifting away from the bet side is money leaving the table, so ``Over`` reads
    green on a falling line and ``Under`` on a rising one. A line that never moved draws
    a flat gray rule rather than an empty cell — "the app held" is itself the answer, and
    on a same-day slate it is the answer for most offers.

    Three states occur in the data, and only ``n_moves`` separates them: never repriced,
    moved and stayed, and moved and came back. The last two both net to a delta that can
    be zero, and ``series`` cannot settle it either — it is a fixed-width resample, so a
    move that reverted between two of its ticks leaves no trace in it at all. ``n_moves``
    counts every raw observation before resampling, so it is the one honest witness.
    """
    values = [float(v) for v in series]
    if not values:
        return ""
    open_line, close_line = values[0], values[-1]
    if not n_moves:
        return _svg(_rule(_HEIGHT / 2, theme.GRAY), f"Line held at {close_line:.10g}")
    move = close_line - open_line
    if not move:
        color, title = theme.GRAY, f"Line left {close_line:.10g} and came back"
    else:
        favorable = move < 0 if bet.lower().startswith("o") else move > 0
        color = theme.GREEN if favorable else theme.RED
        title = f"Line {open_line:.10g} → {close_line:.10g} ({move:+.10g})"
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
