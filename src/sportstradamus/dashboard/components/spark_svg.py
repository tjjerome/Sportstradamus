"""Inline-SVG sparklines shared by the Board grid and the constellation hover card.

Both consumers need a finished ``<svg>`` string rather than a chart object: AG Grid 34
only renders markup handed to it through a cellRenderer's ``getGui`` DOM node, and the
constellation's hand-authored frontend drops the card's body straight into ``innerHTML``.
Keeping every coordinate here means neither surface writes sparkline geometry in
JavaScript, where no gate we run would read it.

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
    """The player's recent results against this offer's line, with the line as the rule.

    The rule shares the trace's scale, so a run of bars clearing it reads as clearing it.
    """
    vals = [float(v) for v in values]
    if not vals:
        return ""
    lo, hi = min([*vals, line]), max([*vals, line])
    hits = sum(1 for v in vals if v >= line)
    body = _rule(_scale_y(line, lo, hi), theme.BORDER) + _trace(
        _plot_points(vals, lo, hi), theme.GREEN
    )
    return _svg(body, f"{hits}/{len(vals)} over {line:.10g}")
