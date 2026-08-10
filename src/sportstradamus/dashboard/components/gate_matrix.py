"""Ship-gate matrix builder + thin render for the Lab Training surface.

``gate_matrix_frame`` is the pure sort/annotate step (unit-tested directly against
synthetic frames); :func:`render_gate_matrix` is the one-line Streamlit call that
turns its output into a themed grid.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.components.grid import render_themed_grid

GATE_COLS = ("g1_pass", "g2_pass", "g3_pass", "g4_pass", "g5_pass", "g6_pass")

# rail tiers, ascending sort priority: amber (one gate short) first, then red
# (multi-fail, itself sorted worst-first by n_fails), then none (fully passing) last.
_TIER_AMBER, _TIER_RED, _TIER_NONE = 0, 1, 2


def _rail(n_fails: int) -> str:
    if n_fails == 1:
        return "amber"
    if n_fails >= 2:
        return "red"
    return "none"


def _tier(n_fails: int) -> int:
    if n_fails == 1:
        return _TIER_AMBER
    if n_fails >= 2:
        return _TIER_RED
    return _TIER_NONE


def gate_matrix_frame(stats: pd.DataFrame) -> pd.DataFrame:
    """Cells × gates, worst-first: one-gate-short cells pinned to the top
    (amber rail), then by fails desc (multi-fail = red rail), then BSS asc.
    Adds n_fails + rail ∈ {amber, red, none}. Pure.

    "Worst-first" means most-actionable-first: a cell one gate away from shipping
    is the highest-priority row, ranked above cells failing multiple gates, which
    in turn rank above cells that are fully passing.
    """
    out = stats.copy()
    # Gates are tri-state: pd.NA until the scorecard has run over that cell's test set,
    # or when its artifacts are too stale to bind one. `astype(bool)` raises on those,
    # and an ungraded gate is not a pass — count it as a fail so the cell sorts toward
    # the actionable end rather than crashing the page.
    fails = ~out[list(GATE_COLS)].fillna(False).astype(bool)
    out["n_fails"] = fails.sum(axis=1)
    out["rail"] = out["n_fails"].map(_rail)
    out["_tier"] = out["n_fails"].map(_tier)
    out = out.sort_values(
        ["_tier", "n_fails", "brier_skill_score"],
        ascending=[True, False, True],
        kind="stable",
    )
    return out.drop(columns=["_tier"])


def render_gate_matrix(matrix: pd.DataFrame, *, height: int = 560) -> None:
    """Render the gate matrix as a themed grid: ● pass (green) / ○ fail (red) glyphs,
    amber/red row rails from :func:`gate_matrix_frame`'s ``rail`` column.

    ``rail`` rides along as a hidden column — ``getRowStyle`` reads it from each row's
    data, so it must stay in the frame ``AgGrid`` renders even though it isn't its own
    visible grid column.
    """
    id_cols = [c for c in ("league", "market", "brier_skill_score") if c in matrix.columns]
    display = matrix[[*id_cols, *GATE_COLS, "n_fails", "rail"]].copy()
    for col in GATE_COLS:
        # pd.NA is ambiguous in a boolean test, and an ungraded gate is not a pass —
        # same reading `gate_matrix_frame` counts it under.
        display[col] = display[col].fillna(False).map(lambda passed: "●" if passed else "○")
    render_themed_grid(
        display,
        numeric_cols=["brier_skill_score", "n_fails"],
        decimal_cols=["brier_skill_score"],
        glyph_cols=GATE_COLS,
        glyph_colors={"●": theme.GREEN, "○": theme.RED},
        hidden_cols=["rail"],
        rail_col="rail",
        rail_colors={"amber": theme.ORANGE, "red": theme.RED},
        height=height,
    )
