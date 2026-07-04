"""Shared diagnostic-filter panel over ``stat_meta.json`` axes (DESIGN spec §3.7).

Diagnostics, Correlations, and Training are one "Lab" made of three surfaces sharing a
single granular filter over the real training-config dimensions (distribution family,
target normalization, post-hoc corrector, and the sparser Optuna search axes). Widget
keys are ``lab_filter_{axis}`` so the selection follows the user across all three pages.

:func:`render_lab_filters` renders the panel (full multiselect grid or a collapsed
one-line chip strip + expander) and returns the selection dict.
:func:`apply_lab_filters` is the pure mask-builder a caller runs against its own frame;
it has no Streamlit dependency so it is unit-tested directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

FILTER_AXES = (
    "dist",
    "target_normalization",
    "posthoc",
    "blending",
    "hpo_selection",
    "count_dispersion_objective",
    "zinb_mode",
    "shipped",
)

# Sentence-case labels for the multiselect widgets and the collapsed chip strip.
_AXIS_LABELS = {
    "dist": "Distribution",
    "target_normalization": "Target norm",
    "posthoc": "Post-hoc",
    "blending": "Blending",
    "hpo_selection": "HPO selection",
    "count_dispersion_objective": "Count dispersion",
    "zinb_mode": "ZINB mode",
    "shipped": "Shipped",
}


def _widget_key(axis: str) -> str:
    return f"lab_filter_{axis}"


def _render_full(meta: pd.DataFrame) -> dict[str, list[str]]:
    sel: dict[str, list[str]] = {}
    cols = st.columns(len(FILTER_AXES))
    for col, axis in zip(cols, FILTER_AXES, strict=True):
        options = sorted(meta[axis].dropna().unique()) if axis in meta.columns else []
        with col:
            sel[axis] = st.multiselect(_AXIS_LABELS[axis], options, key=_widget_key(axis))
    return sel


def _render_collapsed(meta: pd.DataFrame) -> dict[str, list[str]]:
    sel = {axis: st.session_state.get(_widget_key(axis), []) for axis in FILTER_AXES}
    active = [f"{_AXIS_LABELS[axis]}: {', '.join(vals)}" for axis, vals in sel.items() if vals]
    chip_line = " · ".join(active) if active else "No filters active"
    st.caption(chip_line)
    with st.expander("Filters"):
        return _render_full(meta)


def render_lab_filters(meta: pd.DataFrame, *, collapsed: bool) -> dict[str, list[str]]:
    """Render the shared Lab filter panel and return the current selection.

    ``collapsed=False`` (Diagnostics) renders every axis as its own multiselect.
    ``collapsed=True`` (Correlations, Training) renders a one-line chip summary of
    the active selection with an expander that reveals the same multiselects.
    Widget keys are ``lab_filter_{axis}``, so a selection made on one Lab page is
    still selected when the user navigates to another.

    Args:
        meta: Frame from :func:`~sportstradamus.dashboard.data.load_stat_meta`.
        collapsed: Whether to render the compact chip-strip form.

    Returns:
        ``{axis: selected_values}`` for every axis in :data:`FILTER_AXES`.
    """
    return _render_collapsed(meta) if collapsed else _render_full(meta)


def apply_lab_filters(df: pd.DataFrame, meta: pd.DataFrame, sel: dict) -> pd.DataFrame:
    """Filter ``df`` to rows whose ``(league, market)`` cell matches every selection.

    Joins ``df`` to ``meta`` on ``(league, market)`` — ``df`` may use either
    ``league``/``market`` or title-case ``League``/``Market`` column names, detected
    automatically. An axis absent from ``sel`` or mapped to an empty list is not a
    constraint (every cell passes); a non-empty list narrows ``meta`` to the matching
    cells before the join, so only those ``(league, market)`` pairs survive in the
    result. Pure — no Streamlit calls — and returns ``df``'s own columns unchanged.

    Args:
        df: Frame to filter; needs a league and a market column (either casing).
        meta: Frame from :func:`~sportstradamus.dashboard.data.load_stat_meta`.
        sel: ``{axis: selected_values}``, a subset of :data:`FILTER_AXES` keys.

    Returns:
        ``df`` narrowed to rows whose cell matches every non-empty axis selection.
    """
    league_col, market_col = (
        ("League", "Market") if "League" in df.columns else ("league", "market")
    )

    mask = pd.Series(True, index=meta.index)
    for axis, values in sel.items():
        if values:
            mask &= meta[axis].isin(values)
    allowed_cells = meta.loc[mask, ["league", "market"]]

    keys = pd.MultiIndex.from_frame(df[[league_col, market_col]])
    allowed = pd.MultiIndex.from_frame(
        allowed_cells.rename(columns={"league": league_col, "market": market_col})
    )
    return df.loc[keys.isin(allowed)]
