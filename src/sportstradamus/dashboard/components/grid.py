"""Themed AG Grid helpers — the one place the stat grids speak the design tokens.

DESIGN.md §4: numbers right-align in tabular (Plex Mono) numerals, never centered;
conditional heatmaps use the diverging chart ramp (red ↔ neutral ↔ blue), never gold.
The Board builds its grid through here so the rules live once and
``tests/golden/test_grid_options.py`` has a single pure target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from sportstradamus.dashboard import theme

_MONO = "IBM Plex Mono, monospace"
# Static right-aligned mono style for non-heatmap numeric columns (no JS needed).
_RIGHT_STYLE = {"textAlign": "right", "fontFamily": _MONO}
# Same style as a JS object literal, for the heatmap expression's null / neutral branches.
_RIGHT_EXPR = "{'textAlign':'right','fontFamily':'IBM Plex Mono, monospace'}"

# Light text (textColor token) reads on every painted bucket because only the saturated
# ramp ends are painted — the near-neutral band stays unpainted.
_HEAT_TEXT = "#E6E9EF"
# Saturated diverging buckets from theme.DIVERGING_COLORS: strong/mild red below the
# centre (negative edge), strong/mild blue above it. The pale middle is left unpainted.
_NEG_STRONG, _NEG = theme.DIVERGING_COLORS[0], theme.DIVERGING_COLORS[1]
_POS, _POS_STRONG = theme.DIVERGING_COLORS[7], theme.DIVERGING_COLORS[8]
# Fractions of the column's max deviation that split mild/strong and the neutral band.
_HEAT_STRONG_FRAC = 0.6
_HEAT_NEUTRAL_FRAC = 0.2


def _heat_expr(bg: str) -> str:
    return (
        "{'backgroundColor':'"
        + bg
        + "','color':'"
        + _HEAT_TEXT
        + "','textAlign':'right','fontFamily':'IBM Plex Mono, monospace'}"
    )


def _heatmap_cellstyle(values: pd.Series, center: float) -> JsCode | dict:
    """A diverging cellStyle ``JsCode`` over ``params.value``, bounds baked from the column.

    Paints only the saturated tails (the outliers DESIGN §4 wants surfaced) so the
    neutral band stays clean and the light text keeps contrast. Falls back to the plain
    right-aligned style when the column has no spread. ``JsCode`` requires the AgGrid call
    to pass ``allow_unsafe_jscode=True`` — without it st_aggrid drops the function.
    """
    dev = (pd.to_numeric(values, errors="coerce") - center).abs()
    span = float(dev.max()) if len(dev) else 0.0
    if not span or pd.isna(span):
        return dict(_RIGHT_STYLE)
    lo2, lo1 = center - _HEAT_STRONG_FRAC * span, center - _HEAT_NEUTRAL_FRAC * span
    hi1, hi2 = center + _HEAT_NEUTRAL_FRAC * span, center + _HEAT_STRONG_FRAC * span
    expr = (
        "params.value==null||isNaN(params.value) ? "
        + _RIGHT_EXPR
        + " : params.value <= "
        + repr(lo2)
        + " ? "
        + _heat_expr(_NEG_STRONG)
        + " : params.value <= "
        + repr(lo1)
        + " ? "
        + _heat_expr(_NEG)
        + " : params.value < "
        + repr(hi1)
        + " ? "
        + _RIGHT_EXPR
        + " : params.value < "
        + repr(hi2)
        + " ? "
        + _heat_expr(_POS)
        + " : "
        + _heat_expr(_POS_STRONG)
    )
    return JsCode("function(params) { return (" + expr + "); }")


def build_themed_grid_options(
    df: pd.DataFrame,
    *,
    numeric_cols: Sequence[str],
    heatmap_col: str | None = None,
    heatmap_center: float = 0.0,
    header_help: Mapping[str, str] | None = None,
    selection_mode: str = "single",
    sparkline_col: str | None = None,  # L1 scar hook — line-movement sparklines, not built
) -> dict:
    """Token-themed ``gridOptions``: right-aligned mono numerals, an optional diverging
    heatmap on ``heatmap_col``, and per-column header tooltips. Pure — no Streamlit call.
    """
    help_map = dict(header_help or {})
    present = set(df.columns)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode=selection_mode, use_checkbox=False)
    # enableBrowserTooltips renders the header tooltips as native browser titles (reliable;
    # AG Grid's own tooltip component otherwise needs extra wiring and a long show delay).
    gb.configure_grid_options(rowStyle={"cursor": "pointer"}, enableBrowserTooltips=True)
    for col in numeric_cols:
        if col not in present:
            continue
        if col == heatmap_col:
            cell_style = _heatmap_cellstyle(df[col], heatmap_center)
        else:
            cell_style = dict(_RIGHT_STYLE)
        kwargs = {"cellStyle": cell_style}
        if help_map.get(col):
            kwargs["headerTooltip"] = help_map[col]
        gb.configure_column(col, **kwargs)
    for col, tip in help_map.items():
        if col not in numeric_cols and col in present:
            gb.configure_column(col, headerTooltip=tip)
    return gb.build()


def render_themed_grid(
    df: pd.DataFrame,
    *,
    numeric_cols: Sequence[str],
    heatmap_col: str | None = None,
    heatmap_center: float = 0.0,
    header_help: Mapping[str, str] | None = None,
    selection_mode: str = "single",
    height: int = 720,
    key: str | None = None,
) -> list[dict]:
    """Render the themed grid; return the selected rows as dicts (empty when none).

    ``key`` disambiguates multiple grids rendered on one page.
    """
    options = build_themed_grid_options(
        df,
        numeric_cols=numeric_cols,
        heatmap_col=heatmap_col,
        heatmap_center=heatmap_center,
        header_help=header_help,
        selection_mode=selection_mode,
    )
    grid = AgGrid(
        df,
        gridOptions=options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        height=height,
        width="stretch",
        key=key,
    )
    selected = grid.selected_rows
    # Newer streamlit-aggrid returns a DataFrame; older versions return a list.
    if isinstance(selected, pd.DataFrame):
        return selected.to_dict("records")
    return selected or []
