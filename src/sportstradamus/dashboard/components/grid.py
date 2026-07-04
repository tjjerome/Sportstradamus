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
from sportstradamus.dashboard.narrative import bet_arrow

_MONO = "IBM Plex Mono, monospace"
# Static right-aligned mono style for non-heatmap numeric columns (no JS needed).
_RIGHT_STYLE = {"textAlign": "right", "fontFamily": _MONO}
# Same style as a JS object literal, for the heatmap expression's null / neutral branches.
_RIGHT_EXPR = "{'textAlign':'right','fontFamily':'IBM Plex Mono, monospace'}"
# Append a "%" to a numeric cell for display only — the underlying value stays numeric so
# sorting and the heatmap still see the number (the Board's edge columns are EV − 1, x100).
_PERCENT_FORMATTER = JsCode(
    "function(params){return (params.value==null||isNaN(params.value))?'':params.value.toFixed(1)+'%';}"
)

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

# Client-side row hover (spec §4.3): a gold left-rail + faint gold wash, painted by AG
# Grid's own row-hover class so hovering never triggers a Streamlit rerun.
_HOVER_CSS = {
    ".ag-row-hover": {
        "box-shadow": "inset 3px 0 0 #C9A227 !important",
        "background-color": "rgba(201,162,39,.06) !important",
    }
}


def _arrow_cellrenderer() -> JsCode:
    """Prefix the row's ``Bet``-keyed Over/Under arrow ahead of the cell's own value.

    ``narrative.bet_arrow`` is the one source for the SVG strings; baking both branches
    in as JS literals here keeps that Python string the source of truth without a
    second SVG definition living in JS.
    """
    up, down = bet_arrow("Over"), bet_arrow("Under")
    return JsCode(
        "function(params){return (params.data.Bet==='Over'?"
        + repr(up)
        + ":"
        + repr(down)
        + ")+params.value;}"
    )


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


def _numeric_col_kwargs(
    df: pd.DataFrame,
    col: str,
    *,
    heatmap_col: str | None,
    heatmap_center: float,
    pct: set[str],
    arrow_col: str | None,
    has_bet: bool,
    tip: str | None,
) -> dict:
    """``configure_column`` kwargs for one numeric column: heatmap-or-plain cellStyle,
    an optional "%" formatter, an optional arrow cellRenderer, an optional tooltip.
    """
    if col == heatmap_col:
        cell_style = _heatmap_cellstyle(df[col], heatmap_center)
    else:
        cell_style = dict(_RIGHT_STYLE)
    kwargs = {"cellStyle": cell_style}
    if col in pct:
        kwargs["valueFormatter"] = _PERCENT_FORMATTER
    if col == arrow_col and has_bet:
        kwargs["cellRenderer"] = _arrow_cellrenderer()
    if tip:
        kwargs["headerTooltip"] = tip
    return kwargs


def build_themed_grid_options(
    df: pd.DataFrame,
    *,
    numeric_cols: Sequence[str],
    heatmap_col: str | None = None,
    heatmap_center: float = 0.0,
    header_help: Mapping[str, str] | None = None,
    selection_mode: str = "single",
    sparkline_col: str | None = None,  # L1 scar hook — line-movement sparklines, not built
    percent_cols: Sequence[str] = (),
    arrow_col: str | None = None,
    hidden_cols: Sequence[str] = (),
) -> dict:
    """Token-themed ``gridOptions``: right-aligned mono numerals, an optional diverging
    heatmap on ``heatmap_col``, per-column header tooltips, and a "%" display suffix on
    ``percent_cols`` (kept numeric underneath). Pure — no Streamlit call.

    ``arrow_col`` prefixes that column's cells with the row's Over/Under arrow, keyed
    off a ``Bet`` column in the row data. ``hidden_cols`` stays in the row data (so
    selection callbacks and JS renderers can still read it) without rendering as its
    own grid column — e.g. ``Bet`` for the arrow renderer, or a logic-only slug column
    that a display column already covers.
    """
    help_map = dict(header_help or {})
    pct = set(percent_cols)
    present = set(df.columns)
    has_bet = "Bet" in present
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode=selection_mode, use_checkbox=False)
    # enableBrowserTooltips renders the header tooltips as native browser titles (reliable;
    # AG Grid's own tooltip component otherwise needs extra wiring and a long show delay).
    gb.configure_grid_options(rowStyle={"cursor": "pointer"}, enableBrowserTooltips=True)
    for col in numeric_cols:
        if col not in present:
            continue
        kwargs = _numeric_col_kwargs(
            df,
            col,
            heatmap_col=heatmap_col,
            heatmap_center=heatmap_center,
            pct=pct,
            arrow_col=arrow_col,
            has_bet=has_bet,
            tip=help_map.get(col),
        )
        gb.configure_column(col, **kwargs)
    for col, tip in help_map.items():
        if col not in numeric_cols and col in present:
            gb.configure_column(col, headerTooltip=tip)
    for col in hidden_cols:
        if col in present:
            gb.configure_column(col, hide=True)
    return gb.build()


def render_themed_grid(
    df: pd.DataFrame,
    *,
    numeric_cols: Sequence[str],
    heatmap_col: str | None = None,
    heatmap_center: float = 0.0,
    header_help: Mapping[str, str] | None = None,
    selection_mode: str = "single",
    percent_cols: Sequence[str] = (),
    arrow_col: str | None = None,
    hidden_cols: Sequence[str] = (),
    height: int = 720,
    key: str | None = None,
) -> list[dict]:
    """Render the themed grid; return the selected rows as dicts (empty when none).

    ``key`` disambiguates multiple grids rendered on one page. Row hover is a
    client-side gold rail (``_HOVER_CSS``) — it never triggers a Streamlit rerun.
    """
    options = build_themed_grid_options(
        df,
        numeric_cols=numeric_cols,
        heatmap_col=heatmap_col,
        heatmap_center=heatmap_center,
        header_help=header_help,
        selection_mode=selection_mode,
        percent_cols=percent_cols,
        arrow_col=arrow_col,
        hidden_cols=hidden_cols,
    )
    grid = AgGrid(
        df,
        gridOptions=options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        custom_css=_HOVER_CSS,
        height=height,
        width="stretch",
        key=key,
    )
    selected = grid.selected_rows
    # Newer streamlit-aggrid returns a DataFrame; older versions return a list.
    if isinstance(selected, pd.DataFrame):
        return selected.to_dict("records")
    return selected or []
