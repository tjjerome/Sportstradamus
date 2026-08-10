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
# Same as above but a leading "+" on positives — the edge columns read as signed deltas
# (a +6.0% edge, a -3.5% edge). The plain formatter is for always-positive percents (Win %).
_SIGNED_PERCENT_FORMATTER = JsCode(
    "function(params){if(params.value==null||isNaN(params.value))return '';"
    "return (params.value>0?'+':'')+params.value.toFixed(1)+'%';}"
)
# Fixed 3-decimal display for score columns (e.g. the gate matrix's Brier Skill Score)
# that otherwise render as long floats. The value stays numeric so the grid still sorts.
_FIXED3_FORMATTER = JsCode(
    "function(params){return (params.value==null||isNaN(params.value))?'':params.value.toFixed(3);}"
)

# Light text (textColor token) reads on every painted bucket because only the saturated
# ramp ends are painted — the near-neutral band stays unpainted.
_HEAT_TEXT = "#E6E9EF"
# Saturated diverging buckets from theme.DIVERGING_COLORS: strong/mild red below the
# centre (negative edge), mild/strong blue above it. The deepest ramp ends ([0]/[9]) and
# the neutral middle stay unpainted so only genuine outliers carry a tint.
_NEG_STRONG, _NEG_MILD = theme.DIVERGING_COLORS[1], theme.DIVERGING_COLORS[2]
_POS_MILD, _POS_STRONG = theme.DIVERGING_COLORS[6], theme.DIVERGING_COLORS[7]
# Absolute |edge − centre| thresholds (in the heatmap column's own units — percentage
# points for the Board's Model Edge) that split the buckets: below MILD stays unpainted,
# MILD..STRONG paints the mild bucket, at/beyond STRONG the saturated one. Absolute, not a
# fraction of the column max — the old fraction scheme painted the whole first screen the
# moment the board was edge-sorted (the max set the scale, so everything looked extreme).
_HEAT_MILD_EDGE = 4.0
_HEAT_STRONG_EDGE = 10.0

# Obsidian Tablet — the owner-selected table treatment (2026-07-15). The themed grids render as a
# polished dark-glass slab (a surface→background gradient) with a gold hairline frame + corner
# brackets, engraved small-caps headers, gold-etched row separators, and a faint top sheen. It is
# injected into AG Grid's OWN DOM through custom_css because the grid renders in an iframe — a
# page-level wrapper div can't reach inside it, so the frame has to live on .ag-root-wrapper. The
# app's Cinzel @import lives in the parent document and does NOT cross into the iframe, so the
# engraved header degrades to a serif; small-caps + tracking + the engrave shadow carry it. Every
# color is a design token: the slab is surface/background, gold is chrome only (frame, brackets,
# rules, hover — never a data mark, DESIGN §2/§6), header ink is the neutral gray token.
_SLAB_GRADIENT = "linear-gradient(158deg,#1A1D24 0%,#0E1117 100%)"  # surface → background tokens
_FRAME_GOLD = "rgba(201,162,39,0.45)"  # 1px tablet frame (the GOLD token in rgba form)
_FRAME_HALO = "rgba(201,162,39,0.12)"  # faint ring just outside the frame
_HEADER_RULE_GOLD = "rgba(201,162,39,0.28)"  # heavier gold rule under the header band
_ETCH_GOLD = "rgba(201,162,39,0.10)"  # hairline gold-etched row separators
_ZEBRA_WASH = "rgba(255,255,255,0.018)"  # even-row lift, a hair above the slab
_ROW_HOVER_GOLD = "rgba(201,162,39,0.09)"  # gold row hover, composited over the slab
# Four L-shaped corner marks as a multi-stop background: a 12px horizontal + 12px vertical bar
# at each corner. pointer-events:none on the ::after keeps it from swallowing a row click.
_BRACKET_GOLD = "rgba(201,162,39,0.55)"
_CORNER_BRACKETS = ", ".join(
    f"linear-gradient({_BRACKET_GOLD},{_BRACKET_GOLD}) {corner} / {size} no-repeat"
    for corner in ("left top", "right top", "left bottom", "right bottom")
    for size in ("12px 1px", "1px 12px")
)

# AG Grid v34 paints row-hover through an absolutely-positioned ::before overlay that covers any
# .ag-row-hover{background-color} rule, so hover is driven by AG Grid's own --ag-row-hover-color
# variable (set on .ag-root-wrapper) plus a gold first-cell rail; both are painted by AG Grid's
# hover class client-side, so they never trigger a Streamlit rerun.
_OBSIDIAN_CSS = {
    ".ag-root-wrapper": {
        "background": _SLAB_GRADIENT,
        "border": "1px solid " + _FRAME_GOLD,
        "border-radius": "4px",
        "box-shadow": "0 0 0 1px " + _FRAME_HALO + ", 0 14px 34px rgba(0,0,0,0.55)",
        "position": "relative",
        "overflow": "hidden",
        "--ag-background-color": "transparent",
        "--ag-header-background-color": "transparent",
        "--ag-odd-row-background-color": "transparent",
        "--ag-row-border-color": _ETCH_GOLD,
        "--ag-row-hover-color": _ROW_HOVER_GOLD,
    },
    ".ag-root-wrapper::before": {
        "content": "''",
        "position": "absolute",
        "top": "0",
        "left": "0",
        "right": "0",
        "height": "38%",
        "background": "linear-gradient(180deg,rgba(255,255,255,0.05),transparent)",
        "pointer-events": "none",
    },
    ".ag-root-wrapper::after": {
        "content": "''",
        "position": "absolute",
        "inset": "5px",
        "background": _CORNER_BRACKETS,
        "pointer-events": "none",
    },
    ".ag-header": {
        "background": "transparent",
        "border-bottom": "1px solid " + _HEADER_RULE_GOLD,
    },
    ".ag-header-cell-text": {
        "font-family": "'Cinzel', Georgia, serif",
        "font-weight": "600",
        "letter-spacing": "0.13em",
        "text-transform": "uppercase",
        "color": theme.GRAY,
        "text-shadow": "0 1px 0 rgba(0,0,0,0.7), 0 -1px 0 rgba(255,255,255,0.05)",
    },
    # Manuscript type (DESIGN §2): text cells take the serif body face; numeric cells carry their own
    # inline Plex Mono (set per-column below), which beats this class rule so figures stay tabular.
    # The grid runs in an isolated iframe the parent @import can't reach, so Spectral resolves to the
    # web-safe Georgia serif here -- the same graceful degrade as the Cinzel headers.
    ".ag-cell": {"font-family": "'Spectral', Georgia, serif"},
    ".ag-row-even": {"background-color": _ZEBRA_WASH},
    ".ag-row-hover .ag-cell:first-child": {"box-shadow": "inset 3px 0 0 " + theme.GOLD},
}


def _arrow_cellrenderer() -> JsCode:
    """A class-based AG Grid cellRenderer prefixing the row's ``Bet``-keyed Over/Under arrow
    ahead of the cell's own value.

    Must be a class exposing ``getGui`` (not a plain function): a function renderer that
    returns an SVG string renders it as an *escaped* text node in AG Grid 34 — the ``<svg>``
    shows as literal markup — whereas ``getGui``'s DOM element takes the SVG through
    ``innerHTML``. The cell value is left untouched, so the Line column still sorts
    numerically. ``narrative.bet_arrow`` stays the one source of the SVG strings.
    """
    up, down = bet_arrow("Over"), bet_arrow("Under")
    return JsCode(
        "class ArrowCellRenderer{init(p){"
        "this.eGui=document.createElement('span');"
        "this.eGui.innerHTML=(p.data.Bet==='Over'?" + repr(up) + ":" + repr(down) + ")"
        "+(p.value==null?'':p.value);"
        "}getGui(){return this.eGui;}}"
    )


def _heat_expr(bg: str) -> str:
    return (
        "{'backgroundColor':'"
        + bg
        + "','color':'"
        + _HEAT_TEXT
        + "','textAlign':'right','fontFamily':'IBM Plex Mono, monospace'}"
    )


def _heatmap_cellstyle(center: float) -> JsCode:
    """A diverging cellStyle ``JsCode`` over ``params.value``, thresholds baked absolutely.

    Paints only the saturated tails (the outliers DESIGN §4 wants surfaced): a cell within
    ``_HEAT_MILD_EDGE`` of ``center`` stays unpainted, the mild bucket runs out to
    ``_HEAT_STRONG_EDGE``, and beyond that the saturated bucket — so an edge-sorted first
    screen shows a few tinted outliers over a mostly-clean column. ``JsCode`` requires the
    AgGrid call to pass ``allow_unsafe_jscode=True`` — without it st_aggrid drops the function.
    """
    lo2, lo1 = center - _HEAT_STRONG_EDGE, center - _HEAT_MILD_EDGE
    hi1, hi2 = center + _HEAT_MILD_EDGE, center + _HEAT_STRONG_EDGE
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
        + _heat_expr(_NEG_MILD)
        + " : params.value < "
        + repr(hi1)
        + " ? "
        + _RIGHT_EXPR
        + " : params.value < "
        + repr(hi2)
        + " ? "
        + _heat_expr(_POS_MILD)
        + " : "
        + _heat_expr(_POS_STRONG)
    )
    return JsCode("function(params) { return (" + expr + "); }")


def _numeric_col_kwargs(
    col: str,
    *,
    heatmap_col: str | None,
    heatmap_center: float,
    pct: set[str],
    signed_pct: set[str],
    decimal: set[str],
    arrow_col: str | None,
    has_bet: bool,
    tip: str | None,
) -> dict:
    """``configure_column`` kwargs for one numeric column: heatmap-or-plain cellStyle, an
    optional "%" formatter (signed for ``signed_pct``, plain for ``pct``) or a 3-decimal
    formatter (``decimal``), an optional arrow cellRenderer, an optional tooltip.
    """
    cell_style = _heatmap_cellstyle(heatmap_center) if col == heatmap_col else dict(_RIGHT_STYLE)
    kwargs = {"cellStyle": cell_style}
    if col in signed_pct:
        kwargs["valueFormatter"] = _SIGNED_PERCENT_FORMATTER
    elif col in pct:
        kwargs["valueFormatter"] = _PERCENT_FORMATTER
    elif col in decimal:
        kwargs["valueFormatter"] = _FIXED3_FORMATTER
    if col == arrow_col and has_bet:
        kwargs["cellRenderer"] = _arrow_cellrenderer()
    if tip:
        kwargs["headerTooltip"] = tip
    return kwargs


def _get_row_style(flag_col: str, flag_below: float) -> JsCode:
    """A per-row ``getRowStyle`` callback painting the amber rail when ``flag_col`` is low.

    Mirrors ``_OBSIDIAN_CSS``'s row-hover gold rail shape (inset box-shadow), orange and
    row-conditional instead of gold and hover-conditional — AG Grid's ``rowStyle`` is a
    static dict applied to every row, so a data-dependent style needs this callback form.
    """
    return JsCode(
        "function(params){return params.data && params.data["
        + repr(flag_col)
        + "] < "
        + repr(flag_below)
        + " ? {boxShadow: 'inset 3px 0 0 "
        + theme.ORANGE
        + "'} : null;}"
    )


def _get_row_style_for_rail(rail_col: str, rail_colors: Mapping[str, str]) -> JsCode:
    """A per-row ``getRowStyle`` callback painting a rail color keyed off a precomputed
    categorical ``rail_col`` value (e.g. the gate matrix's ``amber``/``red``/``none``).

    Same inset-box-shadow shape as :func:`_get_row_style`, but a value lookup rather
    than a numeric threshold — ``flag_col``'s single-color/single-threshold contract
    doesn't fit a multi-state category, so this is a separate callback rather than a
    ``flag_col`` generalization.
    """
    branches = "".join(
        "params.data[" + repr(rail_col) + "]===" + repr(value) + " ? {boxShadow: "
        "'inset 3px 0 0 " + color + "'} : "
        for value, color in rail_colors.items()
    )
    return JsCode("function(params){return params.data ? " + branches + "null : null;}")


def _glyph_expr(color: str) -> str:
    """Colored-text style for a glyph cell: the glyph itself is tinted, the cell stays
    unfilled. Distinct from :func:`_heat_expr`, which paints the whole cell background —
    a background swatch here would hide the ●/○ mark (DESIGN §4: the glyph is the datum).
    """
    return (
        "{'color':'"
        + color
        + "','fontWeight':600,'textAlign':'right','fontFamily':'IBM Plex Mono, monospace'}"
    )


def _glyph_cellstyle(glyph_colors: Mapping[str, str]) -> JsCode:
    """A ``cellStyle`` ``JsCode`` coloring a cell's own literal glyph value (e.g. ``●``/``○``)
    as tinted text — green pass / red fail — right-aligned mono, never a filled cell.
    """
    branches = "".join(
        "params.value===" + repr(glyph) + " ? " + _glyph_expr(color) + " : "
        for glyph, color in glyph_colors.items()
    )
    return JsCode("function(params) { return (" + branches + _RIGHT_EXPR + "); }")


def _row_style_options(
    flag_col: str | None,
    flag_below: float,
    rail_col: str | None,
    rail_colors: Mapping[str, str] | None,
) -> dict:
    """The ``getRowStyle`` grid-options fragment for whichever rail kind was requested.

    ``flag_col`` (numeric threshold) and ``rail_col`` (categorical lookup) both paint a
    row rail but via different callbacks; a caller passing both gets ``rail_col``'s
    style — checked first, always the winner regardless of kwarg order — rather than
    a silent combination of the two.
    """
    if rail_col is not None:
        return {"getRowStyle": _get_row_style_for_rail(rail_col, rail_colors or {})}
    if flag_col is not None:
        return {"getRowStyle": _get_row_style(flag_col, flag_below)}
    return {}


def _configure_remaining_columns(
    gb: GridOptionsBuilder,
    present: set[str],
    *,
    numeric_cols: Sequence[str],
    glyph_cols: Sequence[str],
    glyph_colors: Mapping[str, str] | None,
    help_map: Mapping[str, str],
    hidden_cols: Sequence[str],
) -> None:
    """The non-numeric column passes: glyph cellStyles, header tooltips on columns
    ``numeric_cols`` didn't already cover, and ``hide`` on ``hidden_cols``.
    """
    for col in glyph_cols:
        if col in present:
            gb.configure_column(col, cellStyle=_glyph_cellstyle(glyph_colors or {}))
    for col, tip in help_map.items():
        if col not in numeric_cols and col in present:
            gb.configure_column(col, headerTooltip=tip)
    for col in hidden_cols:
        if col in present:
            gb.configure_column(col, hide=True)


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
    signed_percent_cols: Sequence[str] = (),
    decimal_cols: Sequence[str] = (),
    arrow_col: str | None = None,
    hidden_cols: Sequence[str] = (),
    flag_col: str | None = None,
    flag_below: float = 0.0,
    rail_col: str | None = None,
    rail_colors: Mapping[str, str] | None = None,
    glyph_cols: Sequence[str] = (),
    glyph_colors: Mapping[str, str] | None = None,
) -> dict:
    """Token-themed ``gridOptions``: right-aligned mono numerals, an optional diverging
    heatmap on ``heatmap_col``, per-column header tooltips, and a "%" display suffix on
    ``percent_cols`` (kept numeric underneath). ``signed_percent_cols`` is the same suffix
    with a leading "+" on positives, for signed deltas like the edge columns.
    ``decimal_cols`` renders at a fixed 3 decimals (e.g. a Brier Skill Score), value kept
    numeric so sorting is unaffected. Pure — no Streamlit call.

    ``arrow_col`` prefixes that column's cells with the row's Over/Under arrow, keyed
    off a ``Bet`` column in the row data. ``hidden_cols`` stays in the row data (so
    selection callbacks and JS renderers can still read it) without rendering as its
    own grid column — e.g. ``Bet`` for the arrow renderer, or a logic-only slug column
    that a display column already covers. ``flag_col`` paints an amber left-rail on any
    row whose value in that column is below ``flag_below`` (e.g. a negative Brier Skill
    Score) via AG Grid's ``getRowStyle``. ``rail_col``/``rail_colors`` is the categorical
    sibling of ``flag_col`` — a precomputed string column (e.g. ``amber``/``red``/``none``)
    mapped straight to a rail color, for cases a single numeric threshold can't express.
    Only one of ``flag_col``/``rail_col`` should be passed per call — both set
    ``getRowStyle`` and ``rail_col`` always wins if both are given, regardless of
    kwarg order (a fixed precedence, not "last kwarg wins").
    ``glyph_cols``/``glyph_colors`` color cells by their own literal value (e.g. a
    ●/○ pass/fail glyph) rather than by a numeric heatmap.
    """
    help_map = dict(header_help or {})
    pct = set(percent_cols)
    signed_pct = set(signed_percent_cols)
    decimal = set(decimal_cols)
    present = set(df.columns)
    has_bet = "Bet" in present
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode=selection_mode, use_checkbox=False)
    # enableBrowserTooltips renders the header tooltips as native browser titles (reliable;
    # AG Grid's own tooltip component otherwise needs extra wiring and a long show delay).
    grid_opts = {"rowStyle": {"cursor": "pointer"}, "enableBrowserTooltips": True}
    grid_opts.update(_row_style_options(flag_col, flag_below, rail_col, rail_colors))
    gb.configure_grid_options(**grid_opts)
    for col in numeric_cols:
        if col not in present:
            continue
        kwargs = _numeric_col_kwargs(
            col,
            heatmap_col=heatmap_col,
            heatmap_center=heatmap_center,
            pct=pct,
            signed_pct=signed_pct,
            decimal=decimal,
            arrow_col=arrow_col,
            has_bet=has_bet,
            tip=help_map.get(col),
        )
        gb.configure_column(col, **kwargs)
    _configure_remaining_columns(
        gb,
        present,
        numeric_cols=numeric_cols,
        glyph_cols=glyph_cols,
        glyph_colors=glyph_colors,
        help_map=help_map,
        hidden_cols=hidden_cols,
    )
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
    signed_percent_cols: Sequence[str] = (),
    decimal_cols: Sequence[str] = (),
    arrow_col: str | None = None,
    hidden_cols: Sequence[str] = (),
    flag_col: str | None = None,
    flag_below: float = 0.0,
    rail_col: str | None = None,
    rail_colors: Mapping[str, str] | None = None,
    glyph_cols: Sequence[str] = (),
    glyph_colors: Mapping[str, str] | None = None,
    height: int = 720,
    key: str | None = None,
) -> list[dict]:
    """Render the themed grid; return the selected rows as dicts (empty when none).

    ``key`` disambiguates multiple grids rendered on one page. The grid wears the Obsidian
    tablet skin (``_OBSIDIAN_CSS``); row hover is a client-side gold rail that never reruns.
    """
    options = build_themed_grid_options(
        df,
        numeric_cols=numeric_cols,
        heatmap_col=heatmap_col,
        heatmap_center=heatmap_center,
        header_help=header_help,
        selection_mode=selection_mode,
        percent_cols=percent_cols,
        signed_percent_cols=signed_percent_cols,
        decimal_cols=decimal_cols,
        arrow_col=arrow_col,
        hidden_cols=hidden_cols,
        flag_col=flag_col,
        flag_below=flag_below,
        rail_col=rail_col,
        rail_colors=rail_colors,
        glyph_cols=glyph_cols,
        glyph_colors=glyph_colors,
    )
    grid = AgGrid(
        df,
        gridOptions=options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        custom_css=_OBSIDIAN_CSS,
        height=height,
        width="stretch",
        key=key,
    )
    selected = grid.selected_rows
    # Newer streamlit-aggrid returns a DataFrame; older versions return a list.
    if isinstance(selected, pd.DataFrame):
        return selected.to_dict("records")
    return selected or []
