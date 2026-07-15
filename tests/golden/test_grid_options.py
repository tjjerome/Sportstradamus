"""Pin the themed-grid contract: design-token colors, right-aligned numerals, lens math.

``dashboard/components/grid.py`` is the one place the stat grids turn a frame into
``gridOptions``. These goldens hold its three promises: numbers never center (DESIGN.md §4),
every color it paints is a committed design token, and the board lenses / edge derivation
read the EV columns as edge-vs-DFS (Model Edge = ``Model EV − 1``, Consensus Edge =
``Market EV − 1``; lens book-agreement keys off ``Market EV``).
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.columns import (
    CONSENSUS_EDGE,
    LABELS,
    MODEL_EDGE,
    add_edges,
    add_market_display,
    add_match_column,
)
from sportstradamus.dashboard.components.grid import (
    _HEAT_MILD_EDGE,
    _HEAT_STRONG_EDGE,
    _HEAT_TEXT,
    _HOVER_CSS,
    build_themed_grid_options,
)
from sportstradamus.dashboard.lenses import LENSES, apply_lens

_HEX = re.compile(r"#[0-9A-Fa-f]{6}")
_GRID_DF = pd.DataFrame(
    {
        "Player": ["A", "B", "C", "D"],
        MODEL_EDGE: [0.18, -0.05, 0.00, 0.50],
        CONSENSUS_EDGE: [0.02, 0.00, 0.05, -0.10],
        "Kelly": [0.10, 0.00, 0.02, 0.25],
    }
)
_NUMERIC = [MODEL_EDGE, CONSENSUS_EDGE, "Kelly"]


def _dumps(options: dict) -> str:
    """JSON-dump gridOptions, rendering st_aggrid ``JsCode`` cellStyles as their js text."""
    return json.dumps(options, default=lambda o: getattr(o, "js_code", str(o)))


def _column_defs(options: dict) -> dict[str, dict]:
    return {c["field"]: c for c in options["columnDefs"]}


def _cellstyle_text(cell_style) -> str:
    """The cellStyle as text whether it's a ``JsCode`` function or a plain style dict."""
    return getattr(cell_style, "js_code", None) or json.dumps(cell_style)


def test_numeric_cols_right_aligned_never_centered() -> None:
    options = build_themed_grid_options(_GRID_DF, numeric_cols=_NUMERIC, heatmap_col=MODEL_EDGE)
    defs = _column_defs(options)
    for col in _NUMERIC:
        style = _cellstyle_text(defs[col]["cellStyle"])
        assert "'textAlign':'right'" in style or '"textAlign": "right"' in style, (
            f"{col} numeric cell is not right-aligned"
        )
    assert "center" not in _dumps(options), "a grid cell centers a numeral (DESIGN §4)"


def test_grid_hexes_are_design_tokens() -> None:
    options = build_themed_grid_options(_GRID_DF, numeric_cols=_NUMERIC, heatmap_col=MODEL_EDGE)
    allowed = set(theme.DIVERGING_COLORS) | {_HEAT_TEXT}
    found = set(_HEX.findall(_dumps(options)))
    assert found, "heatmap baked no colors — expected the diverging ramp"
    assert found <= allowed, f"grid paints a non-token color: {found - allowed}"


def test_heatmap_is_jscode_on_its_column_only() -> None:
    options = build_themed_grid_options(_GRID_DF, numeric_cols=_NUMERIC, heatmap_col=MODEL_EDGE)
    defs = _column_defs(options)
    # The heatmap column must be a JsCode function (st_aggrid needs allow_unsafe_jscode);
    # a plain dict here means the heatmap silently never renders.
    assert hasattr(defs[MODEL_EDGE]["cellStyle"], "js_code"), "heatmap cellStyle is not JsCode"
    assert "backgroundColor" in defs[MODEL_EDGE]["cellStyle"].js_code
    for col in (CONSENSUS_EDGE, "Kelly"):
        assert "backgroundColor" not in _cellstyle_text(defs[col]["cellStyle"]), (
            f"{col} is painted but only the heatmap column should be"
        )


def test_percent_cols_get_a_value_formatter_value_stays_numeric() -> None:
    options = build_themed_grid_options(
        _GRID_DF,
        numeric_cols=_NUMERIC,
        heatmap_col=MODEL_EDGE,
        percent_cols=[MODEL_EDGE, CONSENSUS_EDGE],
    )
    defs = _column_defs(options)
    for col in (MODEL_EDGE, CONSENSUS_EDGE):
        fmt = defs[col].get("valueFormatter")
        assert fmt is not None and "%" in fmt.js_code  # display-only suffix
    assert "valueFormatter" not in defs["Kelly"]  # only percent_cols get it


def test_signed_percent_cols_get_a_sign_prefixed_formatter() -> None:
    options = build_themed_grid_options(
        _GRID_DF,
        numeric_cols=_NUMERIC,
        signed_percent_cols=[MODEL_EDGE],
        percent_cols=[CONSENSUS_EDGE],
    )
    defs = _column_defs(options)
    signed = defs[MODEL_EDGE]["valueFormatter"].js_code
    plain = defs[CONSENSUS_EDGE]["valueFormatter"].js_code
    # The signed formatter (edge columns) prefixes a "+" on positives; the plain one doesn't.
    assert ">0?'+'" in signed
    assert ">0?'+'" not in plain
    assert "%" in signed and "%" in plain


def test_decimal_cols_get_a_three_decimal_formatter_value_stays_numeric() -> None:
    """Gate matrix's Brier Skill Score renders at 3 dp (P8 Task R5.e) via a ``toFixed(3)``
    valueFormatter; the underlying value stays numeric so the grid still click-sorts on it.
    Only ``decimal_cols`` get it — an integer count column (n_fails) stays formatter-free.
    """
    df = pd.DataFrame({"brier_skill_score": [0.123456, -0.05], "n_fails": [1, 2]})
    options = build_themed_grid_options(
        df, numeric_cols=["brier_skill_score", "n_fails"], decimal_cols=["brier_skill_score"]
    )
    defs = _column_defs(options)
    assert "toFixed(3)" in defs["brier_skill_score"]["valueFormatter"].js_code
    assert "valueFormatter" not in defs["n_fails"]


def test_heatmap_thresholds_are_absolute_not_fraction_of_max() -> None:
    options = build_themed_grid_options(
        _GRID_DF, numeric_cols=_NUMERIC, heatmap_col=MODEL_EDGE, heatmap_center=0.0
    )
    js = _column_defs(options)[MODEL_EDGE]["cellStyle"].js_code
    # Bucket boundaries are the fixed ±MILD/±STRONG edges baked absolutely, independent of
    # the column's own max — so an edge-sorted first screen no longer paints solid (the old
    # fraction-of-column-max bug, where the max set the scale and everything looked extreme).
    assert repr(-_HEAT_STRONG_EDGE) in js and repr(_HEAT_STRONG_EDGE) in js
    assert repr(-_HEAT_MILD_EDGE) in js


def test_header_tooltips_attached() -> None:
    help_map = {MODEL_EDGE: "your edge vs DFS", CONSENSUS_EDGE: "book's view"}
    options = build_themed_grid_options(
        _GRID_DF, numeric_cols=_NUMERIC, heatmap_col=MODEL_EDGE, header_help=help_map
    )
    defs = _column_defs(options)
    assert defs[MODEL_EDGE]["headerTooltip"] == "your edge vs DFS"
    assert defs[CONSENSUS_EDGE]["headerTooltip"] == "book's view"


def test_apply_lens_each_narrows() -> None:
    df = pd.DataFrame(
        {
            "Model EV": [1.20, 0.95, 1.05, 1.40, 1.01],
            "Market EV": [1.02, 1.00, 1.05, 0.90, 0.98],
            "Boost": [1.0, 1.0, 1.0, 3.0, 1.0],
            "Date": ["2026-07-04", "2026-07-04", "2026-07-05", "2026-07-04", "2026-07-05"],
        }
    )
    assert apply_lens(df, "All").equals(df)
    for lens in LENSES:
        if lens == "All":
            continue
        out = apply_lens(df, lens)
        assert set(out.index).issubset(df.index)
        assert len(out) < len(df), f"{lens} did not narrow the board"
    # You play against DFS: Contrarian = model +EV vs DFS where the book is under break-even;
    # Consensus = both the model and the book score it over break-even.
    assert set(apply_lens(df, "Contrarian").index) == {3, 4}
    assert set(apply_lens(df, "Consensus").index) == {0, 2}
    # Tonight = the soonest slate date present, so it degrades without wall-clock/timezone
    # parsing — here the minimum Date is 2026-07-04.
    assert set(apply_lens(df, "Tonight").index) == {0, 1, 3}


def test_apply_lens_missing_cols_unchanged() -> None:
    df = pd.DataFrame({"Model EV": [1.2, 1.0]})
    assert apply_lens(df, "Sharp edges").equals(df)
    assert apply_lens(df, "nonsense-lens").equals(df)


def test_tonight_lens_missing_date_col_unchanged() -> None:
    df = pd.DataFrame({"Model EV": [1.2, 1.0], "Market EV": [1.0, 1.0], "Boost": [1.0, 1.0]})
    assert apply_lens(df, "Tonight").equals(df)


def test_add_edges_are_ev_minus_one() -> None:
    df = pd.DataFrame({"Model EV": [1.20, 1.00], "Market EV": [1.05, 0.90]})
    out = add_edges(df)
    assert out[MODEL_EDGE].tolist() == [pytest.approx(0.20), pytest.approx(0.00)]
    assert out[CONSENSUS_EDGE].tolist() == [pytest.approx(0.05), pytest.approx(-0.10)]
    # Each edge needs only its own EV column.
    assert MODEL_EDGE in add_edges(df.drop(columns=["Market EV"])).columns
    assert CONSENSUS_EDGE not in add_edges(df.drop(columns=["Market EV"])).columns
    assert CONSENSUS_EDGE in add_edges(df.drop(columns=["Model EV"])).columns
    assert MODEL_EDGE not in add_edges(df.drop(columns=["Model EV"])).columns


def test_add_match_column_player_team_first() -> None:
    df = pd.DataFrame(
        {
            "Team": ["LVA", "NYK"],
            "Opponent": ["IND", "SAS"],
            "Home": [True, False],
        }
    )
    out = add_match_column(df)
    assert out["Match"].tolist() == ["LVA v IND", "NYK @ SAS"]


def test_add_match_column_missing_home_defaults_to_away() -> None:
    # current_offers.parquet snapshots predating the upstream Home passthrough
    # fix lack the column entirely; must degrade to "@", not crash.
    df = pd.DataFrame({"Team": ["LVA", "NYK"], "Opponent": ["IND", "SAS"]})
    out = add_match_column(df)
    assert out["Match"].tolist() == ["LVA @ IND", "NYK @ SAS"]


def test_add_market_display_keeps_slug_column() -> None:
    df = pd.DataFrame({"League": ["NBA", "NFL"], "Market": ["PTS", "qb-yards"]})
    out = add_market_display(df)
    assert "Market" in out.columns and out["Market"].tolist() == ["PTS", "qb-yards"]
    assert "Market Display" in out.columns
    # An unmapped slug falls back to itself (helpers.market_display_name's own contract).


def test_arrow_col_adds_cellrenderer_and_hidden_cols_hide() -> None:
    df = pd.DataFrame({"Line": [23.5, 8.5], "Bet": ["Over", "Under"], "Market": ["PTS", "AST"]})
    options = build_themed_grid_options(
        df, numeric_cols=["Line"], arrow_col="Line", hidden_cols=["Bet", "Market"]
    )
    defs = _column_defs(options)
    renderer = defs["Line"]["cellRenderer"].js_code
    assert "Bet" in renderer
    # Must be a class renderer exposing getGui — a plain-function renderer returning an SVG
    # string renders it as escaped markup in AG Grid 34 (the exact P8 Board arrow bug).
    assert "getGui" in renderer, "arrow renderer is not class-based (would escape the SVG)"
    assert defs["Bet"]["hide"] is True
    assert defs["Market"]["hide"] is True


def test_arrow_col_noop_without_bet_column() -> None:
    df = pd.DataFrame({"Line": [23.5, 8.5]})
    options = build_themed_grid_options(df, numeric_cols=["Line"], arrow_col="Line")
    defs = _column_defs(options)
    assert "cellRenderer" not in defs["Line"]
    assert "Bet" not in defs


def test_hover_css_is_gold_rail_client_side() -> None:
    # AG Grid v34 paints the row-hover background through an absolutely-positioned ::before
    # overlay that covers a .ag-row-hover{background-color} rule, so the hover wash must
    # drive AG Grid's own --ag-row-hover-color variable; the gold rail is a first-cell inset.
    assert _HOVER_CSS[".ag-root-wrapper"]["--ag-row-hover-color"] == "rgba(201,162,39,0.09)"
    assert (
        _HOVER_CSS[".ag-row-hover .ag-cell:first-child"]["box-shadow"] == "inset 3px 0 0 #C9A227"
    )


def test_flag_col_adds_get_row_style_jscode() -> None:
    """Lab Diagnostics' amber rail (P8 Task B4): a row-conditional style needs AG Grid's
    ``getRowStyle`` callback, since the existing ``rowStyle`` dict is static/uniform.
    """
    df = pd.DataFrame({"Market": ["A", "B"], "BSS": [-0.02, 0.03]})
    options = build_themed_grid_options(df, numeric_cols=["BSS"], flag_col="BSS", flag_below=0.0)
    assert hasattr(options["getRowStyle"], "js_code")
    assert theme.ORANGE in options["getRowStyle"].js_code
    assert "boxShadow" in options["getRowStyle"].js_code


def test_flag_col_omitted_by_default() -> None:
    df = pd.DataFrame({"Market": ["A", "B"], "BSS": [-0.02, 0.03]})
    options = build_themed_grid_options(df, numeric_cols=["BSS"])
    assert "getRowStyle" not in options


def test_rail_col_adds_get_row_style_jscode_for_each_mapped_value() -> None:
    """Gate matrix's 3-state rail (P8 Task B5): a categorical string column, unlike
    ``flag_col``'s numeric threshold — needs its own ``getRowStyle`` keyed off a
    precomputed ``rail`` value rather than a live comparison.
    """
    df = pd.DataFrame({"Market": ["A", "B", "C"], "rail": ["amber", "red", "none"]})
    options = build_themed_grid_options(
        df,
        numeric_cols=[],
        rail_col="rail",
        rail_colors={"amber": theme.ORANGE, "red": theme.RED},
    )
    assert hasattr(options["getRowStyle"], "js_code")
    assert theme.ORANGE in options["getRowStyle"].js_code
    assert theme.RED in options["getRowStyle"].js_code
    assert "boxShadow" in options["getRowStyle"].js_code


def test_rail_col_survives_as_row_data_when_also_hidden() -> None:
    """A ``hide``-configured column still rides along in AG Grid's row data (only its own
    visible grid column is suppressed) — ``getRowStyle`` reads ``params.data[rail_col]``
    from that row data, so a caller that both hides ``rail`` (it's not meant to display)
    and uses it as ``rail_col`` must not lose the value the callback depends on. Regression
    for a real P8 Task B5 bug: an earlier ``gate_matrix.render_gate_matrix`` dropped the
    ``rail`` column from the frame entirely before calling ``AgGrid``, so the rail never
    painted despite ``getRowStyle`` being wired correctly — `hide=True` keeps the column
    (and its data) present, unlike dropping it from the DataFrame outright.
    """
    df = pd.DataFrame({"Market": ["A", "B"], "rail": ["amber", "none"]})
    options = build_themed_grid_options(
        df,
        numeric_cols=[],
        hidden_cols=["rail"],
        rail_col="rail",
        rail_colors={"amber": theme.ORANGE},
    )
    defs = _column_defs(options)
    assert defs["rail"]["hide"] is True
    assert hasattr(options["getRowStyle"], "js_code")
    assert theme.ORANGE in options["getRowStyle"].js_code


def test_rail_col_omitted_by_default() -> None:
    df = pd.DataFrame({"Market": ["A", "B"], "rail": ["amber", "none"]})
    options = build_themed_grid_options(df, numeric_cols=[])
    assert "getRowStyle" not in options


def test_glyph_cols_color_pass_fail_glyphs_green_red() -> None:
    """Gate matrix's ●/○ pass/fail glyphs (P8 Task B5) get a boolean-keyed cellStyle —
    green for the pass glyph, red for the fail glyph — never the gold accent, since
    this is a data mark (DESIGN §4a binds gold to decoration/interactive-highlight only).
    """
    df = pd.DataFrame({"g1_pass": ["●", "○"], "g2_pass": ["○", "●"]})
    options = build_themed_grid_options(
        df,
        numeric_cols=[],
        glyph_cols=["g1_pass", "g2_pass"],
        glyph_colors={"●": theme.GREEN, "○": theme.RED},
    )
    defs = _column_defs(options)
    for col in ("g1_pass", "g2_pass"):
        style = _cellstyle_text(defs[col]["cellStyle"])
        assert theme.GREEN in style
        assert theme.RED in style
        assert theme.GOLD not in style
        # R5.d: the glyph is colored TEXT, not a filled cell — a background swatch would
        # hide the ●/○ mark. The token must ride on 'color', never 'backgroundColor'.
        assert "backgroundColor" not in style


def test_glyph_cols_noop_without_kwarg() -> None:
    df = pd.DataFrame({"g1_pass": ["●", "○"]})
    options = build_themed_grid_options(df, numeric_cols=[])
    defs = _column_defs(options)
    assert "cellStyle" not in defs["g1_pass"]


def test_market_display_labels_back_to_market_header() -> None:
    """Market Display carries the slug's prose label but the grid header reads "Market"
    (mockup ``docs/mockups/p8-board.html:104``) — columns.LABELS renames it back, same
    mechanism as Consensus Edge -> Cons Edge and Win Prob -> Win %.
    """
    assert LABELS["Market Display"] == "Market"
    df = pd.DataFrame({"Market Display": ["Points", "Assists"]}).rename(columns=LABELS)
    options = build_themed_grid_options(df, numeric_cols=[])
    defs = _column_defs(options)
    assert "Market" in defs and "Market Display" not in defs


def test_board_grid_omits_team_opponent_kelly_date() -> None:
    """Board 14->10: Team/Opponent/Bet/Kelly/Date must never resurface as displayed
    grid columns — Match/Market/Platform/League already cover what a viewer needs
    (Tonight lens covers the recency Date carried). Mirrors surfaces/board.py's own
    MAIN_COLS + hidden-cols contract without executing the Streamlit script (importing
    board.py runs the whole page against real snapshot data).
    """
    main_cols = [
        "League",
        "Match",
        "Player",
        "Market Display",
        "Line",
        "Boost",
        "Win Prob",
        "Model Edge",
        "Consensus Edge",
        "Platform",
    ]
    df = pd.DataFrame(
        {
            **{c: ["x", "y"] for c in main_cols},
            "Bet": ["Over", "Under"],
            "Market": ["PTS", "AST"],
            # Columns that must NOT survive into display_cols even if present upstream.
            "Team": ["NYK", "LVA"],
            "Opponent": ["SAS", "IND"],
            "Kelly": [0.1, 0.2],
            "Date": ["2026-07-04", "2026-07-05"],
        }
    )
    display_cols = [c for c in [*main_cols, "Bet", "Market"] if c in df.columns]
    grid_df = df[display_cols].rename(columns={"Market": "Market Slug"})
    grid_df = grid_df.rename(columns=LABELS)
    options = build_themed_grid_options(
        grid_df,
        numeric_cols=["Line", "Boost"],
        arrow_col="Line",
        hidden_cols=["Bet", "Market Slug"],
    )
    defs = _column_defs(options)
    for dropped in ("Team", "Opponent", "Kelly", "Date"):
        assert dropped not in defs, f"{dropped} leaked into the Board grid's columnDefs"
    assert "Match" in defs and "Market" in defs
