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
from sportstradamus.dashboard.columns import CONSENSUS_EDGE, MODEL_EDGE, add_edges
from sportstradamus.dashboard.components.grid import _HEAT_TEXT, build_themed_grid_options
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


def test_apply_lens_missing_cols_unchanged() -> None:
    df = pd.DataFrame({"Model EV": [1.2, 1.0]})
    assert apply_lens(df, "Sharp edges").equals(df)
    assert apply_lens(df, "nonsense-lens").equals(df)


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
