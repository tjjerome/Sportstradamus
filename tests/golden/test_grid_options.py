"""Pin the themed-grid contract: design-token colors, right-aligned numerals, lens math.

``dashboard/components/grid.py`` is the one place the stat grids (Board, Slips Pick'em)
turn a frame into ``gridOptions``. These goldens hold its three promises: numbers never
center (DESIGN.md §4), every color it paints is a committed design token, and the board
lenses / Edge derivation key off the renamed EV columns (``Model EV − Market EV``).
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.columns import add_edge, edge_series
from sportstradamus.dashboard.components.grid import _HEAT_TEXT, build_themed_grid_options
from sportstradamus.dashboard.lenses import LENSES, apply_lens

_HEX = re.compile(r"#[0-9A-Fa-f]{6}")
_GRID_DF = pd.DataFrame(
    {
        "Player": ["A", "B", "C", "D"],
        "Model EV": [1.20, 0.95, 1.05, 1.40],
        "Market EV": [1.02, 1.00, 1.05, 0.90],
        "Edge": [0.18, -0.05, 0.00, 0.50],
        "Kelly": [0.10, 0.00, 0.02, 0.25],
    }
)
_NUMERIC = ["Model EV", "Market EV", "Edge", "Kelly"]


def _column_defs(options: dict) -> dict[str, dict]:
    return {c["field"]: c for c in options["columnDefs"]}


def _cellstyle_text(cell_style: dict) -> str:
    """The cellStyle as text whether it's a plain dict or a ``{"function": expr}`` form."""
    return cell_style.get("function", json.dumps(cell_style))


def test_numeric_cols_right_aligned_never_centered() -> None:
    options = build_themed_grid_options(_GRID_DF, numeric_cols=_NUMERIC, heatmap_col="Edge")
    defs = _column_defs(options)
    for col in _NUMERIC:
        style = _cellstyle_text(defs[col]["cellStyle"])
        assert "'textAlign':'right'" in style or '"textAlign": "right"' in style, (
            f"{col} numeric cell is not right-aligned"
        )
    assert "center" not in json.dumps(options), "a grid cell centers a numeral (DESIGN §4)"


def test_grid_hexes_are_design_tokens() -> None:
    options = build_themed_grid_options(_GRID_DF, numeric_cols=_NUMERIC, heatmap_col="Edge")
    allowed = set(theme.DIVERGING_COLORS) | {_HEAT_TEXT}
    found = set(_HEX.findall(json.dumps(options)))
    assert found, "heatmap baked no colors — expected the diverging ramp"
    assert found <= allowed, f"grid paints a non-token color: {found - allowed}"


def test_heatmap_paints_only_its_column() -> None:
    options = build_themed_grid_options(_GRID_DF, numeric_cols=_NUMERIC, heatmap_col="Edge")
    defs = _column_defs(options)
    assert "backgroundColor" in _cellstyle_text(defs["Edge"]["cellStyle"])
    for col in ("Model EV", "Market EV", "Kelly"):
        assert "backgroundColor" not in _cellstyle_text(defs[col]["cellStyle"]), (
            f"{col} is painted but only the heatmap column should be"
        )


def test_apply_lens_each_narrows() -> None:
    df = pd.DataFrame(
        {
            "Model EV": [1.20, 0.95, 1.05, 1.40, 1.01],
            "Market EV": [1.02, 1.00, 1.05, 0.90, 1.03],
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
    # Edge-keyed lenses split on Model EV − Market EV, not on probability.
    assert set(apply_lens(df, "Contrarian").index) == {0, 3}
    assert set(apply_lens(df, "Consensus").index) == {2, 4}


def test_apply_lens_missing_cols_unchanged() -> None:
    df = pd.DataFrame({"Model EV": [1.2, 1.0]})
    assert apply_lens(df, "Sharp edges").equals(df)
    assert apply_lens(df, "nonsense-lens").equals(df)


def test_add_edge_is_model_minus_market() -> None:
    df = pd.DataFrame({"Model EV": [1.20, 1.00], "Market EV": [1.05, 0.90]})
    out = add_edge(df)
    assert out["Edge"].tolist() == [pytest.approx(0.15), pytest.approx(0.10)]
    assert edge_series(df).tolist() == [pytest.approx(0.15), pytest.approx(0.10)]
    assert "Edge" not in add_edge(df.drop(columns=["Market EV"])).columns
