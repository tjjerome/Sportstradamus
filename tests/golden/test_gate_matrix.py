"""Pin the gate-matrix sort contract: one-gate-short pinned above multi-fail, stable ties.

``gate_matrix_frame`` is "worst-first" in the actionability sense, not the brokenness
sense: a cell one gate away from shipping is the highest-priority row (a near-miss),
ranked above cells failing multiple gates, which rank above fully-passing cells.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from sportstradamus.dashboard.components.gate_matrix import (
    GATE_COLS,
    gate_matrix_frame,
    render_gate_matrix,
)


def _cell(market: str, fails: set[str], bss: float) -> dict:
    row = {"league": "NBA", "market": market, "brier_skill_score": bss}
    for col in GATE_COLS:
        row[col] = col not in fails
    return row


def _stats_every_failcount() -> pd.DataFrame:
    # One cell per fail-count 0..6, plus two extra n_fails==2 cells (different original
    # row order + different BSS) to exercise the within-tier BSS tiebreak and stability.
    rows = [
        _cell("ZERO", set(), 0.10),
        _cell("ONE", {"g3_pass"}, 0.05),
        _cell("TWO_A", {"g1_pass", "g2_pass"}, 0.02),
        _cell("TWO_B", {"g4_pass", "g5_pass"}, 0.02),
        _cell("THREE", {"g1_pass", "g2_pass", "g3_pass"}, -0.01),
        _cell("FOUR", {"g1_pass", "g2_pass", "g3_pass", "g4_pass"}, -0.02),
        _cell("FIVE", {"g1_pass", "g2_pass", "g3_pass", "g4_pass", "g5_pass"}, -0.03),
        _cell("SIX", set(GATE_COLS), -0.04),
    ]
    return pd.DataFrame(rows)


def test_one_gate_short_pinned_above_two_fail():
    out = gate_matrix_frame(_stats_every_failcount()).reset_index(drop=True)
    one_fail_pos = out.index[out["market"] == "ONE"][0]
    two_plus_pos = out.index[out["n_fails"] >= 2]
    assert out.loc[one_fail_pos, "n_fails"] == 1
    assert len(two_plus_pos) == 6  # TWO_A, TWO_B, THREE, FOUR, FIVE, SIX
    # Every n_fails==1 row must sort strictly above every n_fails>=2 row.
    assert one_fail_pos < two_plus_pos.min()


def test_multi_fail_block_sorted_by_fails_descending():
    out = gate_matrix_frame(_stats_every_failcount()).reset_index(drop=True)
    multi = out.loc[out["n_fails"] >= 2].reset_index(drop=True)
    fails_seq = multi["n_fails"].tolist()
    assert fails_seq == sorted(fails_seq, reverse=True), (
        f"multi-fail block not sorted by n_fails desc: {fails_seq}"
    )
    assert fails_seq[0] == 6
    assert fails_seq[-1] == 2


def test_fully_passing_cells_sort_last():
    out = gate_matrix_frame(_stats_every_failcount()).reset_index(drop=True)
    zero_pos = out.index[out["market"] == "ZERO"][0]
    assert zero_pos == len(out) - 1, "a fully-passing cell must sort after every failing cell"
    assert out.loc[zero_pos, "rail"] == "none"


def test_rail_values_match_tier():
    out = gate_matrix_frame(_stats_every_failcount())
    rails = dict(zip(out["market"], out["rail"], strict=True))
    assert rails["ONE"] == "amber"
    for m in ("TWO_A", "TWO_B", "THREE", "FOUR", "FIVE", "SIX"):
        assert rails[m] == "red", f"{m} should carry the red multi-fail rail"
    assert rails["ZERO"] == "none"


def test_within_tier_ties_break_on_bss_ascending():
    # TWO_A (bss=0.02) and TWO_B (bss=0.02) tie on n_fails and BSS by construction here —
    # rebuild with distinct BSS to check the ascending tiebreak fires.
    rows = [
        _cell("HIGH_BSS", {"g1_pass", "g2_pass"}, 0.50),
        _cell("LOW_BSS", {"g3_pass", "g4_pass"}, -0.50),
    ]
    out = gate_matrix_frame(pd.DataFrame(rows)).reset_index(drop=True)
    assert out["market"].tolist() == ["LOW_BSS", "HIGH_BSS"], (
        "within a tier, worse (lower) BSS must sort first"
    )


def test_stable_sort_preserves_original_order_on_full_ties():
    # Two n_fails==2 rows tied on n_fails AND brier_skill_score — only original row
    # order can distinguish them. Without kind="stable" pandas' default quicksort
    # is not guaranteed to preserve this, so this test would be flaky/wrong without it.
    rows = [
        _cell("FIRST", {"g1_pass", "g2_pass"}, 0.00),
        _cell("SECOND", {"g3_pass", "g4_pass"}, 0.00),
    ]
    df = pd.DataFrame(rows)
    out = gate_matrix_frame(df).reset_index(drop=True)
    assert out["market"].tolist() == ["FIRST", "SECOND"], (
        "rows tied on every sort key must keep their original relative order"
    )
    # Reversed input must still come out in ITS original (now-reversed) order — proves
    # the assertion above isn't coincidentally matching input order for other reasons.
    out_rev = gate_matrix_frame(df.iloc[::-1].reset_index(drop=True)).reset_index(drop=True)
    assert out_rev["market"].tolist() == ["SECOND", "FIRST"]


def test_n_fails_counts_false_gate_cols():
    df = pd.DataFrame([_cell("X", {"g2_pass", "g4_pass", "g6_pass"}, 0.0)])
    out = gate_matrix_frame(df)
    assert out.iloc[0]["n_fails"] == 3


def test_pure_no_mutation_of_input():
    df = _stats_every_failcount()
    original = df.copy()
    gate_matrix_frame(df)
    pd.testing.assert_frame_equal(df, original)


def test_render_gate_matrix_keeps_rail_in_row_data_as_hidden_column():
    """Regression: an earlier ``render_gate_matrix`` selected columns for display
    without including ``rail``, so ``getRowStyle`` (which reads ``params.data.rail``)
    always saw ``undefined`` and no row ever painted — despite ``rail_col="rail"``
    being passed correctly. ``rail`` must ride along in the frame AgGrid renders,
    hidden via ``hidden_cols`` rather than dropped from the DataFrame outright.
    """
    matrix = gate_matrix_frame(_stats_every_failcount())
    with patch("sportstradamus.dashboard.components.gate_matrix.render_themed_grid") as mock_render:
        render_gate_matrix(matrix)
    displayed_df = mock_render.call_args.args[0]
    kwargs = mock_render.call_args.kwargs
    assert "rail" in displayed_df.columns, "rail dropped from the frame AgGrid renders"
    assert kwargs.get("rail_col") == "rail"
    assert "rail" in kwargs.get("hidden_cols", ()), "rail must be hidden, not absent"
    assert set(GATE_COLS) == set(kwargs.get("glyph_cols", ()))
    for col in GATE_COLS:
        assert set(displayed_df[col].unique()) <= {"●", "○"}, f"{col} not glyph-mapped"
