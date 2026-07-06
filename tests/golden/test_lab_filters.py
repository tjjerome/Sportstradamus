"""Golden pins for the shared Lab filter panel (§3.7) and its stat_meta loader.

``apply_lab_filters`` is the pure, directly-testable half — mask logic and
join-casing are pinned against synthetic ``meta`` frames. ``load_stat_meta``
is pinned against the real ``stat_meta.json`` so the sparse-key defaulting
matches production data, not just an idealized fixture.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.components.lab_filters import (
    FILTER_AXES,
    apply_lab_filters,
    render_lab_filters,
)
from sportstradamus.dashboard.data import load_stat_meta

_META = pd.DataFrame(
    [
        {
            "league": "NBA",
            "market": "AST",
            "dist": "SkewNormal",
            "target_normalization": "centered_additive_mean10",
            "posthoc": "none",
            "blending": "crps",
            "hpo_selection": "none",
            "count_dispersion_objective": "none",
            "zinb_mode": "none",
            "shipped": "devel",
        },
        {
            "league": "NBA",
            "market": "PTS",
            "dist": "SkewNormal",
            "target_normalization": "none",
            "posthoc": "roe_mean",
            "blending": "none",
            "hpo_selection": "calibrated",
            "count_dispersion_objective": "none",
            "zinb_mode": "none",
            "shipped": "withheld",
        },
        {
            "league": "WNBA",
            "market": "FTM",
            "dist": "ZINB",
            "target_normalization": "none",
            "posthoc": "none",
            "blending": "none",
            "hpo_selection": "none",
            "count_dispersion_objective": "pit_ks",
            "zinb_mode": "hurdle",
            "shipped": "devel",
        },
        {
            "league": "MLB",
            "market": "H",
            "dist": "ZINB",
            "target_normalization": "ratio_meanyr",
            "posthoc": "cdf_recal_isotonic",
            "blending": "none",
            "hpo_selection": "none",
            "count_dispersion_objective": "none",
            "zinb_mode": "none",
            "shipped": "withheld",
        },
    ]
)


def _offers_lower() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"league": "NBA", "market": "AST", "row": 1},
            {"league": "NBA", "market": "PTS", "row": 2},
            {"league": "WNBA", "market": "FTM", "row": 3},
            {"league": "MLB", "market": "H", "row": 4},
        ]
    )


def _offers_title() -> pd.DataFrame:
    df = _offers_lower()
    return df.rename(columns={"league": "League", "market": "Market"})


def test_single_axis_selection_narrows_to_matching_cells():
    df = apply_lab_filters(_offers_lower(), _META, {"dist": ["ZINB"]})
    assert sorted(zip(df["league"], df["market"], strict=True)) == [
        ("MLB", "H"),
        ("WNBA", "FTM"),
    ]


def test_multiple_axes_and_the_constraints():
    sel = {"dist": ["ZINB"], "zinb_mode": ["hurdle"]}
    df = apply_lab_filters(_offers_lower(), _META, sel)
    assert sorted(zip(df["league"], df["market"], strict=True)) == [("WNBA", "FTM")]


def test_multiple_axes_with_no_overlap_returns_empty():
    sel = {"dist": ["SkewNormal"], "zinb_mode": ["hurdle"]}
    df = apply_lab_filters(_offers_lower(), _META, sel)
    assert df.empty


def test_empty_selection_list_means_no_constraint():
    df = apply_lab_filters(_offers_lower(), _META, {"dist": []})
    assert len(df) == len(_offers_lower())


def test_missing_axis_key_means_no_constraint():
    df = apply_lab_filters(_offers_lower(), _META, {})
    assert len(df) == len(_offers_lower())


def test_all_empty_selections_is_a_pass_through_of_every_row():
    sel = {axis: [] for axis in FILTER_AXES}
    df = apply_lab_filters(_offers_lower(), _META, sel)
    assert len(df) == len(_offers_lower())


def test_join_casing_title_case_matches_lower_case_behavior():
    sel = {"dist": ["ZINB"]}
    lower = apply_lab_filters(_offers_lower(), _META, sel)
    title = apply_lab_filters(_offers_title(), _META, sel)
    assert sorted(zip(lower["league"], lower["market"], strict=True)) == sorted(
        zip(title["League"], title["Market"], strict=True)
    )


def test_join_casing_preserves_input_columns_and_casing():
    df = apply_lab_filters(_offers_title(), _META, {"dist": ["ZINB"]})
    assert list(df.columns) == ["League", "Market", "row"]


def test_load_stat_meta_nba_ast_sparse_defaults():
    meta = load_stat_meta()
    row = meta.loc[(meta["league"] == "NBA") & (meta["market"] == "AST")].iloc[0]
    assert row["dist"] == "SkewNormal"
    assert row["shipped"] == "devel"
    assert row["target_normalization"] == "centered_additive_mean10"
    assert row["posthoc"] == "none"
    assert row["blending"] == "crps"
    assert row["hpo_selection"] == "none"
    assert row["count_dispersion_objective"] == "none"
    assert row["zinb_mode"] == "none"


def test_load_stat_meta_wnba_ftm_sparse_defaults():
    meta = load_stat_meta()
    row = meta.loc[(meta["league"] == "WNBA") & (meta["market"] == "FTM")].iloc[0]
    assert row["dist"] == "ZINB"
    assert row["shipped"] == "devel"
    assert row["target_normalization"] == "none"
    assert row["posthoc"] == "none"
    assert row["blending"] == "none"
    assert row["hpo_selection"] == "none"
    assert row["count_dispersion_objective"] == "pit_ks"
    assert row["zinb_mode"] == "hurdle"


def test_load_stat_meta_shape_and_no_nulls():
    meta = load_stat_meta()
    assert len(meta) == 98
    assert list(meta.columns) == [
        "league",
        "market",
        "dist",
        "target_normalization",
        "posthoc",
        "blending",
        "hpo_selection",
        "count_dispersion_objective",
        "zinb_mode",
        "shipped",
    ]
    assert not meta.isna().any().any()


def test_render_lab_filters_smoke(monkeypatch):
    """Minimal smoke check — full widget rendering needs the AppTest harness A7 adds."""
    import streamlit as st

    monkeypatch.setattr(st, "session_state", {})
    sel = render_lab_filters(_META, collapsed=False)
    assert set(sel.keys()) <= set(FILTER_AXES)
    for values in sel.values():
        assert isinstance(values, list)
