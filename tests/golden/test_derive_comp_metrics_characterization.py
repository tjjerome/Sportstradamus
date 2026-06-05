"""Characterization of :func:`nfl_fp_loader.derive_comp_metrics`.

A pure ``profile -> profile`` transform (no I/O) decomposed from a single CC-34
function into a flat orchestrator plus position-group helpers. These pins freeze
its two observable contracts:

* **Derived-column order** — the function appends derived columns in a fixed
  insertion order, which becomes the output column order and is part of the
  frozen schema (STYLE_GUIDE §18.9); a position-grouped rewrite must not reorder
  them. Pinned exactly on a full-column and a QB-only profile.
* **Per-position masking + arithmetic** — each metric is written only on its
  position mask (QB / RB / non-QB) and skipped when its source column is absent.
  Spot values are independent literals (not recomputed via the function).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.stats.nfl_fp_loader import derive_comp_metrics

_INPUT_COLUMNS = [
    "position",
    "player_game_count",
    "pass_adv_DB",
    "rush_basic_ATT",
    "rec_adv_RTE",
    "pass_adv_SCRM",
    "pass_basic_YDS.1",
    "rec_basic_REC",
    "rec_adv_CTGT_pct",
    "rec_sep_cov_WIN_RATE.4",
    "rec_adv_YPRR",
    "rec_mz_YPRR.1",
    "rec_mz_YPRR.2",
    "rec_sep_align_SEP_SCORE",
    "rec_sep_align_WIN_RATE",
    "rec_sep_route_SEP_SCORE",
    "rec_sep_route_SEP_SCORE.1",
    "qb_cov_QB_MAN_pct",
    "off_faced_man_pct",
    "rush_adv_Success_pct",
    "rush_adv_EXP_RUN_pct",
    "rush_bellcow_ATT_pct",
    "rush_bellcow_TGT_pct",
]

_DERIVED_ORDER = [
    "dropbacks",
    "attempts",
    "routes",
    "dropbacks_per_game",
    "scrambles_per_dropback",
    "designed_yards_per_game",
    "total_touches_per_game",
    "contested_target_rate",
    "deep_contested_target_rate",
    "man_yprr_diff",
    "zone_yprr_diff",
    "sep_overall",
    "win_rate_overall",
    "sep_routes_mean",
    "cov_man_share",
    "success_rate",
    "exp_run_share",
    "bellcow_score",
]


def _full_profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["QB", "RB", "WR"],
            "player_game_count": [10.0, 12.0, 15.0],
            "pass_adv_DB": [350.0, np.nan, np.nan],
            "rush_basic_ATT": [20.0, 200.0, 5.0],
            "rec_adv_RTE": [np.nan, 50.0, 400.0],
            "pass_adv_SCRM": [30.0, np.nan, np.nan],
            "pass_basic_YDS.1": [180.0, np.nan, np.nan],
            "rec_basic_REC": [2.0, 40.0, 90.0],
            "rec_adv_CTGT_pct": [np.nan, 0.10, 0.25],
            "rec_sep_cov_WIN_RATE.4": [np.nan, 0.40, 0.55],
            "rec_adv_YPRR": [np.nan, 1.2, 2.1],
            "rec_mz_YPRR.1": [np.nan, 1.5, 2.4],
            "rec_mz_YPRR.2": [np.nan, 1.0, 1.9],
            "rec_sep_align_SEP_SCORE": [np.nan, 2.5, 3.1],
            "rec_sep_align_WIN_RATE": [np.nan, 0.45, 0.60],
            "rec_sep_route_SEP_SCORE": [np.nan, 2.2, 2.9],
            "rec_sep_route_SEP_SCORE.1": [np.nan, 2.4, 3.0],
            "qb_cov_QB_MAN_pct": [0.32, np.nan, np.nan],
            "off_faced_man_pct": [0.28, 0.30, 0.31],
            "rush_adv_Success_pct": [np.nan, 0.47, np.nan],
            "rush_adv_EXP_RUN_pct": [np.nan, 0.12, np.nan],
            "rush_bellcow_ATT_pct": [np.nan, 0.65, np.nan],
            "rush_bellcow_TGT_pct": [np.nan, 0.15, np.nan],
        },
        index=["QB1", "RB1", "WR1"],
    )


def test_full_profile_column_order_and_values():
    out = derive_comp_metrics(_full_profile())

    assert list(out.columns) == _INPUT_COLUMNS + _DERIVED_ORDER

    assert round(out.loc["QB1", "dropbacks_per_game"], 6) == 35.0
    assert round(out.loc["QB1", "scrambles_per_dropback"], 6) == 0.085714
    assert round(out.loc["QB1", "designed_yards_per_game"], 6) == 18.0
    assert pd.isna(out.loc["RB1", "dropbacks_per_game"])  # qb mask excludes RB

    assert round(out.loc["RB1", "total_touches_per_game"], 6) == 20.0  # (200 + 40) / 12
    assert round(out.loc["RB1", "success_rate"], 6) == 0.47
    assert round(out.loc["RB1", "exp_run_share"], 6) == 0.12
    assert round(out.loc["RB1", "bellcow_score"], 6) == 0.8  # 0.65 + 0.15
    assert pd.isna(out.loc["WR1", "total_touches_per_game"])  # rb mask excludes WR
    assert pd.isna(out.loc["QB1", "total_touches_per_game"])

    assert round(out.loc["RB1", "contested_target_rate"], 6) == 0.10
    assert round(out.loc["WR1", "contested_target_rate"], 6) == 0.25
    assert round(out.loc["RB1", "man_yprr_diff"], 6) == 0.3  # 1.5 - 1.2
    assert round(out.loc["RB1", "zone_yprr_diff"], 6) == -0.2  # 1.0 - 1.2
    assert round(out.loc["WR1", "sep_overall"], 6) == 3.1
    assert round(out.loc["RB1", "sep_routes_mean"], 6) == 2.3  # mean(2.2, 2.4)
    assert pd.isna(out.loc["QB1", "contested_target_rate"])

    assert round(out.loc["QB1", "cov_man_share"], 6) == 0.32
    assert round(out.loc["RB1", "cov_man_share"], 6) == 0.30
    assert round(out.loc["WR1", "cov_man_share"], 6) == 0.31

    assert round(out.loc["QB1", "dropbacks"], 6) == 350.0
    assert round(out.loc["RB1", "attempts"], 6) == 200.0
    assert round(out.loc["WR1", "routes"], 6) == 400.0


def test_partial_profile_skips_absent_groups():
    """QB-only columns: receiver / RB metrics are skipped, order is still fixed."""
    qb_cols = [
        "position",
        "player_game_count",
        "pass_adv_DB",
        "pass_adv_SCRM",
        "pass_basic_YDS.1",
        "qb_cov_QB_MAN_pct",
    ]
    out = derive_comp_metrics(_full_profile()[qb_cols])

    assert list(out.columns) == [
        *qb_cols,
        "dropbacks",
        "dropbacks_per_game",
        "scrambles_per_dropback",
        "designed_yards_per_game",
        "cov_man_share",
    ]
    assert round(out.loc["QB1", "dropbacks_per_game"], 6) == 35.0
    assert round(out.loc["QB1", "cov_man_share"], 6) == 0.32
    assert pd.isna(out.loc["RB1", "dropbacks_per_game"])


def test_no_position_returns_unchanged():
    no_pos = _full_profile().drop(columns=["position"])
    out = derive_comp_metrics(no_pos)
    assert list(out.columns) == list(no_pos.columns)
    pd.testing.assert_frame_equal(out, no_pos)
