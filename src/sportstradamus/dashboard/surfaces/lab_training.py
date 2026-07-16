"""Per-cell training diagnostics from the latest `meditate` run.

Renders the wide one-row-per-`(league, market)` schema written by
``training.report.report()`` with the lifecycle state joined in at read
time via ``training.graduation.lifecycle_table()``. Tabs split the slate
by metric family; each column carries a one-line help string indicating
whether higher or lower is better (or that the field is informational).
"""

import datetime as dt

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.gate_matrix import gate_matrix_frame, render_gate_matrix
from sportstradamus.dashboard.components.hero import page_hero
from sportstradamus.dashboard.components.lab_filters import apply_lab_filters, render_lab_filters
from sportstradamus.dashboard.data import (
    format_ts,
    load_model_stats,
    load_stat_meta,
    sport_filtered,
)
from sportstradamus.dashboard.surfaces.lab_training_charts import lifecycle_funnel
from sportstradamus.dashboard.viewport import is_mobile
from sportstradamus.helpers.io import LIVE_METRICS_PATH, MODEL_STATS_PATH
from sportstradamus.training.graduation import lifecycle_table

# Single source of truth for the per-column direction help strings rendered
# through Streamlit's ``column_config``. Columns absent from this map don't
# get an annotation.
HIGHER = "↑ higher is better"
LOWER = "↓ lower is better"
INFO = "informational"
DIRECTIONS: dict[str, str] = {
    # Scoring rules
    "brier_book": LOWER,
    "brier_model": LOWER,
    "brier_skill_score": HIGHER,
    "log_loss_book": LOWER,
    "log_loss_model": LOWER,
    "nll": LOWER,
    "ece_equal_mass": LOWER,
    "ece_debiased": LOWER,
    # Discrimination
    "roc_auc": HIGHER,
    "accuracy": HIGHER,
    "precision_over": HIGHER,
    "precision_under": HIGHER,
    "prediction_std": INFO,
    # Rates
    "predicted_over_rate": INFO,
    "empirical_over_rate": INFO,
    "frac_ev_gt_line": INFO,
    "over_pct_ev_gt": HIGHER,
    "over_pct_ev_lt": LOWER,
    # EV & lines
    "model_ev": INFO,
    "mean_line": INFO,
    "result_mean": INFO,
    "mean_ev_diff": HIGHER,
    "median_ev_diff": HIGHER,
    # Kelly & blending
    "kelly_shrinkage": HIGHER,
    "model_weight": INFO,
    # Dispersion
    "model_shape": INFO,
    "empirical_shape": INFO,
    "shape_ratio": INFO,
    "marginal_shape": INFO,
    "dispersion_cal": INFO,
    "skew_cal": INFO,
    # Ship gates
    "g1_brier_diff_mean": LOWER,
    "g1_brier_diff_ci_lo": LOWER,
    "g1_brier_diff_ci_hi": LOWER,
    "g2_star_z": LOWER,
    "g3_bench_z": LOWER,
    "g4_iqr_ratio": INFO,
    "g5_ece_debiased": LOWER,
    "g6_star_ci_hi": LOWER,
    "g6_star_ref": INFO,
    "g6_recent_corr": LOWER,
    "g1_pass": INFO,
    "g2_pass": INFO,
    "g3_pass": INFO,
    "g4_pass": INFO,
    "g5_pass": INFO,
    "g6_pass": INFO,
    "ship": INFO,
    # Hyperparameters
    "hp_rounds": INFO,
    "hp_leaves": INFO,
    "hp_lr": INFO,
    "hp_min_child": INFO,
    "hp_l1": INFO,
    "hp_l2": INFO,
    "cv": INFO,
    "std": INFO,
    "historical_zero_rate": INFO,
    # Lifecycle / identity
    "lifecycle_state": INFO,
    "n_validation": INFO,
}

ID_COLS = ["league", "market", "distribution", "shipped", "lifecycle_state"]

TAB_COLUMNS: dict[str, list[str]] = {
    "Overview": [
        "n_validation",
        "brier_skill_score",
        "kelly_shrinkage",
        "ship",
    ],
    "Scoring rules": [
        "brier_book",
        "brier_model",
        "brier_skill_score",
        "log_loss_book",
        "log_loss_model",
        "nll",
        "ece_equal_mass",
        "ece_debiased",
    ],
    "Discrimination": [
        "roc_auc",
        "accuracy",
        "precision_over",
        "precision_under",
        "prediction_std",
    ],
    "Rates": [
        "predicted_over_rate",
        "empirical_over_rate",
        "frac_ev_gt_line",
        "over_pct_ev_gt",
        "over_pct_ev_lt",
    ],
    "EV & lines": [
        "model_ev",
        "mean_line",
        "result_mean",
        "mean_ev_diff",
        "median_ev_diff",
    ],
    "Kelly & blending": ["kelly_shrinkage", "model_weight", "brier_skill_score"],
    "Dispersion": [
        "model_shape",
        "empirical_shape",
        "shape_ratio",
        "marginal_shape",
        "dispersion_cal",
        "skew_cal",
    ],
    "Ship gates": [
        "g1_brier_diff_mean",
        "g1_brier_diff_ci_lo",
        "g1_brier_diff_ci_hi",
        "g2_star_z",
        "g3_bench_z",
        "g4_iqr_ratio",
        "g5_ece_debiased",
        "g6_star_ci_hi",
        "g6_star_ref",
        "g6_recent_corr",
        "g1_pass",
        "g2_pass",
        "g3_pass",
        "g4_pass",
        "g5_pass",
        "g6_pass",
        "ship",
    ],
    "Hyperparameters": [
        "hp_rounds",
        "hp_leaves",
        "hp_lr",
        "hp_min_child",
        "hp_l1",
        "hp_l2",
        "cv",
        "std",
        "historical_zero_rate",
    ],
}

TAB_CAPTIONS: dict[str, str] = {
    "Overview": ("One row per (league, market). Columns are annotated ↑/↓/informational."),
    "Scoring rules": (
        "Proper scoring rules on the validation set. The model's `*_model` column"
        " should beat the `*_book` baseline; the derived `brier_skill_score` is the"
        " single-number gate the kelly chain reads. Columns are annotated ↑/↓/informational."
    ),
    "Discrimination": (
        "How well the model separates winners from losers. Columns are annotated ↑/↓/informational."
    ),
    "Rates": (
        "Rates and conditional Over% slices. Compare `predicted_over_rate` to"
        " `empirical_over_rate` for bias; `over_pct_ev_gt` / `over_pct_ev_lt` show"
        " whether EV-positive picks actually win at higher rates."
        " Columns are annotated ↑/↓/informational."
    ),
    "EV & lines": (
        "Expected-value diagnostics vs. the bookmaker line. Mean/median EV diffs are"
        " the model's edge over the line. Columns are annotated ↑/↓/informational."
    ),
    "Kelly & blending": (
        "Skill score relative to the book and the derived Kelly shrinkage."
        " `kelly_shrinkage = clip(brier_skill_score, 0, 1)`."
        " Columns are annotated ↑/↓/informational."
    ),
    "Dispersion": (
        "Shape calibration diagnostics. `shape_ratio≈1.0` means model dispersion"
        " matches the empirical outcome dispersion. Columns are informational."
    ),
    "Ship gates": (
        "Six offline ship gates (training.scorecard.compute_gates). G1 = paired"
        " Brier CI vs book (ci_hi < 0 ⇒ model wins). G2/G3 = top-decile/bottom-quartile"
        " bias-over-spread z. G4 = predicted vs realized IQR ratio (≈1.0 ideal)."
        " G5 = Roelofs-debiased equal-mass ECE. G6 = anti-shrinkage (recent-form CI"
        " vs. a league reference ratio). `ship` is the AND of all six gates."
    ),
    "Hyperparameters": (
        "Optuna-tuned LightGBMLSS hyperparameters and per-market scale references."
        " Informational only."
    ),
}


def _column_config(view: pd.DataFrame, cols: list[str]) -> dict[str, st.column_config.Column]:
    """Streamlit column_config: a 3-decimal NumberColumn for float metric columns, a plain
    Column (carrying the ↑/↓ direction help) for everything else. Float detection is by
    dtype, so integer counts and boolean gate flags keep their natural rendering.
    """
    config: dict[str, st.column_config.Column] = {}
    for c in cols:
        help_str = DIRECTIONS.get(c)
        if c in view.columns and pd.api.types.is_float_dtype(view[c]):
            config[c] = st.column_config.NumberColumn(help=help_str, format="%.3f")
        elif help_str:
            config[c] = st.column_config.Column(help=help_str)
    return config


def _render_tab(tab_name: str, metric_cols: list[str], view: pd.DataFrame) -> None:
    st.caption(TAB_CAPTIONS[tab_name])
    cols = [c for c in ID_COLS + metric_cols if c in view.columns]
    st.dataframe(
        view[cols].sort_values(["league", "market"]),
        width="stretch",
        height=560,
        hide_index=True,
        column_config=_column_config(view, cols),
    )


mtime = (
    format_ts(
        dt.datetime.fromtimestamp(MODEL_STATS_PATH.stat().st_mtime).isoformat(timespec="seconds")
    )
    if MODEL_STATS_PATH.is_file()
    else "no meditate run on record"
)
page_hero("MODEL LAB · TRAINING", "Model Training Diagnostics", mtime)
if is_mobile():
    st.caption("Best at a desk — this page keeps its desktop layout.")


stats = load_model_stats()
if stats.empty:
    st.info("No model stats found. Run `poetry run meditate` to generate `model_stats.parquet`.")
    st.stop()

# The global sport switch narrows every other surface; apply it here too so the training
# board follows the selected league (stats carries the lowercase `league` column).
stats = sport_filtered(stats)
if stats.empty:
    st.info("No model stats for the selected sport.")
    st.stop()

# Lifecycle state joins offline Gate-1 (this parquet) with live Gate-2
# (live_metrics_per_market.parquet, written daily by `reflect`). Computed
# at read time so the lifecycle classification reflects today's live data
# rather than whatever it was at the last meditate run.
lifecycle = lifecycle_table(MODEL_STATS_PATH, LIVE_METRICS_PATH)
if not lifecycle.empty:
    stats = stats.merge(
        lifecycle[["league", "market", "lifecycle_state"]],
        on=["league", "market"],
        how="left",
    )
else:
    stats["lifecycle_state"] = pd.NA

# The former body Leagues/Markets multiselects duplicated the shared Lab filter panel
# (rendered below) and defaulted to every market — a wall of chips. Keep only the
# Lifecycle axis here; it isn't a stat_meta config dimension the shared panel covers.
lifecycle_states = sorted(stats["lifecycle_state"].dropna().unique())
sel_lifecycle = (
    st.multiselect("Lifecycle", lifecycle_states, default=lifecycle_states, key="lab_lifecycle")
    if lifecycle_states
    else []
)
view = stats
if sel_lifecycle:
    view = stats.loc[stats["lifecycle_state"].isin(sel_lifecycle)]

stat_meta = load_stat_meta()
lab_sel = render_lab_filters(stat_meta, collapsed=True)
view = apply_lab_filters(view, stat_meta, lab_sel)

st.caption(f"Showing **{len(view):,}** cells.")

matrix = gate_matrix_frame(view)
one_gate_short = int((matrix["n_fails"] == 1).sum())

st.subheader("Run at a Glance")
tile_cells, tile_ship, tile_short, tile_withheld = st.columns(4)
tile_cells.metric("Cells trained", f"{len(view):,}")
tile_ship.metric("Shipping (all gates)", f"{int(view['ship'].eq(True).sum()):,}")
tile_short.metric("One gate short", f"{one_gate_short:,}")
tile_withheld.metric("Withheld", f"{int(view['shipped'].eq('withheld').sum()):,}")

st.plotly_chart(
    lifecycle_funnel(
        trained=len(view),
        g1_pass=int(view["g1_pass"].eq(True).sum()),
        all_gates=int(view["ship"].eq(True).sum()),
        graduated=int(view["lifecycle_state"].eq("graduated").sum()),
    ),
    width="stretch",
)

st.subheader("Gate Matrix")
if not matrix.empty:
    render_gate_matrix(matrix)

tabs = st.tabs(list(TAB_COLUMNS.keys()))
for tab, name in zip(tabs, TAB_COLUMNS.keys(), strict=True):
    with tab:
        _render_tab(name, TAB_COLUMNS[name], view)
