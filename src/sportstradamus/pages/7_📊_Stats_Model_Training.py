"""Per-cell training diagnostics from the latest `meditate` run.

Renders the wide one-row-per-`(league, market)` schema written by
``training.report.report()`` with the lifecycle state joined in at read
time via ``training.graduation.lifecycle_table()``. Tabs split the slate
by metric family; each column carries a one-line help string indicating
whether higher or lower is better (or that the field is informational).
"""

import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from sportstradamus.dashboard_data import format_ts, load_model_stats, render_banner
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
    # Ship gates
    "g1_brier_diff_mean": LOWER,
    "g1_brier_diff_ci_lo": LOWER,
    "g1_brier_diff_ci_hi": LOWER,
    "g2_star_z": LOWER,
    "g3_bench_z": LOWER,
    "g4_iqr_ratio": INFO,
    "g5_ece_debiased": LOWER,
    "g1_pass": INFO,
    "g2_pass": INFO,
    "g3_pass": INFO,
    "g4_pass": INFO,
    "g5_pass": INFO,
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
    ],
    "Ship gates": [
        "g1_brier_diff_mean",
        "g1_brier_diff_ci_lo",
        "g1_brier_diff_ci_hi",
        "g2_star_z",
        "g3_bench_z",
        "g4_iqr_ratio",
        "g5_ece_debiased",
        "g1_pass",
        "g2_pass",
        "g3_pass",
        "g4_pass",
        "g5_pass",
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
    "Overview": (
        "One row per (league, market). Columns are annotated ↑/↓/informational."
    ),
    "Scoring rules": (
        "Proper scoring rules on the validation set. The model's `*_model` column"
        " should beat the `*_book` baseline; the derived `brier_skill_score` is the"
        " single-number gate the kelly chain reads. Columns are annotated ↑/↓/informational."
    ),
    "Discrimination": (
        "How well the model separates winners from losers."
        " Columns are annotated ↑/↓/informational."
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
        "Five offline ship gates (training.scorecard.compute_gates). G1 = paired"
        " Brier CI vs book (ci_hi < 0 ⇒ model wins). G2/G3 = top-decile/bottom-quartile"
        " bias-over-spread z. G4 = predicted vs realized IQR ratio (≈1.0 ideal)."
        " G5 = Roelofs-debiased equal-mass ECE. `ship` is the AND of all five gates."
    ),
    "Hyperparameters": (
        "Optuna-tuned LightGBMLSS hyperparameters and per-market scale references."
        " Informational only."
    ),
}


def _column_config(cols: list[str]) -> dict[str, st.column_config.Column]:
    """Streamlit column_config with direction help for each annotated column."""
    return {c: st.column_config.Column(help=DIRECTIONS[c]) for c in cols if c in DIRECTIONS}


def _render_tab(tab_name: str, metric_cols: list[str], view: pd.DataFrame) -> None:
    st.caption(TAB_CAPTIONS[tab_name])
    cols = [c for c in ID_COLS + metric_cols if c in view.columns]
    st.dataframe(
        view[cols].sort_values(["league", "market"]),
        use_container_width=True,
        height=560,
        hide_index=True,
        column_config=_column_config(cols),
    )


st.set_page_config(page_title="Stats — Model Training", layout="wide")
st.title("Model Training Diagnostics")

mtime = (
    format_ts(
        dt.datetime.fromtimestamp(MODEL_STATS_PATH.stat().st_mtime).isoformat(timespec="seconds")
    )
    if MODEL_STATS_PATH.is_file()
    else "no meditate run on record"
)
render_banner("stats", f"last meditated {mtime}")

stats = load_model_stats()
if stats.empty:
    st.info("No model stats found. Run `poetry run meditate` to generate `model_stats.parquet`.")
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

with st.sidebar:
    st.header("Filters")
    leagues = sorted(stats["league"].dropna().unique())
    sel_leagues = st.multiselect("Leagues", leagues, default=leagues)
    distributions = sorted(stats["distribution"].dropna().unique())
    sel_dists = st.multiselect("Distributions", distributions, default=distributions)
    shipped_states = sorted(stats["shipped"].dropna().unique()) if "shipped" in stats.columns else []
    sel_shipped = (
        st.multiselect("Shipped", shipped_states, default=shipped_states) if shipped_states else []
    )
    lifecycle_states = sorted(stats["lifecycle_state"].dropna().unique())
    sel_lifecycle = (
        st.multiselect("Lifecycle", lifecycle_states, default=lifecycle_states)
        if lifecycle_states
        else []
    )

scope = stats["league"].isin(sel_leagues) & stats["distribution"].isin(sel_dists)
if sel_shipped:
    scope &= stats["shipped"].isin(sel_shipped)
if sel_lifecycle:
    scope &= stats["lifecycle_state"].isin(sel_lifecycle)
view = stats.loc[scope]

st.caption(f"Showing **{len(view):,}** cells.")

tabs = st.tabs(list(TAB_COLUMNS.keys()))
for tab, name in zip(tabs, TAB_COLUMNS.keys(), strict=True):
    with tab:
        _render_tab(name, TAB_COLUMNS[name], view)
