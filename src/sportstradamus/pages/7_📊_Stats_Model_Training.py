"""Per-market training diagnostics from the latest `meditate` run.

Phase 3 §4.d: tabs are split by metric family (Scoring rules, Discrimination,
Rates, Kelly & blending, Dispersion, EV & lines, Hyperparameters). Each tab
renders a one-row "Book-only baseline" table above the per-market model rows
so readers see what taking the book's odds gets you before reading model
performance. Every metric column carries a one-line help string indicating
whether higher or lower is better (or that the field is informational).
"""

import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from sportstradamus.dashboard_data import format_ts, load_model_stats, render_banner
from sportstradamus.helpers.io import MODEL_STATS_PATH

# The "calibrated" row carries the post-correction performance the model
# actually deploys at; the other rows (raw, corrected) are diagnostic.
DEFAULT_METRIC_ROW = "calibrated"

# Direction annotations rendered as Streamlit `column_config` help strings.
# Single source of truth — column groupings below pull their help text from here.
HIGHER = "↑ higher is better"
LOWER = "↓ lower is better"
INFO = "informational"
DIRECTIONS: dict[str, str] = {
    # Scoring rules
    "brier_score": LOWER,
    "log_loss": LOWER,
    "nll": LOWER,
    "expected_calibration_error": LOWER,
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
    # Kelly & blending
    "brier_skill_score": HIGHER,
    "kelly_shrinkage": HIGHER,
    "model_weight": INFO,
    # Dispersion
    "shape_ratio": INFO,
    "dispersion_cal": INFO,
    "model_shape": INFO,
    "empirical_shape": INFO,
    # EV & lines
    "model_ev": INFO,
    "mean_line": INFO,
    "result_mean": INFO,
    "mean_ev_diff": HIGHER,
    "median_ev_diff": HIGHER,
    "cf_over_pct": INFO,
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
}

ID_COLS = ["league", "market", "distribution"]

TAB_COLUMNS: dict[str, list[str]] = {
    "Scoring rules": ["brier_score", "log_loss", "nll", "expected_calibration_error"],
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
    "Kelly & blending": ["brier_skill_score", "kelly_shrinkage", "model_weight"],
    "Dispersion": ["shape_ratio", "dispersion_cal", "model_shape", "empirical_shape"],
    "EV & lines": [
        "model_ev",
        "mean_line",
        "result_mean",
        "mean_ev_diff",
        "median_ev_diff",
        "cf_over_pct",
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
    "Scoring rules": (
        "Proper scoring rules on the validation set. The pinned baseline shows what"
        " predicting the book's implied probability scores; model rows beneath should beat it."
        " Columns are annotated ↑/↓/informational."
    ),
    "Discrimination": (
        "How well the model separates winners from losers. Baseline = book-as-model."
        " Columns are annotated ↑/↓/informational."
    ),
    "Rates": (
        "Rates and conditional Over% slices. Baseline = book-as-model where applicable;"
        " EV-conditional fields are model-only and read NaN on the baseline."
        " Columns are annotated ↑/↓/informational."
    ),
    "Kelly & blending": (
        "Skill score relative to the book and the derived Kelly shrinkage."
        " Baseline pins to skill=0, shrinkage=0 (no edge over the book)."
        " Columns are annotated ↑/↓/informational."
    ),
    "Dispersion": (
        "Shape calibration diagnostics. shape_ratio≈1.0 means model dispersion matches"
        " the empirical outcome dispersion. Columns are informational."
    ),
    "EV & lines": (
        "Expected-value diagnostics vs. the bookmaker line. Mean/median EV diffs are"
        " the model's edge over the line. Columns are annotated ↑/↓/informational."
    ),
    "Hyperparameters": (
        "Optuna-tuned LightGBMLSS hyperparameters and per-market scale references."
        " Informational only."
    ),
}


def _column_config(cols: list[str]) -> dict[str, st.column_config.Column]:
    """Build Streamlit column_config with direction help for each metric column."""
    cfg: dict[str, st.column_config.Column] = {}
    for c in cols:
        if c in DIRECTIONS:
            cfg[c] = st.column_config.Column(help=DIRECTIONS[c])
    return cfg


def _render_tab(
    tab_name: str,
    metric_cols: list[str],
    model_view: pd.DataFrame,
    book_view: pd.DataFrame,
) -> None:
    """Render the pinned book-baseline table above the per-market model table."""
    st.caption(TAB_CAPTIONS[tab_name])
    cols = ID_COLS + metric_cols
    available = [c for c in cols if c in model_view.columns]

    book_available = [c for c in cols if c in book_view.columns]
    if not book_view.empty and book_available:
        st.markdown("**Book-only baseline (what taking the book's odds gets you):**")
        st.dataframe(
            book_view[book_available].sort_values(["league", "market"]),
            use_container_width=True,
            hide_index=True,
            column_config=_column_config(book_available),
        )
    else:
        st.info("No book baseline rows for the current filters.")

    st.dataframe(
        model_view[available].sort_values(["league", "market"]),
        use_container_width=True,
        height=560,
        hide_index=True,
        column_config=_column_config(available),
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

# Backfill row_kind for parquets written before §4.c (treat all rows as model).
if "row_kind" not in stats.columns:
    stats = stats.assign(row_kind="model")

with st.sidebar:
    st.header("Filters")
    leagues = sorted(stats["league"].dropna().unique())
    sel_leagues = st.multiselect("Leagues", leagues, default=leagues)
    distributions = sorted(stats["distribution"].dropna().unique())
    sel_dists = st.multiselect("Distributions", distributions, default=distributions)

    model_rows_df = stats.loc[stats["row_kind"] == "model"]
    metric_rows = sorted(model_rows_df["metric_row"].dropna().unique())
    if not metric_rows:
        st.warning("No model rows in `model_stats.parquet`.")
        st.stop()
    sel_metric_row = st.selectbox(
        "Metric row",
        metric_rows,
        index=metric_rows.index(DEFAULT_METRIC_ROW) if DEFAULT_METRIC_ROW in metric_rows else 0,
    )

scope = stats["league"].isin(sel_leagues) & stats["distribution"].isin(sel_dists)
model_view = stats.loc[
    scope & (stats["row_kind"] == "model") & (stats["metric_row"] == sel_metric_row)
]
book_view = stats.loc[scope & (stats["row_kind"] == "book_baseline")]

st.caption(
    f"Showing **{len(model_view):,}** model rows at `{sel_metric_row}` and "
    f"**{len(book_view):,}** book-baseline rows."
)

tabs = st.tabs(list(TAB_COLUMNS.keys()))
for tab, name in zip(tabs, TAB_COLUMNS.keys(), strict=True):
    with tab:
        _render_tab(name, TAB_COLUMNS[name], model_view, book_view)
