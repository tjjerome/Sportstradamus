"""Per-market training diagnostics from the latest `meditate` run."""

import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import streamlit as st

from sportstradamus.dashboard_data import load_model_stats, render_banner
from sportstradamus.helpers.io import MODEL_STATS_PATH

# The "calibrated" row carries the post-correction performance the model
# actually deploys at; the other rows (raw, corrected) are diagnostic.
DEFAULT_METRIC_ROW = "calibrated"

st.set_page_config(page_title="Stats — Model Training", layout="wide")
st.title("Model Training Diagnostics")

mtime = (
    dt.datetime.fromtimestamp(MODEL_STATS_PATH.stat().st_mtime).isoformat(timespec="seconds")
    if MODEL_STATS_PATH.is_file()
    else "no meditate run on record"
)
render_banner("stats", f"last meditated {mtime}")

stats = load_model_stats()
if stats.empty:
    st.info(
        "No model stats found. Run `poetry run meditate` to generate "
        "`model_stats.parquet`."
    )
    st.stop()

with st.sidebar:
    st.header("Filters")
    leagues = sorted(stats["league"].dropna().unique())
    sel_leagues = st.multiselect("Leagues", leagues, default=leagues)
    distributions = sorted(stats["distribution"].dropna().unique())
    sel_dists = st.multiselect("Distributions", distributions, default=distributions)
    metric_rows = sorted(stats["metric_row"].dropna().unique())
    sel_metric_row = st.selectbox(
        "Metric row",
        metric_rows,
        index=metric_rows.index(DEFAULT_METRIC_ROW) if DEFAULT_METRIC_ROW in metric_rows else 0,
    )

view = stats.loc[
    stats["league"].isin(sel_leagues)
    & stats["distribution"].isin(sel_dists)
    & (stats["metric_row"] == sel_metric_row)
]
st.caption(f"Showing **{len(view):,}** markets at row `{sel_metric_row}`")

tab_acc, tab_diag, tab_hp = st.tabs(["Per-market accuracy", "Diagnostics", "Hyperparameters"])

with tab_acc:
    cols = ["league", "market", "distribution", "accuracy", "over_prec", "under_prec", "over_pct", "sharpness", "nll"]
    st.dataframe(
        view[cols].sort_values(["league", "market"]),
        use_container_width=True,
        height=620,
    )

with tab_diag:
    cols = [
        "league",
        "market",
        "distribution",
        "model_weight",
        "model_calib",
        "shape_ratio",
        "dispersion_cal",
        "model_ev",
        "mean_line",
        "result_mean",
        "mean_ev_diff",
        "median_ev_diff",
        "frac_ev_gt_line",
        "over_pct_ev_gt",
        "over_pct_ev_lt",
        "cf_over_pct",
    ]
    st.dataframe(
        view[cols].sort_values(["league", "market"]),
        use_container_width=True,
        height=620,
    )

with tab_hp:
    cols = [
        "league",
        "market",
        "distribution",
        "hp_rounds",
        "hp_leaves",
        "hp_lr",
        "hp_min_child",
        "hp_l1",
        "hp_l2",
        "cv",
        "std",
        "historical_zero_rate",
    ]
    st.dataframe(
        view[cols].sort_values(["league", "market"]),
        use_container_width=True,
        height=620,
    )
