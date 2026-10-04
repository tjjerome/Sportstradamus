"""Receipts — the "prove it" surface.

Every number prices ``realized.settled_offers``: sides the platform posted, one bet per
platform, each graded at the platform's real payout. The hero is the model's recommended
legs (the rule the story menu and the nightly ledger share) with every posted side as
context; beneath it the recommended legs' cumulative units, skeptic checks (CLV beat rate,
Brier, worst month — losers shown, never hidden), the realized-by-side panel, your tracked
slips, accuracy trends alongside the standard-vs-alt-line reliability diagram, the full CLV
breakdown, and the folded-in strategy simulator. A skeptic should be able to verify
profitability unaided.
"""

from datetime import UTC, datetime

import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss

from sportstradamus import clv, realized
from sportstradamus.analysis import compute_book_brier_skill_score
from sportstradamus.dashboard.components.by_side import render_by_side
from sportstradamus.dashboard.components.hero import desk_only_notice, page_hero
from sportstradamus.dashboard.components.profit_sim import (
    render_profit_sim,
    render_profit_sim_summary,
)
from sportstradamus.dashboard.components.receipts_hero import (
    WINDOW_LABELS,
    WINDOW_OPTIONS,
    render_receipts_hero,
    window_offers,
)
from sportstradamus.dashboard.components.tickets import build_tickets, render_tickets
from sportstradamus.dashboard.data import (
    format_ts,
    load_calibration_summary,
    load_history,
    load_profit_sim_summary,
    load_resolve_meta,
    load_user_slips,
    posted_offers_or_stop,
    sidebar_filters,
    sport_filtered,
)
from sportstradamus.dashboard.surfaces.receipts_charts import (
    cumulative_profit_chart,
    reliability_diagram,
    rolling_accuracy_chart,
)

page_hero("THE RECEIPTS", "Receipts")
desk_only_notice()


history = load_history()

if history.empty:
    st.warning("No prediction history found. Run `prophecize` first.")
    st.stop()

meta = load_resolve_meta()
if meta.get("last_run"):
    st.caption(f"Data last resolved: {format_ts(meta['last_run'])}")
else:
    st.warning(
        "Nightly resolution has not run yet. Run `poetry run reflect` to resolve prediction outcomes."
    )

# Sport narrow runs before the sidebar builds its league multiselect, so the
# sidebar only offers that sport's leagues.
history = sport_filtered(history)
if history.empty:
    st.info("No resolved predictions match the current sport filter.")
    st.stop()

filters = sidebar_filters(history, key_prefix="receipts_")
df = posted_offers_or_stop(history, filters)
df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

# The window scopes the hero, its cumulative units and worst month, and the by-side
# panel; every other df-consuming section below keeps reading the full filtered df.
window = (
    st.segmented_control(
        "Window",
        WINDOW_OPTIONS,
        default="All",
        format_func=lambda w: WINDOW_LABELS.get(w, w),
        key="receipts_window",
    )
    or "All"
)
df_windowed = window_offers(df, window, datetime.now(UTC))
recommended = df_windowed[df_windowed["Recommended"]]
render_receipts_hero(realized.cohort_summary(recommended), realized.cohort_summary(df_windowed))

wm = realized.worst_month(recommended)
daily_profit = recommended.groupby("_date").agg(Profit=("Unit", "sum")).reset_index()
daily_profit["Cumulative Profit"] = daily_profit["Profit"].cumsum()
st.plotly_chart(cumulative_profit_chart(daily_profit, wm), width="stretch")

st.download_button(
    "Export filtered history (CSV)",
    df.to_csv(index=False),
    "history_filtered.csv",
    "text/csv",
)

st.subheader("Skeptic checks")
clv_summary = clv.summarize(df)
brier = brier_score_loss(df["Hit"], df["Win Prob"].clip(0, 1))
book_skill = compute_book_brier_skill_score(df)

s1, s2, s3 = st.columns(3)
s1.metric(
    "CLV beat rate",
    f"{clv_summary['frac_beat_close']:.1%}" if clv_summary["n"] else "—",
    f"{clv_summary['n']:,} legs with close" if clv_summary["n"] else "no close data yet",
)
s2.metric(
    "Brier",
    f"{brier:.4f}",
    f"{book_skill:+.1%} vs book" if pd.notna(book_skill) else "book baseline n/a",
)
if wm:
    s3.metric(
        "Worst month",
        wm["month"],
        f"{wm['units']:+.1f}u · {wm['n']} bets",
        delta_color="normal",  # negative units render red — losers are shown, never hidden
    )
else:
    s3.metric("Worst month", "—")

st.subheader("Realized by side — platform payouts")
render_by_side(realized.by_split(recommended))

st.subheader("Your slips")
user_slips = load_user_slips()
if user_slips.empty:
    st.caption("Lock in a slip on the Build or Board surface to track it here.")
else:
    has_status = "status" in user_slips.columns
    pending_n = int((user_slips["status"] == "pending").sum()) if has_status else len(user_slips)
    graded = (
        user_slips.loc[user_slips["status"] != "pending"] if has_status else user_slips.iloc[:0]
    )
    if not graded.empty:
        wins = int((graded["status"] == "won").sum())
        st.caption(
            f"{len(graded)} graded · {wins} won · {pending_n} pending — your record vs the model's."
        )
    else:
        st.caption(f"{pending_n} pending — graded nightly by `reflect`.")
    tickets = build_tickets(user_slips.sort_values("saved_at", ascending=False))
    render_tickets(tickets)

trend_col, cal_col = st.columns(2)

with trend_col:
    st.subheader("Rolling 30-Day Accuracy by League")
    breakeven = realized.cohort_summary(df)["breakeven_rate"]
    st.plotly_chart(rolling_accuracy_chart(df, breakeven), width="stretch")

with cal_col:
    st.subheader("Calibration — predicted vs realized hit rate")
    cal_summary = load_calibration_summary()
    # A snapshot written before the cohort split has no Cohort column; the next nightly
    # reflect rewrites it.
    if cal_summary.empty or "Cohort" not in cal_summary.columns:
        st.info("No calibration data yet. Populates after `reflect` runs against resolved history.")
    else:
        st.plotly_chart(reliability_diagram(cal_summary), width="stretch")
        posted_cal = cal_summary.loc[cal_summary["Cohort"] == "posted"]
        std_split = posted_cal.loc[~posted_cal["Alt Line"]]
        alt_split = posted_cal.loc[posted_cal["Alt Line"]]
        e1, e2, e3 = st.columns(3)
        e1.metric(
            "Realized ECE",
            f"{std_split['ECE'].iloc[0]:.1%}" if not std_split.empty else "—",
        )
        e2.metric(
            "Standard ROI",
            f"{std_split['ROI'].iloc[0]:+.1%}" if not std_split.empty else "—",
        )
        e3.metric(
            "Alt / ladder ROI",
            f"{alt_split['ROI'].iloc[0]:+.1%}" if not alt_split.empty else "—",
        )
        # A recommended leg's Win Prob clears the first bin edge (a 5% edge at a payout of
        # 2.5x or less needs 0.42), so N-weighting the bins of both alt-line splits gives
        # the whole cohort's ECE and ROI.
        rec_cal = cal_summary.loc[cal_summary["Cohort"] == "recommended"]
        rec_n = rec_cal["N"].sum()
        rec_error = (rec_cal["N"] * (rec_cal["Predicted"] - rec_cal["Actual"]).abs()).sum()
        r1, r2 = st.columns(2)
        r1.metric("Recommended ECE", f"{rec_error / rec_n:.1%}" if rec_n else "—")
        r2.metric(
            "Recommended ROI",
            f"{(rec_cal['N'] * rec_cal['ROI']).sum() / rec_n:+.1%}" if rec_n else "—",
        )
        st.caption(
            "On the diagonal = honest probabilities. On posted legs the model is calibrated "
            "bucket by bucket; on the recommended tail it is not — the hero sets its read "
            "against its hit rate. The panel reads the nightly snapshot of every resolved "
            "leg, so the sidebar and window do not narrow it."
        )

st.subheader("Closing Line Value")
if clv_summary["n"] == 0:
    st.info(
        "No legs with closing-line data yet. CLV populates after `reflect` "
        "runs against archives that contain post-lock odds."
    )
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Close coverage",
        f"{clv_summary['n'] / len(df):.1%}",
        f"{clv_summary['n']:,} of {len(df):,} legs",
    )
    c2.metric("Mean Market CLV", f"{clv_summary['market_clv_mean']:+.3f}")
    c3.metric(
        "Mean Model CLV",
        f"{clv_summary['model_clv_mean']:+.3f}" if pd.notna(clv_summary["model_clv_mean"]) else "—",
    )
    c4.metric("Beat-close rate", f"{clv_summary['frac_beat_close']:.1%}")
    st.caption(
        "Model CLV compares the model's win probability to the closing probability at "
        "lock; Market CLV does the same for the market's own price."
    )

    segments = clv_summary["segments"]
    if not segments.empty:
        st.caption(
            f"Segments with at least {clv.CLV_SEGMENT_MIN_N} legs, sorted by mean Market CLV."
        )
        st.dataframe(
            segments.style.format({"market_clv": "{:+.3f}"}),
            width="stretch",
            hide_index=True,
        )

st.subheader("Strategy simulator — retrospective (hindsight)")
st.caption(
    "If you'd staked the model's positive-edge bets under each strategy. Computed nightly — "
    "ROI / Sharpe / drawdown / win-rate over each look-back window."
)
st.caption("Forward paper-trading ledger lands with the sim-bettor-ledger lane.")
render_profit_sim_summary(load_profit_sim_summary())
with st.expander("Customize & run a live backtest"):
    render_profit_sim(history)
