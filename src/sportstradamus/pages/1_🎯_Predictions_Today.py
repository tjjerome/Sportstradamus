"""Today's scored offers from the latest `prophecize` run."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from sportstradamus.dashboard_data import (
    format_ts,
    load_current_history,
    load_current_meta,
    load_current_offers,
    render_banner,
)
from sportstradamus.helpers.distributions import get_odds

st.set_page_config(page_title="Predictions — Today", layout="wide")
st.title("Today's Predictions")

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
render_banner("predictions", f"generated {generated}")

offers = load_current_offers()
if offers.empty:
    st.info(
        "No current predictions found. Run `poetry run prophecize` to "
        "generate `current_offers.parquet`."
    )
    st.stop()

history = load_current_history()

# Hot-row highlight thresholds
EV_HOT_THRESHOLD = 1.02
BOOKS_HOT_THRESHOLD = 0.95
BOOST_HOT_CEILING = 2.5
HIGHLIGHT_BG = "#1f4e79"

# Main table columns
MAIN_COLS = [
    "League",
    "Date",
    "Team",
    "Opponent",
    "Player",
    "Market",
    "Bet",
    "Line",
    "Boost",
    "Model P",
    "Model",
    "Books",
    "Platform",
]

# Numeric columns for range filtering
RANGE_COLS = ["Model P", "Model", "Books"]

# Defensive: drop rows with no signal (all zero edge/boost)
signal_cols = [c for c in ["Boost", "Model", "Books"] if c in offers.columns]
if signal_cols:
    signal = offers[signal_cols].fillna(0)
    offers = offers.loc[(signal != 0).any(axis=1)]

# --- In-page filters ---
col1, col2, col3 = st.columns(3)
with col1:
    leagues = sorted(offers["League"].dropna().unique())
    selected_leagues = st.multiselect("League", leagues, default=leagues)
with col2:
    platforms = sorted(offers["Platform"].dropna().unique()) if "Platform" in offers else []
    selected_platforms = st.multiselect("Platform", platforms, default=platforms)
with col3:
    markets = sorted(offers["Market"].dropna().unique())
    selected_markets = st.multiselect("Market", markets, default=markets)

player_query = st.text_input("Player search", placeholder="e.g. Jokic")

filtered = offers
if selected_leagues:
    filtered = filtered.loc[filtered["League"].isin(selected_leagues)]
if selected_platforms and "Platform" in filtered:
    filtered = filtered.loc[filtered["Platform"].isin(selected_platforms)]
if selected_markets:
    filtered = filtered.loc[filtered["Market"].isin(selected_markets)]
if player_query:
    filtered = filtered.loc[filtered["Player"].str.contains(player_query, case=False, na=False)]

# Numeric range filters
range_cols = [c for c in RANGE_COLS if c in filtered.columns]
if range_cols:
    st.caption("Numeric range filters")
    rcols = st.columns(len(range_cols))
    for slot, col in zip(rcols, range_cols, strict=False):
        series = pd.to_numeric(filtered[col], errors="coerce").dropna()
        if series.empty:
            continue
        lo = float(series.min())
        hi = float(series.max())
        if lo == hi:
            continue
        sel = slot.slider(col, lo, hi, (lo, hi), step=(hi - lo) / 100 or 0.01)
        vals = pd.to_numeric(filtered[col], errors="coerce")
        filtered = filtered.loc[vals.between(sel[0], sel[1]) | vals.isna()]

# Default ordering: descending by Model
if "Model" in filtered.columns:
    filtered = filtered.sort_values("Model", ascending=False)

filtered = filtered.reset_index(drop=True)
st.caption(f"Showing **{len(filtered):,}** of {len(offers):,} offers")

# --- Session state for detail popup navigation ---
if "detail_stack" not in st.session_state:
    st.session_state.detail_stack = []


# --- Helper functions for charts ---


def _history_chart(df: pd.DataFrame, line: float, title: str) -> alt.Chart:
    """Render bar chart of game history with dotted line at betting threshold."""
    df = df.copy()
    df["color"] = np.where(df["StatValue"] >= line, "Over", "Under")
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("GameDate:T", title="Date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("StatValue:Q", title=""),
            color=alt.Color(
                "color:N",
                scale=alt.Scale(domain=["Over", "Under"], range=["#4CAF50", "#F44336"]),
                legend=None,
            ),
            tooltip=["GameDate:T", "Opponent:N", "StatValue:Q"],
        )
        .properties(title=title)
    )
    rule = (
        alt.Chart(pd.DataFrame({"Line": [line]}))
        .mark_rule(strokeDash=[6, 3], color="#FFFFFF", strokeWidth=1.5)
        .encode(y="Line:Q")
    )
    return bars + rule


def _to_american(p: float) -> str:
    """Convert probability to American odds format."""
    if not isinstance(p, float) or np.isnan(p) or p <= 0 or p >= 1:
        return "N/A"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    return f"+{round((1 - p) / p * 100)}"


def _parse_corr(s: str, max_n: int = 3) -> list[tuple[str, float]]:
    """Parse comma-separated correlation string into (desc, multiplier) tuples."""
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for item in s.split(",")[:max_n]:
        item = item.strip()
        if "(" in item and item.endswith(")"):
            desc, raw = item.rsplit("(", 1)
            try:
                mult = float(raw.rstrip("x)"))
            except ValueError:
                mult = 1.0
            out.append((desc.strip(), mult))
        else:
            out.append((item, 1.0))
    return out


def _strength_badge(mult: float) -> str:
    """Return emoji badge for correlation strength."""
    if mult >= 1.25:
        return "🔴 Strong"
    if mult >= 1.1:
        return "🟡 Moderate"
    return "⚪ Mild"


def _find_corr_row_idx(desc: str, filtered: pd.DataFrame) -> int | None:
    """Find row index in filtered DataFrame matching a correlation description."""
    for direction in ("Over", "Under"):
        if direction in desc:
            player_name = desc.split(direction)[0].strip()
            matches = filtered[filtered["Player"].str.lower() == player_name.lower()]
            if not matches.empty:
                return matches.index[0]
    return None


def _render_corr_cards(
    items: list[tuple[str, float]], group_label: str, filtered: pd.DataFrame, tab_key_prefix: str
) -> None:
    """Render correlated bet cards as clickable buttons."""
    if not items:
        return
    st.markdown(f"**{group_label}**")
    for i, (desc, mult) in enumerate(items):
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"**{desc}**")
        col2.markdown(_strength_badge(mult) + f" {mult:.2f}×")
        if st.button("View →", key=f"{tab_key_prefix}_{i}"):
            idx = _find_corr_row_idx(desc, filtered)
            if idx is not None:
                st.session_state.detail_stack.append(idx)
                st.rerun()


@st.dialog("Offer detail", width="large")
def _show_detail(row: pd.Series, history: pd.DataFrame, filtered: pd.DataFrame) -> None:
    """Render detailed view of a single offer with charts and correlations."""
    # Back button for navigation stack
    if len(st.session_state.detail_stack) > 1:
        if st.button("← Back"):
            st.session_state.detail_stack.pop()
            st.rerun()

    # Header
    st.subheader(f"{row.get('Player', '?')} — {row.get('Market', '?')}")
    st.write(
        f"**{row.get('Bet', '?')} {row.get('Line', '?')}** · "
        f"{row.get('Team', '?')} vs {row.get('Opponent', '?')} · "
        f"{row.get('League', '?')} · {row.get('Platform', '?')}"
    )

    # Context metrics: Moneyline (as American odds), O/U (raw total), DVPOA (as %)
    col1, col2, col3 = st.columns(3)
    with col1:
        ml = row.get("Moneyline")
        ml_str = _to_american(ml) if pd.notna(ml) else "N/A"
        st.metric("Moneyline", ml_str)
    with col2:
        ou = row.get("O/U")
        ou_str = f"{ou:.1f}" if pd.notna(ou) and isinstance(ou, int | float) else "N/A"
        st.metric("O/U Total", ou_str)
    with col3:
        dvpoa = row.get("DVPOA")
        dvpoa_str = (
            f"{dvpoa * 100:+.1f}%" if pd.notna(dvpoa) and isinstance(dvpoa, int | float) else "N/A"
        )
        st.metric("DVPOA", dvpoa_str)

    # Three tabs: History, Model Distribution, Correlated Bets
    tab1, tab2, tab3 = st.tabs(["📈 History", "〜 Model", "🔗 Correlated"])

    with tab1:
        stat_key = row.get("Stat", row.get("Market"))
        line = row.get("Line")
        player_hist = history.loc[
            (history["Player"] == row["Player"])
            & (history["League"] == row["League"])
            & (history["Market"] == stat_key)
        ].sort_values("GameDate")

        if not player_hist.empty:
            recent = player_hist.tail(15)
            st.altair_chart(_history_chart(recent, line, "Last 15 games"), use_container_width=True)

            h2h = player_hist.loc[player_hist["Opponent"] == row["Opponent"]]
            if not h2h.empty:
                st.altair_chart(
                    _history_chart(h2h, line, f"vs {row['Opponent']}"), use_container_width=True
                )
            else:
                st.caption(f"No H2H games vs {row['Opponent']} in current season window.")
        else:
            st.caption("No history available.")

    with tab2:
        dist = row.get("Dist")
        ev = row.get("Model EV")
        cv = row.get("CV")
        line = row.get("Line")

        if pd.notna(dist) and pd.notna(ev) and pd.notna(cv):
            std = row.get("Model STD") or ev * 0.3
            lo = max(0.0, ev - 4 * std)
            hi = ev + 4 * std
            step = 0.5 if dist in ("Gamma", "ZAGamma", "SkewNormal") else 1.0
            xs = np.arange(lo, hi + step, step)

            kw = {
                k: row.get(k)
                for k in ("Model R", "Model Alpha", "Model Sigma", "Model Skew", "Gate")
            }
            kw = {k: v for k, v in kw.items() if pd.notna(v) or (not isinstance(v, float))}

            try:
                cdf_vals = [get_odds(x, ev, dist, cv, **kw) for x in xs]
                pmf_vals = np.diff([0.0, *cdf_vals])
                xs_mid = xs[: len(pmf_vals)]

                df_pdf = pd.DataFrame(
                    {
                        "x": xs_mid,
                        "P": pmf_vals,
                        "Side": ["Over" if x >= line else "Under" for x in xs_mid],
                    }
                )

                chart = (
                    alt.Chart(df_pdf)
                    .mark_bar(size=max(4, 400 // max(len(xs_mid), 1)))
                    .encode(
                        x=alt.X("x:Q", title=row["Market"]),
                        y=alt.Y("P:Q", title="Probability"),
                        color=alt.Color(
                            "Side:N",
                            scale=alt.Scale(domain=["Over", "Under"], range=["#2196F3", "#FF7043"]),
                            legend=alt.Legend(orient="top"),
                        ),
                        tooltip=["x:Q", "P:Q", "Side:N"],
                    )
                )
                rule = (
                    alt.Chart(pd.DataFrame({"Line": [line]}))
                    .mark_rule(strokeDash=[6, 3], color="#FFFFFF", strokeWidth=1.5)
                    .encode(x="Line:Q")
                )
                st.altair_chart(chart + rule, use_container_width=True)
            except Exception as e:
                st.error(f"Error computing distribution: {e}")
        else:
            st.caption("Distribution parameters unavailable — re-run `prophecize` to refresh.")

    with tab3:
        same_items = _parse_corr(row.get("Team Correlation"))
        opp_items = _parse_corr(row.get("Opp Correlation"))

        _render_corr_cards(same_items, f"Same team — {row['Team']}", filtered, "corr_same")
        _render_corr_cards(opp_items, f"Opponent — {row['Opponent']}", filtered, "corr_opp")

        if not same_items and not opp_items:
            st.caption("No correlated legs cleared the display thresholds for this offer.")


# --- AgGrid table ---
display_cols = [c for c in MAIN_COLS if c in filtered.columns]

gb = GridOptionsBuilder.from_dataframe(filtered[display_cols])
gb.configure_selection(selection_mode="single", use_checkbox=False)
gb.configure_grid_options(rowStyle={"cursor": "pointer"})

# Hot-row styling on Model column
gb.configure_column(
    "Model",
    cellStyle={
        "function": (
            f"params.data.Model > {EV_HOT_THRESHOLD} && "
            f"params.data.Books > {BOOKS_HOT_THRESHOLD} && "
            f"params.data.Boost <= {BOOST_HOT_CEILING} "
            f"? {{'backgroundColor': '{HIGHLIGHT_BG}', 'color': 'white'}} : {{}}"
        )
    },
)

go = gb.build()
ag = AgGrid(
    filtered[display_cols],
    gridOptions=go,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    fit_columns_on_grid_load=True,
    height=720,
    use_container_width=True,
)

selected = ag.selected_rows
# Newer streamlit-aggrid returns a DataFrame; older versions return a list.
if isinstance(selected, pd.DataFrame):
    selected_rows = selected.to_dict("records")
else:
    selected_rows = selected or []

if selected_rows:
    row_data = selected_rows[0]
    player = row_data["Player"]
    market = row_data.get("Market")
    mask = filtered["Player"] == player
    if market:
        mask &= filtered["Market"] == market
    matches = filtered.loc[mask]
    if not matches.empty:
        idx = matches.index[0]
        if not st.session_state.detail_stack or st.session_state.detail_stack[-1] != idx:
            st.session_state.detail_stack = [idx]

if st.session_state.detail_stack:
    row_idx = st.session_state.detail_stack[-1]
    _show_detail(filtered.loc[row_idx], history, filtered)
else:
    st.caption("Click a row to see charts and correlated bets.")

with st.expander("Snapshot info"):
    st.json(meta)
