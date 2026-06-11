"""Offer-detail dialog and navigation for the dashboard."""

import numpy as np
import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.deep_dive_charts import (
    _DIST_PARAM_COLS,
    build_h2h_history,
    build_recent_history,
    distribution_chart,
    distribution_frame,
    history_chart,
    strength_badge,
)
from sportstradamus.dashboard.data import GAMELOG_SCHEMA


def init_detail_state() -> None:
    """Initialise the session-state keys the detail dialog navigates with."""
    if "detail_stack" not in st.session_state:
        st.session_state.detail_stack = []
    if "last_grid_key" not in st.session_state:
        st.session_state.last_grid_key = None
    if "corr_nav" not in st.session_state:
        st.session_state.corr_nav = False


def _to_american(p: float) -> str:
    if not isinstance(p, float) or np.isnan(p) or p <= 0 or p >= 1:
        return "N/A"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    return f"+{round((1 - p) / p * 100)}"


def _parse_corr(s: str, max_n: int = 3) -> list[tuple[str, float]]:
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


def _find_corr_row_idx(desc: str, filtered: pd.DataFrame) -> int | None:
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
    if not items:
        return
    st.markdown(f"**{group_label}**")
    for i, (desc, mult) in enumerate(items):
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"**{desc}**")
        col2.markdown(strength_badge(mult) + f" {mult:.2f}×")
        if st.button("View", icon=":material/arrow_forward:", key=f"{tab_key_prefix}_{i}"):
            idx = _find_corr_row_idx(desc, filtered)
            if idx is not None:
                st.session_state.detail_stack.append(idx)
                st.session_state.corr_nav = True
                st.rerun()


def _select_history_df(
    hist_df: pd.DataFrame, h2h_df: pd.DataFrame, league: str, opponent: str, row_id: int
) -> pd.DataFrame:
    if hist_df.empty or league == "NFL" or h2h_df.empty:
        return hist_df
    filter_opt = st.radio(
        "Filter by opponent:",
        options=["All games", f"vs {opponent}"],
        horizontal=True,
        key=f"h2h_filter_{row_id}",
    )
    return h2h_df if filter_opt == f"vs {opponent}" else hist_df


def _render_history_tab(row: pd.Series) -> None:
    stat_key = row.get("Stat") or row.get("Market")
    line = row.get("Line")
    league = row.get("League", "")
    opponent = row.get("Opponent", "")
    schema = GAMELOG_SCHEMA.get(league, {})

    hist_df = pd.DataFrame()
    h2h_df = pd.DataFrame()
    if stat_key and schema:
        hist_df = build_recent_history(row, league, stat_key, line, schema)
        if league != "NFL" and opponent:
            h2h_df = build_h2h_history(row, league, stat_key, line, opponent, schema)

    display_df = _select_history_df(hist_df, h2h_df, league, opponent, id(row))
    if not display_df.empty:
        st.altair_chart(history_chart(display_df, line), use_container_width=True)
    elif hist_df.empty:
        st.caption("No history available for this player/stat.")


def _render_model_tab(row: pd.Series) -> None:
    dist = row.get("Dist")
    ev = row.get("Model EV")
    cv = row.get("CV")
    line = row.get("Line")
    if not (pd.notna(dist) and pd.notna(ev) and pd.notna(cv)):
        st.caption("Distribution parameters unavailable — re-run `prophecize` to refresh.")
        return

    params = {
        param: row.get(col) for col, param in _DIST_PARAM_COLS.items() if pd.notna(row.get(col))
    }
    std = row.get("Model STD") or ev * 0.3
    try:
        df_pdf, y_title, is_continuous = distribution_frame(dist, ev, std, params, line)
        chart = distribution_chart(df_pdf, is_continuous, line, row["Market"], y_title)
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error computing distribution: {e}")


def _render_context_metrics(row: pd.Series) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        ml = row.get("Moneyline")
        st.metric("Moneyline", _to_american(ml) if pd.notna(ml) else "N/A")
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


def _render_nav() -> None:
    nav = st.columns([1, 1, 6])
    if nav[0].button("Close", icon=":material/close:"):
        st.session_state.detail_stack = []
        st.session_state.last_grid_key = None
        st.rerun()
    if len(st.session_state.detail_stack) > 1:
        if nav[1].button("Back", icon=":material/arrow_back:"):
            st.session_state.detail_stack.pop()
            st.rerun()


def _render_corr_tab(row: pd.Series, filtered: pd.DataFrame) -> None:
    same_items = _parse_corr(row.get("Team Correlation"))
    opp_items = _parse_corr(row.get("Opp Correlation"))
    _render_corr_cards(same_items, f"Same team — {row['Team']}", filtered, "corr_same")
    _render_corr_cards(opp_items, f"Opponent — {row['Opponent']}", filtered, "corr_opp")
    if not same_items and not opp_items:
        st.caption("No correlated legs cleared the display thresholds for this offer.")


@st.dialog("Offer detail", width="large")
def show_detail(row: pd.Series, filtered: pd.DataFrame) -> None:
    _render_nav()

    st.subheader(f"{row.get('Player', '?')} — {row.get('Market', '?')}")
    st.write(
        f"**{row.get('Bet', '?')} {row.get('Line', '?')}** · "
        f"{row.get('Team', '?')} vs {row.get('Opponent', '?')} · "
        f"{row.get('League', '?')} · {row.get('Platform', '?')}"
    )
    _render_context_metrics(row)

    tab1, tab2, tab3 = st.tabs(
        [":material/history: History", ":material/query_stats: Model", ":material/hub: Correlated"]
    )
    with tab1:
        _render_history_tab(row)
    with tab2:
        _render_model_tab(row)
    with tab3:
        _render_corr_tab(row, filtered)
