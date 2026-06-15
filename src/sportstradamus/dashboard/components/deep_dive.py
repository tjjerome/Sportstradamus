"""Offer-detail dialog and navigation for the dashboard.

The evidence-chain view a user opens on any offer: a header edge badge, the model's
"case" (the per-offer ``Why``), a game-context strip, a live market-trust read with a
deep-link to the Model Lab cell, and five tabs (History, Model, Comps, Other stats,
Correlated). The Comps / Other-stats tabs read the ``current_offer_details`` sidecar
prerendered at ``prophecize`` time so the server never recomputes them live.
"""

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.deep_dive_charts import (
    DIST_PARAM_COLS,
    build_h2h_history,
    build_recent_history,
    distribution_chart,
    distribution_frame,
    history_chart,
    sparkline,
    strength_badge,
)
from sportstradamus.dashboard.components.slip_state import add_to_simple_slip
from sportstradamus.dashboard.data import (
    GAMELOG_SCHEMA,
    load_current_game_context,
    load_current_offer_details,
    load_live_metrics,
)
from sportstradamus.dashboard.narrative import SHAPE_HELP, context_strip

# Market-trust reads the 30-day live-settlement window; below this many settled legs
# the live hit-rate is too sparse to show (honest "needs more data" caption instead).
_LIVE_WINDOW_DAYS = 30
_MIN_SETTLED = 30
_DETAIL_KEY = ["League", "Date", "Player", "Market", "Opponent"]


def init_detail_state() -> None:
    """Initialise the session-state keys the detail dialog navigates with."""
    if "detail_stack" not in st.session_state:
        st.session_state.detail_stack = []
    if "last_grid_key" not in st.session_state:
        st.session_state.last_grid_key = None
    if "corr_nav" not in st.session_state:
        st.session_state.corr_nav = False


def to_american(p: float) -> str:
    """Format a win probability as American odds, or ``"N/A"`` outside (0, 1)."""
    if not isinstance(p, float) or np.isnan(p) or p <= 0 or p >= 1:
        return "N/A"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    return f"+{round((1 - p) / p * 100)}"


def render_offer_card(row: Mapping) -> None:
    """Compact offer card — the Streamlit twin of the constellation's JS hover card.

    Player + bet/line/market + win/boost/Kelly. The headshot (a person icon) and
    the last-5 line are scarred placeholders, matching the constellation card: the
    offer snapshot carries no player-id or gamelog columns to fill them yet.
    """
    win = float(row.get("Win Prob", 0.0) or 0.0)
    boost = float(row.get("Boost", 1.0) or 1.0)
    kelly = float(row.get("Kelly", 0.0) or 0.0)
    st.markdown(f":material/account_circle: **{row['Player']}**")
    st.caption(f"{row['Bet']} {float(row['Line']):.10g} {row['Market']}")
    st.markdown(f"Win **{win:.0%}** · **{boost:.2f}×** · Kelly **{kelly:.0%}**")
    st.caption(":gray[Last 5 — coming soon]")


def _edge_badge(model_ev: float) -> str:
    """Board-style Model Edge (``Model EV − 1``) as a semantic colored chip."""
    edge = model_ev - 1.0
    color = "green" if edge >= 0 else "red"
    return f":{color}[**{edge:+.0%} edge**]"


def _render_header(row: pd.Series) -> None:
    head, badge = st.columns([4, 1])
    head.subheader(f"{row.get('Player', '?')} — {row.get('Market', '?')}")
    head.caption(
        f"**{row.get('Bet', '?')} {row.get('Line', '?')}** · "
        f"{row.get('Team', '?')} vs {row.get('Opponent', '?')} · "
        f"{row.get('League', '?')} · {row.get('Platform', '?')}"
    )
    model_ev = row.get("Model EV")
    if pd.notna(model_ev):
        badge.markdown(_edge_badge(float(model_ev)))


def _render_case(row: pd.Series) -> None:
    """The model's case for the pick — the per-offer ``Why`` prose, if present."""
    why = row.get("Why")
    if isinstance(why, str) and why.strip():
        st.info(why)


def _pct(v) -> str:
    return f"{v * 100:+.1f}%" if pd.notna(v) and isinstance(v, int | float) else "N/A"


def _team_total_metric(col, ou, strip: dict | None) -> None:
    if not (pd.notna(ou) and isinstance(ou, int | float)):
        col.metric("Implied team total", "N/A")
        return
    delta = None
    if strip and not pd.isna(strip["baseline_total"]):
        delta = f"{ou - strip['baseline_total'] / 2:+.1f} vs avg"
    col.metric("Implied team total", f"{ou:.1f}", delta=delta, delta_color="off")


def _render_context_strip(row: pd.Series, game_context: pd.DataFrame) -> None:
    strip = (
        context_strip(game_context, game=row.get("Game", ""), date=row.get("Date", ""))
        if not game_context.empty
        else None
    )
    c1, c2, c3, c4 = st.columns(4)
    _team_total_metric(c1, row.get("O/U"), strip)
    ml = row.get("Moneyline")
    c2.metric("Moneyline", to_american(ml) if pd.notna(ml) else "N/A")
    shape = strip["shape"] if strip else None
    c3.metric("Game shape", str(shape).title() if shape else "N/A", help=SHAPE_HELP)
    c4.metric("DVPOA", _pct(row.get("DVPOA")))


def _live_read(live: pd.DataFrame, league, market, bet) -> tuple[float | None, int]:
    """``(bet-side live precision, n_settled)`` for the cell's 30-day window.

    Selects ``precision_over_live`` for an Over bet, ``precision_under_live`` otherwise;
    ``(None, 0)`` when the cell has no live row. Pure — the panel formats the result.
    """
    if live.empty:
        return None, 0
    sub = live[
        (live["league"] == league)
        & (live["market"] == market)
        & (live["window_days"] == _LIVE_WINDOW_DAYS)
    ]
    if sub.empty:
        return None, 0
    r = sub.iloc[0]
    prec = r["precision_over_live"] if bet == "Over" else r["precision_under_live"]
    return (float(prec) if pd.notna(prec) else None), int(r["n_settled"])


def _market_trust_metric(col, live: pd.DataFrame, league, market, bet) -> None:
    prec, n = _live_read(live, league, market, bet)
    if n == 0:
        col.caption("Market trust: no live settlement history yet.")
    elif n >= _MIN_SETTLED and prec is not None:
        col.metric(f"{bet} hit rate · live 30d", f"{prec:.0%}", help=f"{n} settled legs")
    else:
        col.caption(f"Market trust: needs ≥{_MIN_SETTLED} settled ({n} so far).")


def _render_market_trust(row: pd.Series, live: pd.DataFrame) -> None:
    league, market, bet = row.get("League"), row.get("Market"), row.get("Bet")
    info, action = st.columns([3, 1])
    _market_trust_metric(info, live, league, market, bet)
    if action.button("View in Model Lab", icon=":material/open_in_new:", key="nav_lab_cell"):
        st.session_state["nav_cell"] = (league, market)
        st.switch_page("surfaces/lab_training.py")


def _detail_row(row: pd.Series, details: pd.DataFrame) -> pd.Series | None:
    """The prerendered detail row matching this offer's key, or ``None`` if absent."""
    if details.empty:
        return None
    mask = pd.Series(True, index=details.index)
    for col in _DETAIL_KEY:
        mask &= details[col].astype(str) == str(row.get(col))
    sub = details[mask]
    return sub.iloc[0] if not sub.empty else None


def _decode(detail: pd.Series | None, col: str) -> list:
    if detail is None:
        return []
    raw = detail.get(col)
    return json.loads(raw) if isinstance(raw, str) and raw else []


def _render_comps_tab(detail: pd.Series | None, row: pd.Series) -> None:
    comps = _decode(detail, "comps_vs_opp")
    opp = row.get("Opponent", "")
    if not comps:
        st.caption(f"No comparable player has faced {opp} in the last 300 days.")
        return
    df = pd.DataFrame(comps)
    df["pct_diff"] = (df["pct_diff"] * 100).map(lambda v: f"{v:+.0f}%")
    df = df.rename(
        columns={
            "comp": "Comp",
            "n_games": "Games",
            "avg_vs_opp": f"Avg vs {opp}",
            "pct_diff": "vs their avg",
        }
    )
    st.caption(
        f"KNN comps who faced {opp} in the last 300 days — their average then, "
        "against their own 300-day norm."
    )
    st.dataframe(df, hide_index=True, width="stretch")


def _render_other_stats_tab(detail: pd.Series | None, row: pd.Series) -> None:
    volume = _decode(detail, "volume_trend")
    others = _decode(detail, "other_stats")
    if not volume and not others:
        st.caption("No prerendered supporting-stat detail — re-run `prophecize` to refresh.")
        return
    if volume:
        st.markdown(f"**Volume trend** · last {len(volume)}")
        st.altair_chart(sparkline(volume), width="stretch")
    for o in others:
        c1, c2, c3 = st.columns([2, 1, 2])
        c1.markdown(f"**{o['stat']}**")
        c2.markdown(f"{o['value']:.3g}")
        pct = o.get("percentile")
        if pct is not None:
            c3.caption(f"{pct:.0%} pctile · position, this week")
        if o.get("series"):
            c3.altair_chart(sparkline(o["series"]), width="stretch")


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
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        col1.markdown(f"**{desc}**")
        col2.markdown(strength_badge(mult) + f" {mult:.2f}×")
        idx = _find_corr_row_idx(desc, filtered)
        if col3.button("View", icon=":material/arrow_forward:", key=f"{tab_key_prefix}_view_{i}"):
            if idx is not None:
                st.session_state.detail_stack.append(idx)
                st.session_state.corr_nav = True
                st.rerun()
        if idx is not None and col4.button(
            "+ slip", icon=":material/add:", key=f"{tab_key_prefix}_slip_{i}"
        ):
            add_to_simple_slip(filtered.loc[idx])
            st.toast(f"Added {filtered.loc[idx]['Player']} to slip")


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
        st.altair_chart(history_chart(display_df, line), width="stretch")
    elif hist_df.empty:
        st.caption("No history available for this player/stat.")


def _render_model_tab(row: pd.Series) -> None:
    dist = row.get("Dist")
    ev = row.get("Projection")
    cv = row.get("CV")
    line = row.get("Line")
    if not (pd.notna(dist) and pd.notna(ev) and pd.notna(cv)):
        st.caption("Distribution parameters unavailable — re-run `prophecize` to refresh.")
        return

    params = {
        param: row.get(col) for col, param in DIST_PARAM_COLS.items() if pd.notna(row.get(col))
    }
    std = row.get("Projection STD") or ev * 0.3
    try:
        df_pdf, y_title, is_continuous = distribution_frame(dist, ev, std, params, line)
        chart = distribution_chart(
            df_pdf,
            is_continuous,
            line,
            row["Market"],
            y_title,
            projection=ev,
            consensus_line=row.get("Consensus Line"),
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            ":material/circle: app line (dashed) · :gray[consensus line (solid)] · "
            ":orange[model projection (dot)]"
        )
    except Exception as e:
        st.error(f"Error computing distribution: {e}")


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
    _render_header(row)
    _render_case(row)
    _render_context_strip(row, load_current_game_context())
    _render_market_trust(row, load_live_metrics())

    detail = _detail_row(row, load_current_offer_details())
    tabs = st.tabs(
        [
            ":material/history: History",
            ":material/query_stats: Model",
            ":material/group: Comps",
            ":material/insights: Other stats",
            ":material/hub: Correlated",
        ]
    )
    with tabs[0]:
        _render_history_tab(row)
    with tabs[1]:
        _render_model_tab(row)
    with tabs[2]:
        _render_comps_tab(detail, row)
    with tabs[3]:
        _render_other_stats_tab(detail, row)
    with tabs[4]:
        _render_corr_tab(row, filtered)
