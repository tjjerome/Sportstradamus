"""Offer-detail dialog and navigation for the dashboard.

The evidence-chain view a user opens on any offer: a themed header (Cinzel kicker,
player + market, side line, edge badge, gold rule), the model's "case" (the per-offer
``Why``), a one-line game-context strip, and five tabs (History, Model, Comps, Other
stats, Correlated). The Comps / Other-stats tabs read the ``current_offer_details``
sidecar prerendered at ``prophecize`` time so the server never recomputes them live.
"""

import html

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.deep_dive_tabs import (
    render_comps_tab,
    render_corr_tab,
    render_history_tab,
    render_model_tab,
    render_other_stats_tab,
)
from sportstradamus.dashboard.components.glyphs import game_shape_glyph
from sportstradamus.dashboard.data import (
    load_current_game_context,
    load_current_offer_details,
)
from sportstradamus.dashboard.narrative import SHAPE_HELP, bet_arrow, context_strip
from sportstradamus.dashboard.theme import GOLD, GRAY

_DETAIL_KEY = ["League", "Date", "Player", "Market", "Opponent"]

# Compact glyph for the one-line context strip — smaller than Tonight's cards.
_STRIP_GLYPH_SIZE = 22


def init_detail_state() -> None:
    """Initialise the session-state keys the detail dialog navigates with."""
    if "detail_stack" not in st.session_state:
        st.session_state.detail_stack = []
    if "last_grid_key" not in st.session_state:
        st.session_state.last_grid_key = None
    if "corr_nav" not in st.session_state:
        st.session_state.corr_nav = False


def drop_detail_on_page_change(state, page_id: str) -> None:
    """Close an offer-detail dialog left open when the active surface changes.

    ``detail_stack`` is global session state shared by every dialog-owning surface
    (Board, Games), so a dialog left open on one page would re-open the moment you
    return to another. The app entry calls this once per run with the active page id;
    within a page (id unchanged) the stack and its correlation-nav history survive
    across reruns.
    """
    if state.get("_active_page") != page_id:
        state["detail_stack"] = []
        state["last_grid_key"] = None
        state["_active_page"] = page_id


def _edge_badge(model_ev: float) -> str:
    """Board-style Model Edge (``Model EV − 1``) as a semantic colored chip."""
    edge = model_ev - 1.0
    color = "green" if edge >= 0 else "red"
    return f":{color}[**{edge:+.0%} edge**]"


def _esc_field(row: pd.Series, key: str) -> str:
    """HTML-escaped ``row[key]``, or escaped ``"?"`` when absent."""
    return html.escape(str(row.get(key, "?")))


def _render_header(row: pd.Series) -> None:
    """Themed-workbench header: Cinzel kicker, Plex name + market, side line, gold rule."""
    market = html.escape(str(row.get("Market Display", row.get("Market", "?"))))
    bet = _esc_field(row, "Bet")
    arrow = bet_arrow(bet) if bet in ("Over", "Under") else ""
    head, badge = st.columns([4, 1])
    head.markdown(
        f'<div class="celestial-kicker">◈ {_esc_field(row, "League")} · '
        f"{_esc_field(row, 'Platform')}</div>"
        f'<div style="font-size:19px;font-weight:700;margin:2px 0 1px">'
        f"{_esc_field(row, 'Player')} — {market}</div>"
        f'<div style="color:{GRAY};font-size:13px">{arrow} '
        f"{bet} {_esc_field(row, 'Line')} · "
        f"{_esc_field(row, 'Team')} vs "
        f"{_esc_field(row, 'Opponent')}</div>",
        unsafe_allow_html=True,
    )
    model_ev = row.get("Model EV")
    if pd.notna(model_ev):
        badge.markdown(_edge_badge(float(model_ev)))
    st.markdown(
        f'<hr style="height:1px;border:0;margin:10px 0 4px;'
        f'background:linear-gradient(90deg,{GOLD},transparent)">',
        unsafe_allow_html=True,
    )


def _render_case(row: pd.Series) -> None:
    """The model's case for the pick — the per-offer ``Why`` prose, if present."""
    why = row.get("Why")
    if isinstance(why, str) and why.strip():
        st.info(why)


def _pct(v) -> str:
    return f"{v * 100:+.1f}%" if pd.notna(v) and isinstance(v, int | float) else "N/A"


def _team_total_text(ou, strip: dict | None) -> str:
    """``"88.2 team total (+3.1)"`` — implied team total and its delta vs the season avg."""
    if not (pd.notna(ou) and isinstance(ou, int | float)):
        return "team total N/A"
    if strip and not pd.isna(strip["baseline_total"]):
        return f"{ou:.1f} team total ({ou - strip['baseline_total'] / 2:+.1f})"
    return f"{ou:.1f} team total"


def _win_prob_text(strip: dict | None) -> str:
    """``"64% win prob"`` from the game moneyline favorite prob, or ``"win prob N/A"``."""
    prob = strip["ml_fav_prob"] if strip else None
    if prob is not None and pd.notna(prob):
        return f"{prob:.0%} win prob"
    return "win prob N/A"


def _render_context_strip(row: pd.Series, game_context: pd.DataFrame) -> None:
    strip = (
        context_strip(game_context, game=row.get("Game", ""), date=row.get("Date", ""))
        if not game_context.empty
        else None
    )
    shape = strip["shape"] if strip else None
    glyph = (
        game_shape_glyph(str(shape), size=_STRIP_GLYPH_SIZE)
        if shape
        else f'<span style="color:{GRAY}">shape N/A</span>'
    )
    st.markdown(
        f'<div title="{SHAPE_HELP}" style="font-family:\'IBM Plex Mono\',monospace;'
        f"color:{GRAY};font-size:12px;display:flex;align-items:center;gap:8px;"
        f'flex-wrap:wrap;margin:2px 0 6px">'
        f"<span>{_team_total_text(row.get('O/U'), strip)}</span><span>·</span>"
        f"<span>{_win_prob_text(strip)}</span><span>·</span>"
        f"{glyph}<span>·</span>"
        f"<span>DVPOA {_pct(row.get('DVPOA'))}</span></div>",
        unsafe_allow_html=True,
    )


def _detail_row(row: pd.Series, details: pd.DataFrame) -> pd.Series | None:
    """The prerendered detail row matching this offer's key, or ``None`` if absent."""
    if details.empty:
        return None
    mask = pd.Series(True, index=details.index)
    for col in _DETAIL_KEY:
        mask &= details[col].astype(str) == str(row.get(col))
    sub = details[mask]
    return sub.iloc[0] if not sub.empty else None


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


@st.dialog("Offer detail", width="large")
def show_detail(row: pd.Series, filtered: pd.DataFrame) -> None:
    _render_nav()
    _render_header(row)
    _render_case(row)
    _render_context_strip(row, load_current_game_context())

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
        render_history_tab(row)
    with tabs[1]:
        render_model_tab(row)
    with tabs[2]:
        render_comps_tab(detail, row)
    with tabs[3]:
        render_other_stats_tab(detail, row)
    with tabs[4]:
        render_corr_tab(row, filtered)
