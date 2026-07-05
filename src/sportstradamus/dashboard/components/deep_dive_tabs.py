"""The five tab bodies of the offer-detail dialog (History, Model, Comps, Other stats,
Correlated).

``deep_dive.show_detail`` builds the header/case/context strip and the ``st.tabs``
shell, then calls the five public renderers here. Split out of ``deep_dive.py`` so the
orchestration and the tab content live in one purpose each; the private helpers
(``_decode``, ``_heat_css``, ``_gauge_svg``, ``_lift_survivors``, ``_render_stat_row``,
``_render_corr_cards``, ``_select_history_df``, ``_p_over``) are used only within this
file. The Comps / Other-stats tabs read the ``current_offer_details`` sidecar
prerendered at ``prophecize`` time; the Correlated tab prices each candidate live via
the copula scorer (``slip_engine.ev_lift``).
"""

import json

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.deep_dive_charts import (
    DIST_PARAM_COLS,
    build_h2h_history,
    build_recent_history,
    distribution_chart,
    distribution_frame,
    history_chart,
    resolve_std,
    sparkline,
)
from sportstradamus.dashboard.components.slip_state import add_to_simple_slip
from sportstradamus.dashboard.data import (
    GAMELOG_SCHEMA,
    load_current_game_corr,
    load_stat_tooltips,
)
from sportstradamus.dashboard.legs import find_offer_idx
from sportstradamus.dashboard.slip_engine import ev_lift
from sportstradamus.dashboard.theme import DIVERGING_COLORS, GRAY, SEQUENTIAL_COLORS
from sportstradamus.leg_schema import build_leg, leg_label

# Comps-tab heatmap on the "vs their avg" column: same diverging-ramp bucket semantics
# as grid.py so a painted cell reads the same everywhere (DESIGN §4). Only the saturated
# tails paint (red below the comp's own average, blue above); the near-neutral middle
# stays unpainted, keeping the light text readable on painted cells. Center is 0 because
# pct_diff is already a signed deviation from the comp's own norm.
_HEAT_TEXT = "#E6E9EF"
_NEG_STRONG, _NEG = DIVERGING_COLORS[0], DIVERGING_COLORS[1]
_POS, _POS_STRONG = DIVERGING_COLORS[7], DIVERGING_COLORS[8]
_HEAT_STRONG_FRAC = 0.6
_HEAT_NEUTRAL_FRAC = 0.2

# Comps table is a plain HTML table (borderColor token, hardcoded per grid.py's own
# precedent — the token isn't a theme.py export) rather than st.dataframe: the DataFrame
# grid renders to a <canvas>, so zebra striping and per-column mono/right-align (DESIGN
# §4) aren't reachable via CSS/column_config on it. A real <table> gets all three
# directly, matching the mockup's own markup for this tab.
_TABLE_BORDER = "#2A2E37"
_TABLE_MONO = "font-family:'IBM Plex Mono',monospace;text-align:right"

# Other-stats vertical percentile gauge: a SEQUENTIAL_COLORS fill (magnitude, not
# good/bad — a high turnovers percentile is not "good"), never gold (DESIGN §2: gold is
# never a data mark). Step 4 of the 10-step blue ramp reads clearly against the
# border-token track while steering clear of the primary-button blue (SEQUENTIAL_COLORS[6]).
# Geometry mirrors the mockup's 10×30 viewBox two-rect gauge.
_GAUGE_FILL = SEQUENTIAL_COLORS[3]
_GAUGE_TRACK = "#2A2E37"
_GAUGE_TRACK_Y = 2.0
_GAUGE_TRACK_H = 26.0


def _p_over(row: pd.Series) -> float | None:
    """Model P(outcome over the line), or ``None`` if ``Win Prob`` is absent.

    ``Win Prob`` is ``max(Model Over, Model Under)`` with ``Bet`` its argmax, so it
    approximates P(over) on an Over pick and its complement on an Under pick. The raw
    ``Model Over`` is not persisted to the offer snapshot; this recovers it. Note:
    ``_finalize_records`` clips each side to 90% independently before taking the max,
    so on a >90%-confidence Under pick the derived complement can slightly overstate
    true P(over).
    """
    win = row.get("Win Prob")
    if not pd.notna(win):
        return None
    return float(win) if row.get("Bet") == "Over" else 1.0 - float(win)


def _decode(detail: pd.Series | None, col: str):
    """Parse a sidecar JSON column. ``[]`` when absent/empty; the payload otherwise
    (a list for comps/other_stats, a single dict for volume_trend, ``None`` for null).
    """
    if detail is None:
        return []
    raw = detail.get(col)
    return json.loads(raw) if isinstance(raw, str) and raw else []


def _heat_css(value: float, span: float) -> str:
    """Diverging-ramp cell CSS for a signed ``pct_diff`` (center 0), or ``""`` neutral.

    Mirrors ``grid.py``'s bucket thresholds: below the neutral band paints red (mild →
    strong), above it blue, and light text on any painted cell keeps contrast.
    """
    if not span or pd.isna(value):
        return ""
    if value <= -_HEAT_STRONG_FRAC * span:
        bg = _NEG_STRONG
    elif value <= -_HEAT_NEUTRAL_FRAC * span:
        bg = _NEG
    elif value < _HEAT_NEUTRAL_FRAC * span:
        return ""
    elif value < _HEAT_STRONG_FRAC * span:
        bg = _POS
    else:
        bg = _POS_STRONG
    return f"background-color: {bg}; color: {_HEAT_TEXT}"


def _table_cell(text: str, *, mono: bool = False, extra: str = "") -> str:
    style = f"padding:7px 10px;border-bottom:1px solid {_TABLE_BORDER}"
    if mono:
        style += f";{_TABLE_MONO}"
    if extra:
        style += f";{extra}"
    return f'<td style="{style}">{text}</td>'


def _comps_table_html(comps: list[dict], avg_label: str, span: float) -> str:
    """A zebra-striped HTML table: right-aligned Plex Mono numerals, the last column
    (``avg_label``'s % deviation) heatmapped via ``_heat_css``. ``comps`` iterates in
    the order given — the caller preserves the KNN closest-first order, not this table.
    """
    labels = ["Comp", "Games", avg_label, "vs their avg"]
    header = "".join(
        f'<th style="padding:7px 10px;text-align:{"left" if lbl == "Comp" else "right"};'
        f'color:{GRAY};font-size:11px;border-bottom:1px solid {_TABLE_BORDER}">{lbl}</th>'
        for lbl in labels
    )
    rows = []
    for i, c in enumerate(comps):
        pct = c["pct_diff"]
        cells = (
            _table_cell(c["comp"])
            + _table_cell(str(c["n_games"]), mono=True)
            + _table_cell(f"{c['avg_vs_opp']:.1f}", mono=True)
            + _table_cell(f"{pct * 100:+.0f}%", mono=True, extra=_heat_css(pct, span))
        )
        zebra = "background:rgba(255,255,255,.015)" if i % 2 else ""
        rows.append(f'<tr style="{zebra}">{cells}</tr>')
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render_comps_tab(detail: pd.Series | None, row: pd.Series) -> None:
    """KNN comps who faced this opponent, heat-mapped against their own 300-day norm."""
    comps = _decode(detail, "comps_vs_opp")
    opp = row.get("Opponent", "")
    if not comps:
        st.caption(f"No comparable player has faced {opp} in the last 300 days.")
        return
    # comps_vs_opponent already returns closest-comp-first (KNN order); preserve it.
    span = float(max(abs(c["pct_diff"]) for c in comps))
    st.caption(
        f"KNN comps who faced {opp} in the last 300 days — their average then, "
        "against their own 300-day norm."
    )
    st.markdown(_comps_table_html(comps, f"Avg vs {opp}", span), unsafe_allow_html=True)


def _gauge_svg(percentile: float) -> str:
    """A slim vertical percentile gauge: a SEQUENTIAL_COLORS fill over a track, bottom-
    anchored so its height is the percentile (0..1). Inline SVG for st.markdown.
    """
    fill_h = _GAUGE_TRACK_H * percentile
    fill_y = _GAUGE_TRACK_Y + _GAUGE_TRACK_H - fill_h
    return (
        '<svg viewBox="0 0 10 30" width="10" height="30">'
        f'<rect x="2" y="{_GAUGE_TRACK_Y}" width="6" height="{_GAUGE_TRACK_H}" rx="3" '
        f'fill="{_GAUGE_TRACK}"/>'
        f'<rect x="2" y="{fill_y:.2f}" width="6" height="{fill_h:.2f}" rx="3" '
        f'fill="{_GAUGE_FILL}"/></svg>'
    )


def _render_stat_row(entry: dict, tooltip: str | None) -> None:
    gauge_col, val_col, spark_col = st.columns([1, 3, 4])
    pct = entry.get("percentile")
    if pct is not None:
        gauge_col.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px">'
            f"{_gauge_svg(pct)}"
            f"<span style=\"font-family:'IBM Plex Mono',monospace;font-size:10px;"
            f'color:{GRAY}">{pct * 100:.0f}</span></div>',
            unsafe_allow_html=True,
        )
    val_col.metric(entry["stat"], f"{entry['value']:.3g}", help=tooltip)
    if entry.get("series"):
        spark_col.altair_chart(sparkline(entry["series"]), width="stretch")


def render_other_stats_tab(detail: pd.Series | None, row: pd.Series) -> None:
    """Volume trend plus supporting stats, each with a same-position percentile gauge."""
    volume = _decode(detail, "volume_trend")
    volume = volume if isinstance(volume, dict) else None
    others = _decode(detail, "other_stats")
    if not volume and not others:
        st.caption("No prerendered supporting-stat detail — re-run `prophecize` to refresh.")
        return
    tips = load_stat_tooltips().get(row.get("League", ""), {})
    for entry in ([volume] if volume else []) + list(others):
        _render_stat_row(entry, tips.get(entry["stat"]))
    st.caption("Gauge: percentile among same-position slate players, low → high.")


def _lift_survivors(
    items: list[dict],
    filtered: pd.DataFrame,
    focus_leg: dict,
    corr: pd.DataFrame,
    platform: str | None,
) -> list[tuple[dict, int, float]]:
    """``(item, offer_idx, lift)`` for each partner with positive EV lift.

    Resolves each partner back to its offer row (``find_offer_idx`` pinned to the
    displayed offer's book), builds a canonical candidate leg, and prices the pair
    against the focus alone. A partner that isn't in the current grid (``idx is None``,
    e.g. a Board selection narrower than the offer pool) has no computable lift and is
    dropped, as is any candidate whose lift isn't positive (``ev_lift`` returns a raw EV
    multiplier, so "positive lift" is ``> 1.0`` — the ``_edge_badge`` convention).
    """
    survivors = []
    for item in items:
        idx = find_offer_idx(item, filtered, platform=platform)
        if idx is None:
            continue
        lift = ev_lift(focus_leg, build_leg(filtered.loc[idx]), corr, platform=platform or "")
        if lift > 1.0:
            survivors.append((item, idx, lift))
    return survivors


def _render_corr_cards(
    survivors: list[tuple[dict, int, float]],
    group_label: str,
    filtered: pd.DataFrame,
    tab_key_prefix: str,
) -> None:
    """Render one correlated-partner group. ``survivors`` are the positive-lift partners
    ``_lift_survivors`` already resolved to their offer rows, so the row's Boost/View
    lookup is fixed to the displayed offer's own book (``find_correlation`` runs once per
    platform; ``filtered`` can hold both books for a game, so the resolution is pinned).
    """
    if not survivors:
        return
    st.markdown(f"**{group_label}**")
    for i, (item, idx, lift) in enumerate(survivors):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        col1.markdown(f"**{leg_label(item)}**")
        col2.markdown(f":green[**{lift - 1.0:+.1%} EV**]")
        if col3.button("View", icon=":material/arrow_forward:", key=f"{tab_key_prefix}_view_{i}"):
            st.session_state.detail_stack.append(idx)
            st.session_state.corr_nav = True
            st.rerun()
        if col4.button("+ slip", icon=":material/add:", key=f"{tab_key_prefix}_slip_{i}"):
            add_to_simple_slip(filtered.loc[idx])
            st.toast(f"Added {filtered.loc[idx]['Player']} to slip")


def _select_history_df(
    hist_df: pd.DataFrame, h2h_df: pd.DataFrame, league: str, opponent: str, row_id: int
) -> pd.DataFrame:
    if hist_df.empty or league == "NFL" or h2h_df.empty:
        return hist_df
    filter_opt = st.segmented_control(
        "Filter by opponent",
        options=["All games", f"vs {opponent}"],
        default="All games",
        key=f"h2h_filter_{row_id}",
    )
    return h2h_df if filter_opt == f"vs {opponent}" else hist_df


def render_history_tab(row: pd.Series) -> None:
    """Recent-history sparkline, with an opponent filter when a H2H sample exists."""
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


def render_model_tab(row: pd.Series) -> None:
    """The model's fitted distribution vs. the app/consensus lines, plus P(over)."""
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
    std = resolve_std(row.get("Projection STD"), ev, cv)
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
        p_over = _p_over(row)
        if p_over is not None:
            st.markdown(f":green[**P(over) {p_over:.0%}**]")
        st.caption(
            ":green[over] / :red[under] mass · app line (dashed white) · "
            ":gray[consensus line (solid)] · model projection (gold dot)"
        )
    except Exception as e:
        st.error(f"Error computing distribution: {e}")


def render_corr_tab(row: pd.Series, filtered: pd.DataFrame) -> None:
    """Same-team and opponent partners that clear the positive-EV-lift threshold."""
    # A parquet round-trip decodes a list<struct> cell to a numpy ndarray, not a
    # list — `x or []` on a >1-element ndarray raises (ambiguous truth value), so
    # branch on identity and convert explicitly instead of relying on truthiness.
    raw_same = row.get("Corr Same")
    raw_opp = row.get("Corr Opp")
    same_items = [] if raw_same is None else list(raw_same)
    opp_items = [] if raw_opp is None else list(raw_opp)
    platform = row.get("Platform")
    focus_leg = build_leg(row)
    corr = load_current_game_corr()
    same = _lift_survivors(same_items, filtered, focus_leg, corr, platform)
    opp = _lift_survivors(opp_items, filtered, focus_leg, corr, platform)
    _render_corr_cards(same, f"Same team — {row['Team']}", filtered, "corr_same")
    _render_corr_cards(opp, f"Opponent — {row['Opponent']}", filtered, "corr_opp")
    if not same and not opp:
        st.caption("No correlated legs cleared the positive-EV-lift threshold for this offer.")
