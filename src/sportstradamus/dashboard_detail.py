"""Shared offer-detail dialog and parlay-leg helpers for the dashboard pages.

Extracted verbatim (behaviour-preserving) from the Predictions Today page so the
Parlays page can reuse the same per-offer detail dialog.  Also hosts the
parlay-leg parser, offer lookup, and the offline distinctive-family-name
heuristic — all token-free, derived at render time from data already on disk.
"""

import hashlib
from collections import Counter, defaultdict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from sportstradamus.dashboard_data import GAMELOG_SCHEMA, load_gamelog
from sportstradamus.helpers import stat_map

# Distribution families, mirrored from the prediction pipeline.
_CONTINUOUS = ("Gamma", "ZAGamma", "SkewNormal")
_ZERO_INFLATED = ("ZAGamma", "ZINB")

# Map a market name to a coarse stat category so the phrase bank can pick a
# verb that fits the stat ("dominates the glass" only makes sense for boards).
# Needles cover every leg vocabulary in play: pretty names ("3-Pointers
# Made"), Sleeper snake keys ("threes_made"), and canonical codes ("FG3M").
_STAT_CATEGORY = {
    "scoring": (
        "point", "pts", "3-p", "3pt", "three", "fg3", "fg ", "fga", "fgm",
        "pass yd", "pass td", "rush yd", "rec yd", "goal", "shots",
    ),
    "boards": ("rebound", "reb", "board", "glass"),
    "playmaking": ("assist", "ast", "playmak"),
    "k's": ("strikeout", "pitcher k", "ks", "_k", "saves"),
}

# Deterministic, offline phrase bank keyed by (direction, stat_category).
# A stable hash of (player, family) selects the variant so names never flip
# between Streamlit reruns.
_PHRASES = {
    ("Over", "scoring"): ["{p} goes nuclear", "{p} can't miss", "{p} fills it up"],
    ("Over", "boards"): ["{p} dominates the glass", "{p} crashes the boards", "{p} cleans up"],
    ("Over", "playmaking"): ["{p} runs the show", "{p} dishes all night", "{p} sets the table"],
    ("Over", "k's"): ["{p} mows them down", "{p} racks up Ks", "{p} is unhittable"],
    ("Over", "production"): ["{p} goes off", "{p} stuffs the stat sheet", "{p} takes over"],
    ("Under", "scoring"): ["{p} ice cold", "{p} held in check", "{p} can't buy a bucket"],
    ("Under", "boards"): ["{p} boxed out", "{p} off the glass", "{p} disappears inside"],
    ("Under", "playmaking"): ["{p} can't find anyone", "{p} bottled up", "{p} held quiet"],
    ("Under", "k's"): ["{p} gets hit around", "{p} no swing-and-miss", "{p} laboring"],
    ("Under", "production"): ["{p} struggles", "{p} no-shows", "{p} quiet night"],
}



def init_detail_state() -> None:
    """Initialise the session-state keys the detail dialog navigates with."""
    if "detail_stack" not in st.session_state:
        st.session_state.detail_stack = []
    if "last_grid_key" not in st.session_state:
        st.session_state.last_grid_key = None
    if "corr_nav" not in st.session_state:
        st.session_state.corr_nav = False


# --- Helper functions for charts ---


def _history_chart(df: pd.DataFrame, line: float) -> alt.Chart:
    """Bar chart of recent games with a dotted betting-line rule.

    ``df`` must have columns ``Label`` (ordered x-axis string),
    ``StatValue`` (y), and ``Hit`` (bool).  The most-recent game should be
    last so the bars read left-to-right chronologically.
    """
    df = df.copy()
    df["color"] = np.where(df["Hit"], "Hit", "Miss")
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Label:N", sort=None, title="", axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("StatValue:Q", title=""),
            color=alt.Color(
                "color:N",
                scale=alt.Scale(domain=["Hit", "Miss"], range=["#4CAF50", "#F44336"]),
                legend=None,
            ),
            tooltip=["Label:N", "StatValue:Q"],
        )
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
                st.session_state.corr_nav = True
                st.rerun()


# --- Parlay-leg parsing & offer lookup (shared by legs + family names) ---


def parse_leg(leg: str) -> dict | None:
    """Parse a parlay leg string into its offer components.

    Leg format is ``"{Player} {Bet} {Line} {Market} - {Model P}%, {Boost}x"``
    (see ``prediction/correlation.py``).  Splitting on the unambiguous
    ``" Over "`` / ``" Under "`` token yields the player name even when it
    contains spaces.  Returns ``None`` for anything that does not parse.
    """
    if not isinstance(leg, str) or not leg.strip():
        return None
    head = leg.split(" - ", 1)[0].strip()
    for bet in ("Over", "Under"):
        token = f" {bet} "
        if token not in head:
            continue
        player, rest = head.split(token, 1)
        rest = rest.strip().split()
        if not rest:
            return None
        try:
            line = float(rest[0])
        except ValueError:
            return None
        market = " ".join(rest[1:]).strip()
        if not player.strip() or not market:
            return None
        return {"Player": player.strip(), "Bet": bet, "Line": line, "Market": market}
    return None


def _candidate_markets(market: str, platform: str | None) -> set[str]:
    """Canonical-code aliases for a leg's display market under a platform.

    Parlay legs carry the platform's source/display market label (Underdog
    "Pts + Rebs + Asts", Sleeper "pts_reb_ast") while ``current_offers.parquet``
    stores the canonical code ("PRA"). ``stat_map[platform]`` is the codebase's
    display→code table; the space-stripped lookup covers Underdog's spaced
    names ("Pts + Rebs + Asts" vs the table key "Pts+Rebs+Asts").
    """
    out = {market}
    pmap = stat_map.get(platform, {}) if platform else {}
    if market in pmap:
        out.add(pmap[market])
    nospace = market.replace(" ", "")
    if nospace in pmap:
        out.add(pmap[nospace])
    return out


def find_offer_idx(
    parsed: dict, offers: pd.DataFrame, platform: str | None = None
) -> int | None:
    """Find the offers-frame index for a parsed leg, or ``None`` if it moved.

    ``platform`` is the parlay's book; it lets the leg's display market be
    translated to the canonical code stored in the offers snapshot.
    """
    if not parsed or offers.empty:
        return None
    markets = _candidate_markets(parsed["Market"], platform)
    mask = (
        (offers["Player"] == parsed["Player"])
        & (offers["Bet"] == parsed["Bet"])
        & (offers["Market"].isin(markets))
        & (pd.to_numeric(offers["Line"], errors="coerce") == parsed["Line"])
    )
    matches = offers.index[mask]
    return int(matches[0]) if len(matches) else None


def _stat_category(market: str) -> str:
    """Coarse stat bucket for the phrase bank."""
    m = (market or "").lower()
    for cat, needles in _STAT_CATEGORY.items():
        if any(n in m for n in needles):
            return cat
    return "production"


def _phrase(player: str, family: float, direction: str, category: str) -> str:
    """Pick a stable phrase-bank variant for a family headliner."""
    bank = _PHRASES.get((direction, category)) or _PHRASES[(direction, "production")]
    seed = f"{player}|{family}".encode()
    idx = int(hashlib.md5(seed).hexdigest(), 16) % len(bank)
    return bank[idx].format(p=player)


def family_labels_for_game(game_group: pd.DataFrame) -> dict[float, str]:
    """Map each family in one game's parlays to a fun, distinct headline.

    "Star carousel": rank the game's players by stardom — number of distinct
    stat markets they're offered in (stars get props everywhere), tie-broken by
    how big their biggest line is relative to that market (stars carry high
    lines).  Hand the #1 star to the family they most drive (most legs), the
    #2 star to the next family, and so on, so every family gets a recognizable,
    different headliner.  Families left after the stars are exhausted fall
    through to their own next-best player — never a neutral placeholder.
    """
    leg_cols = [c for c in game_group.columns if c.startswith("Leg ")]
    families = sorted(game_group["Family"].dropna().unique())
    if not families:
        return {}

    # Per (family, player): list of (bet, market, line) legs.
    fam_legs: dict[float, dict[str, list]] = {f: defaultdict(list) for f in families}
    player_markets: dict[str, set] = defaultdict(set)
    player_total: Counter = Counter()
    market_lines: dict[str, list] = defaultdict(list)

    for _, row in game_group.iterrows():
        fam = row["Family"]
        if fam not in fam_legs:
            continue
        for col in leg_cols:
            p = parse_leg(row.get(col))
            if not p:
                continue
            fam_legs[fam][p["Player"]].append((p["Bet"], p["Market"], p["Line"]))
            player_markets[p["Player"]].add(p["Market"])
            player_total[p["Player"]] += 1
            market_lines[p["Market"]].append(p["Line"])

    if not player_total:
        return dict.fromkeys(families, "Mixed bag")

    def line_pct(market: str, line: float) -> float:
        vals = market_lines[market]
        return sum(1 for v in vals if v <= line) / len(vals) if vals else 0.0

    # Stardom: distinct-market breadth, then biggest line vs its market,
    # then raw volume — all free proxies for "this is a featured player".
    def stardom(player: str) -> tuple:
        best_pct = max(
            (
                line_pct(mk, ln)
                for fam in families
                for _, mk, ln in fam_legs[fam].get(player, [])
            ),
            default=0.0,
        )
        return (len(player_markets[player]), best_pct, player_total[player], player)

    ranked = sorted(player_total, key=stardom, reverse=True)

    labels: dict[float, str] = {}
    taken: set[str] = set()

    def name(fam: float, player: str) -> str:
        legs = fam_legs[fam][player]
        over = sum(1 for b, _, _ in legs if b == "Over")
        direction = "Over" if over * 2 >= len(legs) else "Under"
        category = Counter(_stat_category(m) for _, m, _ in legs).most_common(1)[0][0]
        return _phrase(player, fam, direction, category)

    # #1 star → family they most drive, #2 → next family, ...
    for player in ranked:
        if len(labels) == len(families):
            break
        cands = [f for f in families if f not in labels and player in fam_legs[f]]
        if not cands:
            continue
        fam = max(cands, key=lambda f: (len(fam_legs[f][player]), -f))
        labels[fam] = name(fam, player)
        taken.add(player)

    # Stars exhausted: give the family its own best remaining (then any) player.
    for fam in families:
        if fam in labels:
            continue
        pool = sorted(fam_legs[fam], key=stardom, reverse=True)
        pick = next((p for p in pool if p not in taken), pool[0] if pool else None)
        labels[fam] = name(fam, pick) if pick else "Mixed bag"
        if pick:
            taken.add(pick)
    return labels


@st.dialog("Offer detail", width="large")
def _show_detail(row: pd.Series, filtered: pd.DataFrame) -> None:
    """Render detailed view of a single offer with charts and correlations."""
    # Navigation: always-visible close, plus back when the stack is deep.
    nav = st.columns([1, 1, 6])
    if nav[0].button("✕ Close"):
        st.session_state.detail_stack = []
        st.session_state.last_grid_key = None
        st.rerun()
    if len(st.session_state.detail_stack) > 1:
        if nav[1].button("← Back"):
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
        stat_key = row.get("Stat") or row.get("Market")
        line = row.get("Line")
        league = row.get("League", "")
        opponent = row.get("Opponent", "")
        schema = GAMELOG_SCHEMA.get(league, {})

        hist_df = pd.DataFrame()
        if stat_key and schema:
            gl = load_gamelog(league)
            pcol = schema["player"]
            dcol = schema.get("date")
            ocol = schema.get("opp")
            hcol = schema.get("home")

            if not gl.empty and pcol in gl.columns and stat_key in gl.columns:
                pg = gl[gl[pcol] == row["Player"]].copy()
                if dcol and dcol in pg.columns:
                    pg = pg.sort_values(dcol)
                pg = pg.tail(10)

                # Build x-axis label: "@OPP, MM/DD" away, "OPP, MM/DD" home
                if ocol and hcol and ocol in pg.columns and hcol in pg.columns:
                    opp_prefix = np.where(pg[hcol].astype(bool), "", "@")
                    labels = opp_prefix + pg[ocol].astype(str)
                elif ocol and ocol in pg.columns:
                    labels = pg[ocol].astype(str)
                else:
                    labels = pd.Series(
                        [f"Wk {w}" for w in pg["week"]] if "week" in pg.columns
                        else [str(i + 1) for i in range(len(pg))],
                        index=pg.index,
                    )

                if dcol and dcol in pg.columns:
                    dates = pd.to_datetime(pg[dcol]).dt.strftime("%m/%d")
                    labels = labels + ", " + dates

                hist_df = pd.DataFrame({
                    "Label": labels.values,
                    "StatValue": pg[stat_key].values,
                    "Hit": pg[stat_key].values >= line,
                    "Opponent": pg[ocol].values if ocol and ocol in pg.columns else "",
                })

        # Radio filter for H2H if not NFL and opponent data exists
        display_df = hist_df
        h2h_df = pd.DataFrame()

        if league != "NFL" and opponent and stat_key and schema:
            # Precompute H2H data from full gamelog (not just last-10-games)
            gl = load_gamelog(league)
            pcol = schema["player"]
            ocol = schema.get("opp")
            dcol = schema.get("date")
            hcol = schema.get("home")

            if not gl.empty and all([pcol, ocol, stat_key]) and pcol in gl.columns:
                h2h_games = gl[(gl[pcol] == row["Player"]) & (gl[ocol] == opponent)].copy()
                if dcol and dcol in h2h_games.columns:
                    h2h_games = h2h_games.sort_values(dcol)
                h2h_games = h2h_games.tail(10)

                if not h2h_games.empty:
                    # Build labels
                    if hcol and hcol in h2h_games.columns:
                        opp_prefix = np.where(h2h_games[hcol].astype(bool), "", "@")
                        h2h_labels = pd.Series([f"{p}{opponent}" for p in opp_prefix], index=h2h_games.index)
                    else:
                        h2h_labels = pd.Series([opponent] * len(h2h_games), index=h2h_games.index)

                    if dcol and dcol in h2h_games.columns:
                        dates = pd.to_datetime(h2h_games[dcol]).dt.strftime("%m/%d")
                        h2h_labels = h2h_labels + ", " + dates

                    h2h_df = pd.DataFrame({
                        "Label": h2h_labels.values,
                        "StatValue": h2h_games[stat_key].values,
                        "Hit": h2h_games[stat_key].values >= line,
                        "Opponent": h2h_games[ocol].values if ocol in h2h_games.columns else opponent,
                    })

        if not hist_df.empty and league != "NFL" and not h2h_df.empty:
            filter_opt = st.radio(
                "Filter by opponent:",
                options=["All games", f"vs {opponent}"],
                horizontal=True,
                key=f"h2h_filter_{id(row)}",
            )
            display_df = h2h_df if filter_opt == f"vs {opponent}" else hist_df

        if not display_df.empty:
            st.altair_chart(_history_chart(display_df, line), use_container_width=True)
        elif hist_df.empty:
            st.caption("No history available for this player/stat.")

    with tab2:
        dist = row.get("Dist")
        ev = row.get("Model EV")
        cv = row.get("CV")
        line = row.get("Line")

        if pd.notna(dist) and pd.notna(ev) and pd.notna(cv):
            is_continuous = dist in _CONTINUOUS

            _PARAM_MAP = {
                "Model R": "r",
                "Model Alpha": "alpha",
                "Model Sigma": "sigma",
                "Model Skew": "skew_alpha",
                "Gate": "gate",
            }
            params = {
                param: row.get(col)
                for col, param in _PARAM_MAP.items()
                if pd.notna(row.get(col))
            }

            try:
                if is_continuous:
                    # Continuous: use 300 linspace points and scipy PDF
                    std = row.get("Model STD") or ev * 0.3
                    lo = max(0.0, ev - 4 * std)
                    hi = ev + 4 * std
                    xs = np.linspace(lo, hi, 300)

                    if dist == "Gamma":
                        alpha = params.get("alpha")
                        if not alpha or alpha <= 0:
                            raise ValueError("Gamma requires Model Alpha > 0")
                        scale = ev / alpha
                        pdf_vals = stats.gamma.pdf(xs, alpha, scale=scale)
                    elif dist == "ZAGamma":
                        alpha = params.get("alpha")
                        if not alpha or alpha <= 0:
                            raise ValueError("ZAGamma requires Model Alpha > 0")
                        gate = params.get("gate") or 0
                        scale = ev / alpha
                        pdf_vals = (1 - gate) * stats.gamma.pdf(xs, alpha, scale=scale)
                    elif dist == "SkewNormal":
                        sigma = params.get("sigma")
                        if not sigma or sigma <= 0:
                            sigma = ev * 0.3
                        skew = params.get("skew_alpha") or 0
                        pdf_vals = stats.skewnorm.pdf(xs, skew, loc=ev, scale=sigma)
                    else:
                        raise ValueError(f"Unknown continuous dist: {dist}")

                    y_title = "Density"
                else:
                    # Discrete: only evaluate at integer values
                    std = row.get("Model STD") or ev * 0.3
                    hi = int(np.ceil(ev + 4 * std)) + 1
                    xs = np.arange(0, hi + 1, dtype=int)

                    if dist == "Poisson":
                        pdf_vals = stats.poisson.pmf(xs, ev)
                    elif dist == "NegBin":
                        r = params.get("r")
                        if not r or r <= 0:
                            raise ValueError("NegBin requires Model R > 0")
                        p = r / (r + ev)
                        pdf_vals = stats.nbinom.pmf(xs, r, p)
                    elif dist == "ZINB":
                        r = params.get("r")
                        if not r or r <= 0:
                            raise ValueError("ZINB requires Model R > 0")
                        gate = params.get("gate") or 0
                        p = r / (r + ev)
                        pmf = stats.nbinom.pmf(xs, r, p)
                        pdf_vals = np.where(xs == 0, gate + (1 - gate) * pmf, (1 - gate) * pmf)
                    else:
                        raise ValueError(f"Unknown discrete dist: {dist}")

                    y_title = "Probability"

                df_pdf = pd.DataFrame(
                    {
                        "x": xs,
                        "P": pdf_vals,
                        "Side": ["Over" if x >= line else "Under" for x in xs],
                    }
                )

                x_enc = alt.X("x:Q", title=row["Market"])
                y_enc = alt.Y("P:Q", title=y_title)

                if is_continuous:
                    # Smooth PDF curve with line and area, colored by side of betting line
                    color_enc = alt.Color("Side:N",
                        scale=alt.Scale(domain=["Over","Under"], range=["#2196F3","#FF7043"]),
                        legend=alt.Legend(orient="top"))
                    chart = alt.layer(
                        alt.Chart(df_pdf)
                        .mark_area(opacity=0.3)
                        .encode(x=x_enc, y=y_enc, color=color_enc, tooltip=["x:Q", "P:Q", "Side:N"]),
                        alt.Chart(df_pdf)
                        .mark_line(strokeWidth=2)
                        .encode(x=x_enc, y=y_enc, color=color_enc),
                    )
                else:
                    # Discrete: bars at integer values, touching each other using explicit boundaries
                    df_pdf["x_start"] = df_pdf["x"] - 0.5
                    df_pdf["x_end"] = df_pdf["x"] + 0.5
                    color_enc = alt.Color("Side:N",
                        scale=alt.Scale(domain=["Over","Under"], range=["#2196F3","#FF7043"]),
                        legend=alt.Legend(orient="top"))
                    chart = (
                        alt.Chart(df_pdf)
                        .mark_rect(stroke="#444", strokeWidth=1)
                        .encode(x=alt.X("x_start:Q", title=row["Market"], axis=alt.Axis(tickMinStep=1)),
                                x2="x_end:Q",
                                y=alt.Y("P:Q", title=y_title),
                                color=color_enc,
                                tooltip=["x:Q", "P:Q", "Side:N"])
                    )

                betting_line = (
                    alt.Chart(pd.DataFrame({"Line": [line]}))
                    .mark_rule(strokeDash=[6, 3], color="#FFFFFF", strokeWidth=1.5)
                    .encode(x="Line:Q")
                )
                combined = chart + betting_line
                st.altair_chart(combined, use_container_width=True)
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
