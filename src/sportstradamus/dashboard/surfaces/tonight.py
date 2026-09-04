"""Tonight — game-card slate showing today's offers grouped by game."""

import html
from urllib.parse import quote

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.glyphs import game_shape_glyph
from sportstradamus.dashboard.components.hero import page_hero
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_game_context,
    load_current_game_stories,
    load_current_meta,
    load_current_offers,
    load_current_parlays,
    sport_filtered,
)
from sportstradamus.dashboard.narrative import (
    SHAPE_CAPTION,
    SHAPE_DISPLAY,
    context_strip,
    game_headline,
    home_away,
    storyless_prophecy,
)

# Card-side glyph is the largest on this page; the legend row below shrinks to fit
# five across one line of st.columns(5).
_CARD_GLYPH_SIZE = 58
_LEGEND_GLYPH_SIZE = 28

# A game tipping off within this many minutes is "urgent" — it rises to the top of the
# slate and its kicker turns red. Needs the Commence tip-off timestamp on the offers;
# absent it (older snapshots, Sleeper), every game reads as non-urgent.
_URGENT_MINUTES = 60


def _format_countdown(minutes: float) -> str:
    """Minutes-to-tip as "locks in 2h 05m" / "locks in 43m"; "in progress" once started."""
    if minutes <= 0:
        return "in progress"
    hours, mins = divmod(int(minutes), 60)
    return f"locks in {hours}h {mins:02d}m" if hours else f"locks in {mins}m"


def _render_shape_legend() -> None:
    """Compact 5-column glyph + name + caption row, shared vocabulary for the cards below."""
    cols = st.columns(len(SHAPE_CAPTION))
    for col, (shape, sub) in zip(cols, SHAPE_CAPTION.items(), strict=True):
        with col:
            st.markdown(game_shape_glyph(shape, size=_LEGEND_GLYPH_SIZE), unsafe_allow_html=True)
            st.caption(f"**{SHAPE_DISPLAY[shape]}**  \n{sub}")


meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
page_hero("TONIGHT'S SLATE", "Tonight", generated)

offers = sport_filtered(load_current_offers())
parlays = load_current_parlays()
stories = load_current_game_stories()
game_context = load_current_game_context()

if offers.empty:
    st.info("No current predictions. Run `poetry run prophecize` to generate offers.")
    st.caption("Prophecies arrive with the next data wave.")
    st.stop()

_render_shape_legend()
st.divider()

# Group on the canonical Game key (not Team/Opponent) so a matchup's two
# per-team orderings collapse to one card.
game_key_cols = [c for c in ["League", "Date", "Game"] if c in offers.columns]
if "Game" not in offers.columns:
    st.info("Offer data missing game grouping columns.")
    st.stop()

now = pd.Timestamp.now(tz="UTC")
cards = []
for game_key, group in offers.groupby(game_key_cols, sort=False):
    if isinstance(game_key, str):
        game_key = (game_key,)
    key_dict = dict(zip(game_key_cols, game_key, strict=False))
    game = str(group["Game"].iloc[0])
    date_raw = key_dict.get("Date", "")
    home, away = home_away(group)

    # Favored legs: the model-liked picks (Kelly > 0) the constellation draws as stars,
    # deduped to distinct Player/Market/Bet across books and alt-lines.
    kelly = pd.to_numeric(group["Kelly"], errors="coerce") if "Kelly" in group else None
    favored = (
        group.loc[kelly > 0].drop_duplicates(["Player", "Market", "Bet"]).shape[0]
        if kelly is not None
        else 0
    )
    # Top edge = the best model edge vs the DFS payout in the game (Model EV − 1).
    edges = pd.to_numeric(group["Model EV"], errors="coerce") - 1 if "Model EV" in group else None
    top_edge = edges.max() if edges is not None and edges.notna().any() else None
    strip = context_strip(game_context, game=game, date=date_raw)
    # Commence is the full tip-off timestamp threaded from the scrapers; absent it (older
    # snapshots, Sleeper offers) minutes-to-tip is inf, so the game reads as non-urgent.
    commence = (
        pd.to_datetime(group["Commence"].iloc[0], utc=True, errors="coerce")
        if "Commence" in group.columns
        else pd.NaT
    )
    minutes = (commence - now).total_seconds() / 60 if pd.notna(commence) else float("inf")
    cards.append(
        {
            "league": key_dict.get("League", ""),
            "home": home,
            "away": away,
            "date": date_raw,
            "favored": favored,
            "offer_count": len(group),
            "top_edge": top_edge,
            "headline": game_headline(stories, parlays, game=game, date=date_raw),
            "shape": strip["shape"] if strip else "",
            "minutes": minutes,
            "urgent": 0 < minutes < _URGENT_MINUTES,
        }
    )

# Games tipping off within the hour rise to the top; then most favored legs first,
# soonest tip-off as the tiebreak.
cards.sort(key=lambda c: (0 if c["urgent"] else 1, -c["favored"], c["minutes"]))

for c in cards:
    league, home, away = c["league"], c["home"], c["away"]
    top_edge, favored, offer_count = c["top_edge"], c["favored"], c["offer_count"]
    shape, minutes = c["shape"], c["minutes"]
    # A game with no context row has no shape at all, which is the clouded reading too.
    shape_name = SHAPE_DISPLAY.get(shape, SHAPE_DISPLAY["even"])

    # A positive top edge lights the card; without one it drops to the muted "no edge"
    # state. The prophecy is the oracle voice — the story headline, or a dim placeholder.
    has_edge = top_edge is not None and pd.notna(top_edge) and top_edge > 0
    card_class = "tonight-card" if has_edge else "tonight-card muted"
    edge_badge = (
        f'<span class="tc-badge tc-g">+{top_edge:.0%} top edge</span>'
        if has_edge
        else '<span class="tc-badge tc-gray">no edge</span>'
    )
    prophecy = (
        f'<div class="tc-prophecy">“{html.escape(c["headline"])}”</div>'
        if c["headline"]
        else f'<div class="tc-prophecy tc-dim">{storyless_prophecy(favored)}</div>'
    )
    countdown = _format_countdown(minutes) if minutes != float("inf") else ""
    league_esc = html.escape(league)
    kicker = f"{league_esc} &nbsp;·&nbsp; {countdown}" if countdown else league_esc
    kicker_cls = "tc-kicker tc-urgent" if c["urgent"] else "tc-kicker"
    plural = "s" if offer_count != 1 else ""
    # The whole card is the View-game link; Date keeps doubleheaders distinct and the
    # label must match the Games page's game-picker (its ?game= deep link).
    game_param = f"{home} vs {away} · {c['date']}" if c["date"] else f"{home} vs {away}"
    home_esc, away_esc = html.escape(home, quote=True), html.escape(away, quote=True)
    st.markdown(
        f'<div class="{card_class}">'
        f'<a class="tc-cardlink" href="games?game={quote(game_param)}" target="_self" '
        f'aria-label="View {home_esc} vs {away_esc}"></a>'
        f'<div class="tc-main"><div class="{kicker_cls}">{kicker}</div>'
        f'<div class="tc-matchup">{home_esc} <span class="tc-vs">vs</span> {away_esc}</div>'
        f"{prophecy}"
        f'<div class="tc-foot">{edge_badge}<span class="tc-meta">'
        f"<b>{offer_count}</b> offer{plural} &middot; <b>{favored}</b> favored</span>"
        f'<span class="tc-viewcue">View game →</span></div></div>'
        f'<div class="tc-side">{game_shape_glyph(shape, size=_CARD_GLYPH_SIZE)}'
        f'<span class="tc-shapename">{shape_name}</span></div></div>',
        unsafe_allow_html=True,
    )
