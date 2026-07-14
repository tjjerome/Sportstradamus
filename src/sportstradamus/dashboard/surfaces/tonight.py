"""Tonight — game-card slate showing today's offers grouped by game."""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.glyphs import game_shape_glyph
from sportstradamus.dashboard.components.hero import page_hero
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_game_context,
    load_current_meta,
    load_current_offers,
    load_current_parlays,
    sport_filtered,
)
from sportstradamus.dashboard.narrative import context_strip, home_away, top_thesis

# Legend glyph size: smaller than the default (40px) card glyph so five fit across
# one row of st.columns(5).
_LEGEND_GLYPH_SIZE = 28

# (shape, display name, mockup .sub caption) for the legend row, in the same order
# as docs/mockups/p8-glyphs.html.
_SHAPE_LEGEND = [
    ("shootout", "Shootout", "comet · fast total"),
    ("blowout", "Blowout", "supernova · lopsided"),
    ("coinflip", "Coinflip", "scales · close ML"),
    ("even", "Even", "lone star · neutral"),
    ("grind", "Grind", "hourglass · low total"),
]


def _render_shape_legend() -> None:
    """Compact 5-column glyph + name + caption row, shared vocabulary for the cards below."""
    cols = st.columns(len(_SHAPE_LEGEND))
    for col, (shape, name, sub) in zip(cols, _SHAPE_LEGEND, strict=False):
        with col:
            st.markdown(game_shape_glyph(shape, size=_LEGEND_GLYPH_SIZE), unsafe_allow_html=True)
            st.caption(f"**{name}**  \n{sub}")


meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
page_hero("TONIGHT'S SLATE", "Tonight", generated)

offers = sport_filtered(load_current_offers())
parlays = load_current_parlays()
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

for game_key, group in offers.groupby(game_key_cols, sort=False):
    if isinstance(game_key, str):
        game_key = (game_key,)
    key_dict = dict(zip(game_key_cols, game_key, strict=False))

    league = key_dict.get("League", "")
    date_raw = key_dict.get("Date", "")
    home, away = home_away(group)
    lock_time = format_ts(str(date_raw)) if date_raw else ""

    offer_count = len(group)
    # Legs past the Kelly filter (Kelly > 0) — the model-liked picks the constellation
    # draws as stars; deduped to distinct Player/Market/Bet across books and alt-lines.
    kelly = pd.to_numeric(group["Kelly"], errors="coerce") if "Kelly" in group else None
    liked_count = (
        group.loc[kelly > 0].drop_duplicates(["Player", "Market", "Bet"]).shape[0]
        if kelly is not None
        else 0
    )
    # Top edge = the best model edge vs the DFS payout in the game (Model EV − 1).
    edges = pd.to_numeric(group["Model EV"], errors="coerce") - 1 if "Model EV" in group else None
    top_edge = edges.max() if edges is not None and edges.notna().any() else None
    headline = top_thesis(parlays, game=group["Game"].iloc[0], date=date_raw)
    strip = context_strip(game_context, game=key_dict["Game"], date=date_raw)
    shape = strip["shape"] if strip else ""

    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{home} vs {away}**  ·  {league}")
            if headline:
                st.markdown(headline)
            if lock_time:
                st.caption(lock_time)
            st.caption(
                f"{offer_count} offer{'s' if offer_count != 1 else ''} · {liked_count} model-liked"
            )
            if top_edge is not None and pd.notna(top_edge):
                st.caption(f"Top edge: {top_edge:+.1%}")
        with right:
            st.markdown(game_shape_glyph(shape), unsafe_allow_html=True)
            game_label = f"{home} vs {away}"
            # Date keeps doubleheaders distinct; format must match the Games
            # page's game-picker labels.
            game_param = f"{game_label} · {date_raw}" if date_raw else game_label
            if st.button("View game", key=f"tonight_{league}_{date_raw}_{game_label}"):
                # switch_page drops query params set the same run; hand off via
                # session state and let the Games picker read it first (?game=
                # is the deep link).
                st.session_state["nav_game"] = game_param
                st.switch_page("surfaces/games.py")

st.caption("Prophecies arrive with the next data wave.")
