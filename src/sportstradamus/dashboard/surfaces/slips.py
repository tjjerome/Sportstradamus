"""Build a slip — story- or game-seeded constellation builder.

Folds the former Game surface in: a slip can be seeded from a curated story or
from any game's offers, and the builder carries that game's Total / Spread /
Shape context banner.
"""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_builder import render_constellation_builder
from sportstradamus.dashboard.components.slip_state import seed_from_legs, seed_from_story
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_game_context,
    load_current_game_corr,
    load_current_game_stories,
    load_current_meta,
    load_current_offers,
    render_banner,
    sport_filtered,
)
from sportstradamus.prediction.stories.context import ctxs_from_frame


def _render_story_cascade(stories: pd.DataFrame, offers: pd.DataFrame) -> None:
    """Platform → game → story → objective; seed the constellation builder."""
    platform = st.selectbox(
        "Platform", sorted(stories["platform"].dropna().unique()), key="cascade_platform"
    )
    sub = stories.loc[stories["platform"] == platform]
    games = sorted(sub["Game"].dropna().unique())
    if not games:
        st.info("No stories for this platform.")
        return
    game = st.selectbox("Game", games, key="cascade_game")
    gsub = sub.loc[sub["Game"] == game]
    story_ids = sorted(
        gsub["story_id"].dropna().unique(), key=lambda s: int(str(s).rsplit("#", 1)[-1])
    )

    def _story_label(sid: str) -> str:
        rows = gsub.loc[gsub["story_id"] == sid]
        return str(rows.iloc[0]["headline"]) or sid

    story_id = st.selectbox("Story", story_ids, format_func=_story_label, key="cascade_story")
    rows = gsub.loc[gsub["story_id"] == story_id]
    objective = st.radio(
        "Start from",
        ["builder", "moon"],
        format_func=lambda o: "Bankroll Builder" if o == "builder" else "Shoot the Moon",
        horizontal=True,
        key="cascade_objective",
    )
    orow = rows.loc[rows["objective"] == objective]
    if orow.empty:
        return
    o = orow.iloc[0]
    st.caption(f"{o['headline']} · {int(o['bet_size'])} legs · EV {o['model_ev'] - 1:+.1%}")
    if st.button("Seed builder", key="cascade_seed", type="primary"):
        seed_from_story(o["legs"], platform, offers)
        st.rerun()


def _apply_game_preselect(games: list[str]) -> None:
    """Pre-select the Tonight "View game" handoff: session key wins, then ``?game=``.

    A stale value (e.g. the sport switch narrowed the slate) is dropped so the
    picker falls back to the first game instead of an empty selection.
    """
    preselect = st.session_state.pop("nav_game", "") or st.query_params.get("game", "")
    if preselect in games:
        st.session_state["game_seed_select"] = preselect
    if st.session_state.get("game_seed_select") not in games:
        st.session_state.pop("game_seed_select", None)


def _render_game_seed(offers: pd.DataFrame) -> None:
    """Pick a game and platform, choose its legs, seed the constellation builder.

    The game label (with a date suffix to keep a doubleheader's two games
    distinct) must match tonight.py's handoff format.
    """
    if not {"Team", "Opponent"} <= set(offers.columns):
        return
    labelled = offers.assign(_game=offers["Team"] + " vs " + offers["Opponent"])
    if "Date" in labelled.columns:
        labelled["_game"] += " · " + labelled["Date"].astype(str)
    games = labelled["_game"].dropna().unique().tolist()
    if not games:
        return

    _apply_game_preselect(games)
    game = st.selectbox("Game", games, key="game_seed_select")
    pool = labelled.loc[labelled["_game"] == game]
    platforms = sorted(pool["Platform"].dropna().unique())
    if not platforms:
        return
    platform = st.selectbox("Platform", platforms, key="game_seed_platform")
    ppool = pool.loc[pool["Platform"] == platform]
    leg_rows = {
        f"{r['Player']} {r['Bet']} {float(r['Line']):.10g} {r['Market']}": r
        for r in ppool.to_dict("records")
        if pd.notna(r.get("Win Prob"))
    }
    picks = st.multiselect("Legs", list(leg_rows), key="game_seed_legs")
    if len(picks) >= 2 and st.button("Seed builder", type="primary", key="game_seed_btn"):
        seed_from_legs([leg_rows[p] for p in picks], platform, "constellation")
        st.rerun()


st.title("Build a slip")
meta = load_current_meta()
render_banner("predictions", f"generated {format_ts(meta.get('generated_at', 'no run on record'))}")

offers = load_current_offers().reset_index(drop=True)
corr = load_current_game_corr()
game_context = load_current_game_context()
ctxs = ctxs_from_frame(game_context, corr)
stories = sport_filtered(load_current_game_stories())

editing = (
    st.session_state.get("edit_slip_id") is not None
    and st.session_state.get("slip_builder") == "constellation"
)
if not editing:
    if stories.empty:
        st.info("No story menu found. Run `poetry run prophecize` to generate stories.")
    else:
        st.subheader("Start from a story")
        _render_story_cascade(stories, offers)
    st.subheader("Start from a game")
    _render_game_seed(sport_filtered(offers))

st.divider()
render_constellation_builder(offers, corr, ctxs, game_context=game_context)
