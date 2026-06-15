"""Games — pick a game, see its constellation, build a slip.

Folds the former Game + Slips surfaces together: a shared platform + game picker
drives the Total / Spread / Shape banner and the interactive constellation. A slip
is then built by clicking candidate stars from scratch, or pre-seeded from a story.
"""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_builder import render_constellation_builder
from sportstradamus.dashboard.components.slip_state import (
    _BUILDER,
    _LEGS,
    _PLATFORM,
    seed_from_story,
)
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
from sportstradamus.dashboard.narrative import context_strip, home_away
from sportstradamus.prediction.stories.context import ctxs_from_frame

_SHAPE_HELP = (
    "Projected game script: shootout (high total), grind (low total), blowout "
    "(lopsided), or coinflip (tight). It tilts which counting stats run hot."
)


def _candidate_games(offers: pd.DataFrame, platform: str) -> dict[str, dict]:
    """Deduped ``{label: {game, date}}`` for games with a model-liked leg on ``platform``.

    One entry per ``(Game, Date)`` — the canonical key collapses a matchup's two
    per-team orderings — labelled ``"{home} vs {away} · {date}"`` (home first via
    :func:`home_away`), matching tonight.py's View-game handoff format.
    """
    kelly = pd.to_numeric(offers.get("Kelly"), errors="coerce")
    pool = offers[(offers["Platform"] == platform) & (kelly > 0)]
    out: dict[str, dict] = {}
    for _, group in pool.groupby(["Game", "Date"], sort=True):
        home, away = home_away(group)
        date = str(group["Date"].iloc[0])
        out[f"{home} vs {away} · {date}"] = {"game": str(group["Game"].iloc[0]), "date": date}
    return out


def _apply_game_preselect(labels: list[str]) -> None:
    """Pre-select the Tonight "View game" handoff: session key wins, then ``?game=``.

    A stale value (e.g. the sport or platform switch narrowed the slate) is dropped
    so the picker falls back to the first game instead of an empty selection.
    """
    preselect = st.session_state.pop("nav_game", "") or st.query_params.get("game", "")
    if preselect in labels:
        st.session_state["game_select"] = preselect
    if st.session_state.get("game_select") not in labels:
        st.session_state.pop("game_select", None)


def _render_banner(game_context: pd.DataFrame, game: str, date: str) -> None:
    """Total / Spread / Shape strip for the chosen game (blank if context is unknown)."""
    strip = context_strip(game_context, game=game, date=date)
    if not strip:
        return
    fav, spread = strip["fav_team"], strip["spread"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", f"{strip['game_total']:.1f}")
    c2.metric("Spread", f"{fav} -{spread:.1f}" if fav and spread > 0 else "Even")
    c3.metric("Shape", str(strip["shape"]).title(), help=_SHAPE_HELP)


def _render_story_preloader(
    stories: pd.DataFrame, platform: str, game: str, offers: pd.DataFrame
) -> None:
    """Optional: pre-load a story's legs into the constellation for the chosen game."""
    sub = stories.loc[(stories["platform"] == platform) & (stories["Game"] == game)]
    if sub.empty:
        return
    story_ids = sorted(
        sub["story_id"].dropna().unique(), key=lambda s: int(str(s).rsplit("#", 1)[-1])
    )

    story_id = st.selectbox(
        "Preload a story (optional)",
        story_ids,
        format_func=lambda sid: str(sub.loc[sub["story_id"] == sid].iloc[0]["headline"]) or sid,
        key="cascade_story",
    )
    rows = sub.loc[sub["story_id"] == story_id]
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


def _platform_selector(offers: pd.DataFrame) -> str:
    platforms = sorted(offers["Platform"].dropna().unique())
    if not platforms:
        return st.session_state[_PLATFORM]
    index = platforms.index("Underdog") if "Underdog" in platforms else 0
    return st.selectbox("Platform", platforms, index=index, key="game_platform")


def _select_slip_label(games: dict[str, dict], slip_game: str) -> None:
    """Point the game selectbox at the active slip's game when it's a listed candidate."""
    for label, info in games.items():
        if info["game"] == slip_game:
            st.session_state["game_select"] = label
            return


def _game_picker(
    games: dict[str, dict],
    game_context: pd.DataFrame,
    stories: pd.DataFrame,
    offers: pd.DataFrame,
    legs: list[dict],
) -> str:
    """Render the game picker + context banner + story preloader; return the focus game.

    The picker stays visible whether or not a slip is seeded. A live slip pins the focus
    (and the selectbox, when listed) to its own game so the map keeps matching the slip;
    clearing the slip frees the picker. Otherwise the Tonight "View game" handoff wins.
    """
    slip_game = legs[0]["Game"] if legs else ""
    if not games:
        if not slip_game:
            st.info("No model-liked legs on this platform right now.")
        return slip_game
    if slip_game:
        _select_slip_label(games, slip_game)
    else:
        _apply_game_preselect(list(games))
    choice = st.selectbox("Game", list(games), key="game_select")
    if slip_game:
        focus_game, date = slip_game, str(legs[0]["Date"])
    else:
        focus_game, date = games[choice]["game"], games[choice]["date"]
    _render_banner(game_context, focus_game, date)
    _render_story_preloader(stories, st.session_state[_PLATFORM], focus_game, offers)
    return focus_game


st.title("Games")
meta = load_current_meta()
render_banner("predictions", f"generated {format_ts(meta.get('generated_at', 'no run on record'))}")

offers = sport_filtered(load_current_offers()).reset_index(drop=True)
corr = load_current_game_corr()
game_context = load_current_game_context()
ctxs = ctxs_from_frame(game_context, corr)
stories = sport_filtered(load_current_game_stories())

# The picker (platform + game + banner + story preloader) stays visible whether or not
# a slip is seeded; a live slip pins it to that slip's game (see _game_picker).
legs = st.session_state[_LEGS]
if offers.empty:
    st.info("No current predictions. Run `poetry run prophecize` to generate offers.")
    focus_game = ""
else:
    st.session_state[_PLATFORM] = _platform_selector(offers)
    st.session_state[_BUILDER] = "constellation"
    games = _candidate_games(offers, st.session_state[_PLATFORM])
    focus_game = _game_picker(games, game_context, stories, offers, legs)

st.divider()
render_constellation_builder(offers, corr, ctxs, focus_game=focus_game)
