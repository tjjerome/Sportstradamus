"""Games — pick a game, see its constellation, build a slip.

Folds the former Game + Slips surfaces together: a shared platform + game picker
drives the nebula hero (matchup, shape glyph, prophecy subline, Total/Spread/Shape)
and the interactive constellation. A slip is then built by clicking candidate stars
from scratch, or pre-seeded from a story.
"""

import html

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.constellation_slate import GameShape, slate_shapes
from sportstradamus.dashboard.components.glyphs import game_shape_glyph
from sportstradamus.dashboard.components.hero import page_hero
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
    load_current_parlays,
    load_game_ctxs,
    sport_filtered,
)
from sportstradamus.dashboard.narrative import SHAPE_HELP, context_strip, game_headline, home_away
from sportstradamus.dashboard.theme import GOLD, GRAY, SEQUENTIAL_COLORS, team_name

# Hero shape glyph — matches the mockup's 74px .glyphwrap.
_HERO_GLYPH_SIZE = 74
# Nebula wash (DESIGN.md §3): blue radial stop off chartSequentialColors, gold held
# at 7% opacity, both well under the hero-card 12% gold ceiling — literal values
# ported from the mockup's own .hero background (docs/mockups/p8-games.html:35-39).
_HERO_BG = (
    f"radial-gradient(ellipse at 82% -10%, {SEQUENTIAL_COLORS[6]}29, transparent 46%),"
    f"radial-gradient(ellipse at 12% 120%, {GOLD}12, transparent 45%),#1A1D24"
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
    # Consume the ?game= deep link once (Tonight's whole-card links set it): left in the
    # URL it would re-force game_select back to it every rerun, freezing the selectbox.
    if "game" in st.query_params:
        del st.query_params["game"]
    if preselect in labels:
        st.session_state["game_select"] = preselect
    if st.session_state.get("game_select") not in labels:
        st.session_state.pop("game_select", None)


def _hero_stat(label: str, value: str, *, title: str = "", mono: bool = True) -> str:
    """One Cinzel-kicker stat span for the hero's stats row.

    Numeric values (Total, Spread) stay Plex Mono; ``mono=False`` rides the Spectral body serif
    instead, for a word value like the game shape that reads as prose, not a figure.
    """
    attr = f' title="{title}"' if title else ""
    value_font = "'IBM Plex Mono',monospace" if mono else "'Spectral',Georgia,serif"
    return (
        f"<span{attr}><div style=\"font-family:'Cinzel',serif;font-size:9px;"
        f'letter-spacing:.14em;text-transform:uppercase;color:{GRAY}">{label}</div>'
        f'<div style="font-family:{value_font};font-size:16px;'
        f'font-weight:600;margin-top:1px">{html.escape(value)}</div></span>'
    )


def _render_hero(
    game_context: pd.DataFrame,
    parlays: pd.DataFrame,
    stories: pd.DataFrame,
    league: str,
    game: str,
    date: str,
    home: str,
    away: str,
) -> None:
    """Nebula game hero: matchup, shape glyph, prophecy subline, Total/Spread/Shape.

    Blank if the game has no context row (unknown game — same graceful no-op as the
    banner this replaces). The matchup line speaks full team names
    (:func:`theme.team_name`, abbrev fallback); the prophecy subline is the story
    engine's headline for the game (:func:`narrative.game_headline`).
    """
    strip = context_strip(game_context, game=game, date=date)
    if not strip:
        return
    fav, spread = strip["fav_team"], strip["spread"]
    spread_text = f"{fav} -{spread:.1f}" if fav and spread > 0 else "Even"
    headline = game_headline(stories, parlays, game=game, date=date)
    glyph = game_shape_glyph(str(strip["shape"]), size=_HERO_GLYPH_SIZE)
    subline = (
        '<div class="celestial-headline" style="font-size:17px;margin-top:2px">'
        f"{html.escape(headline)}</div>"
        if headline
        else ""
    )
    stats = "".join(
        (
            _hero_stat("Total", f"{strip['game_total']:.1f}"),
            _hero_stat("Spread", spread_text),
            _hero_stat("Shape", str(strip["shape"]).title(), title=SHAPE_HELP, mono=False),
        )
    )
    st.markdown(
        f'<div style="position:relative;overflow:hidden;border:1px solid #2A2E37;'
        f'border-radius:5px;padding:16px 18px;background:{_HERO_BG}">'
        f'<div style="position:absolute;top:8px;right:14px;width:{_HERO_GLYPH_SIZE}px;'
        f'height:{_HERO_GLYPH_SIZE}px;opacity:.9">{glyph}</div>'
        f'<div style="font-size:20px;font-weight:700;letter-spacing:.01em">'
        f"{html.escape(team_name(league, away))} "
        f'<span style="color:{GRAY};font-size:15px;font-weight:400">@</span> '
        f"{html.escape(team_name(league, home))}</div>"
        f"{subline}"
        f'<div style="display:flex;gap:22px;margin-top:12px">{stats}</div></div>',
        unsafe_allow_html=True,
    )


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
    # dek is story-level (identical on both objective rows); pre-dek snapshots lack it.
    dek = str(rows.iloc[0]["dek"]) if "dek" in rows.columns and rows.iloc[0]["dek"] else ""
    if dek:
        st.caption(dek)
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
    st.caption(f"{int(o['bet_size'])} legs · EV {o['model_ev'] - 1:+.1%}")
    if st.button("Seed constellation", key="cascade_seed", type="primary"):
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


def _game_select(games: dict[str, dict], legs: list[dict]) -> tuple[str, str]:
    """The game selectbox; returns the focus ``(game, date)``.

    Stays visible whether or not a slip is seeded. A live slip pins the focus (and the
    selectbox, when listed) to its own game so the map keeps matching the slip; clearing
    the slip frees the picker. Otherwise the Tonight "View game" handoff wins.
    """
    slip_game = legs[0]["game"] if legs else ""
    if not games:
        if not slip_game:
            st.info("No model-liked legs on this platform right now.")
        return slip_game, (str(legs[0]["date"]) if legs else "")
    if slip_game:
        _select_slip_label(games, slip_game)
    else:
        _apply_game_preselect(list(games))
    choice = st.selectbox("Game", list(games), key="game_select")
    if slip_game:
        return slip_game, str(legs[0]["date"])
    return games[choice]["game"], games[choice]["date"]


# Constellation lenses (P8 Task C6): a multi-select segmented control — tight, and gold
# when active via theme's shared `segmented_controlActive` rule. Each option maps to a
# session-state bool (`deep_pool` / `wider_groups`) that render_constellation_builder reads.
_LENS_LABELS = {"lens_deep": "Look deeper", "lens_wider": "Look wider"}


def _render_lens_toggles() -> None:
    """Multi-select lens control over the map — either, both, or neither overlay.

    Replaces the old "Add a leg from another game" / "Add a leg the model doesn't
    like" expanders — both are now modes of the constellation figure itself
    (``constellation_figure``'s ``deep_pool`` / ``wider_groups``). Sets the two
    session-state bools ``render_constellation_builder`` reads.
    """
    selected = st.segmented_control(
        "Lenses",
        list(_LENS_LABELS),
        format_func=_LENS_LABELS.get,
        selection_mode="multi",
        key="lens_seg",
    )
    for state_key in _LENS_LABELS:
        st.session_state[state_key] = state_key in (selected or [])


def _render_tuning_cockpit(shapes: dict[str, GameShape], focus_game: str) -> None:
    """Every reading behind tonight's dealing, so thresholds get set against numbers.

    The classifier's six metrics are what ``constellation_shapes.json``'s ``tuning``
    block is compared to, and the catalog reloads on file mtime — so the loop is:
    read a column here, edit the JSON, rerun the browser. No dashboard restart, and
    no caching on this path: a stale table would be worse than none.
    """
    with st.expander("Constellation tuning", expanded=False):
        table = pd.DataFrame(
            [
                {
                    "Game": ("▸ " if game == focus_game else "") + game,
                    "Class": shape.topology,
                    "Shape": shape.label or "— spring layout —",
                    "Stars": shape.n_supernodes,
                    "cross_share": shape.readings["cross_share"],
                    "top_share": shape.readings["top_share"],
                    "mean_degree": shape.readings["mean_degree"],
                    "diameter_frac": shape.readings["diameter_frac"],
                    "density": shape.readings["density"],
                }
                for game, shape in sorted(shapes.items())
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "Thresholds live in `data/config/constellation_shapes.json` → `tuning`. "
            "Edit and rerun — the catalog reloads on file change, no restart. "
            "`variety_lambda` is the knob for a night that comes up all one class."
        )


meta = load_current_meta()
page_hero("THE CONSTELLATION", "Games", format_ts(meta.get("generated_at", "no run on record")))

slate = load_current_offers()
offers = sport_filtered(slate).reset_index(drop=True)
corr = load_current_game_corr()
game_context = load_current_game_context()
ctxs = load_game_ctxs()
stories = sport_filtered(load_current_game_stories())
parlays = load_current_parlays()

# The picker (platform + game selectboxes side by side, then hero + story preloader full
# width) stays visible whether or not a slip is seeded; a live slip pins it to that slip's
# game (see _game_select).
legs = st.session_state[_LEGS]
shapes: dict[str, GameShape] = {}
if offers.empty:
    st.info("No current predictions. Run `poetry run prophecize` to generate offers.")
    focus_game = ""
else:
    st.session_state[_BUILDER] = "constellation"
    pcol, gcol = st.columns([1, 2])
    with pcol:
        st.session_state[_PLATFORM] = _platform_selector(offers)
    games = _candidate_games(offers, st.session_state[_PLATFORM])
    with gcol:
        focus_game, focus_date = _game_select(games, legs)
    if focus_game:
        group = offers.loc[
            (offers["Game"] == focus_game) & (offers["Date"].astype(str) == focus_date)
        ]
        league = str(group["League"].iloc[0]) if not group.empty else ""
        home, away = home_away(group) if not group.empty else ("", "")
        # Dealt from the whole night — every league, every platform — so a game's shape
        # can't change under the user when they switch Underdog <-> Sleeper or narrow
        # the sport filter, and two games showing at once can't wear the same shape.
        shapes = slate_shapes(slate.loc[slate["Date"].astype(str) == focus_date], corr, focus_date)
        _render_hero(game_context, parlays, stories, league, focus_game, focus_date, home, away)
        _render_story_preloader(stories, st.session_state[_PLATFORM], focus_game, offers)

st.divider()
if focus_game:
    _render_lens_toggles()
render_constellation_builder(
    offers,
    corr,
    ctxs,
    focus_game=focus_game,
    shape=shapes[focus_game].template if focus_game in shapes else None,
)
if shapes:
    _render_tuning_cockpit(shapes, focus_game)

# Reconciler handoff: the Modifiers page defaults to the session rail, so the
# current slip arrives loaded. Two legs is the reconciler's minimum.
if len(st.session_state[_LEGS]) >= 2:
    st.page_link(
        "surfaces/lab_modifiers.py",
        label="Payout incorrect? Report it",
        icon=":material/report:",
    )
