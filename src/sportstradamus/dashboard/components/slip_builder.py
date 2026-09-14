"""The slip builders — main-page interactive editors over a session-state slip.

Two render entry points share the ``slip_state`` API (session-state contract,
seed/add/lock primitives) and one live scorer (``slip_engine.score_slip``):

* :func:`render_constellation_builder` — same-game, correlation-aware, with a
  live deterministic thesis headline. Hosted on the Games surface; seeded from a
  story (Bankroll Builder / Shoot the Moon), a game-seed selection, or a sidebar
  edit. Carries the focus game's Total / Spread / Shape context banner.
* :func:`render_simple_builder` — any-game, grade-only (no thesis); hosted on the
  Board over a cross-game selection.

Both end in **Lock it in!**, which upserts the slip to ``user_slips.parquet``
(status ``pending``) for the sidebar shelf and nightly grading. Money is
``Decimal``; legs are snapshotted from ``current_offers`` at seed/add time so
scoring never re-reads the frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import partial

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.assets import headshot_uris
from sportstradamus.dashboard.components.astrolabe_component import render_astrolabe
from sportstradamus.dashboard.components.constellation import constellation_figure
from sportstradamus.dashboard.components.constellation_component import render_constellation
from sportstradamus.dashboard.components.deep_dive import init_detail_state, show_detail
from sportstradamus.dashboard.components.form_spark import form_sparks, move_sparks
from sportstradamus.dashboard.components.pair_note import render_pair_note
from sportstradamus.dashboard.components.satellite_picker import (
    render_added_legs,
    satellite_groups,
)
from sportstradamus.dashboard.components.slip_state import (
    _BANKROLL,
    _BUILDER,
    _LEGS,
    _PLATFORM,
    clear_slip,
    lock_in,
    remove_leg,
)
from sportstradamus.dashboard.data import load_model_stats
from sportstradamus.dashboard.legs import corr_key, find_offer_idx
from sportstradamus.dashboard.slip_engine import (
    SlipScore,
    astrolabe_payload,
    banned_partners,
    score_slip,
    slip_headline,
)
from sportstradamus.dashboard.viewport import is_mobile
from sportstradamus.leg_schema import build_leg, leg_label
from sportstradamus.prediction.stories.legs import validate_parlay_legs

# The selected-legs list rides in a fixed-height scroll region so adding or removing a leg never
# grows or shrinks the page under the constellation (owner: that reflow undercut the astrolabe
# sweep). Sized for ~5 legs; a longer slip scrolls inside it.
_LEG_PANEL_HEIGHT = 150


def slip_shrinkage(legs: Sequence[Mapping]) -> float:
    """Worst-cell kelly_shrinkage across the slip's legs (1.0 when stats are absent)."""
    stats = load_model_stats()
    if stats.empty or "kelly_shrinkage" not in stats.columns:
        return 1.0
    vals = []
    for leg in legs:
        cell = stats.loc[(stats["league"] == leg["league"]) & (stats["market"] == leg["market"])]
        if not cell.empty and pd.notna(cell.iloc[0]["kelly_shrinkage"]):
            vals.append(float(cell.iloc[0]["kelly_shrinkage"]))
    return min(vals) if vals else 1.0


def _active_lenses(
    offers: pd.DataFrame,
    legs: list[dict],
    pool: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
) -> tuple[pd.DataFrame | None, list[tuple[str, list[dict]]] | None]:
    """The two lens overlays for ``constellation_figure``, read off ``games.py``'s toggles.

    "Look deeper" hands back the same unfiltered ``pool`` frame the map already
    has — the figure keeps whatever rows it doesn't already know, and its tier is
    those plus the model-liked legs its own star cut left behind — so no separate
    query. "Look wider" reruns the existing ``satellite_groups`` query. Either
    lens off resolves ``None``.
    """
    deep_pool = pool if st.session_state.get("lens_deep", False) else None
    if not st.session_state.get("lens_wider", False):
        return deep_pool, None
    exclude = {corr_key(leg) for leg in legs}
    wider_groups = satellite_groups(
        offers, focus_game=focus_game, platform=platform, exclude_keys=exclude
    )
    return deep_pool, wider_groups


def render_constellation_builder(
    offers: pd.DataFrame,
    ctxs: Mapping,
    mods: Mapping,
    *,
    focus_game: str,
    key_prefix: str = "cb",
    shape: dict | None = None,
) -> None:
    """Same-game correlation-aware editor with a live deterministic thesis headline.

    Draws ``focus_game``'s star map whether or not a slip is seeded: the game's
    strongest candidates are clickable stars and the rest of its legs wait behind
    the *deeper* lens, so a slip can be built from scratch by clicking or pre-seeded
    from a story; a star's hover card opens the full offer detail without disturbing
    the slip. Other-game slip legs show as satellites. Two lens toggles
    (``games.py``'s ``lens_deep`` / ``lens_wider`` session-state bools) turn on the
    map's "look deeper" / "look wider" overlays. ``mods`` is the pair-modifier map:
    stars the platform won't pair with the slip wear an orange ×, the note under the
    readout names the refused and repriced pairs, and a refused slip can't be locked.
    The caller draws the game's context banner above this, and deals ``shape`` —
    this game's constellation template for the night — from the whole league
    slate, so it can't depend on which platform is showing.

    Below the map the elements keep one order at every leg count — leg panel, notes,
    astrolabe, then the detail dialog, which inserts a block while open — so Streamlit
    never moves the astrolabe's iframe and remounts it. Below two legs the notes hold
    the prompt caption and the astrolabe rests.
    """
    if not focus_game:
        st.info("Pick a game above to see its constellation.")
        return
    legs = st.session_state[_LEGS]
    platform = st.session_state[_PLATFORM]
    pool = offers.loc[(offers["Game"] == focus_game) & (offers["Platform"] == platform)]
    # Every focus-game leg goes to the map: the figure promotes anything outside its
    # default star cut — a model-passed leg included — so a slip leg is always lit
    # somewhere on the map rather than vanishing from it on add.
    focus_legs = [leg for leg in legs if leg["game"] == focus_game]
    deep_pool, wider_groups = _active_lenses(
        offers, legs, pool, focus_game=focus_game, platform=platform
    )
    _render_constellation(
        offers,
        focus_legs,
        rho=ctxs[focus_game].rho if focus_game in ctxs else {},
        pool=pool,
        key_prefix=key_prefix,
        deep_pool=deep_pool,
        wider_groups=wider_groups,
        shape=shape,
        bans=banned_partners(legs, mods, platform=platform),
    )
    with st.container():
        if legs:
            with st.container(height=_LEG_PANEL_HEIGHT, border=False, key="constellation_legpanel"):
                _render_leg_list(key_prefix, focus_game=focus_game, removable=False, columns=2)
                _render_non_star_legs(legs, focus_game=focus_game, key_prefix=key_prefix)
    valid, reason = validate_parlay_legs(legs)
    notes = st.container()
    if len(legs) < 2:
        notes.caption(
            (reason or "Tap a star to add another leg.")
            if legs
            else "Tap a star to start a slip from this game."
        )
        render_astrolabe({"legs": len(legs)}, key=f"{key_prefix}_astrolabe")
    else:
        if not valid:
            notes.warning(reason)
        headline = slip_headline(focus_legs, offers, ctxs)
        if headline:
            notes.markdown(f"#### {headline}")
        _render_price(
            legs,
            ctxs,
            mods,
            platform=platform,
            valid=valid,
            key_prefix=key_prefix,
            headline=headline,
        )
    _draw_detail_dialog(offers)


def render_simple_builder(
    offers: pd.DataFrame, ctxs: Mapping, mods: Mapping, *, key_prefix: str = "sb"
) -> None:
    """Any-game grade-only editor (no thesis); legs come from a Board selection."""
    legs = st.session_state[_LEGS]
    if not legs or st.session_state[_BUILDER] != "simple":
        return
    _render_leg_list(key_prefix)
    if len(legs) < 2:
        st.caption("Select at least two legs to price the slip.")
        return
    platform = st.session_state[_PLATFORM]
    # A Board slip spans games, so the both-teams rule doesn't apply; a repeated player
    # still blocks the lock, and the pair note leaves saying so to this warning.
    valid, reason = validate_parlay_legs(legs, require_both_teams=False)
    if not valid:
        st.warning(reason)
    _render_price(legs, ctxs, mods, platform=platform, valid=valid, key_prefix=key_prefix)


def _render_price(
    legs: list[dict],
    ctxs: Mapping,
    mods: Mapping,
    *,
    platform: str,
    valid: bool,
    key_prefix: str,
    headline: str = "",
) -> None:
    """Both builders' ending: price the slip, then its readout, pair note and lock.

    ``headline`` only rides into the lock; the constellation builder draws it above, in its
    notes slot. **Lock it in!** stays off unless ``valid`` holds and the platform refuses
    no pair on the slip.
    """
    shrink = slip_shrinkage(legs)
    score = score_slip(
        legs,
        ctxs,
        mods,
        platform=platform,
        bankroll=Decimal(str(st.session_state[_BANKROLL])),
        shrinkage=shrink,
    )
    _render_metrics(score, key_prefix=key_prefix)
    render_pair_note(score, legs, platform)
    _render_lock_in(score, headline, shrink, key_prefix, can_lock=valid and not score.banned)


def _render_leg_list(
    key_prefix: str, *, focus_game: str | None = None, removable: bool = True, columns: int = 1
) -> None:
    """List slip legs. ``focus_game`` shows only that game's legs — every one of them is
    a star on the map, so the satellite picker lists only the other games' legs.
    ``removable=False`` drops the button column because a star leg is removed by clicking
    it on the map, and ``columns`` then flows that read-only list across that many
    side-by-side columns.
    """
    legs = st.session_state[_LEGS]
    cols = st.columns(columns) if not removable and columns > 1 else None
    shown = 0
    for i, leg in enumerate(legs):
        if focus_game is not None and leg["game"] != focus_game:
            continue
        line = f"{leg_label(leg)}  ·  {leg['league']}"
        if not removable:
            (cols[shown % columns] if cols else st).write(line)
            shown += 1
            continue
        text_col, rm_col = st.columns([8, 1])
        text_col.write(line)
        if rm_col.button(":material/close:", key=f"{key_prefix}_rm_{i}", help="Remove leg"):
            remove_leg(i)
            st.rerun()


def _render_non_star_legs(legs: list[dict], *, focus_game: str, key_prefix: str) -> None:
    """List every slip leg that isn't a star on the map — the other games' legs — with a
    remove control.

    A satellite is never drawn on the focus game's map, so this list is its only
    removal path; it's also the only place it stays visible once the *wider* lens that
    revealed it toggles back off and its trace leaves the figure.
    """
    non_star = [(i, leg) for i, leg in enumerate(legs) if leg["game"] != focus_game]
    action = render_added_legs(non_star, key_prefix, caption="Other slip legs", infix="added")
    if action and "remove" in action:
        remove_leg(action["remove"])
        st.rerun()


def _render_constellation(
    offers: pd.DataFrame,
    legs: list[dict],
    *,
    rho: Mapping[frozenset, float],
    pool: pd.DataFrame,
    key_prefix: str,
    deep_pool: pd.DataFrame | None,
    wider_groups: list[tuple[str, list[dict]]] | None,
    shape: dict | None,
    bans: dict[str, str],
) -> None:
    """Draw the interactive star map and apply the star intents and detail opens it sends.

    :func:`_apply_constellation_action` runs first as the map's ``on_change``, which
    Streamlit calls before the next script run draws anything, so a click redraws the page
    once, with the new slip. It runs again right after the map to catch a value whose
    callback was lost, and then reruns the page once more. The map gets the last applied
    ``seq`` back as its ``ack``. ``deep_pool``/``wider_groups`` are the two lens overlays —
    a clicked deep star resolves against the same ``pool`` frame as any other star on the
    map; a sky star from another game resolves against ``wider_groups``. The hover card's
    last-five and line-movement rows are keyed by star, and its headshots by player, since
    a wider dot from another game has no row in ``pool``. ``bans`` crosses each star the
    platform won't pair with the slip and fills its card's "won't pair" row.
    """
    mobile = is_mobile()
    # The wider lens draws other games' legs as sky stars, and their cards open like any
    # other — so they need sparks too, and they need not share the focus game's league.
    wider_rows = [row for _, rows in wider_groups or [] for row in rows]
    sparked = pd.concat([pool, pd.DataFrame(wider_rows)]) if wider_rows else pool
    key = f"{key_prefix}_constellation"
    render_constellation(
        constellation_figure(
            legs,
            rho,
            pool,
            deep_pool=deep_pool,
            wider_groups=wider_groups,
            mobile=mobile,
            shape=shape,
            banned=bans,
        ),
        key=key,
        ack=st.session_state.get(f"{key}_seq", 0),
        sparks=form_sparks(sparked),
        moves=move_sparks(sparked),
        shots=headshot_uris(sparked),
        bans=bans,
        on_change=partial(_apply_constellation_action, key, offers, pool, wider_groups),
        mobile=mobile,
    )
    # A click that interrupts the run before the map registers loses its on_change for the
    # next run, so it is applied here and the page redrawn once.
    if _apply_constellation_action(key, offers, pool, wider_groups):
        st.rerun()


def _draw_detail_dialog(offers: pd.DataFrame) -> None:
    """Draw the offer-detail dialog for whichever offer is on the detail stack.

    Shared by every star on the map, including a wider dot from another game —
    ``_open_offer_detail`` always resolves to a full ``offers``-frame index before
    pushing it, so this reads the same way regardless of which lens revealed the star.
    """
    stack = st.session_state.get("detail_stack")
    if not stack or stack[-1] not in offers.index:
        return
    row = offers.loc[stack[-1]]
    game_pool = offers[
        (offers["Game"] == row["Game"]) & (offers["Platform"] == st.session_state[_PLATFORM])
    ]
    show_detail(row, game_pool)


def _apply_constellation_action(
    component_key: str,
    offers: pd.DataFrame,
    pool: pd.DataFrame,
    wider_groups: list[tuple[str, list[dict]]] | None,
) -> bool:
    """Apply the map's ``{seq, lit, detail}`` value if its ``seq`` is new; return whether it did.

    ``lit`` re-sends every intent the frontend hasn't seen acknowledged, so each intent
    only moves the slip toward its state and a repeat is a no-op: a lit star missing from
    the slip is added from its ``pool`` row (an ordinary or deep star), else its
    ``wider_groups`` row (a sky star from another game); an unlit star in the slip is
    removed. Matches run against the canonical slip (``st.session_state[_LEGS]``) — the
    map is drawn over the focus game's *filtered* leg view, so matching that throwaway
    copy would drop an add and mis-index a remove once a slip has cross-game legs.
    ``detail`` seeds the offer dialog without touching the slip. The applied ``seq`` is
    kept under ``<component_key>_seq``, which the map reads back as its ``ack``.
    """
    value = st.session_state.get(component_key)
    applied_key = f"{component_key}_seq"
    if value is None or value["seq"] <= st.session_state.get(applied_key, 0):
        return False
    st.session_state[applied_key] = value["seq"]
    legs = st.session_state[_LEGS]
    for star, lit in value["lit"].items():
        keys = [corr_key(leg) for leg in legs]
        if (star in keys) == lit:
            continue
        if not lit:
            remove_leg(keys.index(star))
            continue
        match = _pool_match_for_key(pool, star)
        row = match[1] if match else _wider_row_for_key(wider_groups, star)
        if row is not None:
            legs.append(build_leg(row))
    if value["detail"]:
        _open_offer_detail(value["detail"], offers, pool, wider_groups)
    return True


def _open_offer_detail(
    key: str,
    offers: pd.DataFrame,
    pool: pd.DataFrame,
    wider_groups: list[tuple[str, list[dict]]] | None,
) -> None:
    """Reuses the Board's ``deep_dive`` dialog and its ``detail_stack`` navigation; the
    slip lives in session state, so opening detail never clears the parlay. A wider
    dot's row came from ``satellite_groups``'s own ``to_dict("records")`` (no frame
    index attached), so it's re-resolved against the full ``offers`` frame by its
    fields, the same way the old satellite popover's "Full detail" button did.
    """
    match = _pool_match_for_key(pool, key)
    if match is not None:
        init_detail_state()
        st.session_state.detail_stack = [match[0]]
        return
    row = _wider_row_for_key(wider_groups, key)
    if row is None:
        return
    idx = find_offer_idx(row, offers, str(row["Platform"]))
    if idx is not None:
        init_detail_state()
        st.session_state.detail_stack = [idx]


def _wider_row_for_key(wider_groups: list[tuple[str, list[dict]]] | None, key: str) -> dict | None:
    if not wider_groups:
        return None
    for _, rows in wider_groups:
        for row in rows:
            if corr_key(row) == key:
                return row
    return None


def _pool_match_for_key(pool: pd.DataFrame, key: str) -> tuple | None:
    for idx, row in zip(pool.index, pool.to_dict("records"), strict=True):
        if corr_key(row) == key:
            return idx, row
    return None


def _render_metrics(score: SlipScore, *, key_prefix: str) -> None:
    """Draw the astrolabe readout and the Kelly stake under it.

    The astrolabe tweens from its current pose on every render and snaps only on first
    mount, so an unrelated rerun that recomputes the identical ``SlipScore`` is a no-op.
    """
    render_astrolabe(astrolabe_payload(score), key=f"{key_prefix}_astrolabe")
    bankroll = float(st.session_state[_BANKROLL])
    st.caption(f"Kelly stake ${score.stake} of ${bankroll:,.0f}")


def _render_lock_in(
    score: SlipScore, headline: str, shrink: float, key_prefix: str, *, can_lock: bool
) -> None:
    lock_col, clear_col = st.columns(2)
    if lock_col.button(
        "Lock it in!", key=f"{key_prefix}_lock", type="primary", disabled=not can_lock
    ):
        lock_in(score, headline, shrink)
    if clear_col.button("Clear", key=f"{key_prefix}_clear"):
        clear_slip()
        st.rerun()
