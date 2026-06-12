"""The slip builders — main-page interactive editors over a session-state slip.

Two render entry points share one slip-state API and one live scorer
(``slip_engine.score_slip``):

* :func:`render_constellation_builder` — same-game, correlation-aware, with a
  live deterministic thesis headline. Hosted on the Slips surface; seeded from a
  story (Bankroll Builder / Shoot the Moon), a Game-tab selection, or a sidebar
  edit. The literal star/edge visual is a later theming step; this is the editor.
* :func:`render_simple_builder` — any-game, grade-only (no thesis); hosted on the
  Board over a cross-game selection.

Both end in **Lock it in!**, which upserts the slip to ``user_slips.parquet``
(status ``pending``) for the sidebar shelf and nightly grading. Money is
``Decimal``; legs are snapshotted from ``current_offers`` at seed/add time so
scoring never re-reads the frame.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.data import load_model_stats
from sportstradamus.dashboard.legs import find_offer_idx, parse_leg
from sportstradamus.dashboard.slip_engine import SlipScore, score_slip, slip_headline
from sportstradamus.helpers.io import upsert_user_slip

_LEGS = "slip_legs"
_PLATFORM = "slip_platform"
_BUILDER = "slip_builder"  # "constellation" | "simple"
_BANKROLL = "slip_bankroll"
_EDIT_ID = "edit_slip_id"

# Leg fields snapshotted from an offer row; the rest of the app reads these.
_SNAPSHOT_COLS = ("Player", "Market", "Bet", "Game", "League", "Platform", "Date")


def init_slip_state() -> None:
    """Seed the plain (non-widget) slip-state keys once per session."""
    ss = st.session_state
    ss.setdefault(_LEGS, [])
    ss.setdefault(_PLATFORM, "Underdog")
    ss.setdefault(_BUILDER, "constellation")
    ss.setdefault(_BANKROLL, 1000.0)
    ss.setdefault(_EDIT_ID, None)


def bankroll_input() -> None:
    """Render the one global bankroll control; mirror it onto the plain slip key.

    Follows the app's widget-key→plain-key convention so the builders read a
    stable non-widget key (``slip_bankroll``) that every Kelly stake scales to.
    """
    value = st.number_input(
        "Bankroll ($)",
        min_value=0.0,
        value=float(st.session_state[_BANKROLL]),
        step=50.0,
        key="_bankroll_widget",
    )
    st.session_state[_BANKROLL] = value


def _leg_from_offer(row: Mapping) -> dict:
    """Snapshot an offer row into a slip leg (numeric fields + canonical Desc).

    The Desc carries the pipeline's ``- {pct}%, {boost}x`` tail so it parses the
    same way parlay legs do (``parse_leg`` for headlines, the grading
    ``LEG_PATTERN`` for outcomes).
    """
    line = float(row["Line"])
    model_p = float(row["Model P"])
    boost = float(row.get("Boost", 1.0) or 1.0)
    leg = {col: row[col] for col in _SNAPSHOT_COLS}
    leg["Line"] = line
    leg["Model P"] = model_p
    leg["Push P"] = float(row.get("Push P", 0.0) or 0.0)
    leg["Boost"] = boost
    leg["Desc"] = (
        f"{row['Player']} {row['Bet']} {line:.10g} {row['Market']} - {model_p * 100:.1f}%, {boost:.2f}x"
    )
    return leg


def _legs_from_descs(descs: Sequence[str], platform: str, offers: pd.DataFrame) -> list[dict]:
    """Resolve stored leg-desc strings against current offers, dropping any that moved."""
    out: list[dict] = []
    for desc in descs:
        parsed = parse_leg(desc)
        idx = find_offer_idx(parsed, offers, platform) if parsed else None
        if idx is not None:
            out.append(_leg_from_offer(offers.loc[idx]))
    return out


def seed_from_story(legs_json: str, platform: str, offers: pd.DataFrame) -> None:
    """Populate the constellation builder from a story objective's leg list."""
    st.session_state[_LEGS] = _legs_from_descs(json.loads(legs_json), platform, offers)
    st.session_state[_PLATFORM] = platform
    st.session_state[_BUILDER] = "constellation"
    st.session_state[_EDIT_ID] = None


def seed_from_legs(rows: Sequence[Mapping], platform: str, builder: str) -> None:
    """Seed a builder from selected offer rows (Game → constellation, Board → simple).

    A slip is single-platform; rows off ``platform`` are dropped.
    """
    st.session_state[_LEGS] = [_leg_from_offer(r) for r in rows if r["Platform"] == platform]
    st.session_state[_PLATFORM] = platform
    st.session_state[_BUILDER] = builder
    st.session_state[_EDIT_ID] = None


def load_slip(slip_row: Mapping, offers: pd.DataFrame) -> None:
    """Reopen a locked slip in its builder for editing (re-lock updates in place)."""
    platform = slip_row["platform"]
    descs = [leg["desc"] for leg in json.loads(slip_row["legs"])]
    st.session_state[_LEGS] = _legs_from_descs(descs, platform, offers)
    st.session_state[_PLATFORM] = platform
    st.session_state[_BUILDER] = slip_row["builder_type"]
    st.session_state[_EDIT_ID] = slip_row["slip_id"]


def add_to_simple_slip(row: Mapping) -> None:
    """Board entry point: append an offer row to the cross-game simple slip.

    Starts a fresh simple slip when one isn't active; ignores a row off the
    slip's platform (a slip is single-platform).
    """
    ss = st.session_state
    if ss[_BUILDER] != "simple" or not ss[_LEGS]:
        ss[_LEGS] = []
        ss[_BUILDER] = "simple"
        ss[_EDIT_ID] = None
        ss[_PLATFORM] = row["Platform"]
    if row["Platform"] != ss[_PLATFORM]:
        return
    ss[_LEGS].append(_leg_from_offer(row))


def remove_leg(i: int) -> None:
    legs = st.session_state[_LEGS]
    if 0 <= i < len(legs):
        legs.pop(i)


def clear_slip() -> None:
    st.session_state[_LEGS] = []
    st.session_state[_EDIT_ID] = None


def lock_in(score: SlipScore, headline: str, shrinkage: float) -> None:
    """Persist the active slip (pending) and reset the builder; the shelf re-reads it."""
    legs = st.session_state[_LEGS]
    games = {leg["Game"] for leg in legs}
    leagues = {leg["League"] for leg in legs}
    row = {
        "slip_id": st.session_state[_EDIT_ID] or str(uuid.uuid4()),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "builder_type": st.session_state[_BUILDER],
        "platform": st.session_state[_PLATFORM],
        "League": leagues.pop() if len(leagues) == 1 else "MULTI",
        "Game": games.pop() if len(games) == 1 else "MULTI",
        "legs": json.dumps(
            [
                {
                    "desc": leg["Desc"],
                    "league": leg["League"],
                    "game": leg["Game"],
                    "date": str(leg["Date"]),
                }
                for leg in legs
            ]
        ),
        "bet_size": int(score.bet_size),
        "play_type": score.play_type,
        "indep_p": score.indep_p,
        "joint_p": score.joint_p,
        "payout_multiplier": score.payout,
        "payout_approximate": score.payout_approximate,
        "model_ev": score.model_ev,
        "bankroll": float(st.session_state[_BANKROLL]),
        "stake": float(score.stake),
        "shrinkage": shrinkage,
        "headline": headline,
        "status": "pending",
    }
    upsert_user_slip(row)
    clear_slip()
    st.rerun()


def _slip_shrinkage(legs: Sequence[Mapping]) -> float:
    """Resolved slip shrinkage = min per-leg ``kelly_shrinkage`` from model_stats (default 1.0)."""
    stats = load_model_stats()
    if stats.empty or "kelly_shrinkage" not in stats.columns:
        return 1.0
    vals = []
    for leg in legs:
        cell = stats.loc[(stats["league"] == leg["League"]) & (stats["market"] == leg["Market"])]
        if not cell.empty and pd.notna(cell.iloc[0]["kelly_shrinkage"]):
            vals.append(float(cell.iloc[0]["kelly_shrinkage"]))
    return min(vals) if vals else 1.0


def render_constellation_builder(
    offers: pd.DataFrame, corr: pd.DataFrame, ctxs: Mapping, *, key_prefix: str = "cb"
) -> None:
    """Same-game correlation-aware editor with a live deterministic thesis headline."""
    legs = st.session_state[_LEGS]
    if not legs:
        st.info("Pick a story above to start a slip, or load one from the rail.")
        return
    _render_leg_list(key_prefix)
    _render_add_leg(offers, key_prefix)
    if len(legs) < 2:
        st.caption("Add at least two legs to price the slip.")
        return
    shrink = _slip_shrinkage(legs)
    score = _score(legs, corr, shrink)
    headline = slip_headline(legs, offers, ctxs)
    if headline:
        st.markdown(f"#### {headline}")
    _render_metrics(score, correlated=True)
    st.caption("Constellation view and pairing-block risk arrive with theming.")
    _render_lock_in(score, headline, shrink, key_prefix)


def render_simple_builder(
    offers: pd.DataFrame, corr: pd.DataFrame, *, key_prefix: str = "sb"
) -> None:
    """Any-game grade-only editor (no thesis); legs come from a Board selection."""
    legs = st.session_state[_LEGS]
    if not legs or st.session_state[_BUILDER] != "simple":
        return
    _render_leg_list(key_prefix)
    if len(legs) < 2:
        st.caption("Select at least two legs to price the slip.")
        return
    shrink = _slip_shrinkage(legs)
    score = _score(legs, corr, shrink)
    _render_metrics(score, correlated=False)
    _render_lock_in(score, "", shrink, key_prefix)


def _score(legs: Sequence[Mapping], corr: pd.DataFrame, shrink: float) -> SlipScore:
    return score_slip(
        legs,
        corr,
        platform=st.session_state[_PLATFORM],
        bankroll=Decimal(str(st.session_state[_BANKROLL])),
        shrinkage=shrink,
    )


def _render_leg_list(key_prefix: str) -> None:
    legs = st.session_state[_LEGS]
    for i, leg in enumerate(legs):
        text_col, rm_col = st.columns([8, 1])
        text_col.write(f"{leg['Desc']}  ·  {leg['League']}")
        if rm_col.button(":material/close:", key=f"{key_prefix}_rm_{i}", help="Remove leg"):
            remove_leg(i)
            st.rerun()


def _render_add_leg(offers: pd.DataFrame, key_prefix: str) -> None:
    """Same-game candidate picker, ranked by edge (story-fit ranking is a later polish)."""
    legs = st.session_state[_LEGS]
    games = {leg["Game"] for leg in legs}
    if len(games) != 1:
        return
    game = games.pop()
    platform = st.session_state[_PLATFORM]
    used = {(leg["Player"], leg["Market"], leg["Bet"]) for leg in legs}
    pool = offers.loc[(offers["Game"] == game) & (offers["Platform"] == platform)]
    cands = [
        r
        for r in pool.to_dict("records")
        if (r["Player"], r["Market"], r["Bet"]) not in used and pd.notna(r.get("Model P"))
    ]
    if not cands:
        return
    cands.sort(key=lambda r: -float(r.get("Model EV", 0.0) or 0.0))
    labels = {f"{c['Player']} {c['Bet']} {float(c['Line']):.10g} {c['Market']}": c for c in cands}
    choice = st.selectbox("Add a leg", ["—", *labels], key=f"{key_prefix}_add")
    if choice != "—" and st.button("Add leg", key=f"{key_prefix}_addbtn"):
        st.session_state[_LEGS].append(_leg_from_offer(labels[choice]))
        st.rerun()


def _render_metrics(score: SlipScore, *, correlated: bool) -> None:
    cols = st.columns(5)
    cols[0].metric("Legs", f"{score.bet_size} · {score.play_type}")
    cols[1].metric("Independent", f"{score.indep_p:.1%}")
    if correlated:
        cols[2].metric(
            "With correlation",
            f"{score.joint_p:.1%}",
            f"{score.joint_p - score.indep_p:+.1%}",
        )
    else:
        cols[2].metric("Joint", f"{score.joint_p:.1%}")
    cols[3].metric("Payout", f"{score.payout:.2f}x")
    cols[4].metric("EV", f"{score.model_ev - 1:+.1%}")
    bankroll = float(st.session_state[_BANKROLL])
    st.caption(f"Kelly stake ${score.stake} of ${bankroll:,.0f}")
    if score.payout_approximate:
        st.caption("Sleeper payout approximate — correlation factor pending.")


def _render_lock_in(score: SlipScore, headline: str, shrink: float, key_prefix: str) -> None:
    lock_col, clear_col = st.columns(2)
    if lock_col.button("Lock it in!", key=f"{key_prefix}_lock", type="primary"):
        lock_in(score, headline, shrink)
    if clear_col.button("Clear", key=f"{key_prefix}_clear"):
        clear_slip()
        st.rerun()
