"""The slip-state API — the session-state contract both builders read and write.

Plain (non-widget) ``st.session_state`` keys hold the active slip: its legs,
platform, builder type, bankroll, and the id of the locked slip being edited.
This module owns seeding (from a story, offer rows, or a locked slip), the
add/remove/clear primitives, and the **Lock it in!** persistence to
``user_slips.parquet``. The render layer lives in ``slip_builder``; the rail and
surfaces drive these functions. Money is ``Decimal`` at the scoring boundary;
legs are snapshotted from ``current_offers`` at seed/add time so scoring never
re-reads the frame.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.legs import find_offer_idx, parse_leg
from sportstradamus.dashboard.slip_engine import SlipScore
from sportstradamus.helpers.io import upsert_user_slip

_LEGS = "slip_legs"
_PLATFORM = "slip_platform"
_BUILDER = "slip_builder"  # "constellation" | "simple"
_BANKROLL = "slip_bankroll"
_EDIT_ID = "edit_slip_id"

# Leg fields snapshotted from an offer row; the rest of the app reads these.
# Team feeds the constellation's both-teams validity gate; K its star-size edge.
_SNAPSHOT_COLS = ("Player", "Market", "Bet", "Game", "League", "Platform", "Date", "Team", "Kelly")


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
    """The Desc carries the pipeline's ``- {pct}%, {boost}x`` tail so it parses the
    same way parlay legs do (``parse_leg`` for headlines, the grading
    ``LEG_PATTERN`` for outcomes).
    """
    line = float(row["Line"])
    model_p = float(row["Win Prob"])
    boost = float(row.get("Boost", 1.0) or 1.0)
    leg = {col: row[col] for col in _SNAPSHOT_COLS}
    leg["Line"] = line
    leg["Win Prob"] = model_p
    leg["Push Prob"] = float(row.get("Push Prob", 0.0) or 0.0)
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
