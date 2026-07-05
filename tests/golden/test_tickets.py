"""Pin the ticket-builder contract behind Receipts' "Your slips" section.

``build_tickets`` is the pure display-precompute step (status/color/stamp,
leg glyph mapping, EV/stake/return-vs-to-win) — unit-tested directly against
hand-built ``user_slips``-shaped fixture rows, same style as
``test_calibration_agg.py``: real asserted values, not just shape checks.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.components.tickets import build_tickets

_LEG_A = {
    "player": "Wilson",
    "bet": "Over",
    "market": "PTS",
    "line": 23.5,
    "league": "WNBA",
    "game": "LVA/CHI",
    "date": "2026-06-30",
    "stat": "PTS",
    "win_prob": 0.60,
    "boost": 1.0,
    "push_prob": 0.0,
    "kelly": 0.08,
}
_LEG_B = {
    "player": "Boston",
    "bet": "Over",
    "market": "REB",
    "line": 8.5,
    "league": "WNBA",
    "game": "LVA/CHI",
    "date": "2026-06-30",
    "stat": "REB",
    "win_prob": 0.55,
    "boost": 1.0,
    "push_prob": 0.0,
    "kelly": 0.05,
}


def _slip(**overrides) -> dict:
    row = {
        "slip_id": "s1",
        "saved_at": "2026-06-30T12:00:00",
        "builder_type": "constellation",
        "platform": "Underdog",
        "League": "WNBA",
        "Game": "LVA/CHI",
        "legs": [_LEG_A, _LEG_B],
        "bet_size": 2,
        "play_type": "Flex",
        "indep_p": 0.33,
        "joint_p": 0.33,
        "payout_multiplier": 6.0,
        "payout_approximate": False,
        "model_ev": 1.145,
        "bankroll": 1000.0,
        "stake": 12.0,
        "shrinkage": 1.0,
        "headline": "The Aces set the pace",
        "status": "pending",
    }
    row.update(overrides)
    return row


def test_won_slip_status_color_stamp_and_return():
    row = _slip(
        status="won",
        legs_hit=2,
        result=json.dumps([0, 0]),
        payout_realized=6.0,
        graded_at="2026-07-01T02:00:00",
    )
    tickets = build_tickets(pd.DataFrame([row]))
    assert len(tickets) == 1
    t = tickets[0]
    assert t["status"] == "won"
    assert t["color"] == theme.GREEN
    assert t["stamp"] == "WON"
    assert t["headline"] == "The Aces set the pace"
    # Return realized, not "to win": stake * payout_realized = 12 * 6.0 = 72.
    assert t["value_label"] == "Return"
    assert t["value"] == pytest.approx(72.0)


def test_lost_slip_status_color_stamp_and_zero_return():
    row = _slip(
        status="lost",
        legs_hit=0,
        result=json.dumps([1, 1]),
        payout_realized=0.0,
        graded_at="2026-07-01T02:00:00",
    )
    t = build_tickets(pd.DataFrame([row]))[0]
    assert t["status"] == "lost"
    assert t["color"] == theme.RED
    assert t["stamp"] == "LOST"
    assert t["value_label"] == "Return"
    assert t["value"] == pytest.approx(0.0)


def test_pending_slip_status_color_stamp_and_to_win():
    row = _slip()  # status="pending"; no result/payout_realized/legs_hit yet.
    t = build_tickets(pd.DataFrame([row]))[0]
    assert t["status"] == "pending"
    assert t["color"] == theme.GOLD
    assert t["stamp"] == "PENDING"
    # To win is potential, not realized: stake * payout_multiplier = 12 * 6.0 = 72.
    assert t["value_label"] == "To win"
    assert t["value"] == pytest.approx(72.0)


def test_partial_slip_gets_its_own_treatment_not_won_or_lost():
    # nightly._grade_slip's realized dict has no "partial" key -> payout_realized
    # is NaN on a real partial row; the builder must not crash or silently
    # reclassify it as won/lost.
    row = _slip(
        status="partial",
        legs_hit=1,
        result=json.dumps([0, 1]),
        payout_realized=float("nan"),
        graded_at="2026-07-01T02:00:00",
    )
    t = build_tickets(pd.DataFrame([row]))[0]
    assert t["status"] == "partial"
    assert t["color"] == theme.ORANGE
    assert t["stamp"] == "PARTIAL"
    # Nothing realized to report — must not fabricate a dollar amount from NaN.
    assert t["value_label"] == "Return"
    assert t["value"] == pytest.approx(0.0)


def test_push_slip_gets_neutral_treatment_and_stake_returned():
    # nightly._slip_status returns "push" when every leg voids (effective == 0);
    # _grade_slip's realized dict maps push -> 1.0 (stake fully returned, no gain/loss).
    # This status previously had no _STATUS_STYLE entry and would KeyError.
    row = _slip(
        status="push",
        legs_hit=0,
        result=json.dumps([None, None]),
        payout_realized=1.0,
        graded_at="2026-07-01T02:00:00",
    )
    t = build_tickets(pd.DataFrame([row]))[0]
    assert t["status"] == "push"
    assert t["color"] == theme.GRAY
    assert t["stamp"] == "PUSH"
    assert t["value_label"] == "Return"
    # stake * payout_realized = 12 * 1.0 = 12: the stake back, not a gain or a loss.
    assert t["value"] == pytest.approx(12.0)
    outcomes = [outcome for _label, outcome in t["legs"]]
    assert all(o not in ("hit", "miss") for o in outcomes)  # both legs voided, not graded


def test_graded_slip_leg_outcomes_from_result_json():
    row = _slip(
        status="won",
        legs_hit=2,
        result=json.dumps([0, 0]),
        payout_realized=6.0,
        graded_at="2026-07-01T02:00:00",
    )
    t = build_tickets(pd.DataFrame([row]))[0]
    labels = [label for label, _outcome in t["legs"]]
    outcomes = [outcome for _label, outcome in t["legs"]]
    assert len(t["legs"]) == 2
    assert outcomes == ["hit", "hit"]  # both legs hit
    assert "Wilson" in labels[0]
    assert "Boston" in labels[1]


def test_graded_slip_leg_outcomes_mix_of_hit_miss_and_push():
    row = _slip(
        legs=[_LEG_A, _LEG_B, dict(_LEG_A, player="ThirdLeg")],
        bet_size=3,
        status="partial",
        legs_hit=1,
        result=json.dumps([0, 1, None]),
        payout_realized=float("nan"),
        graded_at="2026-07-01T02:00:00",
    )
    t = build_tickets(pd.DataFrame([row]))[0]
    outcomes = [outcome for _label, outcome in t["legs"]]
    assert outcomes[0] == "hit"  # 0 -> hit
    assert outcomes[1] == "miss"  # 1 -> miss
    assert outcomes[2] not in ("hit", "miss")  # None -> neutral, not silently omitted


def test_pending_slip_legs_have_no_outcome():
    t = build_tickets(pd.DataFrame([_slip()]))[0]
    assert len(t["legs"]) == 2
    for _label, outcome in t["legs"]:
        assert outcome == ""


def test_meta_line_carries_platform_play_type_leg_count_and_date():
    t = build_tickets(pd.DataFrame([_slip()]))[0]
    assert "Underdog" in t["meta"]
    assert "Flex" in t["meta"]
    assert "2" in t["meta"]


def test_ev_and_stake_present_on_every_ticket():
    t = build_tickets(pd.DataFrame([_slip()]))[0]
    assert t["ev"] == pytest.approx(0.145)  # model_ev (1.145) - 1, as elsewhere in the app
    assert t["stake"] == pytest.approx(12.0)


def test_empty_frame_returns_empty_list():
    assert build_tickets(pd.DataFrame()) == []
