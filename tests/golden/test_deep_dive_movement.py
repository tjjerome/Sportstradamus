"""Movement-tab pins: the step chart's encodings and the tab's empty states.

``movement_chart`` draws an offer's logged polls as two steps: the fair line colored by
``spark_svg.move_color`` from the bet's side, the posted main line gray, under the
dialog's white dashed rule at the offer's *own* line (a Sleeper alt rung's, not the main
line the steps follow). Gold is chrome and never plots data (DESIGN §2). Times must reach
Vega as UTC instants: an offset-less ISO string is read as the viewer's local time, which
shifts every poll by their UTC offset.

``render_movement_tab`` runs against a stubbed ``st`` to pin the branches a live check
rarely reaches: a snapshot written before the fair-line columns, and a single poll.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import streamlit as st

from sportstradamus.dashboard.components.deep_dive_movement import (
    movement_chart,
    render_movement_tab,
)
from sportstradamus.dashboard.theme import GOLD, GRAY, GREEN, RED
from sportstradamus.helpers.io import LINE_MOVEMENT_COLS

# A Sleeper main rung that went 4.5 -> 5.5, its price moving first; the offer is the 3.5
# alt rung. The fair line rises 4.5 -> 5.44: away from an Over, toward an Under.
_CHANGES = [
    {"t": "2026-09-06T15:00:00+00:00", "line": 4.5, "p_over": 0.5, "fair": 4.5},
    {"t": "2026-09-06T19:00:00+00:00", "line": 4.5, "p_over": 0.562, "fair": 4.83},
    {"t": "2026-09-07T01:30:00+00:00", "line": 5.5, "p_over": 0.481, "fair": 5.44},
]
_ALT_LINE = 3.5


def _values(spec: dict, layer: dict) -> list[dict]:
    """A layer's rows; Altair hoists every inline frame into the top-level ``datasets``."""
    return spec["datasets"][layer.get("data", spec["data"])["name"]]


def _rule(spec: dict) -> dict:
    [rule] = [lyr for lyr in spec["layer"] if lyr["mark"]["type"] == "rule"]
    return rule


@pytest.mark.parametrize(("bet", "fair_color"), [("Over", RED), ("Under", GREEN)])
def test_fair_step_takes_the_move_color_and_posted_step_is_gray(bet, fair_color):
    spec = movement_chart(_CHANGES, _ALT_LINE, bet).to_dict()
    steps = {
        lyr["encoding"]["y"]["field"]: lyr for lyr in spec["layer"] if lyr["mark"]["type"] == "line"
    }
    assert steps["fair"]["mark"]["color"] == fair_color
    assert steps["line"]["mark"]["color"] == GRAY
    for step in steps.values():
        assert step["mark"]["interpolate"] == "step-after"
        # Off zero, or a 250.5 passing-yards line flattens against the baseline.
        assert step["encoding"]["y"]["scale"] == {"zero": False}


def test_dashed_rule_sits_at_the_offers_own_line():
    spec = movement_chart(_CHANGES, _ALT_LINE, "Over").to_dict()
    rule = _rule(spec)
    assert rule["mark"]["color"] == "#FFFFFF" and rule["mark"]["strokeDash"] == [6, 3]
    assert rule["encoding"]["y"]["field"] == "Line"
    assert _values(spec, rule) == [{"Line": _ALT_LINE}]


@pytest.mark.parametrize("bet", ["Over", "Under"])
def test_no_gold_and_no_color_legend(bet):
    spec = movement_chart(_CHANGES, _ALT_LINE, bet).to_dict()
    assert GOLD.upper() not in json.dumps(spec).upper()
    # Fixed mark colors only: a color encoding would bring a legend the caption replaces.
    assert not any("color" in lyr["encoding"] for lyr in spec["layer"])


def test_points_carry_the_poll_tooltip():
    spec = movement_chart(_CHANGES, _ALT_LINE, "Over").to_dict()
    [points] = [lyr for lyr in spec["layer"] if "tooltip" in lyr["encoding"]]
    tips = {tip["field"]: tip for tip in points["encoding"]["tooltip"]}
    assert set(tips) == {"t", "line", "p_over", "fair"}
    assert tips["t"]["type"] == "temporal"
    assert tips["p_over"]["format"].endswith("%")


def test_times_reach_vega_as_utc_instants():
    spec = movement_chart(_CHANGES, _ALT_LINE, "Over").to_dict()
    [polls] = [rows for rows in spec["datasets"].values() if "t" in rows[0]]
    stamps = [pd.Timestamp(poll["t"]) for poll in polls]
    assert all(stamp.tzinfo is not None for stamp in stamps)
    assert stamps == [pd.Timestamp(change["t"]) for change in _CHANGES]


def _movement_row(changes: list[dict], n_moves: int, n_price_moves: int) -> pd.Series:
    return pd.Series(
        {
            "series": json.dumps([change["line"] for change in changes]),
            "fair_series": json.dumps([change["fair"] for change in changes]),
            "n_moves": n_moves,
            "n_price_moves": n_price_moves,
            "changes": json.dumps(changes),
        }
    ).reindex(LINE_MOVEMENT_COLS)


def _render(monkeypatch, movement: pd.Series | None) -> list[tuple[str, object]]:
    shown: list[tuple[str, object]] = []
    monkeypatch.setattr(st, "caption", lambda body: shown.append(("caption", body)))
    monkeypatch.setattr(st, "markdown", lambda body, **_: shown.append(("markdown", body)))
    monkeypatch.setattr(st, "altair_chart", lambda chart, **_: shown.append(("chart", chart)))
    render_movement_tab(movement, pd.Series({"Line": _ALT_LINE, "Bet": "Over"}))
    return shown


def test_render_without_history_says_so(monkeypatch):
    # The loader reindexes a pre-fair-line snapshot, so its row arrives with changes NaN.
    old_schema = pd.Series({"open_line": 4.5, "series": "[4.5]"}).reindex(LINE_MOVEMENT_COLS)
    for movement in (None, old_schema):
        [(kind, body)] = _render(monkeypatch, movement)
        assert kind == "caption" and "No line history" in body


def test_render_single_poll_shows_summary_without_chart(monkeypatch):
    shown = _render(monkeypatch, _movement_row(_CHANGES[:1], n_moves=0, n_price_moves=0))
    assert [kind for kind, _ in shown] == ["markdown", "caption"]
    assert "Held at 4.5" in shown[0][1] and "one poll" in shown[1][1]


def test_render_draws_summary_then_chart_at_the_rows_line(monkeypatch):
    shown = _render(monkeypatch, _movement_row(_CHANGES, n_moves=1, n_price_moves=1))
    assert [kind for kind, _ in shown] == ["markdown", "chart", "caption"]
    assert "Fair 4.5 → 5.44 (+0.94) · line 4.5 → 5.5" in shown[0][1]
    spec = shown[1][1].to_dict()
    assert _values(spec, _rule(spec)) == [{"Line": _ALT_LINE}]
