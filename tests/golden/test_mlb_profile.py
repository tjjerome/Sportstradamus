"""MLB profile and matchup inputs (``stats/mlb_profile.py``) on logs small enough to tally by hand.

A rate pools counts (a ratio of sums, not a mean of per-game rates), a game the pitch feed has
not filled adds to neither side of a pitch rate, and each `` short`` twin reads the player's
last games only. The window itself (games strictly before the row's date) is
``Stats.window_short_logs``'s, shared with every profile column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import mlb_profile
from sportstradamus.stats.mlb_profile import lineup_vs_hand, platoon, player_rates

NAN = np.nan


def _log(rows: list[dict]) -> pd.DataFrame:
    defaults = {"starting pitcher": False, "starting batter": False, "battingOrder": 0}
    return pd.DataFrame([defaults | row for row in rows])


def test_player_rates_pool_counts_and_skip_unfilled_games(monkeypatch):
    monkeypatch.setattr(mlb_profile, "PROFILE_SHORT_GAMES", 2)
    start = {"playerName": "Ace", "starting pitcher": True}
    # The second start predates the pitch columns; the relief outing is no start.
    logs = _log(
        [
            start
            | {
                "pitcher strikeouts": 6,
                "batters faced": 24,
                "pitches thrown": 90,
                "walks allowed": 1,
                "pitcher whiffs": 10,
                "pitcher swings": 40,
                "pitcher times through order": 3,
            },
            start
            | {
                "pitcher strikeouts": 3,
                "batters faced": 20,
                "pitches thrown": 80,
                "pitcher whiffs": NAN,
                "pitcher swings": NAN,
                "pitcher times through order": NAN,
            },
            start
            | {
                "pitcher strikeouts": 9,
                "batters faced": 30,
                "pitches thrown": 100,
                "pitcher whiffs": 14,
                "pitcher swings": 46,
                "pitcher times through order": 2,
            },
            {
                "playerName": "Ace",
                "pitcher strikeouts": 2,
                "batters faced": 5,
                "pitches thrown": 20,
                "pitcher whiffs": 3,
                "pitcher swings": 6,
                "pitcher times through order": 1,
            },
            {
                "playerName": "Bat",
                "battingOrder": 2,
                "starting batter": True,
                "batter batted balls": 2,
                "batter hard-hit balls": 1,
            },
            {
                "playerName": "Bat",
                "battingOrder": 4,
                "starting batter": True,
                "batter batted balls": 3,
                "batter hard-hit balls": 3,
            },
            {
                "playerName": "Bat",
                "battingOrder": 7,
                "batter batted balls": 0,
                "batter hard-hit balls": 0,
            },
        ]
    )

    rates = player_rates(logs, "playerName")

    ace, bat = rates.loc["Ace"], rates.loc["Bat"]
    assert ace["pitcher K rate"] == pytest.approx(18 / 74)
    assert ace["pitcher whiff rate"] == pytest.approx(24 / 86)
    assert ace["pitches per start"] == pytest.approx(90)
    assert ace["times through order"] == pytest.approx(2.5)
    assert ace["pitcher K rate short"] == pytest.approx(12 / 50)
    assert ace["pitcher whiff rate short"] == pytest.approx(14 / 46)
    assert np.isnan(ace["hard-hit rate"]), "a pitcher who never batted has no batter rate"
    assert bat["hard-hit rate"] == pytest.approx(4 / 5)
    assert bat["lineup spot"] == pytest.approx(3), "the spot counts only the games he started"
    assert bat["start rate"] == pytest.approx(2 / 3)
    assert bat["start rate short"] == pytest.approx(1 / 2)
    assert np.isnan(bat["pitcher K rate"])


def test_platoon_flags_and_splits():
    # Lefty bats .400 against righties and .100 against lefties; Switch .300 either way,
    # from the left against righties and the right against lefties.
    history = _log(
        [
            {
                "playerName": name,
                "battingOrder": 1,
                "opponent pitcher": arm,
                "opponent pitcher hand": hand,
                "hits": hits,
                "plateAppearances": 10,
            }
            for name, arm, hand, hits in [
                ("Lefty", "Right Arm", "R", 4),
                ("Lefty", "Left Arm", "L", 1),
                ("Switch", "Right Arm", "R", 3),
                ("Switch", "Left Arm", "L", 3),
            ]
        ]
    )
    players = pd.Index(["Lefty", "Switch", "Nobody", "Late"])
    starters = {
        "Lefty": "Right Arm",
        "Switch": "Left Arm",
        "Nobody": "Right Arm",
        "Late": "Mystery Arm",
    }
    bats = {"Lefty": "L", "Switch": "S", "Late": "L"}
    throws = {"Right Arm": "R", "Left Arm": "L"}

    got = platoon(players, history, "hits", starters, bats, throws, "playerName")

    flags = ["platoon advantage", "facing lefty", "bats left", "switch hitter"]
    expected = pd.DataFrame(
        [[1, 0, 1, 0], [1, 1, 0, 1], [0.5, 0, 0.5, 0], [0.5, 0.5, 1, 0]],
        index=players,
        columns=flags,
        dtype=float,
    )
    pd.testing.assert_frame_equal(got[flags], expected)

    league = 11 / 40
    overall = (5 + 50 * league) / (20 + 50)
    assert got.loc["Lefty", "split gap"] == pytest.approx((4 + 150 * overall) / 160 - overall)
    assert got.loc["Lefty", "PA vs hand"] == 10
    # Right Arm faced left-side bats in 20 plate appearances: Lefty's and Switch's.
    assert got.loc["Lefty", "pitcher split gap"] == pytest.approx((7 + 150 * league) / 170 - league)
    assert got.loc["Lefty", "pitcher PA vs side"] == 20
    # Switch bats right against Left Arm, who saw only him from that side.
    assert got.loc["Switch", "pitcher PA vs side"] == 10
    assert got.loc["Nobody", ["PA vs hand", "pitcher PA vs side"]].tolist() == [0, 0]


def test_lineup_vs_hand_pools_the_opposing_lineup_against_the_hand(monkeypatch):
    monkeypatch.setattr(mlb_profile, "_LINEUP_SHORT_GAMES", 2)
    rows = []
    # BBB's two batters in four games: three against righties, the second unfilled by the
    # pitch feed, then one against a lefty.
    for game, (hand, whiffs) in enumerate([("R", 2), ("R", NAN), ("R", 4), ("L", 9)]):
        for slot in (1, 2):
            rows.append(
                {
                    "team": "BBB",
                    "gameId": game,
                    "battingOrder": slot,
                    "opponent pitcher hand": hand,
                    "plateAppearances": 4,
                    "batter strikeouts": game + 1,
                    "walks": 0,
                    "batter whiffs": whiffs,
                    "batter swings": NAN if np.isnan(whiffs) else 10,
                }
            )
    logs = _log(rows)

    got = lineup_vs_hand(pd.Index(["Ace", "Who"]), logs, {"Ace": "BBB", "Who": "BBB"}, {"Ace": "R"})

    ace = got.loc["Ace"]
    assert ace["lineup K rate vs hand"] == pytest.approx(2 * (1 + 2 + 3) / 24)
    assert ace["lineup whiff rate vs hand"] == pytest.approx(2 * (2 + 4) / 40)
    assert ace["lineup K rate vs hand short"] == pytest.approx(2 * (2 + 3) / 16)
    assert ace["lineup whiff rate vs hand short"] == pytest.approx(2 * 4 / 20)
    assert got.loc["Who"].isna().all(), "a pitcher whose hand is not on file has no lineup"
