"""Satellite picker — the pure other-game candidate query.

``render_added_legs`` (the shared non-star leg list) is Streamlit-runtime and
verified manually; only the grouping query is unit-pinned here.
"""

import pandas as pd

from sportstradamus.dashboard.components.satellite_picker import (
    _PER_GAME_CAP,
    satellite_groups,
)
from sportstradamus.dashboard.legs import corr_key


def _offer(player, market, bet, line, game, team, platform, ev, pick=True):
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": line,
        "Game": game,
        "Team": team,
        "Platform": platform,
        # Kelly runs against EV, so ranking on the model's Kelly reverses every order.
        "Kelly": 1.0 - ev,
        "EV": ev,
        "Pick": pick,
        "League": "NBA",
        "Date": "2026-06-14",
        "Win Prob": 0.6,
        "Boost": 2.0,
    }


def test_keeps_only_other_games_picks_on_platform():
    offers = pd.DataFrame(
        [
            _offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.95),  # focus game
            _offer("B", "AST", "Over", 6.5, "BOS/MIA", "BOS", "Underdog", 0.90),  # in
            # Not a pick, though the model's Kelly likes it -> out.
            _offer("C", "REB", "Over", 10.5, "BOS/MIA", "MIA", "Underdog", 0.70, pick=False),
            _offer("D", "PTS", "Over", 30.5, "DEN/LAL", "DEN", "Sleeper", 0.97),  # off platform
        ]
    )
    groups = satellite_groups(offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=set())
    assert [g[0] for g in groups] == ["BOS/MIA"]
    assert [r["Player"] for r in groups[0][1]] == ["B"]


def test_drops_already_slipped_keys():
    offers = pd.DataFrame(
        [
            _offer("B", "AST", "Over", 6.5, "BOS/MIA", "BOS", "Underdog", 0.90),
            _offer("E", "PTS", "Over", 22.5, "BOS/MIA", "MIA", "Underdog", 0.85),
        ]
    )
    exclude = {corr_key({"Player": "B", "Market": "AST", "Bet": "Over"})}
    groups = satellite_groups(
        offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=exclude
    )
    assert [r["Player"] for r in groups[0][1]] == ["E"]


def test_caps_per_game_and_ranks_games_by_best_ev():
    rows = [
        _offer(f"P{i}", "PTS", "Over", 20 + i, "BOS/MIA", "BOS", "Underdog", 0.80 + i * 0.001)
        for i in range(_PER_GAME_CAP + 3)
    ]
    rows.append(_offer("Top", "PTS", "Over", 40.5, "DEN/LAL", "DEN", "Underdog", 0.95))
    groups = satellite_groups(
        pd.DataFrame(rows), focus_game="NYK/SAS", platform="Underdog", exclude_keys=set()
    )
    assert groups[0][0] == "DEN/LAL"  # the highest-EV pick leads
    bos = dict(groups)["BOS/MIA"]
    assert len(bos) == _PER_GAME_CAP  # capped per game
    assert [r["Player"] for r in bos] == [f"P{i}" for i in range(_PER_GAME_CAP + 2, 2, -1)]


def test_empty_when_no_other_game_qualifies():
    offers = pd.DataFrame([_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.95)])
    assert (
        satellite_groups(offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=set())
        == []
    )
