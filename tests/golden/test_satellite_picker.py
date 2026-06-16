"""Satellite picker — the pure other-game candidate query.

Render paths (the expander, chips, auto-open) are Streamlit-runtime and verified
manually; only the grouping query is unit-pinned here.
"""

import pandas as pd

from sportstradamus.dashboard.components.satellite_picker import (
    _PER_GAME_CAP,
    satellite_groups,
)
from sportstradamus.dashboard.legs import corr_key


def _offer(player, market, bet, line, game, team, platform, k):
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": line,
        "Game": game,
        "Team": team,
        "Platform": platform,
        "Kelly": k,
        "League": "NBA",
        "Date": "2026-06-14",
        "Win Prob": 0.6,
        "Boost": 2.0,
    }


def test_keeps_only_other_game_positive_edge_on_platform():
    offers = pd.DataFrame(
        [
            _offer(
                "A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.30
            ),  # focus game -> out
            _offer("B", "AST", "Over", 6.5, "BOS/MIA", "BOS", "Underdog", 0.20),  # in
            _offer("C", "REB", "Over", 10.5, "BOS/MIA", "MIA", "Underdog", 0.0),  # zero edge -> out
            _offer(
                "D", "PTS", "Over", 30.5, "DEN/LAL", "DEN", "Sleeper", 0.40
            ),  # off platform -> out
        ]
    )
    groups = satellite_groups(offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=set())
    assert [g[0] for g in groups] == ["BOS/MIA"]
    assert [r["Player"] for r in groups[0][1]] == ["B"]


def test_drops_already_slipped_keys():
    offers = pd.DataFrame(
        [
            _offer("B", "AST", "Over", 6.5, "BOS/MIA", "BOS", "Underdog", 0.20),
            _offer("E", "PTS", "Over", 22.5, "BOS/MIA", "MIA", "Underdog", 0.10),
        ]
    )
    exclude = {corr_key({"Player": "B", "Market": "AST", "Bet": "Over"})}
    groups = satellite_groups(
        offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=exclude
    )
    assert [r["Player"] for r in groups[0][1]] == ["E"]


def test_caps_per_game_and_ranks_games_by_best_edge():
    rows = [
        _offer(f"P{i}", "PTS", "Over", 20 + i, "BOS/MIA", "BOS", "Underdog", 0.10 + i * 0.001)
        for i in range(_PER_GAME_CAP + 3)
    ]
    rows.append(_offer("Top", "PTS", "Over", 40.5, "DEN/LAL", "DEN", "Underdog", 0.50))
    groups = satellite_groups(
        pd.DataFrame(rows), focus_game="NYK/SAS", platform="Underdog", exclude_keys=set()
    )
    assert groups[0][0] == "DEN/LAL"  # strongest single leg leads
    bos = dict(groups)["BOS/MIA"]
    assert len(bos) == _PER_GAME_CAP  # capped per game
    assert [r["Kelly"] for r in bos] == sorted(
        (r["Kelly"] for r in bos), reverse=True
    )  # sorted by edge


def test_empty_when_no_other_game_qualifies():
    offers = pd.DataFrame([_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.30)])
    assert (
        satellite_groups(offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=set())
        == []
    )
