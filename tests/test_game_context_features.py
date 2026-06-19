"""Derivation pins for the QW-1 game-script context features (model track §6.3).

``Stats._game_context`` derives ``Spread``/``GameTotal``/``Blowout`` from the
team's own implied ``Total`` and the opponent's (``OppTotal``). The train/live
parity gate (``tests/golden/test_train_live_feature_parity.py``) proves the two
branches *agree*; it cannot catch a derivation that is wrong *identically* on both
sides (a flipped Spread sign, a both-pool-the-wrong-total bug). This test pins the
arithmetic and both sides of the blowout threshold on known inputs.

The leakage concern for these as-of features is covered by the parity gate
(archive-baked-at-train == archive-read-at-serve), not the strict ``<``-date
filter that ``test_meanyr_mean10_leakage.py`` guards — game-script context is read
from the target gameday's pre-game line, which is correct, not leaked.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from sportstradamus.stats import StatsNBA
from sportstradamus.stats.base import _BLOWOUT_SPREAD_THRESHOLD

# Two past-date games: a 20-point implied blowout (AAA/BBB) and a 4-point coin-flip
# (CCC/DDD), so the threshold (NBA 10.0) is exercised on both sides in one frame.
GAMEDAY = date(2025, 1, 1)
TEAM_TOTAL = {"AAA": 120.0, "BBB": 100.0, "CCC": 108.0, "DDD": 104.0}
PLAYER_TEAM = {"P1": "AAA", "P2": "BBB", "P3": "CCC", "P4": "DDD"}
OPPONENT = {"AAA": "BBB", "BBB": "AAA", "CCC": "DDD", "DDD": "CCC"}


def _game_context_historical() -> pd.DataFrame:
    stats = StatsNBA()
    ls = stats.log_strings
    stats.gamelog = pd.DataFrame(
        [
            {
                ls["date"]: GAMEDAY.isoformat(),
                ls["player"]: player,
                ls["team"]: team,
                ls["opponent"]: OPPONENT[team],
                ls["home"]: True,
                "moneyline": -110.0,
                "totals": TEAM_TOTAL[team],
            }
            for player, team in PLAYER_TEAM.items()
        ]
    )
    frame = pd.DataFrame(index=list(PLAYER_TEAM))
    teams = dict(PLAYER_TEAM)
    opponents = {p: OPPONENT[t] for p, t in PLAYER_TEAM.items()}
    out, *_ = stats._game_context(frame, [], "PTS", GAMEDAY, teams, opponents)
    return out


def test_spread_gametotal_are_exact_combinations_of_the_two_totals():
    out = _game_context_historical()
    assert out.loc["P1", "OppTotal"] == 100.0  # AAA's opponent is BBB
    assert out.loc["P2", "OppTotal"] == 120.0  # BBB's opponent is AAA
    for player in PLAYER_TEAM:
        total, opp = out.loc[player, "Total"], out.loc[player, "OppTotal"]
        assert out.loc[player, "Spread"] == total - opp
        assert out.loc[player, "GameTotal"] == total + opp


def test_blowout_flag_matches_threshold_on_both_sides():
    out = _game_context_historical()
    thr = _BLOWOUT_SPREAD_THRESHOLD["NBA"]
    # AAA/BBB implied spread is +-20 (> 10) -> blowout; CCC/DDD is +-4 (< 10) -> not.
    assert out.loc["P1", "Blowout"] == 1
    assert out.loc["P2", "Blowout"] == 1
    assert out.loc["P3", "Blowout"] == 0
    assert out.loc["P4", "Blowout"] == 0
    for player in PLAYER_TEAM:
        assert out.loc[player, "Blowout"] == int(abs(out.loc[player, "Spread"]) > thr)
