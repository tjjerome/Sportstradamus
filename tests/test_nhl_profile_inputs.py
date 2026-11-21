"""An NHL skater's profile reads shot attempts, expected goals and the report columns.

Each enters ``stat_types["skater"]`` per game and per 60 minutes on ice (the share of his
team's power-play time per game only), so it reaches ``get_stat_columns`` through
``base_profile``; the per-60 rates are derived on the profile window, one definition for a
training row and a served one. A gamelog the stats service has not filled yet yields the same
columns, read as 0.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from sportstradamus.stats import StatsNHL

REPORTS = {
    "faceOffsTaken": [10.0, 5.0],
    "powerPlayTimeOnIce": [2.0, 1.0],
    "powerPlayShots": [1.0, 0.0],
    "powerPlayShare": [0.5, 0.25],
    "shortHandedTimeOnIce": [1.0, 2.0],
}


def _stats(reports: dict) -> StatsNHL:
    stats = StatsNHL()
    days = [str(date.today() - timedelta(days=back)) for back in (3, 2)]
    stats.gamelog = pd.DataFrame(
        {
            "gameId": [2025020001, 2025020002],
            "gameDate": days,
            "playerId": ["8470001"] * 2,
            "playerName": ["Skater"] * 2,
            "team": ["TOR"] * 2,
            "opponent": ["BOS"] * 2,
            "position": ["C"] * 2,
            "timeOnIce": [20.0, 10.0],
            "shotAttempts": [4.0, 2.0],
            "xGoals": [0.5, 0.25],
        }
        | reports
    )
    stats.teamlog = pd.DataFrame(
        {"gameDate": days, "team": ["TOR"] * 2} | dict.fromkeys(stats.team_stat_types, 0.0)
    )
    return stats


def test_a_skater_profile_reads_each_input_per_game_and_per_60():
    stats = _stats(REPORTS)

    columns = stats.get_stat_columns("shots")

    profile = stats.playerProfile.loc["Skater"]
    assert profile[["shotAttempts", "shotAttempts60", "xGoals60"]].tolist() == [3.0, 12.0, 1.5]
    assert profile[["powerPlayTimeOnIce", "powerPlayTimeOnIce60"]].tolist() == [1.5, 6.0]
    assert profile[["faceOffsTaken60", "shortHandedTimeOnIce60"]].tolist() == [30.0, 7.5]
    assert profile[["powerPlayShots60", "powerPlayShare"]].tolist() == [1.5, 0.375]
    assert {"Player powerPlayShare short", "Player faceOffsTaken60 growth"} <= set(columns)


@pytest.mark.parametrize("filled", [REPORTS, {}], ids=["filled", "unfilled"])
def test_an_unfilled_gamelog_yields_the_same_columns(filled):
    stats = _stats(filled)

    columns = stats.get_stat_columns("shots")

    assert columns == _stats(REPORTS).get_stat_columns("shots")
    assert stats.playerProfile.loc["Skater", "powerPlayTimeOnIce60"] == (6.0 if filled else 0)
