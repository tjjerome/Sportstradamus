"""A WNBA team the league site codes differently from us still gets its roster.

stats.wnba.com names Portland ``PDX`` and ours is ``POR``. ``_build_wnba_player_profiles``
inner-merges the player frame, already on our codes, with the shot-zone shares on player
and team. While the shot rows kept the site's code every Portland player dropped out of
that merge: no roster entry, and no stored game row either, because
``_assemble_wnba_player_rows`` takes each row's position from the same frame.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.stats import StatsWNBA

# A stats.wnba.com shot-locations row is six id columns, then made / attempted / percentage
# for each of eight zones.
_SHOT_ZONE_COLUMNS = [1, 2, 0.5] * 8


def _site_rows(player_id, name, site_team):
    """One player's bios record and shot-locations row, under the league site's team code."""
    bios = {
        "PLAYER_ID": player_id,
        "PLAYER_NAME": name,
        "TEAM_ABBREVIATION": site_team,
        "AGE": 27.0,
        "PLAYER_HEIGHT_INCHES": 70,
        "PLAYER_WEIGHT": "150",
        "USG_PCT": 0.2,
        "TS_PCT": 0.55,
    }
    return bios, [player_id, name, 0, site_team, 27.0, name, *_SHOT_ZONE_COLUMNS]


def test_a_portland_player_keeps_her_roster_under_our_team_code():
    stats = StatsWNBA()
    player_index = pd.DataFrame(
        {"PLAYER_ID": [1, 2], "TEAM_ABBREVIATION": ["POR", "SEA"], "POS": ["G", "F"]}
    )
    portland_bios, portland_shots = _site_rows(1, "Portland Guard", "PDX")
    seattle_bios, seattle_shots = _site_rows(2, "Seattle Forward", "SEA")

    frame = stats._build_wnba_player_profiles(
        player_index, [portland_bios, seattle_bios], {"rowSet": [portland_shots, seattle_shots]}
    )

    assert frame.index.tolist() == ["Portland Guard", "Seattle Forward"]
    assert {team: list(roster) for team, roster in stats.players[stats.season].items()} == {
        "POR": ["Portland Guard"],
        "SEA": ["Seattle Forward"],
    }
