"""NHL player names reach the spelling books and DFS offers carry, in every stored artifact.

Moneypuck drops the non-ASCII letters of some names ("Tim Sttzle" for Stützle) from its
per-game and bio files, so the gamelog and the roster held a spelling no offer carries and
the player was never scored. ``name_map`` aliases that spelling to the books'; ``StatsNHL.load``
applies it to rows stored before the alias.
"""

import pandas as pd
import pytest

from sportstradamus.helpers.text import remove_accents
from sportstradamus.stats import StatsNHL, base

# (moneypuck's or the gamelog's spelling, the spelling books and DFS offers carry)
ALIASES = [
    ("Tim Sttzle", "Tim Stutzle"),
    ("Alexis Lafrenire", "Alexis Lafreniere"),
    ("Juraj Slafkovsk", "Juraj Slafkovsky"),
    ("Michael Brandsegg-Nygrd", "Michael Brandsegg-Nygard"),
    ("Olli Mtt", "Olli Maatta"),
    ("Joshua Mahura", "Josh Mahura"),
    ("Po Joseph", "Pierre-Olivier Joseph"),
    ("Charles Alexis Legault", "Charles-Alexis Legault"),
    ("Thomas Novak", "Tommy Novak"),
    ("Dmitriy Simashev", "Dmitri Simashev"),
]
STUTZLE = "8482116"
MCDAVID = "8478402"


@pytest.mark.parametrize(("stored", "books"), ALIASES)
def test_alias_maps_the_stored_spelling_to_the_books_and_is_stable(stored, books):
    assert remove_accents(stored) == books
    assert remove_accents(books) == books


def _load(monkeypatch, gamelog, players):
    monkeypatch.setattr(
        base,
        "read_gamelog",
        lambda league: {"gamelog": gamelog, "teamlog": pd.DataFrame(), "players": players},
    )
    stats = StatsNHL()
    stats.load()
    return stats


def _stored_artifacts():
    gamelog = pd.DataFrame(
        {
            "gameId": [2025020001, 2026020001, 2026020001],
            "playerId": [STUTZLE, STUTZLE, MCDAVID],
            "playerName": ["Tim Stutzle", "Tim Sttzle", "Connor Mcdavid"],
            "shots": [3.0, 4.0, 5.0],
        }
    )
    players = {
        2025: {STUTZLE: {"playerName": "Tim Sttzle", "position": "C", "team": "OTT"}},
        2026: {
            STUTZLE: {"playerName": "Tim Sttzle", "position": "C", "team": "OTT"},
            MCDAVID: {"playerName": "Connor Mcdavid", "position": "C", "team": "EDM"},
        },
    }
    return gamelog, players


def test_load_respells_the_gamelog_and_every_stored_season(monkeypatch):
    gamelog, players = _stored_artifacts()

    stats = _load(monkeypatch, gamelog.copy(), players)

    assert stats.gamelog["playerName"].tolist() == ["Tim Stutzle", "Tim Stutzle", "Connor Mcdavid"]
    pd.testing.assert_frame_equal(
        stats.gamelog.drop(columns="playerName"), gamelog.drop(columns="playerName")
    )
    assert stats.players[2025][STUTZLE]["playerName"] == "Tim Stutzle"
    assert stats.players[2026][STUTZLE]["playerName"] == "Tim Stutzle"
    assert stats.players[2026][MCDAVID] == {
        "playerName": "Connor Mcdavid",
        "position": "C",
        "team": "EDM",
    }


def test_an_offer_meets_the_roster_the_gamelog_and_the_comp_pool_after_load(monkeypatch):
    gamelog, players = _stored_artifacts()

    stats = _load(monkeypatch, gamelog, players)

    assert "Tim Stutzle" in set(stats._roster_player_df(2026)["playerName"])
    assert "Tim Stutzle" in set(stats.gamelog["playerName"])
    _, _, id_to_name = stats.build_comp_profile(playerDict=stats.players[2026])
    assert id_to_name == {STUTZLE: "Tim Stutzle", MCDAVID: "Connor Mcdavid"}


def test_load_without_stored_nhl_data_leaves_the_frames_empty(monkeypatch):
    stats = _load(monkeypatch, pd.DataFrame(), {})

    assert stats.gamelog.empty
    assert stats.players == {}
