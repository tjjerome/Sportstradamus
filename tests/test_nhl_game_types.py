"""The NHL gamelog keeps regular-season and playoff games only.

The schedule also lists preseason games and the 4 Nations Face-Off (game types 01, 19 and
20, characters 5-6 of the id). Their rows put national teams and tryout rosters into every
NHL matrix, so the schedule walk skips them and ``load`` drops the ones already stored.
"""

from datetime import date

import pandas as pd

from sportstradamus.stats import StatsNHL, base, nhl

PRESEASON = 2023010001
REGULAR = 2024020001
PLAYOFF = 2024030111
FOUR_NATIONS = 2024190001
FOUR_NATIONS_FINAL = 2024200001
GAME_IDS = [PRESEASON, REGULAR, PLAYOFF, FOUR_NATIONS, FOUR_NATIONS_FINAL]


def test_load_drops_stored_rows_of_other_game_types(monkeypatch):
    gamelog = pd.DataFrame(
        {"gameId": GAME_IDS, "playerId": ["8478402"] * 5, "playerName": ["Connor Mcdavid"] * 5}
    )
    teamlog = pd.DataFrame({"gameId": GAME_IDS, "team": ["EDM", "EDM", "EDM", "CAN", "CAN"]})
    monkeypatch.setattr(
        base,
        "read_gamelog",
        lambda league: {"gamelog": gamelog, "teamlog": teamlog, "players": {}},
    )
    stats = StatsNHL()

    stats.load()

    assert stats.gamelog["gameId"].tolist() == [REGULAR, PLAYOFF]
    assert stats.teamlog["gameId"].tolist() == [REGULAR, PLAYOFF]


class _OneWeek:
    def get(self, url):
        return {
            "gameWeek": [{"date": "2025-02-12", "games": [{"id": game} for game in GAME_IDS]}],
            "nextStartDate": "2025-02-19",
        }


def test_schedule_walk_collects_regular_season_and_playoff_games(monkeypatch):
    monkeypatch.setattr(nhl, "scraper", _OneWeek())

    ids = StatsNHL()._collect_game_ids(date(2025, 2, 12), date(2025, 2, 12))

    assert ids == [(REGULAR, "2025-02-12"), (PLAYOFF, "2025-02-12")]
