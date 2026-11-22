from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from sportstradamus.helpers import remove_accents
from sportstradamus.stats import game_starts
from sportstradamus.stats.base import Stats

LOG_STRINGS = {
    "game": "game",
    "player": "player",
    "date": "date",
    "team": "team",
    "opponent": "opponent",
    "home": "home",
    "usage": "usage",
}


@pytest.fixture(autouse=True)
def no_game_starts(tmp_path, monkeypatch):
    """An empty start store whose schedules list nothing: every row reads at 12:00 UTC."""
    monkeypatch.setattr(game_starts, "LEAGUES_DIR", tmp_path)
    for league in ("NBA", "NFL"):
        monkeypatch.setitem(
            game_starts._SOURCES, league, lambda _league, _missing: pd.Series(dtype=object)
        )


def test_training_matrix_skips_gameday_without_reconstructable_stats():
    game_date = date(2024, 9, 8)
    gamelog = pd.DataFrame(
        {
            "game": ["g1"],
            "player": ["Missing Player"],
            "date": [game_date],
            "team": ["AAA"],
            "opponent": ["BBB"],
            "home": [True],
            "usage": [10],
            "market": [3],
        }
    )
    fake = SimpleNamespace(
        league="NFL",
        gamelog=gamelog,
        usage_stat="usage",
        volume_stats=[],
        log_strings=LOG_STRINGS,
        preflight_training_dependencies=lambda _market: None,
        _market_position_filter=lambda frame, _market: frame,
        _dispatch_volume_stats=lambda _offers, _game_date, _market: None,
        get_stats=lambda _market, _offers, _game_date: pd.DataFrame(),
    )

    rebuilt = Stats.get_training_matrix(fake, "market", cutoff_date=date(2024, 1, 1))

    assert rebuilt.empty


def test_training_matrix_keeps_a_player_whose_gamelog_name_has_an_accent():
    """get_stats keys its rows by the short gamelog's accent-free names, as serving does."""
    game_date = date(2024, 9, 8)
    gamelog = pd.DataFrame(
        {
            "game": ["g1", "g1"],
            "player": ["Nikola Jokić", "Plain Name"],
            "date": [game_date, game_date],
            "team": ["AAA", "BBB"],
            "opponent": ["BBB", "AAA"],
            "home": [True, False],
            "usage": [30, 30],
            "market": [25, 10],
        }
    )
    short_log_names = set(gamelog["player"].map(remove_accents))

    def get_stats(_market, offers, _game_date):
        names = [offer["Player"] for offer in offers if offer["Player"] in short_log_names]
        return pd.DataFrame({"Home": [True] * len(names)}, index=names)

    fake = SimpleNamespace(
        league="NBA",
        gamelog=gamelog,
        usage_stat="usage",
        volume_stats=[],
        log_strings=LOG_STRINGS,
        preflight_training_dependencies=lambda _market: None,
        _market_position_filter=lambda frame, _market: frame,
        _dispatch_volume_stats=lambda _offers, _game_date, _market: None,
        get_stats=get_stats,
        resolve_player_market_odds=lambda stats, *_: pd.DataFrame(
            {"Archived": True}, index=stats.index
        ),
    )

    rebuilt = Stats.get_training_matrix(fake, "market", cutoff_date=date(2024, 1, 1))

    assert rebuilt.set_index("Player")["Result"].to_dict() == {"Nikola Jokic": 25, "Plain Name": 10}


def test_an_append_reads_the_usage_floor_over_the_whole_loaded_window():
    """An unquoted row clears the floor a cold rebuild of the same days would set."""
    built_day = date(2024, 9, 1)
    new_day = date(2024, 9, 8)
    history = pd.DataFrame(
        {
            "game": [f"old{i}" for i in range(10)],
            "player": [f"Starter {i}" for i in range(10)],
            "date": [built_day] * 10,
            "team": ["AAA"] * 10,
            "opponent": ["BBB"] * 10,
            "home": [True] * 10,
            "usage": [50 + i for i in range(10)],
            "market": [5] * 10,
        }
    )
    appended = pd.DataFrame(
        {
            "game": ["new"] * 3,
            "player": ["Rotation", "Starter", "Deep Bench"],
            "date": [new_day] * 3,
            "team": ["AAA", "AAA", "BBB"],
            "opponent": ["BBB", "BBB", "AAA"],
            "home": [True, True, False],
            "usage": [30, 55, 20],
            "market": [3, 6, 1],
        }
    )
    fake = SimpleNamespace(
        league="NBA",
        gamelog=pd.concat([history, appended], ignore_index=True),
        usage_stat="usage",
        volume_stats=[],
        log_strings=LOG_STRINGS,
        preflight_training_dependencies=lambda _market: None,
        _market_position_filter=lambda frame, _market: frame,
        _dispatch_volume_stats=lambda _offers, _game_date, _market: None,
        get_stats=lambda _market, offers, _game_date: pd.DataFrame(
            {"Home": [True] * len(offers)}, index=[offer["Player"] for offer in offers]
        ),
        resolve_player_market_odds=lambda stats, *_: pd.DataFrame(
            {"Archived": False}, index=stats.index
        ),
    )

    appended_rows = Stats.get_training_matrix(fake, "market", cutoff_date=built_day)

    # The new day's own 15th percentile is 23, which keeps "Rotation" (30); the window's
    # thirteen rows put it at 46.
    assert appended_rows["Player"].tolist() == ["Starter"]
