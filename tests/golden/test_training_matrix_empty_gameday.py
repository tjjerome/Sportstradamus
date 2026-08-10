from datetime import date
from types import SimpleNamespace

import pandas as pd

from sportstradamus.stats.base import Stats


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
        log_strings={
            "game": "game",
            "player": "player",
            "date": "date",
            "team": "team",
            "opponent": "opponent",
            "home": "home",
            "usage": "usage",
        },
        preflight_training_dependencies=lambda _market: None,
        _market_position_filter=lambda frame, _market: frame,
        _dispatch_volume_stats=lambda _offers, _game_date, _market: None,
        get_stats=lambda _market, _offers, _game_date: pd.DataFrame(),
    )

    rebuilt = Stats.get_training_matrix(fake, "market", cutoff_date=date(2024, 1, 1))

    assert rebuilt.empty
