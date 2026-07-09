"""Pin the per-league training-window row counts and their wiring through
``Stats.trim_gamelog`` — the matrix-depth lever for the MLB/NHL activation."""

import pandas as pd

from sportstradamus.stats import base


class _Stub:
    log_strings = {"usage": "usage", "date": "gameDate"}

    def __init__(self, league, gamelog):
        self.league = league
        self.gamelog = gamelog


def _gamelog(days=300):
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({"gameDate": dates.strftime("%Y-%m-%d"), "usage": 1.0})


def _usable_dates(gamelog):
    dates = pd.to_datetime(gamelog["gameDate"]).dt.date
    cutoff = dates.iloc[0] + pd.Timedelta(days=200)
    return [d for d in dates if d > cutoff]


def test_production_window_values():
    assert base._TRAINING_WINDOW_ROWS == {"NFL": 50_000, "MLB": 95_000, "NHL": 110_000}
    assert base._TRAINING_WINDOW_ROWS_DEFAULT == 21_500


def test_league_window_drives_training_start(monkeypatch):
    monkeypatch.setattr(base, "_TRAINING_WINDOW_ROWS", {"MLB": 30})
    monkeypatch.setattr(base, "_TRAINING_WINDOW_ROWS_DEFAULT", 10)
    log = _gamelog()
    usable = _usable_dates(log)

    mlb_start = base.Stats.trim_gamelog(_Stub("MLB", log.copy()))
    nba_start = base.Stats.trim_gamelog(_Stub("NBA", log.copy()))

    assert mlb_start == usable[-30]
    assert nba_start == usable[-10]
    assert mlb_start < nba_start


def test_window_self_clamps_to_usable_rows(monkeypatch):
    monkeypatch.setattr(base, "_TRAINING_WINDOW_ROWS", {"MLB": 1_000_000})
    log = _gamelog()
    usable = _usable_dates(log)

    assert base.Stats.trim_gamelog(_Stub("MLB", log)) == usable[0]
