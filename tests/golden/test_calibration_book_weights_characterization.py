"""Characterization pins for ``training.calibration.fit_book_weights``.

Golden literals captured from the pre-refactor implementation. The fit
itself is decoupled from SciPy by monkeypatching ``calibration.minimize``
to record the objective/guess/bounds it is handed and return a fixed
``res.x`` — so the assertions pin the function's own branch selection
(per-market result extraction + per-distribution objective construction)
rather than the optimizer's numerics.
"""

import types

import numpy as np
import pandas as pd
import pytest

import sportstradamus.training.calibration as cal
import sportstradamus.training.config as cfg

_LOG_STRINGS = {
    "date": "GAME_DATE",
    "player": "PLAYER_NAME",
    "team": "TEAM_ABBREVIATION",
    "win": "WL",
    "score": "PTS",
}


class _FakeArchive:
    def __init__(self, df):
        self._df = df

    def to_pandas(self, league, market):
        return self._df.copy()


def _player_df(n=12):
    dates = [f"2026-01-{i + 1:02d}" for i in range(n)]
    players = [f"P{i}" for i in range(n)]
    idx = pd.MultiIndex.from_tuples(list(zip(dates, players, strict=True)), names=["date", "player"])
    df = pd.DataFrame(
        {"dk": 0.50 + 0.01 * np.arange(n), "fd": 0.55 + 0.005 * np.arange(n), "Line": 9.5},
        index=idx,
    )
    return df, dates, players


def _player_stat(dates, players, market, results):
    sd = types.SimpleNamespace()
    sd.gamelog = pd.DataFrame({"PLAYER_NAME": players, "GAME_DATE": dates, market: results})
    sd.teamlog = pd.DataFrame()
    sd.log_strings = dict(_LOG_STRINGS)
    return sd


def _team_df(n=12):
    dates = [f"2026-02-{i + 1:02d}" for i in range(n)]
    teams = [f"T{i}" for i in range(n)]
    idx = pd.MultiIndex.from_tuples(list(zip(dates, teams, strict=True)), names=["date", "player"])
    df = pd.DataFrame(
        {"dk": 0.50 + 0.01 * np.arange(n), "fd": 0.55 + 0.005 * np.arange(n), "Line": 0.0},
        index=idx,
    )
    return df, dates, teams


def _team_stat(dates, teams, valcol, vals):
    sd = types.SimpleNamespace()
    sd.gamelog = pd.DataFrame()
    sd.teamlog = pd.DataFrame({"TEAM_ABBREVIATION": teams, "GAME_DATE": dates, valcol: vals})
    sd.log_strings = dict(_LOG_STRINGS)
    return sd


@pytest.fixture
def capture(monkeypatch):
    """Replace ``calibration.minimize`` with a recorder returning fixed res.x."""
    rec = {}

    def fake_minimize(fun, x0, args=(), bounds=None, tol=None, method=None):
        rec["fun"] = fun
        rec["x0"] = np.asarray(x0)
        rec["args"] = args
        rec["bounds"] = bounds
        return types.SimpleNamespace(x=np.array([0.30, 0.70]))

    monkeypatch.setattr(cal, "minimize", fake_minimize)
    return rec


def _set_dist(monkeypatch, dist):
    monkeypatch.setattr(cfg, "load_distribution_config", lambda: {"NBA": {"ZZZ": dist}})


def test_player_gamma_return_and_continuous_objective(capture, monkeypatch):
    _set_dist(monkeypatch, "Gamma")
    df, dates, players = _player_df(12)
    sd = _player_stat(dates, players, "ZZZ", [10.0 + i for i in range(12)])

    out = cal.fit_book_weights("NBA", "ZZZ", sd, _FakeArchive(df), {})

    assert out == {"dk": 0.30, "fd": 0.70}
    assert capture["x0"].tolist() == [0.5, 0.5]
    assert capture["bounds"] == [(0.001, 1), (0.001, 1)]
    obj = float(capture["fun"](capture["x0"], *capture["args"]))
    assert np.isclose(obj, 354.62598643813453)


def test_player_negbin_count_objective(capture, monkeypatch):
    _set_dist(monkeypatch, "NegBin")
    df, dates, players = _player_df(12)
    sd = _player_stat(dates, players, "ZZZ", [10.0 + i for i in range(12)])

    out = cal.fit_book_weights("NBA", "ZZZ", sd, _FakeArchive(df), {})

    assert out == {"dk": 0.30, "fd": 0.70}
    obj = float(capture["fun"](capture["x0"], *capture["args"]))
    assert np.isclose(obj, 38.89894912285565)


def test_moneyline_extraction_and_objective(capture, monkeypatch):
    df, dates, teams = _team_df(12)
    wl = ["W" if i % 2 == 0 else "L" for i in range(12)]
    sd = _team_stat(dates, teams, "WL", wl)

    out = cal.fit_book_weights("NBA", "Moneyline", sd, _FakeArchive(df), {})

    assert out == {"dk": 0.30, "fd": 0.70}
    obj = float(capture["fun"](capture["x0"], *capture["args"]))
    assert np.isclose(obj, 0.7111002428769083)


def test_totals_extraction_continuous_objective(capture, monkeypatch):
    df, dates, teams = _team_df(12)
    sd = _team_stat(dates, teams, "PTS", [200.0 + i for i in range(12)])

    out = cal.fit_book_weights("NBA", "Totals", sd, _FakeArchive(df), {})

    assert out == {"dk": 0.30, "fd": 0.70}
    obj = float(capture["fun"](capture["x0"], *capture["args"]))
    assert np.isclose(obj, 1010.0548156391587)


def test_no_book_columns_returns_empty(capture, monkeypatch):
    _set_dist(monkeypatch, "Gamma")
    df = pd.DataFrame({"Line": [1.0, 2.0]})
    _df2, dates, players = _player_df(12)
    sd = _player_stat(dates, players, "ZZZ", [10.0 + i for i in range(12)])

    assert cal.fit_book_weights("NBA", "ZZZ", sd, _FakeArchive(df), {}) == {}


def test_too_few_samples_returns_empty(capture, monkeypatch):
    _set_dist(monkeypatch, "Gamma")
    df, dates, players = _player_df(5)
    sd = _player_stat(dates, players, "ZZZ", [10.0 + i for i in range(5)])

    assert cal.fit_book_weights("NBA", "ZZZ", sd, _FakeArchive(df), {}) == {}
