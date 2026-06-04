"""Characterization of :meth:`Stats.get_stats` — the per-player feature-vector builder.

``get_stats`` sits on the prediction critical path (callers: ``get_training_matrix``,
``model_prob``, ``scoring``) yet had no behavioral test before this file — the only
other test that names it *stubs* it. It is decomposed from a single 343-line CC-34
method into a flat orchestrator plus phase helpers; these pins exist to prove that
decomposition is behavior-preserving.

Strategy mirrors the heavy-data integration gates (``test_determinism_gate.py``):
load the **real** league gamelog bundle and skip cleanly when it is absent (the
parquet artifacts are gitignored, so they are present locally but not in CI). To
keep the pinned digests stable regardless of gamelog appends:

* Targets are **immutable past dates** — ``base_profile`` builds ``short_gamelog``
  from the ``[date - 300d, date)`` window, which never changes once ``date`` is in
  the past, so newly-scraped games cannot perturb the pin.
* The precondition is :meth:`Stats.get_depth` (not ``get_volume_stats``); both set
  the ``playerProfile`` depth/position/team that ``get_stats`` reads, but
  ``get_depth`` does not join volume-model output, so the digest is independent of
  the (gitignored, retrainable) ``models/`` pickles.
* The upcoming-branch pin freezes ``base.datetime.today()`` so the
  ``date < today`` branch decision is stable across calendar days, supplies a
  synthetic ``upcoming_games``, and monkeypatches ``base.archive`` with a
  hashlib-derived stub (builtin ``hash`` is per-process salted) so the
  archive-sourced ``Moneyline`` / ``Total`` columns are reproducible.

The MLB pin runs the comp-z loop end-to-end (MLB game-context / pitcher-h2h /
pitcherProfile branches plus the per-player comp z-scoring). It previously
``pytest.raises``-pinned a stale ``scores.index = scores.index.droplevel(0)`` line
that crashed the loop — the vectorized ``_zscore_within_player`` returns a
single-level index, so ``droplevel(0)`` raised. That line was deleted (and the comp
frames copied to drop a ``SettingWithCopyWarning``), so the pin now asserts the
produced feature frame.
"""

from __future__ import annotations

import hashlib
from datetime import date as _date
from datetime import datetime as _datetime

import pandas as pd
import pytest

from sportstradamus.stats import StatsMLB, StatsNBA, base

# Low 63 bits of the summed row-hash — a compact, order-sensitive frame digest.
_DIGEST_MASK = (1 << 63) - 1


def _digest(frame: pd.DataFrame) -> int:
    numeric = frame.select_dtypes("number").round(6)
    return int(pd.util.hash_pandas_object(numeric, index=True).sum()) & _DIGEST_MASK


def _col_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256("|".join(frame.columns).encode()).hexdigest()[:16]


def _slate(stats, slate_date: _date, date_str: str) -> list[dict]:
    """Build a get_stats ``offers`` list from one gameday's real gamelog rows."""
    ls = stats.log_strings
    gamelog = stats.gamelog
    day = pd.to_datetime(gamelog[ls["date"]]).dt.date
    rows = gamelog.loc[day == slate_date].drop_duplicates(subset=[ls["player"]])
    return [
        {
            "Player": r[ls["player"]],
            "Team": r[ls["team"]],
            "Opponent": r[ls["opponent"]],
            "Home": bool(r[ls["home"]]),
            "Date": date_str,
        }
        for _, r in rows.iterrows()
    ]


def _load_or_skip(stats_cls, slate_date: _date):
    """Return a loaded league Stats instance, or skip if its real data is absent."""
    stats = stats_cls()
    stats.load()
    if stats.gamelog.empty:
        pytest.skip(f"{stats_cls.__name__} gamelog bundle not present in this environment")
    day = pd.to_datetime(stats.gamelog[stats.log_strings["date"]]).dt.date
    if not (day == slate_date).any():
        pytest.skip(f"{stats_cls.__name__} gamelog does not cover {slate_date}")
    return stats


def _assert_spot(frame: pd.DataFrame, expected: dict[str, dict[str, object]]) -> None:
    for player, cols in expected.items():
        for col, want in cols.items():
            got = frame.loc[player, col]
            if isinstance(want, bool):
                assert bool(got) == want, f"{player!r}[{col!r}] = {got!r}, want {want!r}"
            else:
                assert round(float(got), 6) == want, (
                    f"{player!r}[{col!r}] = {float(got):.6f}, want {want}"
                )


_FROZEN_TODAY = _datetime(2025, 5, 10, 12, 0, 0)


class _FrozenDatetime(_datetime):
    """``datetime`` whose ``today()`` is pinned, so the upcoming-branch decision
    (``date < datetime.today().date()``) is stable across calendar days."""

    @classmethod
    def today(cls):
        return _FROZEN_TODAY


class _FakeArchive:
    """Deterministic, process-stable stand-in for the upcoming-branch archive reads."""

    @staticmethod
    def _h(key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_moneyline(self, league, date, team):
        return round((self._h(f"ml|{league}|{date}|{team}") % 1000) / 1000.0, 4)

    def get_total(self, league, date, team):
        return round(200.0 + (self._h(f"tot|{league}|{team}") % 5000) / 100.0, 2)


# --- NBA historical (date < today: reads gamelog for Home/Moneyline/Total) ----

_NBA_HIST_DATE = _date(2025, 3, 10)
_NBA_HIST_SHAPE = (252, 278)
_NBA_HIST_COL_HASH = "97162805ac120147"
_NBA_HIST_DIGEST = 2820200463821097376
_NBA_HIST_SPOT = {
    "Aaron Holiday": {
        "Avg1": 20.0, "Avg10": 6.5, "MeanYr": 5.170213, "STDYr": 5.345942,
        "DaysOff": 2.0, "GamesPlayed": 47.0, "Home": True, "Moneyline": 0.614821,
        "Total": 108.444714, "AvgH2H": 0.0, "H2HPlayed": 0.0, "Player comps z": -0.643268,
        "Player z": -0.560515, "Defense avg": -0.064504, "Player depth": 1.0,
    },
    "Aaron Nesmith": {
        "Avg1": 14.0, "Avg10": 15.0, "MeanYr": 10.558824, "STDYr": 5.950254,
        "DaysOff": 2.0, "GamesPlayed": 34.0, "Home": False, "Moneyline": 0.668586,
        "Total": 123.767555, "AvgH2H": 27.0, "H2HPlayed": 1.0, "Player comps z": 0.139783,
        "Player z": 0.244317, "Defense avg": 0.061874, "Player depth": 1.0,
    },
}


def test_nba_historical_get_stats():
    stats = _load_or_skip(StatsNBA, _NBA_HIST_DATE)
    offers = _slate(stats, _NBA_HIST_DATE, _NBA_HIST_DATE.strftime("%Y-%m-%d"))
    stats.get_depth(list(offers), _NBA_HIST_DATE)

    out = stats.get_stats("PTS", list(offers), _NBA_HIST_DATE)

    assert out.shape == _NBA_HIST_SHAPE
    assert _col_hash(out) == _NBA_HIST_COL_HASH
    assert str(out["Home"].dtype) == "bool"
    assert _digest(out) == _NBA_HIST_DIGEST
    _assert_spot(out, _NBA_HIST_SPOT)


# --- NBA upcoming (date >= frozen today: reads upcoming_games + archive) -------

_NBA_UP_DATE = _date(2025, 5, 10)  # == frozen today, so `date < today` is False
_NBA_UP_SLATE_DATE = _date(2025, 3, 10)
_NBA_UP_SHAPE = (253, 278)
_NBA_UP_COL_HASH = "97162805ac120147"
_NBA_UP_DIGEST = 3773554675397057556
_NBA_UP_SPOT = {
    "Aaron Holiday": {
        "Avg1": 3.0, "Avg10": 6.0, "MeanYr": 5.5, "DaysOff": 8.0, "GamesPlayed": 64.0,
        "Home": True, "Moneyline": 0.526, "Total": 243.67, "AvgH2H": 2.5,
        "Player comps z": -0.453887, "Player z": -0.541581,
    },
    "Aaron Nesmith": {
        "Avg1": 7.0, "Avg10": 16.5, "MeanYr": 12.490566, "DaysOff": 1.0, "GamesPlayed": 53.0,
        "Home": False, "Moneyline": 0.317, "Total": 249.03, "AvgH2H": 18.0,
        "Player comps z": -0.216736, "Player z": 0.532669,
    },
}


def test_nba_upcoming_get_stats(monkeypatch):
    stats = _load_or_skip(StatsNBA, _NBA_UP_SLATE_DATE)
    offers = _slate(stats, _NBA_UP_SLATE_DATE, _NBA_UP_DATE.strftime("%Y-%m-%d"))
    upcoming = {}
    for offer in offers:
        upcoming.setdefault(offer["Team"], {"Home": offer["Home"], "Opponent": offer["Opponent"]})

    monkeypatch.setattr(base, "datetime", _FrozenDatetime)
    monkeypatch.setattr(base, "archive", _FakeArchive())
    stats.upcoming_games = upcoming
    stats.get_depth(list(offers), _NBA_UP_DATE)
    stats.upcoming_games = upcoming

    out = stats.get_stats("PTS", list(offers), _NBA_UP_DATE)

    assert out.shape == _NBA_UP_SHAPE
    assert _col_hash(out) == _NBA_UP_COL_HASH
    assert str(out["Home"].dtype) == "bool"
    assert _digest(out) == _NBA_UP_DIGEST
    _assert_spot(out, _NBA_UP_SPOT)


# --- MLB historical: comp-z loop populates Player comps z (droplevel(0) fix) ---

_MLB_DATE = _date(2024, 7, 10)
_MLB_SHAPE = (356, 114)
_MLB_COL_HASH = "cc95b8214c1b993e"
_MLB_DIGEST = 8015345811100554101
_MLB_SPOT = {
    "Aaron Judge": {"Player comps z": -0.290427, "Player z": 1.265389, "Home": False},
    "Adam Duvall": {"Player comps z": 0.400125, "Player z": -0.492163, "Home": False},
}


def test_mlb_historical_get_stats():
    stats = _load_or_skip(StatsMLB, _MLB_DATE)
    offers = _slate(stats, _MLB_DATE, _MLB_DATE.strftime("%Y-%m-%d"))

    # MLB skips the non-MLB depth filter, so no get_depth precondition is needed.
    # Regression pin: the comp-z loop used to crash on a stale droplevel(0); it now
    # runs to completion and writes "Player comps z" for every comped player.
    out = stats.get_stats("hits", list(offers), _MLB_DATE)

    assert out.shape == _MLB_SHAPE
    assert _col_hash(out) == _MLB_COL_HASH
    assert str(out["Home"].dtype) == "bool"
    assert _digest(out) == _MLB_DIGEST
    _assert_spot(out, _MLB_SPOT)
