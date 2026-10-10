"""Regression pin for ``Stats._begin_profile_market`` date normalization.

The shared ``profile_market`` preamble was extracted into
``Stats._begin_profile_market`` (base + MLB share it). ``profile_market`` then
passes the returned ``date`` downstream to ``_apply_comp_features`` ->
``_ensure_comps`` -> ``_comp_target_key`` -> ``date.weekday()``, which requires a
``datetime.date`` — a raw ``str`` (the cron/inference path can supply one)
crashes there. So the helper must *return the normalized date*, not just
normalize a local copy. No golden/integration test passes a ``str`` date through
this path, so the breakage was latent; this pins the contract directly.

The last two tests pin the other half of the guard: ``base_profile`` rebuilds the
profile frames for a new date with no market's columns in them, so it must also
forget the market it held. ``StatsMLB.get_depth`` calls it on its own, ahead of
``profile_market``. While the market survived that call, a matrix build that reaches
``get_depth`` (every MLB hitter market, and pitches thrown) read "same market, same
date" on every gameday after its first and left those rows without a player, pitcher
or team-defense profile.
"""

from __future__ import annotations

import datetime
import importlib.resources as pkg_resources

import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.stats import base
from sportstradamus.stats.base import Stats
from sportstradamus.stats.mlb import StatsMLB

_HITTER_MARKET = "total bases"
_OFFER_KEYS = {"playerName": "Player", "team": "Team", "opponent": "Opponent", "home": "Home"}
# An unprofiled gameday has a share of exactly 0; a profiled one measures about 0.9,
# short of 1 by the players new to the 300-day window.
_MIN_PROFILED_SHARE = 0.5


def _bare_stats() -> Stats:
    """A Stats instance with only the attrs ``_begin_profile_market`` touches."""
    obj = Stats.__new__(Stats)
    obj.profiled_market = None
    obj.profile_latest_date = None
    obj.base_profile = lambda _date: None
    return obj


def test_begin_profile_market_normalizes_str_date() -> None:
    obj = _bare_stats()
    result = obj._begin_profile_market("PTS", "2024-01-15")
    assert isinstance(result, datetime.date)
    assert result == datetime.date(2024, 1, 15)
    # weekday() is what the comp path calls; a str would have raised.
    assert result.weekday() == 0


def test_begin_profile_market_normalizes_datetime_to_date() -> None:
    obj = _bare_stats()
    result = obj._begin_profile_market("PTS", datetime.datetime(2024, 1, 15, 9, 30))
    assert type(result) is datetime.date
    assert result == datetime.date(2024, 1, 15)


def test_begin_profile_market_cache_hit_returns_none() -> None:
    obj = _bare_stats()
    obj.profiled_market = "PTS"
    obj.profile_latest_date = datetime.date(2024, 1, 15)
    assert obj._begin_profile_market("PTS", datetime.date(2024, 1, 15)) is None


def test_a_rebuilt_base_profile_forgets_the_market_it_held() -> None:
    obj = Stats.__new__(Stats)
    obj.profiled_market = "hits"
    obj.profile_latest_date = datetime.date(2024, 6, 1)
    obj.positions = []
    obj.window_short_logs = lambda _date: None
    obj._profile_stat_types = lambda: ([], [])
    obj._build_player_profile_stats = lambda _stat_types, _date: pd.DataFrame()
    obj._build_team_defense_profile = lambda _stat_types, _date: None

    obj.base_profile(datetime.date(2024, 6, 2))

    assert obj._begin_profile_market("hits", datetime.date(2024, 6, 2)) == datetime.date(2024, 6, 2)


class _NoArchive:
    """Fails the test if the walk reads the odds archive: golden tests stay off DuckDB."""

    def __getattr__(self, name):
        raise AssertionError(f"archive.{name} read")


@pytest.mark.skipif(
    not (pkg_resources.files(data) / "leagues" / "mlb" / "gamelog.parquet").is_file(),
    reason="MLB gamelog bundle is gitignored; absent in CI",
)
def test_mlb_hitter_market_is_profiled_on_every_gameday_of_a_matrix_build(monkeypatch) -> None:
    """The matrix builder's per-gameday order, on the last two real gamedays.

    For a hitter market ``_dispatch_volume_stats`` resolves lineup slots through
    ``get_depth``, which rebuilds the base profile for the day, and ``get_stats`` then
    profiles the market.
    """
    stats = StatsMLB(load_live_pitchers=False)
    stats.load()
    monkeypatch.setattr(base, "archive", _NoArchive())
    gamelog = stats.gamelog.assign(day=pd.to_datetime(stats.gamelog["gameDate"]).dt.date)
    # A game logged today would take get_depth's upcoming-lineup branch.
    past_days = sorted(day for day in gamelog["day"].unique() if day < datetime.date.today())

    shares = {}
    for day in past_days[-2:]:
        rows = gamelog[gamelog["day"] == day].drop_duplicates("playerName")
        game = rows[rows["gameId"] == rows.groupby("gameId").size().idxmax()]
        offers = game.rename(columns=_OFFER_KEYS)[list(_OFFER_KEYS.values())].to_dict("records")
        stats._dispatch_volume_stats(offers, day, _HITTER_MARKET)
        frame = stats.get_stats(_HITTER_MARKET, offers, day)
        shares[day] = (frame["Player z"] != 0).mean()

    assert min(shares.values()) > _MIN_PROFILED_SHARE, shares
