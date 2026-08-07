"""Regression test for the MeanYr/Mean10 leakage rule in ``Stats.window_short_logs``.

``stats/base.py`` builds ``short_gamelog`` with a strict less-than date filter
(``gameDates < target_date`` in ``Stats.window_short_logs``, the helper
``base_profile`` delegates to), so the target game date is excluded from every
downstream aggregate (``MeanYr``, ``Mean10``, ``GamesPlayed``, ...). Flipping
the operator to ``<=`` would contaminate the per-player baseline features
with the very target the model is trying to predict — fatal for the centered
SkewNormal target in P1 of the GBDT-mean-regression project (and severe for
``Mean10``, where one leaked row pollutes ~10% of the window).

Two layers of guarding:

1. **Static check**: the source of ``Stats.window_short_logs`` must contain the
   literal strict-less-than expression. Catches a textual regression in
   review without needing a runtime fixture.
2. **Functional check**: applies the same filter to a synthetic gamelog and
   asserts the sentinel row at the target date is excluded from MeanYr /
   Mean10 / GamesPlayed aggregates computed the same way the rolling-feature
   block in ``stats/base.py`` does.
"""

from __future__ import annotations

import inspect
from datetime import date, timedelta

import pandas as pd

from sportstradamus.stats.base import Stats

# 365 days back covers a typical season; the production filter uses 300 days
# but the test value just needs to bracket the synthetic rows below.
_LOOKBACK_DAYS = 365


def _filter_matches_production(gamelog: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Replicate the ``Stats.window_short_logs`` mask on the synthetic frame.

    Pure function so the test asserts the exact production semantics; any
    drift between this and the source is caught by the static check below.
    """
    one_year_ago = target_date - timedelta(days=_LOOKBACK_DAYS)
    gameDates = pd.to_datetime(gamelog["GAME_DATE"]).dt.date
    return gamelog[(one_year_ago <= gameDates) & (gameDates < target_date)].copy()


def test_window_short_logs_source_uses_strict_less_than():
    """Guards against a textual ``<`` → ``<=`` regression in the short-log mask."""
    src = inspect.getsource(Stats.window_short_logs)
    # The leakage-critical filter expression. Whitespace-tolerant.
    assert "(gameDates < target_date)" in src.replace("\n", " "), (
        "Stats.window_short_logs must use strict less-than (`gameDates < "
        "target_date`) so the target game date is excluded from short_gamelog. "
        "A change to `<=` would leak the target row into MeanYr/Mean10."
    )


def test_meanyr_mean10_exclude_target_date_sentinel():
    """Synthetic gamelog: sentinel at target_date must not affect MeanYr/Mean10."""
    target = date(2025, 3, 15)
    # 15 prior games at value 10, one target-date row with a wildly different
    # sentinel value. If the filter is correct, MeanYr=10 and Mean10=10.
    # If it leaks, MeanYr drifts toward (15*10 + 1*999)/16 ≈ 71.8.
    prior_dates = [target - timedelta(days=i) for i in range(1, 16)]
    sentinel_value = 999.0
    gamelog = pd.DataFrame(
        {
            "PLAYER_NAME": ["P"] * 16,
            "GAME_DATE": [*prior_dates, target],
            "PTS": [10.0] * 15 + [sentinel_value],
        }
    )

    short = _filter_matches_production(gamelog, target)

    # Target row excluded — sentinel must not appear anywhere downstream.
    assert (pd.to_datetime(short["GAME_DATE"]).dt.date < target).all()
    assert sentinel_value not in short["PTS"].to_numpy()

    # Same aggregate the production code computes at base.py:681-693.
    grp = short.groupby("PLAYER_NAME")["PTS"]
    last10 = short.groupby("PLAYER_NAME").tail(10)
    mean_yr = grp.mean()["P"]
    mean_10 = last10.groupby("PLAYER_NAME")["PTS"].mean()["P"]
    games_played = grp.count()["P"]

    assert mean_yr == 10.0
    assert mean_10 == 10.0
    assert games_played == 15  # strictly < 16 (would be 16 with `<=`).


def test_expanding_features_source_uses_strict_less_than():
    """Guards the M-1 career expanding mean against a ``<`` → ``<=`` regression."""
    src = inspect.getsource(Stats._expanding_features)
    assert "gld < pd.Timestamp(date)" in src.replace("\n", " "), (
        "Stats._expanding_features must use strict less-than (`gld < "
        "pd.Timestamp(date)`) so the target game is excluded from "
        "MeanYr_expanding_shifted / _vsopp / _eb."
    )


def _expanding_matches_production(gamelog: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Replicate ``Stats._expanding_features`` career filter (full history, strict <)."""
    gld = pd.to_datetime(gamelog["GAME_DATE"]).dt.date
    return gamelog[gld < target_date].copy()


def test_expanding_mean_reads_full_history_and_excludes_target():
    """Career expanding mean spans beyond the 1-yr window and drops the target row."""
    target = date(2025, 3, 15)
    # One game 400 days back (OUTSIDE the 300-day short window), 15 recent games,
    # and a target-date sentinel. The expanding mean must include the old game
    # (proving it reads the full gamelog) and exclude the sentinel.
    old_game = target - timedelta(days=400)
    recent = [target - timedelta(days=i) for i in range(1, 16)]
    gamelog = pd.DataFrame(
        {
            "PLAYER_NAME": ["P"] * 17,
            "GAME_DATE": [old_game, *recent, target],
            "PTS": [4.0] + [10.0] * 15 + [999.0],
        }
    )

    career = _expanding_matches_production(gamelog, target)
    assert (pd.to_datetime(career["GAME_DATE"]).dt.date < target).all()
    assert 999.0 not in career["PTS"].to_numpy()

    grp = career.groupby("PLAYER_NAME")["PTS"]
    # (4 + 15*10) / 16 = 9.625 -- includes the >1yr game, excludes the sentinel.
    assert grp.mean()["P"] == (4.0 + 15 * 10.0) / 16
    assert grp.count()["P"] == 16


def test_expanding_vsopp_excludes_target():
    """The head-to-head career variant must also exclude the target-date game."""
    target = date(2025, 3, 15)
    prior = [target - timedelta(days=i) for i in range(1, 4)]
    gamelog = pd.DataFrame(
        {
            "PLAYER_NAME": ["P"] * 4,
            "GAME_DATE": [*prior, target],
            "OPPONENT": ["DET", "DET", "CHI", "DET"],
            "PTS": [12.0, 8.0, 5.0, 999.0],
        }
    )
    career = _expanding_matches_production(gamelog, target)
    h2h = career.loc[career["OPPONENT"] == "DET"]
    assert 999.0 not in h2h["PTS"].to_numpy()
    # Only the two prior DET games (12, 8); the target-date DET game is excluded.
    assert h2h["PTS"].mean() == 10.0
    assert len(h2h) == 2


def test_meanyr_mean10_filter_with_only_target_date_returns_empty():
    """Edge case: a player whose only game is at target_date has no prior history."""
    target = date(2025, 3, 15)
    gamelog = pd.DataFrame(
        {
            "PLAYER_NAME": ["P"],
            "GAME_DATE": [target],
            "PTS": [42.0],
        }
    )

    short = _filter_matches_production(gamelog, target)
    assert short.empty, (
        "A player whose only appearance is on target_date must yield an empty "
        "short_gamelog — otherwise MeanYr would equal the target itself."
    )
