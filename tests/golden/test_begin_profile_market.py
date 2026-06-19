"""Regression pin for ``Stats._begin_profile_market`` date normalization.

The shared ``profile_market`` preamble was extracted into
``Stats._begin_profile_market`` (base + MLB share it). ``profile_market`` then
passes the returned ``date`` downstream to ``_apply_comp_features`` ->
``_ensure_comps`` -> ``_comp_target_key`` -> ``date.weekday()``, which requires a
``datetime.date`` — a raw ``str`` (the cron/inference path can supply one)
crashes there. So the helper must *return the normalized date*, not just
normalize a local copy. No golden/integration test passes a ``str`` date through
this path, so the breakage was latent; this pins the contract directly.
"""

from __future__ import annotations

import datetime

from sportstradamus.stats.base import Stats


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
