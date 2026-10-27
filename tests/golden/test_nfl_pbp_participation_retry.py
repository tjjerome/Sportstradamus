"""The current season's play-by-play must survive its unpublished participation file.

nflverse publishes ``pbp_participation_{year}.parquet`` only after the season, and
``nfl_data_py.import_pbp_data`` reads it by default from 2016 on; its handler for that
404 names an undefined ``Error``, so a ``NameError`` escapes and every 2026 run lost the
whole year's PBP. The current season retries without participation; a past season, whose
file exists, keeps the single attempt.
"""

import pandas as pd

from sportstradamus.stats import nfl_pbp_agg

CURRENT_SEASON = 2026


def _patch(monkeypatch) -> tuple[list[dict], pd.DataFrame]:
    """Fake ``import_pbp_data`` that fails with participation and records each call's kwargs."""
    calls: list[dict] = []
    frame = pd.DataFrame({"play_id": [1]})

    def import_pbp_data(years, **kwargs):
        calls.append(kwargs)
        if kwargs.get("include_participation", True):
            raise NameError("name 'Error' is not defined")
        return frame

    monkeypatch.setattr(nfl_pbp_agg.nflr, "get_current_season", lambda: CURRENT_SEASON)
    monkeypatch.setattr(nfl_pbp_agg.nfl, "import_pbp_data", import_pbp_data)
    return calls, frame


def test_current_season_retries_without_participation(monkeypatch):
    calls, frame = _patch(monkeypatch)

    result = nfl_pbp_agg._safe_import_pbp(CURRENT_SEASON)

    assert result is frame
    assert calls == [{}, {"include_participation": False}]


def test_past_season_does_not_retry(monkeypatch):
    calls, _ = _patch(monkeypatch)

    assert nfl_pbp_agg._safe_import_pbp(CURRENT_SEASON - 1) is None
    assert calls == [{}]
