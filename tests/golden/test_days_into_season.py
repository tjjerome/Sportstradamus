"""``DaysIntoSeason`` counts from the opener of the season the row's date falls in.

Serving counts from the ``season_start`` a class holds, the current season's opener. A training
matrix builds every past gameday with that same held start, so a row of an earlier season finds
its own opener in the gamelog: the log's first game day, or the first one after an offseason.
"""

from datetime import date

import pandas as pd
import pytest

from sportstradamus.stats.base import Stats, _opening_day

_OPENERS = (date(2023, 10, 10), date(2024, 10, 4), date(2025, 10, 7))
# Three seasons, each through its playoffs in June; the middle one stops for 20 days in February.
_GAME_DAYS = [
    *pd.date_range("2023-10-10", "2024-06-24"),
    *pd.date_range("2024-10-04", "2025-02-08"),
    *pd.date_range("2025-02-28", "2025-06-17"),
    *pd.date_range("2025-10-07", "2026-06-20"),
]
_DATE_TYPES = {
    "str": lambda day: day.date().isoformat(),
    # An evening start: the time of day must not count.
    "Timestamp": lambda day: day + pd.Timedelta(hours=19),
    "date": lambda day: day.date(),
}


def _gamelog(as_type=_DATE_TYPES["str"]) -> pd.DataFrame:
    """Each date twice (two players), each run rotated to open mid-season on 2025-03-01.

    A first-row or row-order reading takes that day for an opener.
    """
    start = _GAME_DAYS.index(pd.Timestamp("2025-03-01"))
    days = _GAME_DAYS[start:] + _GAME_DAYS[:start]
    return pd.DataFrame(
        {
            "PLAYER": ["A"] * len(days) + ["B"] * len(days),
            "GAME_DATE": [as_type(day) for day in days * 2],
            "PTS": 10.0,
        }
    )


class _Unread:
    """Stands in for the gamelog's dates: reading them fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"the gamelog was read (.{name})")


@pytest.mark.parametrize("as_type", _DATE_TYPES.values(), ids=list(_DATE_TYPES))
@pytest.mark.parametrize(
    ("held", "day", "opener"),
    [
        pytest.param(_OPENERS[2], date(2024, 6, 10), _OPENERS[0], id="past-season-playoffs"),
        pytest.param(_OPENERS[2], date(2025, 4, 15), _OPENERS[1], id="past-season-after-break"),
        pytest.param(_OPENERS[2], _OPENERS[1], _OPENERS[1], id="past-season-opening-day"),
        pytest.param(_OPENERS[1], date(2026, 1, 15), _OPENERS[2], id="stale-held-start"),
    ],
)
def test_a_day_outside_the_held_season_counts_from_its_gamelog_opener(held, day, opener, as_type):
    assert _opening_day(day, held, _gamelog(as_type)["GAME_DATE"]) == opener


def test_a_day_in_the_held_season_keeps_the_held_start_without_reading_the_log():
    # Hand-set a week before the log's first game of the season, which must not replace it.
    held = date(2025, 9, 30)
    assert _opening_day(date(2026, 6, 1), held, _Unread()) == held


def test_a_past_season_row_counts_from_its_own_opener():
    stats = Stats()
    stats.log_strings = {"player": "PLAYER", "date": "GAME_DATE"}
    stats.season_start = _OPENERS[2]
    stats.gamelog = _gamelog()
    day = date(2025, 4, 15)
    played = stats.gamelog[pd.to_datetime(stats.gamelog["GAME_DATE"]) < pd.Timestamp(day)]

    rows = stats._rolling_features(pd.DataFrame(), played, "PTS", day)

    assert rows["DaysIntoSeason"].tolist() == [(day - _OPENERS[1]).days] * 2
