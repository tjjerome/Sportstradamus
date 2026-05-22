"""Regression tests for NFL gamelog numeric coercion.

The nfl-data-py advanced-stat columns (yards per carry, passer rating, the
target-share metrics, ...) arrive as object-dtype numeric strings. Uncoerced,
they survive ``base_profile``'s ``numeric_only=True`` aggregate by being
silently dropped, but crash the short-window ``tail(5).mean()`` aggregate
(no ``numeric_only`` argument available) with::

    TypeError: Could not convert string '0.00.00.00.00.0' to numeric

``StatsNFL._coerce_numeric_gamelog`` fixes this at the source by coercing every
gamelog column except the known identifier/categorical/boolean columns.
"""

import pandas as pd

from sportstradamus.stats import StatsNFL


def _synthetic_gamelog() -> pd.DataFrame:
    """A two-row gamelog with numeric-as-string advanced-stat columns."""
    return pd.DataFrame(
        {
            "player id": ["00-0000001", "00-0000002"],
            "player display name": ["Alpha Back", "Bravo Catch"],
            "position": ["QB", "RB"],
            "team": ["KC", "BUF"],
            "season type": ["REG", "REG"],
            "opponent": ["BUF", "KC"],
            "gameday": ["2024-09-10", "2024-09-11"],
            "game id": ["2024_01_KC_BUF", "2024_01_BUF_KC"],
            "home": [True, False],
            # The numeric-as-string offenders (object dtype on load):
            "snap pct": ["0.8", "0.5"],
            "completion percentage": ["62.5", "70.0"],
            "yards per carry": ["5.6", "0.0"],
            "completion percentage over expected": ["-2.61", "1.40"],
        }
    )


def test_coerce_numeric_gamelog_converts_numeric_strings():
    """Numeric-as-string columns become numeric dtype after coercion."""
    nfl = StatsNFL()
    nfl.gamelog = _synthetic_gamelog()

    nfl._coerce_numeric_gamelog()

    for col in (
        "snap pct",
        "completion percentage",
        "yards per carry",
        "completion percentage over expected",
    ):
        assert pd.api.types.is_numeric_dtype(nfl.gamelog[col]), col
    # Values preserved through coercion (incl. negative).
    assert nfl.gamelog["completion percentage"].tolist() == [62.5, 70.0]
    assert nfl.gamelog["completion percentage over expected"].tolist() == [-2.61, 1.40]


def test_coerce_numeric_gamelog_preserves_identifier_columns():
    """Identifier / categorical string columns stay object; bool stays bool."""
    nfl = StatsNFL()
    nfl.gamelog = _synthetic_gamelog()

    nfl._coerce_numeric_gamelog()

    for col in (
        "player id",
        "player display name",
        "position",
        "team",
        "season type",
        "opponent",
        "gameday",
        "game id",
    ):
        assert nfl.gamelog[col].dtype == object, col
    # "home" must remain boolean, not be coerced to 0/1 integers.
    assert pd.api.types.is_bool_dtype(nfl.gamelog["home"])
