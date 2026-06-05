"""Characterization pins for the CLV movement-alignment paths in ``clv.py``.

Covers the two branches the existing ``test_clv.py`` leaves untested before the
CC-debt decomposition of ``summarize`` (CC 18), ``fill_from_archive`` (CC 14), and
``_movement_alignment`` (CC 11):

1. ``_movement_alignment`` line-direction logic directly — including the
   redundant ``x is None`` guards that collapse to ``pd.isna`` (``pd.isna(None)``
   is True), the flat-line / model-neutral NaN cases, and the Over/Under sign flip.
2. ``summarize`` WITH an archive supplying ``get_movement`` — the per-row movement
   cache, the ``MoveAligned`` leg field, and the ``frac_lines_moved_toward_model``
   segment column (absent when no archive is passed).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus import clv


def test_movement_alignment_toward():
    assert clv._movement_alignment({"open_line": 10.0, "close_line": 11.0}, 0.6, "Over") == 1.0


def test_movement_alignment_away():
    assert clv._movement_alignment({"open_line": 10.0, "close_line": 9.0}, 0.6, "Over") == 0.0


def test_movement_alignment_under_bet_flips_sign():
    # line down + Under-leaning model (model_p>0.5, bet Under) -> toward
    assert clv._movement_alignment({"open_line": 10.0, "close_line": 9.0}, 0.6, "Under") == 1.0


@pytest.mark.parametrize(
    "movement,model_p,bet",
    [
        (None, 0.6, "Over"),
        ({"open_line": 10.0, "close_line": 10.0}, 0.6, "Over"),  # flat line
        ({"close_line": 11.0}, 0.6, "Over"),  # open None -> pd.isna
        ({"open_line": np.nan, "close_line": 11.0}, 0.6, "Over"),  # open NaN
        ({"open_line": 10.0, "close_line": 11.0}, np.nan, "Over"),  # model_p NaN
        ({"open_line": 10.0, "close_line": 11.0}, 0.5, "Over"),  # model neutral
    ],
)
def test_movement_alignment_nan_cases(movement, model_p, bet):
    assert math.isnan(clv._movement_alignment(movement, model_p, bet))


class _MovementArchive:
    def __init__(self, movement):
        self._movement = movement

    def get_movement(self, league, market, date, player, *, until=None):
        return self._movement


def _filled_history():
    """One row, two filled Over offers (close_p + market_clv present so legs count)."""
    return pd.DataFrame(
        [
            {
                "Player": "Player A",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Platform": "Underdog",
                "Offers": [
                    (10.5, 1.0, "Underdog", "Over", 0.60, 0.55, 0.58, 0.03, 0.02),
                    (11.5, 1.0, "Underdog", "Over", 0.62, 0.54, 0.57, 0.04, 0.03),
                ],
            }
        ]
    )


def test_summarize_with_archive_adds_movement_alignment_column():
    archive = _MovementArchive({"open_line": 10.0, "close_line": 11.0})  # line up -> toward Over
    summary = clv.summarize(_filled_history(), archive=archive)
    grouped = summary["all_segments"]
    assert "frac_lines_moved_toward_model" in grouped.columns
    row = grouped.iloc[0]
    assert row["n"] == 2
    assert row["frac_lines_moved_toward_model"] == pytest.approx(1.0)
    assert summary["n"] == 2


def test_summarize_without_archive_omits_movement_column():
    summary = clv.summarize(_filled_history())
    assert "frac_lines_moved_toward_model" not in summary["all_segments"].columns
    assert summary["n"] == 2
