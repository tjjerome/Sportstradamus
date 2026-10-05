"""Kelly parity between the board and reflect.

``offer_records.finalize_records`` sizes Kelly on the full decimal payout when the board is
scored; ``analysis.annotate_offer_outcomes`` re-derives it from the persisted row, where
Underdog's ``Boost`` is back to the raw multiplier (``prediction/cli.py`` divides the
per-pick baseline out). Both must zero Kelly outside the same ``(1, MAX_FAVORED_PAYOUT]``
window and agree inside it, or reflect sizes bets the board never would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.analysis import annotate_offer_outcomes
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.prediction import offer_records

# (player, raw Over boost, Model Over) per platform; Over always wins the side.
_BOARD = {
    "Underdog": [
        ("flat", 1.0, 0.62),
        # A raw multiplier whose full payout is 0.99x.
        ("under even", 0.99 / UNDERDOG_BOOST_BASELINE, 0.80),
        # Model Over clips to 0.90 before Win Prob is set, so the persisted Win Prob is
        # already the clipped one and the clip cannot split the two rules.
        ("clipped", 1.2, 0.95),
    ],
    "Sleeper": [
        ("in range", 1.5, 0.75),
        ("over cap", offer_records.MAX_FAVORED_PAYOUT + 0.5, 0.50),
    ],
}


class _StubArchive:
    """The one archive attribute finalize_records reads."""

    default_totals = {"NBA": 220.0}


@pytest.fixture(autouse=True)
def _stub_record_archive(monkeypatch):
    # Unpatched, finalize_records opens the real archive and waits on its lock under xdist.
    monkeypatch.setattr(offer_records, "archive", _StubArchive())


def _scored(platform: str) -> pd.DataFrame:
    offers = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Date": "2026-06-03",
                "Team": "LAL",
                "Opponent": "BOS",
                "Player": player,
                "Market": "points",
                "Line": 25.5,
                "Model Over": model_over,
                "Model Under": 0.05,
                "Boost_Over": boost,
                "Boost_Under": np.nan,
                "Market EV": 0.5,
                "Projection": 25.5,
                "Push Prob": 0.0,
                "Market Projection": np.nan,
                "Model Weight": 1.0,
                "Quote Source": None,
                "Quote Authenticity": None,
                "Quote Books": np.nan,
                "Quote Line": np.nan,
                "Quote Observed At": pd.NaT,
            }
            for player, boost, model_over in _BOARD[platform]
        ]
    )
    records = offer_records.finalize_records(
        offers, "NBA", platform, "NegBin", 0.5, 1.0, None, 1.0, "test-version"
    )
    return pd.DataFrame(records).assign(Platform=platform)


def test_annotate_recomputes_the_board_kelly():
    board = pd.concat([_scored(platform) for platform in _BOARD], ignore_index=True)
    underdog = board["Platform"] == "Underdog"
    persisted = board.assign(
        Boost=board["Boost"].where(~underdog, board["Boost"] / UNDERDOG_BOOST_BASELINE)
    )

    annotated = annotate_offer_outcomes(persisted)

    np.testing.assert_allclose(annotated["Kelly"], board["Kelly"], rtol=0, atol=1e-12)
    zeroed = board.set_index("Player")["Kelly"].eq(0)
    assert sorted(zeroed[zeroed].index) == ["over cap", "under even"]
