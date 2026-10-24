"""Gate 1 is scored on sportsbook-priced rows only: a pick'em-only frame is book-less."""

import pandas as pd

from sportstradamus.training import scorecard


def _frame(authenticity: str, n: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Player": [f"p{i}" for i in range(n)],
            "P": 0.55,
            "Odds": 0.5,
            "Line": 4.5,
            "Result": [3, 6] * (n // 2),
            "QuoteAuthenticity": authenticity,
        }
    )


def test_pickem_only_rows_leave_gate1_blank():
    assert scorecard._priced_rows(_frame("pickem")) is None


def test_sportsbook_rows_still_price_gate1():
    assert len(scorecard._priced_rows(_frame("authentic"))) == 24
