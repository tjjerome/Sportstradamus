"""Unit tests for the CLV fill / summary helpers in ``sportstradamus.clv``."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus import clv


class _StubArchive:
    """Minimal stub that mirrors ``Archive.get_ev``'s lookup contract."""

    def __init__(self, table):
        self._table = table

    def get_ev(self, league, market, date, player, *, at=None):
        del at  # the stub ignores the time-series cutoff
        return self._table.get((league, market, date, player), float("nan"))


def test_signed_clv_over_uses_positive_sign():
    assert clv._signed_clv(0.50, 0.55, "Over") == pytest.approx(0.05)


def test_signed_clv_under_flips_sign():
    assert clv._signed_clv(0.50, 0.55, "Under") == pytest.approx(-0.05)


def test_signed_clv_higher_is_over_synonym():
    assert clv._signed_clv(0.40, 0.45, "Higher") == pytest.approx(0.05)


def test_signed_clv_returns_nan_when_open_is_nan():
    assert math.isnan(clv._signed_clv(float("nan"), 0.55, "Over"))


def test_signed_clv_returns_nan_when_close_is_nan():
    assert math.isnan(clv._signed_clv(0.55, float("nan"), "Under"))


def _build_history():
    """One Over offer with archive coverage, one Under offer without."""
    return pd.DataFrame(
        [
            {
                "Player": "Player A",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 10.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Over",
                "Win Prob": 0.60,
                "Market Prob": 0.55,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            },
            {
                "Player": "Player B",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 12.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Under",
                "Win Prob": 0.48,
                "Market Prob": 0.50,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
            },
        ]
    )


def test_fill_from_archive_writes_close_and_clv_for_resolved_leg():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): 0.62})
    df = clv.fill_from_archive(_build_history(), archive)

    row = df.loc[0]
    assert row["Close Market Prob"] == pytest.approx(0.62)
    # Market CLV = +1 * (0.62 - 0.55) = 0.07
    assert row["Market CLV"] == pytest.approx(0.07)
    # Model CLV = +1 * (0.62 - 0.60) = 0.02
    assert row["Model CLV"] == pytest.approx(0.02)


def test_fill_from_archive_leaves_unresolved_leg_nan():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): 0.62})
    df = clv.fill_from_archive(_build_history(), archive)

    row = df.loc[1]
    assert math.isnan(row["Close Market Prob"])
    assert math.isnan(row["Market CLV"])
    assert math.isnan(row["Model CLV"])


def test_fill_from_archive_skips_rows_already_closed():
    """A row with a non-NaN Close Market Prob is not re-queried against archive."""
    history = _build_history()
    history.loc[0, "Close Market Prob"] = 0.99
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): 0.62})
    df = clv.fill_from_archive(history, archive)
    assert df.loc[0, "Close Market Prob"] == pytest.approx(0.99)


def test_summarize_drops_unresolved_legs():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): 0.62})
    df = clv.fill_from_archive(_build_history(), archive)

    summary = clv.summarize(df)
    assert summary["n"] == 1
    assert summary["market_clv_mean"] == pytest.approx(0.07)
    assert summary["model_clv_mean"] == pytest.approx(0.02)
    assert summary["frac_beat_close"] == pytest.approx(1.0)


def test_summarize_returns_zero_n_when_no_legs():
    summary = clv.summarize(
        pd.DataFrame(
            {
                "League": pd.Series(dtype=str),
                "Market": pd.Series(dtype=str),
                "Platform": pd.Series(dtype=str),
                "Bet": pd.Series(dtype=str),
                "Win Prob": pd.Series(dtype=float),
                "Close Market Prob": pd.Series(dtype=float),
                "Market CLV": pd.Series(dtype=float),
                "Model CLV": pd.Series(dtype=float),
                "Date": pd.Series(dtype=str),
                "Player": pd.Series(dtype=str),
            }
        )
    )
    assert summary["n"] == 0
    assert math.isnan(summary["market_clv_mean"])
