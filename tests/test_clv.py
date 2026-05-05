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

    def get_ev(self, league, market, date, player):
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
    """One Over leg with archive coverage, one Under leg without."""
    return pd.DataFrame(
        [
            {
                "Player": "Player A",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Offers": [
                    (10.5, 1.0, "Underdog", "Over", 0.60, 0.55, np.nan, np.nan, np.nan),
                ],
            },
            {
                "Player": "Player B",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Offers": [
                    (12.5, 1.0, "Underdog", "Under", 0.48, 0.50, np.nan, np.nan, np.nan),
                ],
            },
        ]
    )


def test_fill_from_archive_writes_close_and_clv_for_resolved_leg():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): 0.62})
    df = clv.fill_from_archive(_build_history(), archive)

    offer = df.loc[0, "Offers"][0]
    assert offer[6] == pytest.approx(0.62)
    # Market CLV = +1 * (0.62 - 0.55) = 0.07
    assert offer[7] == pytest.approx(0.07)
    # Model CLV = +1 * (0.62 - 0.60) = 0.02
    assert offer[8] == pytest.approx(0.02)


def test_fill_from_archive_leaves_unresolved_leg_nan():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): 0.62})
    df = clv.fill_from_archive(_build_history(), archive)

    offer = df.loc[1, "Offers"][0]
    assert math.isnan(offer[6])
    assert math.isnan(offer[7])
    assert math.isnan(offer[8])


def test_fill_from_archive_pads_legacy_six_tuple():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): 0.60})
    legacy = pd.DataFrame(
        [
            {
                "Player": "Player A",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Offers": [(10.5, 1.0, "Underdog", "Over", 0.55, 0.50)],
            }
        ]
    )
    out = clv.fill_from_archive(legacy, archive)
    offer = out.loc[0, "Offers"][0]
    assert len(offer) == 9
    assert offer[6] == pytest.approx(0.60)


def test_summarize_drops_unresolved_legs():
    archive = _StubArchive({("NBA", "points", "2026-05-04", "Player A"): 0.62})
    df = clv.fill_from_archive(_build_history(), archive)

    summary = clv.summarize(df)
    assert summary["n"] == 1
    assert summary["market_clv_mean"] == pytest.approx(0.07)
    assert summary["model_clv_mean"] == pytest.approx(0.02)
    assert summary["frac_beat_close"] == pytest.approx(1.0)


def test_summarize_returns_zero_n_when_no_legs():
    summary = clv.summarize(pd.DataFrame({"Offers": [[]], "League": ["NBA"], "Market": ["points"]}))
    assert summary["n"] == 0
    assert math.isnan(summary["market_clv_mean"])
