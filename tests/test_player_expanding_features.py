"""Derivation pins for the M-1 player-level expanding build (model track §6.3).

``Stats._expanding_features`` reads the FULL gamelog (strict ``<`` date) per player
to build the career cumulative mean (``MeanYr_expanding_shifted`` -- distinct from
the 1-year-window ``MeanYr``), its head-to-head-vs-opponent variant, and an
Empirical-Bayes shrinkage toward the per-position baseline where the career window
is sparse. The train/live parity gate proves the two branches agree; this test pins
the arithmetic (career mean, vs-opponent subset, the K/(K+n) shrinkage direction,
and the no-history degrade) on known inputs.

Leakage of the temporal feature is guarded in ``test_meanyr_mean10_leakage.py``.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sportstradamus.stats import StatsNBA
from sportstradamus.stats.base import _EXPANDING_EB_PRIOR_K

GAMEDAY = date(2025, 3, 15)
# P1: 4 career games (one >300 days back, proving full-history read) vs DET/CHI.
# P2: 1 career game (sparse -> EB pulls toward the position baseline).
# P3: no career history (only a target-date game) -> shifted NaN, eb == baseline.
PLAYER_POSITION = {"P1": 1, "P2": 1, "P3": 2}
OPPONENT = {"P1": "DET", "P2": "DET", "P3": "CHI"}


def _gamelog_rows():
    d = lambda n: (GAMEDAY - timedelta(days=n)).isoformat()  # noqa: E731
    return [
        ("P1", "DET", d(400), 12.0),
        ("P1", "DET", d(3), 8.0),
        ("P1", "CHI", d(2), 40.0),
        ("P1", "DET", d(1), 10.0),
        ("P1", "DET", GAMEDAY.isoformat(), 999.0),  # target-date sentinel -> excluded
        ("P2", "DET", d(1), 10.0),
        ("P3", "CHI", GAMEDAY.isoformat(), 50.0),  # only game is on the target date
    ]


def _run_expanding() -> pd.DataFrame:
    stats_obj = StatsNBA()
    ls = stats_obj.log_strings
    stats_obj.gamelog = pd.DataFrame(
        [
            {ls["player"]: p, ls["opponent"]: opp, ls["date"]: gd, "PTS": pts}
            for p, opp, gd, pts in _gamelog_rows()
        ]
    )
    frame = pd.DataFrame(
        {"Player position": [PLAYER_POSITION[p] for p in PLAYER_POSITION]},
        index=list(PLAYER_POSITION),
    )
    stats_obj._expanding_features(frame, "PTS", GAMEDAY, dict(OPPONENT))
    return frame


def test_expanding_shifted_is_full_history_career_mean_excluding_target():
    out = _run_expanding()
    # (12 + 8 + 40 + 10) / 4 -- includes the >300-day game, drops the 999 sentinel.
    assert out.loc["P1", "MeanYr_expanding_shifted"] == pytest.approx(70.0 / 4)
    assert out.loc["P1", "MeanYr_expanding_n"] == 4
    assert out.loc["P2", "MeanYr_expanding_shifted"] == pytest.approx(10.0)
    assert pd.isna(out.loc["P3", "MeanYr_expanding_shifted"])  # no prior history


def test_expanding_vsopp_is_the_head_to_head_subset():
    out = _run_expanding()
    # P1 vs DET: 12, 8, 10 (the CHI game and the target-date game are excluded).
    assert out.loc["P1", "MeanYr_expanding_vsopp"] == pytest.approx((12 + 8 + 10) / 3)
    assert out.loc["P1", "MeanYr_expanding_vsopp_n"] == 3


def test_expanding_eb_shrinks_sparse_players_toward_the_position_baseline():
    out = _run_expanding()
    k = _EXPANDING_EB_PRIOR_K
    # Position-1 baseline is the mean of P1/P2 career means; P3's NaN falls back to
    # the global mean of the expanding means (also the P1/P2 mean here).
    pos1_baseline = (70.0 / 4 + 10.0) / 2
    p1_raw, p1_n = 70.0 / 4, 4
    p2_raw, p2_n = 10.0, 1
    exp_p1 = (p1_n / (p1_n + k)) * p1_raw + (k / (p1_n + k)) * pos1_baseline
    exp_p2 = (p2_n / (p2_n + k)) * p2_raw + (k / (p2_n + k)) * pos1_baseline
    assert out.loc["P1", "MeanYr_expanding_eb"] == pytest.approx(exp_p1)
    assert out.loc["P2", "MeanYr_expanding_eb"] == pytest.approx(exp_p2)
    # Sparse P2 is pulled from its raw 10 toward the 13.75 baseline.
    assert p2_raw < out.loc["P2", "MeanYr_expanding_eb"] < pos1_baseline
    # No-history P3 gets the baseline exactly (n=0 -> all prior weight).
    assert out.loc["P3", "MeanYr_expanding_eb"] == pytest.approx(pos1_baseline)
