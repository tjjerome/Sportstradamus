"""Unit tests for ``sportstradamus.analysis`` metric helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from sportstradamus.analysis import (
    compute_book_brier_skill_score,
    compute_brier_skill_score,
)


def _seeded_subset(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic ``(Bet, Result, Model P, Books P)`` frame."""
    rng = np.random.default_rng(seed)
    sides = np.array(["Over", "Under"])
    bet = rng.choice(sides, size=n)
    result = rng.choice(sides, size=n)
    model_p = rng.uniform(0.05, 0.95, size=n)
    books_p = rng.uniform(0.05, 0.95, size=n)
    return pd.DataFrame(
        {
            "Bet": bet,
            "Result": result,
            "Win Prob": model_p,
            "Market Prob": books_p,
        }
    )


def _reference_book_bss(subset: pd.DataFrame) -> float:
    """Hand-computed reference: 1 - brier(model_p) / brier(books_p)."""
    hits = (subset["Bet"] == subset["Result"]).astype(int)
    brier_model = brier_score_loss(hits, subset["Win Prob"].clip(0, 1))
    brier_book = brier_score_loss(hits, subset["Market Prob"].clip(0, 1))
    return 1 - brier_model / brier_book


def test_compute_book_brier_skill_score_matches_reference():
    subset = _seeded_subset(n=100, seed=42)
    expected = _reference_book_bss(subset)
    assert compute_book_brier_skill_score(subset) == pytest.approx(expected, abs=1e-6)


def test_compute_book_brier_skill_score_uses_model_fallback_when_model_p_missing():
    subset = _seeded_subset(n=100, seed=7)
    subset = subset.rename(columns={"Win Prob": "Model EV"})
    hits = (subset["Bet"] == subset["Result"]).astype(int)
    brier_model = brier_score_loss(hits, subset["Model EV"].clip(0, 1))
    brier_book = brier_score_loss(hits, subset["Market Prob"].clip(0, 1))
    expected = 1 - brier_model / brier_book
    assert compute_book_brier_skill_score(subset) == pytest.approx(expected, abs=1e-6)


def test_compute_book_brier_skill_score_returns_nan_for_empty_subset():
    empty = pd.DataFrame(columns=["Bet", "Result", "Win Prob", "Market Prob"])
    assert math.isnan(compute_book_brier_skill_score(empty))


def test_compute_book_brier_skill_score_returns_nan_when_books_p_missing():
    subset = _seeded_subset(n=10, seed=1).drop(columns=["Market Prob"])
    assert math.isnan(compute_book_brier_skill_score(subset))


def test_compute_book_brier_skill_score_returns_nan_when_books_p_all_nan():
    subset = _seeded_subset(n=10, seed=2)
    subset["Market Prob"] = np.nan
    assert math.isnan(compute_book_brier_skill_score(subset))


def test_compute_book_brier_skill_score_returns_nan_when_books_p_perfectly_predicts():
    subset = pd.DataFrame(
        {
            "Bet": ["Over"] * 10,
            "Result": ["Over"] * 10,
            "Win Prob": [0.5] * 10,
            "Market Prob": [1.0] * 10,
        }
    )
    assert math.isnan(compute_book_brier_skill_score(subset))


def test_compute_book_brier_skill_score_is_positive_when_model_beats_book():
    n = 200
    rng = np.random.default_rng(123)
    bet = np.array(["Over"] * n)
    result = rng.choice(["Over", "Under"], size=n, p=[0.7, 0.3])
    model_p = np.where(result == "Over", 0.8, 0.3)
    books_p = np.full(n, 0.5)
    subset = pd.DataFrame(
        {"Bet": bet, "Result": result, "Win Prob": model_p, "Market Prob": books_p}
    )
    assert compute_book_brier_skill_score(subset) > 0


def test_compute_book_brier_skill_score_is_negative_when_book_beats_model():
    n = 200
    rng = np.random.default_rng(321)
    bet = np.array(["Over"] * n)
    result = rng.choice(["Over", "Under"], size=n, p=[0.7, 0.3])
    model_p = np.full(n, 0.5)
    books_p = np.where(result == "Over", 0.8, 0.3)
    subset = pd.DataFrame(
        {"Bet": bet, "Result": result, "Win Prob": model_p, "Market Prob": books_p}
    )
    assert compute_book_brier_skill_score(subset) < 0


def _reference_bss(subset: pd.DataFrame, base_rate: float = 0.5) -> float:
    """Hand-computed reference: 1 - brier(model_p) / brier_ref."""
    hits = (subset["Bet"] == subset["Result"]).astype(int)
    brier = brier_score_loss(hits, subset["Win Prob"].clip(0, 1))
    brier_ref = base_rate * (1 - base_rate)
    return 1 - brier / brier_ref


def test_compute_brier_skill_score_matches_reference():
    subset = _seeded_subset(n=100, seed=42)
    expected = _reference_bss(subset)
    assert compute_brier_skill_score(subset) == pytest.approx(expected, abs=1e-6)


def _seeded_subset_with_passes(n_bet: int, n_pass: int, seed: int) -> pd.DataFrame:
    """Bet rows with full data, plus passed-offer rows: Bet/Win Prob both NaN.

    Reproduces a live (league, market) cell mixing bets and passed offers --
    the shape that crashed ``compute_book_brier_skill_score`` in production
    (Win Prob picked as prob_col because *some* row has it, then the NaN
    passed-offer rows ride along into ``brier_score_loss``).
    """
    rng = np.random.default_rng(seed)
    sides = np.array(["Over", "Under"])
    bet_df = pd.DataFrame(
        {
            "Bet": rng.choice(sides, size=n_bet),
            "Result": rng.choice(sides, size=n_bet),
            "Win Prob": rng.uniform(0.05, 0.95, size=n_bet),
            "Market Prob": rng.uniform(0.05, 0.95, size=n_bet),
        }
    )
    pass_df = pd.DataFrame(
        {
            "Bet": [np.nan] * n_pass,
            "Result": rng.choice(sides, size=n_pass),
            "Win Prob": [np.nan] * n_pass,
            "Market Prob": rng.uniform(0.05, 0.95, size=n_pass),
        }
    )
    return pd.concat([bet_df, pass_df], ignore_index=True)


def test_compute_book_brier_skill_score_ignores_passed_offers_with_nan_win_prob():
    subset = _seeded_subset_with_passes(n_bet=100, n_pass=50, seed=42)
    bet_only = subset[subset["Bet"].notna()]
    expected = _reference_book_bss(bet_only)
    assert compute_book_brier_skill_score(subset) == pytest.approx(expected, abs=1e-6)


def test_compute_brier_skill_score_ignores_passed_offers_with_nan_win_prob():
    subset = _seeded_subset_with_passes(n_bet=100, n_pass=50, seed=99)
    bet_only = subset[subset["Bet"].notna()]
    expected = _reference_bss(bet_only)
    assert compute_brier_skill_score(subset) == pytest.approx(expected, abs=1e-6)


def test_compute_book_brier_skill_score_does_not_crash_when_passes_outnumber_bets():
    subset = _seeded_subset_with_passes(n_bet=20, n_pass=80, seed=7)
    assert math.isfinite(compute_book_brier_skill_score(subset))


def test_compute_brier_skill_score_does_not_crash_when_passes_outnumber_bets():
    subset = _seeded_subset_with_passes(n_bet=20, n_pass=80, seed=8)
    assert math.isfinite(compute_brier_skill_score(subset))
