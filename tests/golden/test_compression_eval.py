"""Unit tests for the Phase-0 compression eval harness.

Exercises the numeric path (decile binning, compression ratio, scorecard,
ship/kill verdict) on synthetic test-set frames so no trained model, network,
or plotting backend is required.
"""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.scripts.compression_eval import (
    MIN_TOP_DECILE_MAE_IMPROVEMENT,
    Scorecard,
    decile_table,
    load_test_set,
    scorecard,
    verdict,
)


def _base_card(**overrides) -> Scorecard:
    """Build a Scorecard with sensible defaults for verdict() unit tests."""
    defaults = dict(
        timestamp="2026-05-19T00:00:00+00:00",
        git_sha="deadbeef",
        strategy="t",
        league="NBA",
        market="PTS",
        pred_col="EV",
        n_rows=100,
        global_mae=1.0,
        top_decile_mae=2.0,
        top_decile_bias=-0.5,
        compression_ratio=0.5,
        top_decile_compression_ratio=0.4,
        pred_meanyr_corr=-0.5,
        result_meanyr_corr=0.0,
        brier_skill_score=None,
    )
    defaults.update(overrides)
    return Scorecard(**defaults)


def _compressed_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Build a frame whose predictions are shrunk toward the global mean.

    Actuals span a wide MeanYr range; predictions pull each row halfway to the
    grand mean — the canonical compression pathology.
    """
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2, 30, n)
    actual = meanyr + rng.normal(0, 3, n)
    grand = actual.mean()
    pred = grand + 0.5 * (actual - grand)
    return pd.DataFrame({"MeanYr": meanyr, "Result": actual, "EV": pred})


def test_decile_table_shape_and_monotone_bias():
    df = _compressed_frame()
    table = decile_table(df, "EV", n_deciles=10)
    assert len(table) == 10
    # Compression => top decile under-predicted (negative bias), bottom
    # decile over-predicted (positive bias).
    assert table.iloc[-1]["bias"] < 0
    assert table.iloc[0]["bias"] > 0


def test_compression_ratio_below_one_for_shrunk_predictions():
    card = scorecard(
        _compressed_frame(), "EV", strategy="t", league="NBA", market="PTS"
    )
    assert 0.45 < card.compression_ratio < 0.55
    assert card.top_decile_mae > 0
    assert card.top_decile_bias < 0


def test_perfect_predictions_have_unit_ratio():
    rng = np.random.default_rng(1)
    meanyr = rng.uniform(2, 30, 1000)
    df = pd.DataFrame({"MeanYr": meanyr, "Result": meanyr, "EV": meanyr})
    card = scorecard(df, "EV", strategy="t", league="NBA", market="PTS")
    assert card.compression_ratio == pytest.approx(1.0, abs=1e-9)
    assert card.global_mae == pytest.approx(0.0, abs=1e-9)


def test_verdict_ships_when_top_decile_improves():
    base = scorecard(
        _compressed_frame(seed=0), "EV", strategy="base", league="NBA", market="PTS"
    )
    # Candidate: predictions much closer to actual (less compression).
    df = _compressed_frame(seed=0)
    df["EV"] = df["Result"].mean() + 0.95 * (df["Result"] - df["Result"].mean())
    cand = scorecard(df, "EV", strategy="cand", league="NBA", market="PTS")
    ship, reason = verdict(base, cand)
    assert ship, reason


def test_verdict_kills_when_no_top_decile_gain():
    base = scorecard(
        _compressed_frame(seed=2), "EV", strategy="base", league="NBA", market="PTS"
    )
    ship, reason = verdict(base, base)
    assert not ship
    assert "KILL" in reason
    assert f"{MIN_TOP_DECILE_MAE_IMPROVEMENT:.0%}" in reason


def test_load_test_set_drops_nonfinite_and_validates_columns(tmp_path):
    good = pd.DataFrame(
        {"MeanYr": [1.0, 2.0, np.inf], "Result": [1.0, 2.0, 3.0], "EV": [1.0, 2.0, 3.0]}
    )
    p = tmp_path / "NBA_PTS.csv"
    good.to_csv(p, index=False)
    loaded = load_test_set(p, "EV")
    assert len(loaded) == 2

    bad = pd.DataFrame({"MeanYr": [1.0], "Result": [1.0]})
    bp = tmp_path / "NBA_AST.csv"
    bad.to_csv(bp, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_test_set(bp, "EV")


def test_load_test_set_keeps_optional_columns_when_present(tmp_path):
    df = pd.DataFrame(
        {
            "MeanYr": [10.0, 12.0, 14.0],
            "Result": [11.0, 13.0, 15.0],
            "EV": [10.5, 12.5, 14.5],
            "P": [0.55, 0.60, 0.50],
            "Odds": [0.45, 0.40, 0.50],
            "Line": [10.0, 12.0, 14.0],
        }
    )
    p = tmp_path / "NBA_PTS.csv"
    df.to_csv(p, index=False)
    loaded = load_test_set(p, "EV")
    assert {"P", "Odds", "Line"}.issubset(loaded.columns)
    assert len(loaded) == 3


def test_load_test_set_handles_missing_optional_columns(tmp_path):
    df = pd.DataFrame(
        {"MeanYr": [10.0, 12.0], "Result": [11.0, 13.0], "EV": [10.5, 12.5]}
    )
    p = tmp_path / "NBA_PTS.csv"
    df.to_csv(p, index=False)
    loaded = load_test_set(p, "EV")
    card = scorecard(loaded, "EV", strategy="t", league="NBA", market="PTS")
    assert card.brier_skill_score is None


def test_brier_skill_score_positive_when_model_beats_book():
    rng = np.random.default_rng(0)
    n = 4000
    meanyr = rng.uniform(2, 30, n)
    line = meanyr.copy()
    # True over rate aligned tightly with model probability; book is near-random.
    p_true = rng.uniform(0.05, 0.95, n)
    outcomes = rng.uniform(size=n) < p_true
    result = np.where(outcomes, line + 1.0, line - 1.0)
    df = pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Result": result,
            "EV": meanyr,
            "Line": line,
            "P": p_true,  # model nails it
            "Odds": np.full(n, 0.5),  # book is 50/50
        }
    )
    card = scorecard(df, "EV", strategy="t", league="NBA", market="PTS")
    assert card.brier_skill_score is not None
    assert card.brier_skill_score > 0.1


def test_brier_skill_score_negative_when_book_beats_model():
    rng = np.random.default_rng(1)
    n = 4000
    meanyr = rng.uniform(2, 30, n)
    line = meanyr.copy()
    p_true = rng.uniform(0.05, 0.95, n)
    outcomes = rng.uniform(size=n) < p_true
    result = np.where(outcomes, line + 1.0, line - 1.0)
    df = pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Result": result,
            "EV": meanyr,
            "Line": line,
            # Model is anti-correlated noise; book is the true probability so
            # book_over = 1 - Odds nails it.
            "P": 1.0 - p_true,
            "Odds": 1.0 - p_true,
        }
    )
    card = scorecard(df, "EV", strategy="t", league="NBA", market="PTS")
    assert card.brier_skill_score is not None
    assert card.brier_skill_score < 0


def test_verdict_kill_on_brier_skill_regression():
    # MAE gates pass (candidate improves top-decile MAE and global MAE), but
    # brier_skill_score regresses — third gate must fire.
    base = _base_card(
        global_mae=1.0, top_decile_mae=2.0, brier_skill_score=0.10
    )
    cand = _base_card(
        global_mae=0.9, top_decile_mae=1.5, brier_skill_score=0.05
    )
    ship, reason = verdict(base, cand)
    assert not ship
    assert "KILL" in reason
    assert "brier_skill_score regressed" in reason


def test_verdict_ship_includes_brier_skill_when_present():
    base = _base_card(
        global_mae=1.0, top_decile_mae=2.0, brier_skill_score=0.05
    )
    cand = _base_card(
        global_mae=0.9, top_decile_mae=1.5, brier_skill_score=0.10
    )
    ship, reason = verdict(base, cand)
    assert ship, reason
    assert "brier_skill" in reason


def test_verdict_skips_brier_skill_gate_when_either_baseline_or_candidate_lacks_it():
    # Baseline has no brier; candidate has a (worse) value — gate must skip,
    # MAE gates alone decide. MAE improves so SHIP.
    base = _base_card(
        global_mae=1.0, top_decile_mae=2.0, brier_skill_score=None
    )
    cand = _base_card(
        global_mae=0.9, top_decile_mae=1.5, brier_skill_score=-0.50
    )
    ship, reason = verdict(base, cand)
    assert ship, reason
    assert "brier_skill" not in reason

    # And symmetric: candidate None.
    base2 = _base_card(
        global_mae=1.0, top_decile_mae=2.0, brier_skill_score=0.50
    )
    cand2 = _base_card(
        global_mae=0.9, top_decile_mae=1.5, brier_skill_score=None
    )
    ship2, reason2 = verdict(base2, cand2)
    assert ship2, reason2
    assert "brier_skill" not in reason2
