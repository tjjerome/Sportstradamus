"""Golden tests for the Kelly-sized profit-sim used by the ship gate.

Phase 2 of the new ship-gate plan ([docs/ship_gate.md](../../docs/ship_gate.md)).
The behavior under test is the dashboard's Monte-Carlo sim lifted to
``strategies/profit_sim.py``; the page now imports from this module, so
these tests pin its contract and guard against drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.strategies.profit_sim import (
    N_MONTE_CARLO_DEFAULT,
    RANKING_MAP,
    compute_payout,
    simulate_kelly_all,
    simulate_strategy,
    summarize_runs,
)

# --------------------------------------------------------------------------- #
# Fixtures


def _make_offers(n_days: int = 3, hit_pattern: list[bool] | None = None) -> pd.DataFrame:
    """Build a tiny exploded-offer frame with deterministic structure.

    One player per day, Model P=0.7, Books=0.55, Boost=1.0, Platform=Underdog.
    ``hit_pattern`` (if given) defines the per-day Hit outcome — defaults to
    alternating True/False.
    """
    if hit_pattern is None:
        hit_pattern = [(i % 2) == 0 for i in range(n_days)]
    rows = []
    for i in range(n_days):
        rows.append(
            {
                "Player": f"P{i}",
                "Market": "points",
                "Platform": "Underdog",
                "Boost": 1.0,
                "Win Prob": 0.7,
                "Model EV": 0.7,
                "Kelly": 0.05,
                "Market EV": 0.55,
                "Hit": hit_pattern[i],
                "_date": pd.Timestamp(f"2026-01-{i + 1:02d}").date(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# compute_payout


def test_payout_underdog_minus_110():
    row = pd.Series({"Platform": "Underdog", "Boost": 1.0})
    assert compute_payout(row) == pytest.approx(100 / 110)


def test_payout_sleeper_returns_boost():
    row = pd.Series({"Platform": "Sleeper", "Boost": 1.0})
    assert compute_payout(row) == pytest.approx(1.0)


def test_payout_unknown_defaults_minus_110():
    row = pd.Series({"Platform": "DraftKings", "Boost": 2.0})
    # default branch applies -110 then multiplies by boost
    assert compute_payout(row) == pytest.approx((100 / 110) * 2.0)


# --------------------------------------------------------------------------- #
# RANKING_MAP


def test_ranking_map_keys_pinned():
    # Pinned because the page's UI selectbox surfaces these strings;
    # changing them is a UI-facing change, not a refactor.
    assert set(RANKING_MAP.keys()) == {"Kelly", "Probability", "EV"}
    assert RANKING_MAP["Kelly"] == "Kelly"
    assert RANKING_MAP["Probability"] == "Win Prob"
    assert RANKING_MAP["EV"] == "Model EV"


# --------------------------------------------------------------------------- #
# simulate_strategy


def test_empty_input_returns_empty():
    df = pd.DataFrame(
        columns=["Player", "Market", "Platform", "Boost", "Win Prob", "Market EV", "Hit", "_date", "Kelly"]
    )
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.5,
        max_bets_day=5,
        sizing_pct=1.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    assert result.empty


def test_threshold_filter_yields_empty_when_unmet():
    df = _make_offers(n_days=2)
    # min_model_p = 0.99 → no row qualifies
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.99,
        min_books_p=0.0,
        max_bets_day=5,
        sizing_pct=1.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    assert result.empty


def test_flat_sizing_one_run_columns_and_progression():
    df = _make_offers(n_days=3, hit_pattern=[True, False, True])
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=10.0,  # 10% of bankroll per bet
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    assert list(result.columns) == ["date", "run", "bankroll", "daily_pnl"]
    assert len(result) == 3
    # Day 1: stake = 100, payout = 100 * (100/110) ≈ 90.91 → bankroll ≈ 1090.91
    payout = 100 / 110
    expected_d1 = 1000.0 + 100.0 * payout
    assert result.iloc[0]["bankroll"] == pytest.approx(expected_d1)
    # Day 2: stake = 109.09, loss → bankroll = 1090.91 - 109.09 ≈ 981.82
    expected_d2 = expected_d1 - expected_d1 * 0.10
    assert result.iloc[1]["bankroll"] == pytest.approx(expected_d2)


def test_kelly_sizing_caps_at_five_percent():
    # Sleeper with Boost=2 → payout=2, b=1; p=0.99 → kelly_f = (1*0.99 - 0.01)/1
    # = 0.98 raw, capped at 0.05. Lost bet → bankroll = 1000 - 50 = 950.
    df = _make_offers(n_days=1, hit_pattern=[False])
    df.loc[0, "Platform"] = "Sleeper"
    df.loc[0, "Boost"] = 2.0
    df.loc[0, "Win Prob"] = 0.99
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=0.0,
        use_kelly=True,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    assert result.iloc[0]["bankroll"] == pytest.approx(950.0)
    assert result.iloc[0]["daily_pnl"] == pytest.approx(-50.0)


def test_kelly_skips_minus_110_payout():
    # Dashboard behavior preserved: payout = 100/110 ≤ 1 triggers the
    # early-exit in the Kelly branch (the formula treats payout as net
    # odds; with -110 net odds < 1, the sim skips the leg). simulate_kelly_all
    # therefore needs payouts > 1 to bet — Phase 3 (S3) feeds Odds-derived
    # payouts directly rather than the dashboard's exploded-history Boost.
    df = _make_offers(n_days=1, hit_pattern=[True])
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=0.0,
        use_kelly=True,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    assert result.iloc[0]["bankroll"] == pytest.approx(1000.0)
    assert result.iloc[0]["daily_pnl"] == pytest.approx(0.0)


def test_kelly_skips_payout_le_one():
    # If Payout <= 1, Kelly skips the bet entirely (avoids divide-by-zero
    # on b = payout - 1).
    df = _make_offers(n_days=1, hit_pattern=[True])
    df.loc[0, "Platform"] = "Sleeper"
    df.loc[0, "Boost"] = 1.0  # Sleeper Boost=1 → Payout=1, skip
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=0.0,
        use_kelly=True,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    # Bet skipped → bankroll unchanged
    assert result.iloc[0]["bankroll"] == pytest.approx(1000.0)
    assert result.iloc[0]["daily_pnl"] == pytest.approx(0.0)


def test_monte_carlo_seed_reproducibility():
    # Two runs with the same default seed produce identical bankroll
    # trajectories — the sim is reproducible.
    df = _make_offers(n_days=4)
    r1 = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=1.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=3,
    )
    r2 = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=1.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=3,
    )
    pd.testing.assert_frame_equal(r1, r2)


def test_caller_rng_overrides_default_seed():
    # Supplying an rng explicitly is honored — the same rng over two calls
    # yields identical results, and the path that consumes rng (
    # ``len(day_bets) > max_bets_day``) does in fact get invoked.
    rows = []
    for i in range(20):
        rows.append(
            {
                "Player": f"P{i}",
                "Market": "points",
                "Platform": "Underdog",
                "Boost": 1.0,
                "Win Prob": 0.6 + 0.01 * i,
                "Model EV": 0.6 + 0.01 * i,
                "Kelly": 0.05 + 0.01 * i,
                "Market EV": 0.55,
                "Hit": (i % 3) == 0,
                "_date": pd.Timestamp("2026-01-01").date(),
            }
        )
    df = pd.DataFrame(rows)
    r1 = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.0,
        min_books_p=0.0,
        max_bets_day=3,
        sizing_pct=1.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=1,
        rng=np.random.default_rng(123),
    )
    r2 = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.0,
        min_books_p=0.0,
        max_bets_day=3,
        sizing_pct=1.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=1,
        rng=np.random.default_rng(123),
    )
    pd.testing.assert_frame_equal(r1, r2)


# --------------------------------------------------------------------------- #
# simulate_kelly_all


def test_kelly_all_is_deterministic():
    df = _make_offers(n_days=3, hit_pattern=[True, False, True])
    r1 = simulate_kelly_all(df, prob_col="Win Prob", initial_bankroll=1000.0)
    r2 = simulate_kelly_all(df, prob_col="Win Prob", initial_bankroll=1000.0)
    pd.testing.assert_frame_equal(r1, r2)


def test_kelly_all_uses_kelly_sizing_no_thresholds():
    # Sleeper Boost=2 (payout=2 > 1) with low p=0.51 → Kelly returns negative
    # ((1 * 0.51 - 0.49) / 1 = 0.02, capped above 0 — that's a tiny stake).
    # Make p=0.40 → (0.40 - 0.60) / 1 = -0.20 → clamp to 0 → no bet placed.
    df = _make_offers(n_days=1, hit_pattern=[True])
    df.loc[0, "Platform"] = "Sleeper"
    df.loc[0, "Boost"] = 2.0
    df.loc[0, "Win Prob"] = 0.40
    result = simulate_kelly_all(df, prob_col="Win Prob", initial_bankroll=1000.0)
    assert result.iloc[0]["bankroll"] == pytest.approx(1000.0)


def test_kelly_all_runs_one_mc_pass():
    df = _make_offers(n_days=5)
    result = simulate_kelly_all(df, prob_col="Win Prob", initial_bankroll=1000.0)
    # n_mc=1 → exactly len(dates) rows, all run==0
    assert len(result) == 5
    assert (result["run"] == 0).all()


# --------------------------------------------------------------------------- #
# summarize_runs


def test_summarize_empty_returns_zero_metrics():
    summary = summarize_runs(pd.DataFrame(), initial_bankroll=1000.0)
    assert summary == {
        "mean_final": 1000.0,
        "roi": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "win_rate": 0.0,
    }


def test_summarize_winning_run_positive_roi():
    df = _make_offers(n_days=3, hit_pattern=[True, True, True])
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=10.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    summary = summarize_runs(result, initial_bankroll=1000.0)
    # 3 winning bets at 10% flat sizing on -110 → mean_final > 1000
    assert summary["roi"] > 0
    assert summary["win_rate"] == pytest.approx(1.0)
    # No drawdown on a monotonically increasing bankroll
    assert summary["max_drawdown"] == pytest.approx(0.0)


def test_summarize_losing_run_negative_roi_and_drawdown():
    df = _make_offers(n_days=3, hit_pattern=[False, False, False])
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=10.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    summary = summarize_runs(result, initial_bankroll=1000.0)
    assert summary["roi"] < 0
    # Peak was initial_bankroll; trough is final bankroll → drawdown > 0
    assert summary["max_drawdown"] > 0
    assert summary["win_rate"] == pytest.approx(0.0)


def test_summarize_sharpe_finite_on_mixed_outcomes():
    df = _make_offers(n_days=4, hit_pattern=[True, False, True, False])
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=5.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=1,
    )
    summary = summarize_runs(result, initial_bankroll=1000.0)
    assert np.isfinite(summary["sharpe"])


def test_summarize_default_n_mc_matches_constant():
    # The dashboard default is 100; preserve that constant so any change
    # to it is a visible diff.
    assert N_MONTE_CARLO_DEFAULT == 100


def test_summarize_aggregates_across_runs():
    df = _make_offers(n_days=2, hit_pattern=[True, False])
    result = simulate_strategy(
        df,
        prob_col="Win Prob",
        ranking="Kelly",
        min_model_p=0.5,
        min_books_p=0.0,
        max_bets_day=10,
        sizing_pct=1.0,
        use_kelly=False,
        initial_bankroll=1000.0,
        n_mc=5,
    )
    summary = summarize_runs(result, initial_bankroll=1000.0)
    # mean_final is the average across the 5 runs' last bankrolls
    final_runs = result.groupby("run").last()
    assert summary["mean_final"] == pytest.approx(float(final_runs["bankroll"].mean()))
