import numpy as np
import pandas as pd
import pytest

from sportstradamus.training.scorecard import (
    _GATE1_NONINF_MARGIN,
    _GATE1_SEED,
    DEFAULT_PRED_COL,
    _bootstrap_mean_ci,
    _bootstrap_mean_ci_clustered,
    _gate1_brier_ci,
    apply_thresholds,
    gate_row,
    load_test_set,
)


def test_load_test_set_keeps_player_and_date_when_present(tmp_path):
    csv = tmp_path / "NFL_passing-tds.csv"
    pd.DataFrame(
        {
            "MeanYr": [1.0, 2.0],
            "Result": [1, 0],
            "EV": [1.1, 1.9],
            "Blended_EV": [1.1, 1.9],
            "P": [0.6, 0.4],
            "Odds": [0.5, 0.5],
            "Line": [1.5, 1.5],
            "Player": ["A", "B"],
            "Date": ["2025-09-07", "2025-09-07"],
        }
    ).to_csv(csv, index=False)
    df = load_test_set(csv, DEFAULT_PRED_COL)
    assert "Player" in df.columns and "Date" in df.columns


def test_load_test_set_keeps_date_when_player_absent(tmp_path):
    csv = tmp_path / "NFL_team_market.csv"
    pd.DataFrame(
        {
            "MeanYr": [1.0, 2.0],
            "Result": [1, 0],
            "EV": [1.1, 1.9],
            "Blended_EV": [1.1, 1.9],
            "P": [0.6, 0.4],
            "Odds": [0.5, 0.5],
            "Line": [1.5, 1.5],
            "Date": ["2025-09-07", "2025-09-07"],
        }
    ).to_csv(csv, index=False)
    df = load_test_set(csv, DEFAULT_PRED_COL)
    assert "Player" not in df.columns
    assert "Date" in df.columns
    assert len(df) == 2  # string Date must not cause non-finite row drops


def test_clustered_ci_is_wider_on_correlated_panel():
    # 40 players x 25 games; the per-event statistic is identical within a player
    # (max within-player correlation). The clustered bootstrap must yield a wider
    # CI than the i.i.d. one, which ignores the correlation and over-credits.
    rng = np.random.default_rng(7)
    player_means = rng.normal(0.0, 0.02, size=40)
    players = np.repeat(np.arange(40), 25)
    values = np.repeat(player_means, 25)  # constant within player
    _, lo_i, hi_i = _bootstrap_mean_ci(values, np.random.default_rng(1))
    _, lo_c, hi_c = _bootstrap_mean_ci_clustered(values, players, np.random.default_rng(1))
    assert (hi_c - lo_c) > (hi_i - lo_i)


def test_gate1_ship_bound_resamples_game_dates():
    # Six slates whose rows go over or under together (a shared game-level shock). The
    # model leans Over and the book Under on every row, so a slate's Brier difference
    # moves with its over rate. Resampling the 1,200 rows as independent clears the tie
    # margin; resampling the six dates does not. The oracle and pre-blend bounds resample
    # the same dates.
    rows_per_date = 200
    over_rates = [0.72, 0.66, 0.58, 0.46, 0.38, 0.32]
    over = np.concatenate([np.arange(rows_per_date) < rate * rows_per_date for rate in over_rates])
    n = len(over)
    dates = pd.date_range("2026-09-13", periods=len(over_rates), freq="7D").strftime("%Y-%m-%d")
    df = pd.DataFrame(
        {
            "MeanYr": np.linspace(5.0, 15.0, n),
            "Result": np.where(over, 12.0, 9.0),
            "EV": np.full(n, 10.5),
            "Line": np.full(n, 10.5),
            "P": np.full(n, 0.55),
            "P_standalone": np.full(n, 0.55),
            "Odds": np.full(n, 0.55),
            "Date": np.repeat(dates, rows_per_date),
        }
    )
    y = over.astype(float)
    _, _, iid_hi = _bootstrap_mean_ci(
        (0.55 - y) ** 2 - (0.45 - y) ** 2, np.random.default_rng(_GATE1_SEED)
    )
    assert iid_hi < _GATE1_NONINF_MARGIN

    row = apply_thresholds(gate_row(df, "EV", league="NFL", market="yards", strategy="t"))
    assert row["g1_brier_diff_ci_hi"] > _GATE1_NONINF_MARGIN
    assert not row["g1_pass"]
    assert row["g1_brier_diff_ci_hi_standalone"] == row["g1_brier_diff_ci_hi"]
    _, oracle_iid_lo, oracle_iid_hi = _bootstrap_mean_ci(
        -((0.45 - y) ** 2), np.random.default_rng(_GATE1_SEED)
    )
    oracle_width = row["g1_brier_diff_ci_hi_oracle"] - row["g1_brier_diff_ci_lo_oracle"]
    assert oracle_width > 2 * (oracle_iid_hi - oracle_iid_lo)


def test_gate1_ship_bound_never_narrower_than_rows():
    # Three slates that mix alike: every date holds the same ten wins and ten losses for
    # the model, so each date's mean difference is the same +.004 and resampling dates
    # can only return it. That collapsed bound would pass the tie margin; the rows'
    # spread says the cell has not shown it, and the row bound holds the gate shut.
    d_win, d_loss = -0.1, 0.108
    per_date = np.repeat([d_win, d_loss], 10)
    y = np.ones(3 * len(per_date))
    p_book = np.full_like(y, 0.5)
    p_model = 1 - np.sqrt(0.25 + np.tile(per_date, 3))
    dates = np.repeat(["2026-09-13", "2026-09-20", "2026-09-27"], len(per_date))
    d = (p_model - y) ** 2 - (p_book - y) ** 2
    _, _, date_hi = _bootstrap_mean_ci_clustered(d, dates, np.random.default_rng(_GATE1_SEED))
    assert date_hi < _GATE1_NONINF_MARGIN

    mean, _lo, hi = _gate1_brier_ci(p_model, p_book, y, dates)

    assert mean == pytest.approx(0.004)
    assert hi > _GATE1_NONINF_MARGIN
