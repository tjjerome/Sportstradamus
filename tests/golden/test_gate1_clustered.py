import numpy as np
import pandas as pd

from sportstradamus.training.scorecard import (
    DEFAULT_PRED_COL,
    _bootstrap_mean_ci,
    _bootstrap_mean_ci_clustered,
    load_test_set,
)


def test_load_test_set_keeps_player_and_date_when_present(tmp_path):
    csv = tmp_path / "NFL_passing-tds.csv"
    pd.DataFrame(
        {
            "MeanYr": [1.0, 2.0],
            "Result": [1, 0],
            "EV": [1.1, 1.9],
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
