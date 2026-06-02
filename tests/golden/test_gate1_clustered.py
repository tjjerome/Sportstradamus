import pandas as pd

from sportstradamus.training.scorecard import DEFAULT_PRED_COL, load_test_set


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
