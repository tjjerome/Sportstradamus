"""``admin play-replay``: the money tables over a window of a history and its parlay history."""

import pandas as pd
from click.testing import CliRunner

from sportstradamus.history_schema import HISTORY_COLS
from sportstradamus.scripts import play_replay

# A settled Underdog even pick the model took Over at a .6 claim, and won.
_OFFER = {
    "Player": "A",
    "League": "NHL",
    "Team": "BOS",
    "Opponent": "NYR",
    "Date": "2026-11-01",
    "Market": "shots",
    "Line": 2.5,
    "Boost": 1.0,
    "Platform": "Underdog",
    "Bet": "Over",
    "Win Prob": 0.6,
    "Payout Over": 1.83,
    "Payout Under": 1.83,
    "Actual": 4.0,
}
# An Underdog 2-pick NFL entry, every leg graded, none missed.
_ENTRY = {
    "Date": "2026-11-01",
    "League": "NFL",
    "Platform": "Underdog",
    "Bet Size": 2,
    "Boost": 3.5,
    "Model EV": 2.1,
    "Legs Resolved": 2.0,
    "Misses": 0.0,
}


def test_play_replay_scores_the_window_and_writes_both_tables(tmp_path):
    history = tmp_path / "history.parquet"
    pd.DataFrame(
        [
            _OFFER,
            _OFFER | {"Player": "B", "Date": "2026-11-03"},  # the window's last day counts
            _OFFER | {"Player": "C", "Date": "2026-11-04"},
            _OFFER | {"Player": "D", "Date": "2026-10-31"},
        ],
        columns=HISTORY_COLS,
    ).to_parquet(history)
    entries = tmp_path / "parlay_hist"
    entries.mkdir()
    pd.DataFrame([_ENTRY, _ENTRY | {"Misses": 1.0}]).to_parquet(entries / "2026-11-01.parquet")
    pd.DataFrame([_ENTRY | {"Date": "2026-11-04"}]).to_parquet(entries / "2026-11-04.parquet")
    out = tmp_path / "out"

    result = CliRunner().invoke(
        play_replay.main,
        [
            *("--history", str(history)),
            *("--entries", str(entries)),
            *("--since", "2026-11-01"),
            *("--until", "2026-11-03"),
            *("--out", str(out)),
        ],
    )

    assert result.exit_code == 0, result.output
    legs = pd.read_parquet(out / "play_record.parquet").set_index(["play", "league", "kind"])
    assert legs.at[("model_edge", "NHL", "underdog_even"), "n"] == 2
    assert legs.at[("all", "all", "all"), "n"] == 4
    graded = pd.read_parquet(out / "entry_record.parquet")
    assert graded[["league", "size", "n", "hit_rate"]].to_dict("records") == [
        {"league": "NFL", "size": 2, "n": 2, "hit_rate": 0.5}
    ]
    assert "model_edge" in result.output
    assert "NFL" in result.output
