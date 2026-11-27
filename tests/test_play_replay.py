"""``admin play-replay``: the money tables over a window of a history and its parlay history."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus import live_fit, plays
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
# Two Sleeper picks. The board recommends the first (.65 x 1.7 clears its edge floor) and not
# the second, which the app's own prices value higher: 1 / (1/1.6 + 1/2.2) = .926 against
# 1 / (1/1.7 + 1/2.0) = .919.
_SLEEPER = _OFFER | {"Platform": "Sleeper"}
_RULE_PICK = _SLEEPER | {
    "Player": "rule",
    "Boost": 1.7,
    "Win Prob": 0.65,
    "Payout Over": 1.7,
    "Payout Under": 2.0,
}
_RANK_PICK = _SLEEPER | {
    "Player": "rank",
    "Boost": 1.6,
    "Win Prob": 0.55,
    "Payout Over": 1.6,
    "Payout Under": 2.2,
}


def _replay(tmp_path, history: pd.DataFrame, *flags: str):
    """Run the replay over ``history`` and one day of entries; the result and its out directory."""
    history.to_parquet(tmp_path / "history.parquet")
    entries = tmp_path / "parlay_hist"
    entries.mkdir()
    pd.DataFrame([_ENTRY]).to_parquet(entries / "2026-11-01.parquet")
    args = ["--history", str(tmp_path / "history.parquet"), "--entries", str(entries)]
    result = CliRunner().invoke(play_replay.main, [*args, "--out", str(tmp_path / "out"), *flags])
    return result, tmp_path / "out"


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


def test_walk_forward_scores_the_walked_dates_and_prints_the_fit_check(tmp_path):
    train = [_OFFER | {"Date": f"2026-10-{day}"} for day in range(10, 24)]
    # Both picks hit on the 24th and miss on the 25th.
    walked = [
        pick | {"Date": date, "Actual": actual}
        for date, actual in (("2026-10-24", 4.0), ("2026-10-25", 1.0))
        for pick in (_RULE_PICK, _RANK_PICK)
    ]
    # A scoring run's own fit columns and pick flags, which the walk replaces: kept, they mark
    # every leg "+EV" and call every leg a pick.
    stale = {"Fit Prob": 0.99, "Fit Prob Low": 0.99, "EV": 9.9, "Pick": True}
    history = pd.DataFrame(train + walked, columns=HISTORY_COLS).assign(
        **{"Book Prob": np.nan, "Book Class": "C"}, **stale
    )

    result, out = _replay(tmp_path, history, "--walk-forward")

    assert result.exit_code == 0, result.output
    legs = pd.read_parquet(out / "play_record.parquet").set_index(["play", "league", "kind"])
    every = legs.loc[("all", "all", "all")]
    assert (every["n"], every["first_date"], every["last_date"]) == (8, "2026-10-24", "2026-10-25")
    assert legs.at[("model_edge", "all", "all"), "roi"] == pytest.approx((0.7 - 1) / 2)
    assert legs.at[("ev_rank", "all", "all"), "roi"] == pytest.approx((0.6 - 1) / 2)
    # One team on each slate, so one place: the pick the app's prices value higher.
    assert legs.at[("ev_pick", "all", "all"), "roi"] == pytest.approx((0.6 - 1) / 2)
    assert "ev_edge" not in legs.index.get_level_values("play")

    # Too few legs for a fit, so each leg is priced at the app's own fair price: no gain.
    gain = pd.read_parquet(out / "fit_gain.parquet").set_index(["league", "kind"])
    assert gain.loc[("all", "all"), ["n_legs", "n_dates"]].tolist() == [4, 2]
    assert gain.at[("all", "all"), "gain"] == pytest.approx(0.0)
    bands = pd.read_parquet(out / "ev_bands.parquet").set_index("band")
    assert bands["n_legs"].to_dict() == {
        "[0.0, 0.9)": 0,
        "[0.9, 0.94)": 4,
        "[0.94, 0.97)": 0,
        "[0.97, 1.0)": 0,
        "[1.0, inf)": 0,
    }
    said = [1 / (1 / 1.7 + 1 / 2.0), 1 / (1 / 1.6 + 1 / 2.2)]
    assert bands.at["[0.9, 0.94)", "says"] == pytest.approx(np.mean(said))
    assert bands.at["[0.9, 0.94)", "paid"] == pytest.approx((1.7 + 1.6) / 4)

    assert "x 10,000" in result.output
    assert "Legs the lower bound clears (Fit Prob Low x payout at least 1): 0 of 4" in result.output
    # The plays win and lose on the same dates, so a date drawn for both leaves a narrow gap:
    # -.1 on two 24ths, 0 on two 25ths. Drawn apart, the gap would run from -1.7 to +1.6.
    assert "ev_rank minus model_edge, return a leg: -0.050 [-0.100, +0.000]" in result.output
    assert "ev_pick minus model_edge, return a leg: -0.050 [-0.100, +0.000]" in result.output


def test_the_fit_check_marks_by_the_lower_bound_and_prints_the_gain_scaled(tmp_path, monkeypatch):
    # Three Sleeper picks at 2.0 a side, which the app prices at one half; the model took each
    # Over and it hit. The walk is stood in for, to set what the fit said of each.
    pick = _SLEEPER | {"Boost": 2.0, "Payout Over": 2.0, "Payout Under": 2.0}
    history = pd.DataFrame([pick | {"Player": name} for name in "ABC"], columns=HISTORY_COLS)
    history = history.assign(**{"Book Prob": np.nan, "Book Class": "C"})
    said = {"Fit Prob": 0.6, "Fit Prob Low": [0.55, 0.5, 0.45], "EV": 1.2}
    monkeypatch.setattr(live_fit, "walk_forward", lambda offers: offers.assign(**said))

    result, out = _replay(tmp_path, history, "--walk-forward")

    assert result.exit_code == 0, result.output
    # A lower bound of .55 or .5 pays 1 or more at 2.0; .45 does not, though the fit's .6 would.
    assert "Legs the lower bound clears (Fit Prob Low x payout at least 1): 2 of 3" in result.output
    gain = pd.read_parquet(out / "fit_gain.parquet").set_index(["league", "kind"])
    assert gain.at[("all", "all"), "gain"] == pytest.approx(np.log(0.6 / 0.5))
    assert "+1823.2" in result.output


def test_a_play_added_to_the_registry_moves_no_other_tables_bounds(tmp_path, monkeypatch):
    # Entries on eight dates, each at its own payout and missed on the odd ones, so a bound
    # depends on the dates drawn.
    days = [f"2026-11-{day:02d}" for day in range(1, 9)]
    history = pd.DataFrame([_OFFER | {"Date": day} for day in days], columns=HISTORY_COLS)
    history.to_parquet(tmp_path / "history.parquet")
    entries = tmp_path / "parlay_hist"
    entries.mkdir()
    for number, day in enumerate(days):
        graded = [_ENTRY | {"Date": day, "Misses": float(number % 2), "Boost": 3 + 0.37 * number}]
        pd.DataFrame(graded).to_parquet(entries / f"{day}.parquet")
    args = ["--history", str(tmp_path / "history.parquet"), "--entries", str(entries)]

    def entry_table(out: str) -> pd.DataFrame:
        result = CliRunner().invoke(play_replay.main, [*args, "--out", str(tmp_path / out)])
        assert result.exit_code == 0, result.output
        return pd.read_parquet(tmp_path / out / "entry_record.parquet")

    before = entry_table("before")
    monkeypatch.setitem(plays.PLAYS, "overs", lambda sides: sides["Bet"].eq("Over"))

    pd.testing.assert_frame_equal(entry_table("after"), before)
    assert before["roi_lo"].iloc[0] < before["roi_hi"].iloc[0]


def test_walk_forward_refuses_a_history_without_the_book_stamp(tmp_path):
    history = pd.DataFrame([_OFFER], columns=HISTORY_COLS).drop(
        columns="Book Class", errors="ignore"
    )

    result, out = _replay(tmp_path, history, "--walk-forward")

    assert result.exit_code == 2
    assert result.output.splitlines()[-1].startswith("Error: ")
    assert "Book Class" in result.output.splitlines()[-1]
    assert not out.exists()
