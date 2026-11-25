"""The money table: posted sides, the play rules and the exact cross-game entry return."""

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from sportstradamus import plays
from sportstradamus.helpers import underdog_payouts


def _offer(**fields) -> dict:
    """One ``realized.settled_offers`` row: an Underdog even pick the model took Over, and won."""
    return {
        "Date": "2026-11-01",
        "League": "NHL",
        "Player": "A",
        "Team": "BOS",
        "Opponent": "NYR",
        "Platform": "Underdog",
        "Bet": "Over",
        "Result": "Over",
        "Boost": 1.0,
        "Payout Over": 1.83,
        "Payout Under": 1.83,
        "Win Prob": 0.6,
        "Recommended": True,
    } | fields


def test_posted_sides_grades_the_other_side_at_its_own_payout():
    offers = pd.DataFrame(
        [_offer(Boost=1.2, Result="Under", **{"Payout Over": 2.196, "Payout Under": 1.5006})]
    )
    chosen, other = plays.posted_sides(offers).to_dict("records")

    assert (chosen["Side"], chosen["Kind"]) == ("Over", "underdog_raised")
    assert chosen["Payout"] == pytest.approx(2.196)
    assert (chosen["Hit"], chosen["Unit"], chosen["Recommended"]) == (False, -1.0, True)
    assert chosen["Win Prob"] == pytest.approx(0.6)

    assert (other["Side"], other["Kind"]) == ("Under", "underdog_cut")
    assert other["Multiplier"] == pytest.approx(0.82)
    assert other["Payout"] == pytest.approx(1.5006)
    assert other["Breakeven"] == pytest.approx(1 / 1.5006)
    assert (other["Hit"], other["Recommended"]) == (True, False)
    assert other["Unit"] == pytest.approx(0.5006)
    assert other["Win Prob"] == pytest.approx(0.4)


@pytest.mark.parametrize("unposted", [0.0, np.nan])
def test_posted_sides_leaves_out_a_side_the_app_never_posted(unposted):
    offers = pd.DataFrame([_offer(**{"Payout Under": unposted})])
    sides = plays.posted_sides(offers)
    assert list(sides["Side"]) == ["Over"]


def test_posted_sides_names_a_game_the_same_from_either_team():
    offers = pd.DataFrame([_offer(), _offer(Player="B", Team="NYR", Opponent="BOS")])
    assert plays.posted_sides(offers)["Game"].nunique() == 1


def test_play_rules_pick_their_sides():
    offers = pd.DataFrame(
        [
            _offer(),
            _offer(Player="B", Recommended=False),
            _offer(Player="C", Boost=1.2, **{"Payout Over": 2.196, "Payout Under": 1.5006}),
        ]
    )
    sides = plays.posted_sides(offers)
    picked = {name: sides[rule(sides)] for name, rule in plays.PLAYS.items()}

    assert list(picked["model_edge"]["Player"]) == ["A", "C"]
    assert picked["model_edge"]["Side"].eq("Over").all()
    assert sorted(picked["under"]["Player"]) == ["A", "B"]
    assert picked["under"]["Side"].eq("Under").all()
    assert len(picked["all"]) == len(sides) == 6


def test_power_entry_returns_equal_brute_force_over_every_one_leg_per_game_entry():
    rng = np.random.default_rng(7)
    n = 12
    legs = pd.DataFrame(
        {
            "Date": "2026-11-01",
            "Platform": "Underdog",
            "Game": rng.choice(["g1", "g2", "g3", "g4", "g5"], n),
            "Multiplier": rng.choice([0.82, 1.0, 1.2], n),
            "Hit": rng.random(n) < 0.6,
        }
    )
    returns = plays.power_entry_returns(legs).set_index("size")

    for size in plays.POWER_ENTRY_SIZES:
        entries = [
            list(combo)
            for combo in combinations(range(n), size)
            if legs["Game"].iloc[list(combo)].nunique() == size
        ]
        paid = [
            underdog_payouts["power"][size]
            * math.prod(legs["Multiplier"].iloc[entry])
            * legs["Hit"].iloc[entry].all()
            for entry in entries
        ]
        assert returns.at[size, "entries"] == len(entries)
        assert returns.at[size, "roi"] == pytest.approx(np.mean(paid) - 1)


def test_both_sides_of_even_picks_return_the_payout_table_over_two_to_the_size():
    rng = np.random.default_rng(3)
    offers = pd.DataFrame(
        [
            _offer(
                Player=f"P{i}",
                Team=f"T{i % 6}",
                Opponent=f"O{i % 6}",
                Result=rng.choice(["Over", "Under"]),
            )
            for i in range(20)
        ]
    )
    returns = plays.power_entry_returns(plays.posted_sides(offers)).set_index("size")["roi"]

    assert returns[2] == pytest.approx(-0.125)
    assert returns[3] == pytest.approx(-0.1875)
    assert returns[4] == pytest.approx(-0.25)


def test_power_entry_returns_skip_a_size_the_platform_has_no_power_table_for():
    legs = pd.DataFrame(
        {
            "Date": "2026-11-01",
            "Platform": "Sleeper",
            "Game": ["g1", "g2", "g3", "g4"],
            "Multiplier": [1.5, 1.6, 1.7, 1.8],
            "Hit": True,
        }
    )
    returns = plays.power_entry_returns(legs).set_index("size")

    assert list(returns.index) == [2, 3]
    # Sleeper pays the product of the legs' own payouts, times its 3-pick bonus.
    assert returns.at[2, "roi"] == pytest.approx(
        np.mean([a * b for a, b in combinations(legs["Multiplier"], 2)]) - 1
    )


def test_power_entry_returns_never_mix_platforms_and_pool_a_date_by_entry():
    legs = pd.DataFrame(
        {
            "Date": "2026-11-01",
            "Platform": ["Underdog", "Underdog", "Underdog", "Sleeper", "Sleeper"],
            "Game": ["g1", "g2", "g3", "g1", "g2"],
            "Multiplier": [1.0, 1.0, 1.0, 1.5, 2.0],
            "Hit": [True, True, False, True, True],
        }
    )
    returns = plays.power_entry_returns(legs).set_index("size")

    # Three Underdog pairs, one of them all hit at 3.5, and the one Sleeper pair at 1.5 x 2.0.
    assert returns.at[2, "entries"] == 4
    assert returns.at[2, "roi"] == pytest.approx((3.5 + 3.0) / 4 - 1)
    assert returns.at[3, "entries"] == 1


def test_sequence_radius_at_the_design_sample_and_its_growth():
    rho = plays.mixture_rho(1000, 0.10)
    assert abs(rho - 37.7) < 1.0
    lo, hi = plays.sequence_bounds(np.zeros(1000), 0.10, 1000)
    assert hi == pytest.approx(0.0437, abs=0.001) and lo == pytest.approx(-hi)
    assert plays.sequence_bounds(np.zeros(400), 0.10, 1000)[1] > hi
    assert plays.sequence_bounds(np.zeros(1000), 0.10 / 3, 1000)[1] > hi
    # Tuned at the sample it reads, the sequence is tighter there than one tuned elsewhere.
    assert (
        plays.sequence_bounds(np.zeros(400), 0.10, 400)[1]
        < plays.sequence_bounds(np.zeros(400), 0.10, 1000)[1]
    )


def _four_recommended_offers() -> pd.DataFrame:
    """Two NHL even picks on one day (a hit and a miss), then an NBA even pick and a Sleeper leg."""
    return pd.DataFrame(
        [
            _offer(Player="A"),
            _offer(Player="B", Team="CHI", Opponent="DET", Result="Under"),
            _offer(Player="C", League="NBA", Team="LAL", Opponent="GSW", Date="2026-11-02"),
            _offer(
                Player="D",
                Platform="Sleeper",
                Date="2026-11-02",
                Boost=1.7,
                **{"Payout Over": 1.7, "Payout Under": 2.0},
            ),
        ]
    )


def test_play_record_scores_a_plays_legs_per_league_and_kind():
    sides = plays.posted_sides(_four_recommended_offers())
    record = plays.play_record(sides, np.random.default_rng(0))

    assert list(record.columns) == plays.PLAY_RECORD_COLS
    nhl_even = record.set_index(["play", "league", "kind"]).loc[
        ("model_edge", "NHL", "underdog_even")
    ]
    assert (nhl_even["n"], nhl_even["dates"]) == (2, 1)
    assert nhl_even["first_date"] == nhl_even["last_date"] == "2026-11-01"
    assert nhl_even["claim"] == pytest.approx(0.6)
    assert nhl_even["hit_rate"] == 0.5
    assert nhl_even["breakeven_rate"] == pytest.approx(1 / 1.83)
    assert nhl_even["roi"] == pytest.approx(0.5 * 1.83 - 1)
    assert nhl_even["excess"] == pytest.approx(0.5 - 1 / 1.83)
    assert nhl_even["cs_lo"] < nhl_even["excess"] < nhl_even["cs_hi"]


def test_play_record_scores_the_days_cross_game_entries():
    sides = plays.posted_sides(_four_recommended_offers())
    record = plays.play_record(sides, np.random.default_rng(0))
    nhl_even = record.set_index(["play", "league", "kind"]).loc[
        ("model_edge", "NHL", "underdog_even")
    ]

    # The day's one 2-pick entry holds the miss, and two games build no 3-pick entry.
    assert nhl_even["power2_roi"] == -1.0
    assert np.isnan(nhl_even["power3_roi"])


def test_play_record_weights_a_dates_entries_by_its_legs():
    hits = [_offer(Player=f"H{i}", Team=f"T{i}", Opponent=f"O{i}") for i in range(2)]
    misses = [
        _offer(Player=f"M{i}", Team=f"T{i}", Opponent=f"O{i}", Result="Under", Date="2026-11-02")
        for i in range(3)
    ]
    sides = plays.posted_sides(pd.DataFrame(hits + misses))
    record = plays.play_record(sides, np.random.default_rng(0))
    nhl_even = record.set_index(["play", "league", "kind"]).loc[
        ("model_edge", "NHL", "underdog_even")
    ]

    # The first date's one pair pays 3.5 and its two legs carry it; the second's three legs lose.
    assert nhl_even["power2_roi"] == pytest.approx((2 * 2.5 + 3 * -1.0) / 5)


def test_play_record_pools_leagues_and_kinds_under_all():
    sides = plays.posted_sides(_four_recommended_offers())
    n = plays.play_record(sides, np.random.default_rng(0)).set_index(["play", "league", "kind"])[
        "n"
    ]

    assert n[("model_edge", "all", "all")] == 4
    assert n[("model_edge", "all", "underdog_even")] == 3
    assert n[("model_edge", "NHL", "all")] == 3
    assert n[("under", "all", "underdog_even")] == 3
    assert n[("all", "all", "all")] == 8


def test_play_record_of_no_sides_is_empty_with_its_columns():
    sides = plays.posted_sides(_four_recommended_offers().iloc[:0])
    record = plays.play_record(sides, np.random.default_rng(0))

    assert record.empty
    assert list(record.columns) == plays.PLAY_RECORD_COLS


def _entry(**fields) -> dict:
    """One parlay-history row: an Underdog 2-pick NFL entry, every leg graded, none missed."""
    return {
        "Date": "2026-11-01",
        "League": "NFL",
        "Platform": "Underdog",
        "Bet Size": 2,
        "Boost": 3.5,
        "Model EV": 2.1,
        "Legs Resolved": 2.0,
        "Misses": 0.0,
    } | fields


def test_entry_record_pays_an_entry_only_when_no_leg_missed():
    entries = pd.DataFrame(
        [
            _entry(),
            _entry(Misses=1.0),
            _entry(Date="2026-11-02", Misses=2.0),
            _entry(Date="2026-11-02", Boost=7.0, Misses=1.0, **{"Model EV": 2.5}),
        ]
    )
    record = plays.entry_record(entries, np.random.default_rng(0))

    assert list(record.columns) == plays.ENTRY_RECORD_COLS
    (row,) = record.to_dict("records")
    assert (row["league"], row["platform"], row["size"]) == ("NFL", "Underdog", 2)
    assert (row["n"], row["dates"]) == (4, 2)
    assert (row["first_date"], row["last_date"]) == ("2026-11-01", "2026-11-02")
    assert row["model_ev"] == pytest.approx(2.2)
    assert row["payout"] == pytest.approx(4.375)
    assert row["hit_rate"] == 0.25
    assert row["breakeven_rate"] == pytest.approx((3 / 3.5 + 1 / 7.0) / 4)
    assert row["roi"] == pytest.approx(3.5 / 4 - 1)
    assert row["excess"] == pytest.approx(0.25 - (3 / 3.5 + 1 / 7.0) / 4)
    assert row["cs_lo"] < row["excess"] < row["cs_hi"]
    assert row["roi_lo"] <= row["roi"] <= row["roi_hi"]


def test_entry_record_leaves_out_an_entry_with_an_ungraded_or_pushed_leg():
    entries = pd.DataFrame(
        [
            _entry(),
            _entry(**{"Legs Resolved": 1.0}),
            _entry(**{"Legs Resolved": np.nan}, Misses=np.nan),
        ]
    )
    record = plays.entry_record(entries, np.random.default_rng(0))
    assert list(record["n"]) == [1]


def test_entry_record_splits_league_platform_and_size():
    entries = pd.DataFrame(
        [
            _entry(),
            _entry(League="NBA"),
            _entry(Platform="Sleeper"),
            _entry(**{"Bet Size": 3, "Legs Resolved": 3.0}),
        ]
    )
    record = plays.entry_record(entries, np.random.default_rng(0))
    assert set(zip(record["league"], record["platform"], record["size"], strict=True)) == {
        ("NFL", "Underdog", 2),
        ("NBA", "Underdog", 2),
        ("NFL", "Sleeper", 2),
        ("NFL", "Underdog", 3),
    }


def test_entry_record_of_no_entries_is_empty_with_its_columns():
    record = plays.entry_record(pd.DataFrame([_entry()]).iloc[:0], np.random.default_rng(0))

    assert record.empty
    assert list(record.columns) == plays.ENTRY_RECORD_COLS
