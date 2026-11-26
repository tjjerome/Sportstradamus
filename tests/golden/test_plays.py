"""The money table: posted sides, the play rules and the exact cross-game entry return."""

import math
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from sportstradamus import plays, realized
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


# What the EV fit priced the model's side at: a .56 chance, .53 at its lower bound.
_FIT = {"Fit Prob": 0.56, "Fit Prob Low": 0.53, "EV": 1.02}


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


def test_posted_sides_keeps_the_fit_on_the_side_the_model_took():
    offers = pd.DataFrame([_offer(**_FIT)])
    chosen, other = plays.posted_sides(offers)[list(_FIT)].to_dict("records")

    assert chosen == _FIT
    assert np.isnan(list(other.values())).all()


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


def test_ev_edge_marks_a_models_side_whose_lower_bound_pays_inside_the_recommended_band():
    cap = realized.RECOMMENDED_PAYOUT_MAX

    def even(player, bound, **fields):
        return _offer(Player=player, **_FIT | {"Fit Prob Low": bound}, **fields)

    def sleeper(player, payout):
        return even(player, 0.5, Platform="Sleeper", Boost=payout, **{"Payout Over": payout})

    offers = pd.DataFrame(
        [
            even("A", 0.55),  # .55 x 1.83 = 1.0065
            even("B", 0.54),  # .54 x 1.83 = .9882
            # The mark does not read the board's own rule.
            even("C", 0.55, Recommended=False),
            sleeper("D", 2.0),  # exactly 1
            sleeper("E", cap),
            sleeper("F", cap + 0.1),
            _offer(Player="G"),  # the fit gave it no price
        ]
    )
    sides = plays.posted_sides(offers)
    marked = sides[plays.PLAYS["ev_edge"](sides)]

    assert list(marked["Player"]) == ["A", "C", "D", "E"]
    # Every offer here has both sides posted, and the fit prices the model's alone.
    assert marked["Side"].eq("Over").all()


def test_ev_rank_takes_as_many_legs_as_model_edge_on_each_date_league_and_kind():
    def leg(player, ev, recommended=False, **fields):
        return _offer(Player=player, Recommended=recommended, **_FIT | {"EV": ev}, **fields)

    raised = {"Boost": 1.2, "Payout Over": 2.196, "Payout Under": 1.5006}
    offers = pd.DataFrame(
        [
            # NHL even picks on the 1st: the board took two, and two others carry the highest EV.
            leg("rule1", 0.90, recommended=True),
            leg("rule2", 0.91, recommended=True),
            leg("top1", 0.97),
            leg("top2", 0.95),
            leg("low", 0.80),
            # The board took none of these, so neither does the rank play, whatever their EV.
            leg("other league", 1.20, League="NBA"),
            leg("other kind", 1.10, **raised),
            # The 2nd: one leg, and a tie goes to the earlier row.
            leg("tie1", 0.85, Date="2026-11-02"),
            leg("tie2", 0.85, recommended=True, Date="2026-11-02"),
            # The 3rd: the board took two the fit did not price, and one priced leg is left.
            leg("unpriced1", np.nan, recommended=True, Date="2026-11-03"),
            leg("unpriced2", np.nan, recommended=True, Date="2026-11-03"),
            leg("priced", 0.70, Date="2026-11-03"),
        ]
    )
    sides = plays.posted_sides(offers)
    ranked = sides[plays.PLAYS["ev_rank"](sides)]

    assert list(ranked["Player"]) == ["top1", "top2", "tie1", "priced"]
    assert ranked["Side"].eq("Over").all()


def test_ev_pick_takes_the_sides_the_board_stamped_as_picks():
    offers = pd.DataFrame(
        [
            _offer(Player="A", Pick=True, **_FIT),
            _offer(Player="B", Pick=False, **_FIT),
            # Scored before the board flagged picks.
            _offer(Player="C", Pick=np.nan, **_FIT),
        ]
    )
    sides = plays.posted_sides(offers)
    picked = sides[plays.PLAYS["ev_pick"](sides)]

    assert list(picked["Player"]) == ["A"]
    # The flag is the model's side's: the other side of a pick is not one.
    assert picked["Side"].eq("Over").all()


def test_a_history_from_before_the_fit_scores_and_the_ev_plays_pick_nothing():
    sides = plays.posted_sides(_four_recommended_offers())
    record = plays.play_record(sides, np.random.default_rng(0))

    assert not plays.PLAYS["ev_edge"](sides).any()
    assert not plays.PLAYS["ev_rank"](sides).any()
    assert not plays.PLAYS["ev_pick"](sides).any()
    assert set(record["play"]) == {"model_edge", "under", "all"}


@pytest.mark.parametrize("earlier_count", [3, 5])
def test_a_play_draws_after_the_plays_registered_before_it(monkeypatch, earlier_count):
    rng = np.random.default_rng(5)
    # Sleeper legs at payouts all their own, so a bound moves with the draws behind it.
    offers = pd.DataFrame(
        [
            _offer(
                Date=f"2026-11-{day:02d}",
                Player=f"P{pick}",
                Team=f"T{pick}",
                Opponent=f"O{pick}",
                Platform="Sleeper",
                Boost=payout,
                Result=rng.choice(["Over", "Under"]),
                Recommended=pick == 0,
                **{"Payout Over": payout, "Payout Under": rng.uniform(1.5, 2.4)},
                **_FIT | {"Fit Prob Low": 0.7 if pick == 1 else 0.3, "EV": 0.9 + pick / 100},
                Pick=pick == 2,
            )
            for day in range(1, 13)
            for pick, payout in enumerate(rng.uniform(1.5, 2.4, 3))
        ]
    )
    sides = plays.posted_sides(offers)
    record = plays.play_record(sides, np.random.default_rng(0))
    registered = list(plays.PLAYS)
    assert registered == ["model_edge", "under", "all", "ev_edge", "ev_rank", "ev_pick"]
    earlier = registered[:earlier_count]
    monkeypatch.setattr(plays, "PLAYS", {name: plays.PLAYS[name] for name in earlier})
    before = plays.play_record(sides, np.random.default_rng(0))

    assert set(registered[earlier_count:]) <= set(record["play"])
    pd.testing.assert_frame_equal(record[record["play"].isin(earlier)], before)


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


def _every_side(record: pd.DataFrame) -> pd.Series:
    return record.set_index(["play", "league", "kind"]).loc[("all", "all", "all")]


def test_trailing_record_scores_the_last_play_window_days():
    now = datetime(2027, 2, 1)
    first_day = (now - timedelta(days=plays.PLAY_WINDOW_DAYS)).strftime("%Y-%m-%d")
    too_old = (now - timedelta(days=plays.PLAY_WINDOW_DAYS + 1)).strftime("%Y-%m-%d")
    offers = pd.DataFrame(
        [
            _offer(Date="2027-01-31"),
            _offer(Date=first_day, Player="B"),
            _offer(Date=too_old, Player="C"),
        ]
    )

    every = _every_side(plays.trailing_record(offers, now, np.random.default_rng(0)))

    assert (every["n"], every["first_date"], every["last_date"]) == (4, first_day, "2027-01-31")


def test_trailing_record_leaves_out_games_from_before_a_prop_counted_once():
    start = realized.COUNT_ONCE_FROM
    offers = pd.DataFrame(
        [
            _offer(Date=start.strftime("%Y-%m-%d")),
            _offer(Date=(start - timedelta(days=1)).strftime("%Y-%m-%d"), Player="B"),
        ]
    )

    record = plays.trailing_record(offers, start + timedelta(days=30), np.random.default_rng(0))

    assert _every_side(record)["first_date"] == start.strftime("%Y-%m-%d")
    assert _every_side(record)["n"] == 2


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
