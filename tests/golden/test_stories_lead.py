"""Prominence and lead-story golden tests (``prediction.stories.lead``).

Pins the owner's 2026-10-03 decisions: ``Star`` comes from existing offer
columns only (rank-1 depth, median-line percentile, market breadth; no MLB depth
term), each side's lead seed is its most prominent strong leg, and the slate
pass flags exactly one story per menu, alternating Over and Under in tip order
from a date-parity start.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.prediction.stories.lead import (
    LEAD_SIDES,
    assign_leads,
    attach_prominence,
    lead_seeds,
)

EVEN, ODD = "2026-10-03", "2026-10-04"


def _offers(*rows):
    return pd.DataFrame(list(rows), columns=["League", "Market", "Player", "Line", "Position"])


@pytest.mark.parametrize(
    ("league", "position", "depth"),
    [
        ("NFL", "WR1", 1.0),
        ("NFL", "WR2", 0.0),
        ("NHL", "G1", 1.0),
        ("NBA", "B1", 1.0),
        ("MLB", "B1", 0.0),
        ("MLB", "B3", 0.0),
        ("MLB", "P", 0.0),
    ],
)
def test_depth_counts_rank_one_labels_only(league, position, depth):
    # A lone player tops his market's line and his league's breadth, so Star - 2 is depth.
    out = attach_prominence(_offers((league, "PTS", "Solo", 10.5, position)))
    assert out["Star"].iloc[0] == depth + 2.0


def test_line_pct_ranks_each_players_median_line_within_its_market():
    # A's alt rungs put his median (45.5) under C and B on yards although his top
    # rung is the market's longest; a mean or max would rank him higher.
    out = attach_prominence(
        _offers(
            ("NFL", "receiving yards", "A", 40.5, "WR2"),
            ("NFL", "receiving yards", "A", 45.5, "WR2"),
            ("NFL", "receiving yards", "A", 99.5, "WR2"),
            ("NFL", "receiving yards", "B", 70.5, "WR2"),
            ("NFL", "receiving yards", "C", 50.5, "WR2"),
            ("NFL", "receptions", "A", 9.5, "WR2"),
            ("NFL", "receptions", "B", 3.5, "WR2"),
            ("NFL", "receptions", "C", 5.5, "WR2"),
        )
    )
    # Everyone posts both markets (breadth 1.0) under a depth-0 label.
    line_pct = out["Star"] - 1.0
    assert line_pct.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0, 2 / 3, 1.0, 1 / 3, 2 / 3])


def test_line_pct_ranks_within_each_league():
    # NHL and WNBA both post "PTS"; pooled, every NHL line would rank under every WNBA one.
    out = attach_prominence(
        _offers(
            ("NHL", "PTS", "Top Center", 1.5, "C2"),
            ("NHL", "PTS", "Fourth Liner", 0.5, "C4"),
            ("WNBA", "PTS", "Top Forward", 25.5, "F3"),
            ("WNBA", "PTS", "Bench Guard", 6.5, "G2"),
        )
    )
    assert (out["Star"] - 1.0).tolist() == pytest.approx([1.0, 0.5, 1.0, 0.5])


def test_breadth_is_distinct_markets_over_the_league_max():
    # Wide quotes receptions on both platforms and yards on two rungs; each market
    # counts once. Every market holds one player, so line_pct is 1.0 throughout,
    # and NHL normalises by its own maximum (two), not NFL's (four).
    out = attach_prominence(
        _offers(
            ("NFL", "receptions", "Wide", 5.5, "WR2"),
            ("NFL", "receptions", "Wide", 5.5, "WR2"),
            ("NFL", "receiving yards", "Wide", 60.5, "WR2"),
            ("NFL", "receiving yards", "Wide", 80.5, "WR2"),
            ("NFL", "tds", "Wide", 0.5, "WR2"),
            ("NFL", "carries", "Wide", 1.5, "WR2"),
            ("NFL", "rushing yards", "Narrow", 45.5, "RB2"),
            ("NHL", "shots", "Solo", 2.5, "W2"),
            ("NHL", "goals", "Solo", 0.5, "W2"),
        )
    )
    breadth = out["Star"] - 1.0
    assert breadth.tolist() == pytest.approx([1.0] * 6 + [0.25] + [1.0] * 2)


def test_line_and_breadth_lift_a_star_buried_in_the_usage_rank():
    # A'ja Wilson posts as F3 by minutes, yet her top PTS line and four markets
    # outrank an F1 whose PTS line sits under the median and who posts two.
    out = attach_prominence(
        _offers(
            ("WNBA", "PTS", "A'ja Wilson", 25.5, "F3"),
            ("WNBA", "REB", "A'ja Wilson", 9.5, "F3"),
            ("WNBA", "PRA", "A'ja Wilson", 38.5, "F3"),
            ("WNBA", "AST", "A'ja Wilson", 2.5, "F3"),
            ("WNBA", "PTS", "Rank-One Forward", 8.5, "F1"),
            ("WNBA", "REB", "Rank-One Forward", 4.5, "F1"),
            ("WNBA", "PTS", "Guard", 14.5, "G2"),
            ("WNBA", "PTS", "Center", 12.5, "C2"),
            ("WNBA", "REB", "Center", 7.5, "C2"),
        )
    )
    pts = out[out["Market"] == "PTS"].set_index("Player")["Star"]
    assert pts["A'ja Wilson"] == pytest.approx(0.0 + 1.0 + 1.0)
    assert pts["Rank-One Forward"] == pytest.approx(1.0 + 0.25 + 0.5)
    star = out.groupby("Player")["Star"]
    assert star.min()["A'ja Wilson"] > star.max()["Rank-One Forward"]


def test_combo_legs_score_without_depth():
    out = attach_prominence(
        _offers(
            ("NFL", "receiving yards", "Ja'Marr Chase", 85.5, "WR1"),
            ("NFL", "receiving yards", "Ja'Marr Chase + Tee Higgins", 140.5, ""),
            ("NHL", "shots", "Connor McDavid vs. Auston Matthews", 0.5, ""),
        )
    )
    # Both combos top their market and their league's breadth; only depth stays at 0.
    assert out["Star"].tolist() == pytest.approx([1.0 + 0.5 + 1.0, 2.0, 2.0])


def test_empty_offers_gain_the_column():
    out = attach_prominence(_offers())
    assert out.empty
    assert out["Star"].dtype == "float64"


def _legs(*legs):
    """``(player, bet, model_ev, star)`` per strong leg -> lead_seeds' (edge, bet_df, star_of)."""
    edge = {i: ev for i, (_, _, ev, _) in enumerate(legs)}
    bet_df = {
        i: {"Player": player, "Bet": bet, "Model EV": ev}
        for i, (player, bet, ev, _) in enumerate(legs)
    }
    star_of = {i: star for i, (*_, star) in enumerate(legs)}
    return edge, bet_df, star_of


@pytest.mark.parametrize(
    ("legs", "seeds"),
    [
        pytest.param(
            [("Role", "Over", 1.40, 1.5), ("Star", "Over", 1.10, 2.5)],
            {"Over": 1},
            id="star-beats-edge",
        ),
        pytest.param(
            [("Low", "Under", 1.10, 2.0), ("High", "Under", 1.20, 2.0)],
            {"Under": 1},
            id="edge-breaks-star-tie",
        ),
        pytest.param(
            [("Zed", "Over", 1.10, 2.0), ("Abe", "Over", 1.10, 2.0)],
            {"Over": 1},
            id="name-breaks-full-tie",
        ),
        pytest.param(
            [("Up", "Over", 1.10, 1.0), ("Down", "Under", 1.30, 2.0)],
            {"Over": 0, "Under": 1},
            id="one-per-side",
        ),
    ],
)
def test_lead_seeds_tie_breaks(legs, seeds):
    assert lead_seeds(*_legs(*legs)) == seeds


def test_lead_seeds_omits_a_side_with_no_strong_leg():
    edge, bet_df, star_of = _legs(("Up", "Over", 1.20, 1.0), ("Weak", "Under", 1.02, 2.9))
    del edge[1]  # under the menu's edge floor, so the caller never marks it strong
    assert lead_seeds(edge, bet_df, star_of) == {"Over": 0}


def _stories(*stories):
    """``(platform, game, date, rank, lead_side, moon_ev)`` per story -> its builder + moon rows.

    Builder rows sit at breakeven, so only the moon EV can rank stories.
    """
    return pd.DataFrame(
        [
            {
                "platform": platform,
                "League": "NFL",
                "Game": game,
                "story_id": f"{game}#{rank}",
                "objective": objective,
                "model_ev": ev,
                "Date": date,
                "lead_side": lead_side,
                "lead": False,
            }
            for platform, game, date, rank, lead_side, moon_ev in stories
            for objective, ev in (("builder", 1.0), ("moon", moon_ev))
        ]
    )


def _two_sided(platform, game, date):
    """An Over-led (rank 0) and an Under-led (rank 1) story for one menu."""
    return [(platform, game, date, rank, side, 2.0) for rank, side in enumerate(LEAD_SIDES)]


def _flagged(stories, column):
    """``{(platform, Game, Date): column}`` read off each flagged moon row."""
    led = stories[stories["lead"] & (stories["objective"] == "moon")]
    keys = list(zip(led["platform"], led["Game"], led["Date"], strict=True))
    assert len(keys) == len(set(keys)), "a menu flagged more than one lead story"
    return dict(zip(keys, led[column], strict=True))


def test_sides_alternate_in_tip_order_from_the_date_parity():
    assert pd.Timestamp(EVEN).toordinal() % 2 == 0
    assert pd.Timestamp(ODD).toordinal() % 2 == 1
    # A series posts both Games on both dates and the tip order flips overnight, so
    # each date walks its own tips. On EVEN the later key tips first, so a walk by
    # key would flip that date's flags too.
    tips = {
        (EVEN, "AAA/BBB"): "2026-10-03T20:25:00Z",
        (EVEN, "ZZZ/YYY"): "2026-10-03T17:00:00Z",
        (ODD, "AAA/BBB"): "2026-10-04T17:00:00Z",
        (ODD, "ZZZ/YYY"): "2026-10-04T20:25:00Z",
    }
    stories = _stories(*(s for date, game in tips for s in _two_sided("Underdog", game, date)))
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "ZZZ/YYY", EVEN): "Over",
        ("Underdog", "AAA/BBB", EVEN): "Under",
        ("Underdog", "AAA/BBB", ODD): "Under",
        ("Underdog", "ZZZ/YYY", ODD): "Over",
    }


def test_one_sided_and_unled_games_leave_the_turn():
    tips = {
        (EVEN, "G1"): "2026-10-03T17:00:00Z",
        (EVEN, "G2"): "2026-10-03T18:00:00Z",
        (EVEN, "G3"): "2026-10-03T19:00:00Z",
        (EVEN, "G4"): "2026-10-03T20:00:00Z",
    }
    stories = _stories(
        ("Underdog", "G1", EVEN, 0, "Under", 2.0),
        *_two_sided("Underdog", "G2", EVEN),
        ("Underdog", "G3", EVEN, 0, "", 2.0),
        *_two_sided("Underdog", "G4", EVEN),
    )
    # Over is wanted first. G1 can only show Under, so G2 still owes the Over; G3 has
    # nothing led, so G4 still owes the Under.
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "G1", EVEN): "Under",
        ("Underdog", "G2", EVEN): "Over",
        ("Underdog", "G3", EVEN): "",
        ("Underdog", "G4", EVEN): "Under",
    }


def test_unled_game_flags_its_best_moon_story():
    stories = _stories(
        ("Underdog", "G1", EVEN, 0, "", 1.6),
        ("Underdog", "G1", EVEN, 1, "", 2.4),
        ("Underdog", "G1", EVEN, 2, "", 1.9),
    )
    flagged = _flagged(assign_leads(stories, {(EVEN, "G1"): ""}), "story_id")
    assert flagged == {("Underdog", "G1", EVEN): "G1#1"}


def test_led_story_outranks_a_richer_unled_one():
    stories = _stories(
        ("Underdog", "G1", EVEN, 0, "", 3.0),
        ("Underdog", "G1", EVEN, 1, "Under", 1.2),
    )
    flagged = _flagged(assign_leads(stories, {(EVEN, "G1"): ""}), "story_id")
    assert flagged == {("Underdog", "G1", EVEN): "G1#1"}


def test_platforms_show_the_same_side_whenever_both_offer_it():
    tips = {
        (EVEN, "G1"): "2026-10-03T17:00:00Z",
        (EVEN, "G2"): "2026-10-03T20:00:00Z",
        (EVEN, "G3"): "2026-10-03T21:00:00Z",
    }
    stories = _stories(
        # Only Underdog leads G1, and its richer Under still yields to the wanted Over.
        # The turn passes for both platforms, so Sleeper's first led game opens on Under.
        ("Underdog", "G1", EVEN, 0, "Over", 2.0),
        ("Underdog", "G1", EVEN, 1, "Under", 2.5),
        ("Sleeper", "G1", EVEN, 0, "", 1.8),
        *_two_sided("Underdog", "G2", EVEN),
        ("Sleeper", "G2", EVEN, 0, "Under", 2.0),
        ("Sleeper", "G2", EVEN, 1, "Over", 2.5),
        # Sleeper has no Over-led G3 story: its Under one beats a richer unled story.
        *_two_sided("Underdog", "G3", EVEN),
        ("Sleeper", "G3", EVEN, 0, "", 3.0),
        ("Sleeper", "G3", EVEN, 1, "Under", 2.0),
    )
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "G1", EVEN): "Over",
        ("Sleeper", "G1", EVEN): "",
        ("Underdog", "G2", EVEN): "Under",
        ("Sleeper", "G2", EVEN): "Under",
        ("Underdog", "G3", EVEN): "Over",
        ("Sleeper", "G3", EVEN): "Under",
    }


def test_untimed_games_walk_after_timed_ones_by_key():
    # "" sorts before every timestamp, so a plain string sort would walk both untimed games first.
    tips = {
        (EVEN, "MMM/NNN"): "",
        (EVEN, "AAA/BBB"): "",
        (EVEN, "ZZZ/YYY"): "2026-10-03T23:00:00Z",
    }
    stories = _stories(
        *_two_sided("Underdog", "ZZZ/YYY", EVEN),
        *_two_sided("Sleeper", "MMM/NNN", EVEN),
        *_two_sided("Sleeper", "AAA/BBB", EVEN),
    )
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "ZZZ/YYY", EVEN): "Over",
        ("Sleeper", "AAA/BBB", EVEN): "Under",
        ("Sleeper", "MMM/NNN", EVEN): "Over",
    }


def test_one_story_per_menu_on_both_rows_and_a_rerun_changes_nothing():
    tips = {(EVEN, "G1"): "2026-10-03T17:00:00Z", (EVEN, "G2"): ""}
    stories = _stories(
        *_two_sided("Underdog", "G1", EVEN),
        *_two_sided("Sleeper", "G1", EVEN),
        ("Underdog", "G1", EVEN, 2, "", 3.0),
        ("Sleeper", "G2", EVEN, 0, "", 1.5),
        ("Sleeper", "G2", EVEN, 1, "", 1.8),
    )
    first = assign_leads(stories, tips)["lead"].copy()
    menu = ["platform", "Game", "Date"]
    # Every story has one builder and one moon row, so two flagged rows sharing one
    # story_id per menu means both objective rows of exactly one story.
    assert stories.groupby(menu)["lead"].sum().eq(2).all()
    assert stories[stories["lead"]].groupby(menu)["story_id"].nunique().eq(1).all()
    stories["lead"] = True  # stale flags a second pass must clear
    assert assign_leads(stories, tips)["lead"].equals(first)


def test_empty_story_frame_gains_a_bool_lead():
    # build_game_stories hands over an empty frame when no game yields a story.
    columns = ["platform", "Game", "story_id", "objective", "model_ev", "Date", "lead_side"]
    out = assign_leads(pd.DataFrame(columns=columns), {})
    assert out.empty
    assert out["lead"].dtype == bool
