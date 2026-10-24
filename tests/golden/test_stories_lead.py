"""Prominence and lead-story golden tests (``prediction.stories.lead``).

Pins the owner's 2026-10-03 decisions: ``Star`` comes from existing offer
columns only (rank-1 depth, median-line percentile, market breadth; no MLB depth
term), each side's lead seed is its most prominent strong leg, and the slate
pass flags exactly one story per menu. Over leads fill toward half of each
date's games, rounded up, the games with the brightest Over-led story first.
The 2026-10-17 shape rule: an Over seeds a lead only in a shootout or a blowout.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sportstradamus.prediction.stories.lead import (
    _pick_sides,
    assign_leads,
    attach_prominence,
    lead_seeds,
)

DAY, NEXT_DAY = "2026-10-03", "2026-10-04"


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
    assert lead_seeds(*_legs(*legs), "shootout") == seeds


def test_lead_seeds_omits_a_side_with_no_strong_leg():
    edge, bet_df, star_of = _legs(("Up", "Over", 1.20, 1.0), ("Weak", "Under", 1.02, 2.9))
    del edge[1]  # under the menu's edge floor, so the caller never marks it strong
    assert lead_seeds(edge, bet_df, star_of, "blowout") == {"Over": 0}


@pytest.mark.parametrize(
    ("shape", "seeds"),
    [
        ("shootout", {"Over": 0, "Under": 1}),
        ("blowout", {"Over": 0, "Under": 1}),
        ("grind", {"Under": 1}),
        ("coinflip", {"Under": 1}),
        ("even", {"Under": 1}),
    ],
)
def test_an_over_seed_needs_a_shootout_or_a_blowout(shape, seeds):
    edge, bet_df, star_of = _legs(("Up", "Over", 1.40, 3.0), ("Down", "Under", 1.10, 0.0))
    assert lead_seeds(edge, bet_df, star_of, shape) == seeds


def _stories(*stories):
    """``(platform, game, date, rank, lead_side, moon_ev, star)`` per story -> builder + moon rows.

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
                "star": star,
                "lead": False,
            }
            for platform, game, date, rank, lead_side, moon_ev, star in stories
            for objective, ev in (("builder", 1.0), ("moon", moon_ev))
        ]
    )


def _two_sided(platform, game, date, over_star=1.0):
    """An Over-led (rank 0) and an Under-led (rank 1) story for one menu."""
    return [
        (platform, game, date, 0, "Over", 2.0, over_star),
        (platform, game, date, 1, "Under", 2.0, 1.0),
    ]


def _flagged(stories, column):
    """``{(platform, Game, Date): column}`` read off each flagged moon row."""
    led = stories[stories["lead"] & (stories["objective"] == "moon")]
    keys = list(zip(led["platform"], led["Game"], led["Date"], strict=True))
    assert len(keys) == len(set(keys)), "a menu flagged more than one lead story"
    return dict(zip(keys, led[column], strict=True))


@pytest.mark.parametrize(("n_games", "n_over"), [(1, 1), (2, 1), (3, 2)])
def test_two_sided_games_take_over_up_to_half_the_dates_games_rounded_up(n_games, n_over):
    # Stars tie, so the earliest tips take the Overs. A lone game leads Over: half
    # rounded down would leave it on Under every time.
    games = [f"G{i}" for i in range(n_games)]
    tips = {(DAY, game): f"2026-10-03T{17 + i}:00:00Z" for i, game in enumerate(games)}
    stories = _stories(*(s for game in games for s in _two_sided("Underdog", game, DAY)))
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", game, DAY): "Over" if i < n_over else "Under" for i, game in enumerate(games)
    }


def test_each_date_fills_from_its_own_games_by_its_own_tips():
    # A series posts both Games on both dates and the tip order flips overnight. Each
    # date has one Over to give, to its own earlier tip. On DAY that is the later
    # key, so an order by key would flip that date's flags.
    tips = {
        (DAY, "AAA/BBB"): "2026-10-03T20:25:00Z",
        (DAY, "ZZZ/YYY"): "2026-10-03T17:00:00Z",
        (NEXT_DAY, "AAA/BBB"): "2026-10-04T17:00:00Z",
        (NEXT_DAY, "ZZZ/YYY"): "2026-10-04T20:25:00Z",
    }
    stories = _stories(*(s for date, game in tips for s in _two_sided("Underdog", game, date)))
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "ZZZ/YYY", DAY): "Over",
        ("Underdog", "AAA/BBB", DAY): "Under",
        ("Underdog", "AAA/BBB", NEXT_DAY): "Over",
        ("Underdog", "ZZZ/YYY", NEXT_DAY): "Under",
    }


def test_the_brightest_over_lead_takes_the_over_first():
    tips = {(DAY, "AAA/BBB"): "2026-10-03T17:00:00Z", (DAY, "ZZZ/YYY"): "2026-10-03T20:00:00Z"}
    stories = _stories(
        # AAA/BBB tips first and holds the slate's brightest story, but that story is
        # Under-led. Its Over lead scores 2.0, under the 2.5 of ZZZ/YYY's Over lead on
        # Sleeper, which counts although Underdog's scores 1.0.
        ("Underdog", "AAA/BBB", DAY, 0, "Over", 2.0, 2.0),
        ("Underdog", "AAA/BBB", DAY, 1, "Under", 2.0, 3.0),
        *_two_sided("Underdog", "ZZZ/YYY", DAY, over_star=1.0),
        *_two_sided("Sleeper", "ZZZ/YYY", DAY, over_star=2.5),
    )
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "AAA/BBB", DAY): "Under",
        ("Underdog", "ZZZ/YYY", DAY): "Over",
        ("Sleeper", "ZZZ/YYY", DAY): "Over",
    }


def test_untimed_games_go_after_timed_ones_by_key():
    # Three games leave two Overs. "" sorts before every timestamp, so a plain string
    # sort would hand both to the untimed games.
    tips = {
        (DAY, "MMM/NNN"): "",
        (DAY, "AAA/BBB"): "",
        (DAY, "ZZZ/YYY"): "2026-10-03T23:00:00Z",
    }
    stories = _stories(
        *_two_sided("Underdog", "ZZZ/YYY", DAY),
        *_two_sided("Sleeper", "MMM/NNN", DAY),
        *_two_sided("Sleeper", "AAA/BBB", DAY),
    )
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "ZZZ/YYY", DAY): "Over",
        ("Sleeper", "AAA/BBB", DAY): "Over",
        ("Sleeper", "MMM/NNN", DAY): "Under",
    }


def test_one_sided_and_unled_games_keep_their_side_and_count_in_the_slate():
    tips = {(DAY, f"G{i}"): f"2026-10-03T{16 + i}:00:00Z" for i in range(1, 6)}
    stories = _stories(
        ("Underdog", "G1", DAY, 0, "Under", 2.0, 1.0),
        *_two_sided("Underdog", "G2", DAY),
        ("Underdog", "G3", DAY, 0, "", 2.0, 1.0),
        *_two_sided("Underdog", "G4", DAY),
        *_two_sided("Underdog", "G5", DAY),
    )
    # G1 can only show Under and G3 has nothing led, yet each counts among the date's
    # five games. Half is three, so all three two-sided games lead Over: without
    # either one it would be two, and G5 would lead Under.
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "G1", DAY): "Under",
        ("Underdog", "G2", DAY): "Over",
        ("Underdog", "G3", DAY): "",
        ("Underdog", "G4", DAY): "Over",
        ("Underdog", "G5", DAY): "Over",
    }


@pytest.mark.parametrize(
    ("n_over_only", "two_sided_sides"),
    [
        # Half of three games is two, and the Over-only game holds one of them.
        pytest.param(1, ["Over", "Under"], id="one-over-left"),
        # Half of six games is three, and the Over-only games already pass it.
        pytest.param(4, ["Under", "Under"], id="half-already-passed"),
    ],
)
def test_over_only_games_count_toward_the_half(n_over_only, two_sided_sides):
    over_only = [f"O{i}" for i in range(n_over_only)]
    games = [*over_only, "T0", "T1"]
    tips = {(DAY, game): f"2026-10-03T{17 + i}:00:00Z" for i, game in enumerate(games)}
    stories = _stories(
        *(("Underdog", game, DAY, 0, "Over", 2.0, 1.0) for game in over_only),
        *_two_sided("Underdog", "T0", DAY),
        *_two_sided("Underdog", "T1", DAY),
    )
    sides = _flagged(assign_leads(stories, tips), "lead_side")
    assert [sides["Underdog", game, DAY] for game in games] == [
        *["Over"] * n_over_only,
        *two_sided_sides,
    ]


def test_unled_game_flags_its_best_moon_story():
    stories = _stories(
        ("Underdog", "G1", DAY, 0, "", 1.6, 1.0),
        ("Underdog", "G1", DAY, 1, "", 2.4, 1.0),
        ("Underdog", "G1", DAY, 2, "", 1.9, 1.0),
    )
    flagged = _flagged(assign_leads(stories, {(DAY, "G1"): ""}), "story_id")
    assert flagged == {("Underdog", "G1", DAY): "G1#1"}


def test_led_story_outranks_a_richer_unled_one():
    stories = _stories(
        ("Underdog", "G1", DAY, 0, "", 3.0, 1.0),
        ("Underdog", "G1", DAY, 1, "Under", 1.2, 1.0),
    )
    flagged = _flagged(assign_leads(stories, {(DAY, "G1"): ""}), "story_id")
    assert flagged == {("Underdog", "G1", DAY): "G1#1"}


def test_platforms_show_the_same_side_whenever_both_offer_it():
    # G1 and G3 tip first and take the two Overs three games allow, so G2 leads Under.
    tips = {
        (DAY, "G1"): "2026-10-03T17:00:00Z",
        (DAY, "G2"): "2026-10-03T21:00:00Z",
        (DAY, "G3"): "2026-10-03T20:00:00Z",
    }
    stories = _stories(
        # Only Underdog leads G1, and its richer Under still yields to the game's Over.
        ("Underdog", "G1", DAY, 0, "Over", 2.0, 1.0),
        ("Underdog", "G1", DAY, 1, "Under", 2.5, 1.0),
        ("Sleeper", "G1", DAY, 0, "", 1.8, 1.0),
        # Sleeper's richer Over yields to the game's Under just the same.
        *_two_sided("Underdog", "G2", DAY),
        ("Sleeper", "G2", DAY, 0, "Under", 2.0, 1.0),
        ("Sleeper", "G2", DAY, 1, "Over", 2.5, 1.0),
        # Sleeper has no Over-led G3 story: its Under one beats a richer unled story.
        *_two_sided("Underdog", "G3", DAY),
        ("Sleeper", "G3", DAY, 0, "", 3.0, 1.0),
        ("Sleeper", "G3", DAY, 1, "Under", 2.0, 1.0),
    )
    assert _flagged(assign_leads(stories, tips), "lead_side") == {
        ("Underdog", "G1", DAY): "Over",
        ("Sleeper", "G1", DAY): "",
        ("Underdog", "G2", DAY): "Under",
        ("Sleeper", "G2", DAY): "Under",
        ("Underdog", "G3", DAY): "Over",
        ("Sleeper", "G3", DAY): "Under",
    }


def test_one_story_per_menu_on_both_rows_and_a_rerun_changes_nothing():
    tips = {(DAY, "G1"): "2026-10-03T17:00:00Z", (DAY, "G2"): ""}
    stories = _stories(
        *_two_sided("Underdog", "G1", DAY),
        *_two_sided("Sleeper", "G1", DAY),
        ("Underdog", "G1", DAY, 2, "", 3.0, 1.0),
        ("Sleeper", "G2", DAY, 0, "", 1.5, 1.0),
        ("Sleeper", "G2", DAY, 1, "", 1.8, 1.0),
    )
    first = assign_leads(stories, tips)["lead"].copy()
    menu = ["platform", "Game", "Date"]
    # Every story has one builder and one moon row, so two flagged rows sharing one
    # story_id per menu means both objective rows of exactly one story.
    assert stories.groupby(menu)["lead"].sum().eq(2).all()
    assert stories[stories["lead"]].groupby(menu)["story_id"].nunique().eq(1).all()
    stories["lead"] = True  # stale flags a second pass must clear
    assert assign_leads(stories, tips)["lead"].equals(first)


def test_replay_of_a_production_slate_leads_over_in_14_of_29_games(fixtures_dir):
    """The rule on production's story menus of 2026-10-08 22:05 UTC, both platforms.

    ``story_leads_2026-10-08.json`` is the moon rows of that run's
    ``current_game_stories.parquet``. ``commence`` is each game's tip as
    ``build_game_stories`` takes it, the first non-empty ``Commence`` on the game,
    read off the same run's ``current_offers.parquet``: every game carried one
    tip, on its Underdog rows.
    """
    moon = pd.DataFrame(json.loads((fixtures_dir / "story_leads_2026-10-08.json").read_text()))
    tips = dict(zip(zip(moon["Date"], moon["Game"], strict=True), moon["commence"], strict=True))
    over = (pd.Series(_pick_sides(moon, tips)) == "Over").groupby(level=0)
    # Over leads of games per date, 14 of 29 in all.
    assert over.agg(["sum", "size"]).to_numpy().tolist() == [[6, 12], [1, 2], [7, 14], [0, 1]]
    stories = assign_leads(moon.assign(objective="moon"), tips)
    assert stories[stories["lead"]].groupby(["platform", "lead_side"]).size().to_dict() == {
        ("Sleeper", "Over"): 11,
        ("Sleeper", "Under"): 15,
        ("Underdog", "Over"): 11,
        ("Underdog", "Under"): 18,
    }


def test_empty_story_frame_gains_a_bool_lead():
    # build_game_stories hands over an empty frame when no game yields a story.
    columns = ["platform", "Game", "story_id", "objective", "model_ev", "Date", "lead_side", "star"]
    out = assign_leads(pd.DataFrame(columns=columns), {})
    assert out.empty
    assert out["lead"].dtype == bool
