"""Pin the per-offer line-movement contract the Board's move column reads.

``build_line_movement`` is the only producer of ``current_line_movement.parquet``; the
dashboard left-joins it onto ``current_offers`` on ``(League, Platform, Market, Player,
Date)``, so the key set, the column set, and the JSON ``series``, ``fair_series`` and
``changes`` shapes are all external contract. Fixtures are ladder rows shaped like
``Archive.get_book_line_histories`` output: every rung of every poll with its de-vigged
``p_over``.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from sportstradamus.helpers.io import LINE_MOVEMENT_COLS
from sportstradamus.prediction.line_movement import _MOVEMENT_POINTS, build_line_movement

_BASE = pd.Timestamp("2026-09-05 08:00:00")
_KEY = {
    "league": "NFL",
    "market": "receptions",
    "game_date": datetime.date(2026, 9, 13),
    "entity": "Travis Etienne",
    "book": "Underdog",
}
_SIGMA = 1.9


def _offers(**overrides) -> pd.DataFrame:
    row = {
        "League": "NFL",
        "Platform": "Underdog",
        "Market": "receptions",
        "Player": "Travis Etienne",
        "Date": "2026-09-13",
        "Line": 3.5,
        "Projection": 3.8,
        "Projection STD": _SIGMA,
        "CV": 0.5,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _ladder(*polls, **overrides) -> pd.DataFrame:
    """Rows for ``(minutes after _BASE, [(line, p_over), ...])`` polls, rungs a second apart."""
    key = {**_KEY, **overrides}
    return pd.DataFrame(
        [
            {
                **key,
                "observed_at": _BASE + pd.Timedelta(minutes=minutes, seconds=i),
                "line": line,
                "p_over": p_over,
            }
            for minutes, rungs in polls
            for i, (line, p_over) in enumerate(rungs)
        ]
    )


def _observations(lines, offsets_minutes, **overrides) -> pd.DataFrame:
    """Single-rung polls at even money, the Underdog shape: the fair line is the posted one."""
    return _ladder(
        *[(minutes, [(line, 0.5)]) for line, minutes in zip(lines, offsets_minutes, strict=True)],
        **overrides,
    )


def test_open_close_and_move_are_the_endpoints():
    movement = build_line_movement(_observations([3.5, 4.5, 4.0], [0, 60, 120]), _offers())

    assert list(movement.columns) == LINE_MOVEMENT_COLS
    assert len(movement) == 1
    row = movement.iloc[0]
    assert row["open_line"] == 3.5
    assert row["close_line"] == 4.0
    assert row["move"] == 0.5
    assert row["first_seen"] == _BASE
    assert row["last_seen"] == _BASE + pd.Timedelta(minutes=120)

    series = json.loads(row["series"])
    assert series[0] == row["open_line"]
    assert series[-1] == row["close_line"]


def test_consecutive_duplicates_do_not_count_as_moves():
    flickering = _observations([3.5, 3.5, 3.5, 4.5, 4.5, 3.5], [0, 10, 20, 30, 40, 50])
    assert build_line_movement(flickering, _offers()).iloc[0]["n_moves"] == 2

    never_repriced = _observations([3.5] * 6, [0, 10, 20, 30, 40, 50])
    assert build_line_movement(never_repriced, _offers()).iloc[0]["n_moves"] == 0


def test_single_observation_yields_a_one_point_series():
    row = build_line_movement(_observations([3.5], [0]), _offers()).iloc[0]

    assert json.loads(row["series"]) == json.loads(row["fair_series"]) == [3.5]
    assert row["move"] == row["fair_move"] == 0.0
    assert row["n_moves"] == row["n_price_moves"] == 0
    assert row["open_line"] == row["close_line"] == 3.5
    assert row["first_seen"] == row["last_seen"] == _BASE
    assert len(json.loads(row["changes"])) == 1


def test_series_samples_evenly_in_time_not_by_observation_index():
    # Three reprices bunched into the first hour, then a nine-hour hold. Sampling
    # by observation index would replay the raw [3.5, 4.5, 5.5, 6.5, 6.5];
    # sampling in time spends four of the five ticks inside the final hold.
    bunched = _observations([3.5, 4.5, 5.5, 6.5], [0, 20, 40, 60])
    held = pd.concat([bunched, _observations([6.5], [600])], ignore_index=True)

    assert json.loads(build_line_movement(held, _offers()).iloc[0]["series"]) == [
        3.5,
        6.5,
        6.5,
        6.5,
        6.5,
    ]


def test_series_is_capped_at_the_sparkline_width():
    steps = list(range(3 * _MOVEMENT_POINTS))
    dense = _observations([1.0 + i for i in steps], [i * 5 for i in steps])

    row = build_line_movement(dense, _offers()).iloc[0]
    assert len(json.loads(row["series"])) == _MOVEMENT_POINTS
    assert len(json.loads(row["fair_series"])) == _MOVEMENT_POINTS
    assert row["n_moves"] == 3 * _MOVEMENT_POINTS - 1


def test_only_offer_keys_survive_the_join():
    other_player = _observations([9.5, 10.5], [0, 60], entity="Bijan Robinson")
    observations = pd.concat([_observations([3.5, 4.5], [0, 60]), other_player], ignore_index=True)

    movement = build_line_movement(observations, _offers())
    assert movement["Player"].tolist() == ["Travis Etienne"]


def test_one_row_per_offer_key_across_platforms():
    offers = pd.concat([_offers(), _offers(Platform="Sleeper")], ignore_index=True)
    observations = pd.concat(
        [
            _observations([3.5, 4.5], [0, 60]),
            _observations([3.5, 2.5], [0, 60], book="Sleeper"),
        ],
        ignore_index=True,
    )

    movement = build_line_movement(observations, offers)
    key = ["League", "Platform", "Market", "Player", "Date"]
    assert not movement.duplicated(subset=key).any()
    assert dict(zip(movement["Platform"], movement["move"], strict=True)) == {
        "Underdog": 1.0,
        "Sleeper": -1.0,
    }


def test_offer_rows_sharing_a_key_yield_one_movement_row():
    # Sleeper lists each alt rung as its own offer row under one key; those rows must
    # not multiply the polls.
    offers = pd.concat(
        [_offers(Platform="Sleeper", Line=2.5), _offers(Platform="Sleeper", Line=4.5)],
        ignore_index=True,
    )
    ladder = _ladder(
        (0, [(2.5, 0.70), (3.5, 0.52), (4.5, 0.30)]),
        (10, [(2.5, 0.72), (3.5, 0.55), (4.5, 0.33)]),
        book="Sleeper",
    )

    movement = build_line_movement(ladder, offers)
    assert len(movement) == 1
    assert json.loads(movement.iloc[0]["series"]) == [3.5, 3.5]


def test_empty_inputs_yield_the_column_stable_frame():
    empty = pd.DataFrame()
    for observations, offers in (
        (empty, _offers()),
        (_observations([3.5], [0]), empty),
        (empty, empty),
    ):
        movement = build_line_movement(observations, offers)
        assert movement.empty
        assert list(movement.columns) == LINE_MOVEMENT_COLS


def test_early_returns_never_read_the_spread_columns():
    # The integration run's synthetic offers carry no Projection STD, and its archive
    # holds no DFS ladder: it must reach the empty frame, not a KeyError.
    bare = _offers().drop(columns=["Projection", "Projection STD", "CV"])
    unmatched = _observations([9.5], [0], entity="Bijan Robinson")
    for observations in (pd.DataFrame(), unmatched):
        movement = build_line_movement(observations, bare)
        assert movement.empty
        assert list(movement.columns) == LINE_MOVEMENT_COLS


def test_main_line_is_the_rung_priced_nearest_even_not_the_lowest():
    # Sleeper posts cheap rungs either side of even money; add_dfs archives the balanced
    # one on the same rule, so its quote and this trend follow the same line.
    ladder = _ladder((0, [(0.5, 0.72), (1.5, 0.48), (2.5, 0.20)]), book="Sleeper")
    row = build_line_movement(ladder, _offers(Platform="Sleeper")).iloc[0]

    assert row["open_line"] == 1.5
    # Interpolated to even money against the 0.5 rung: 1.5 - 0.02 / 0.24.
    assert json.loads(row["fair_series"]) == [1.42]


def test_rungs_staged_under_a_minute_apart_are_one_poll():
    # One run's rungs land seconds apart (chained 15 s and 30 s here); the next run
    # lands minutes later. Read rung by rung, the first poll would hop 0.5 -> 1.5 -> 2.5.
    ladder = _ladder(
        (0, [(0.5, 0.72)]),
        (0.25, [(1.5, 0.48)]),
        (0.75, [(2.5, 0.20)]),
        (7, [(0.5, 0.72), (1.5, 0.48), (2.5, 0.20)]),
        book="Sleeper",
    )
    row = build_line_movement(ladder, _offers(Platform="Sleeper")).iloc[0]

    assert json.loads(row["series"]) == [1.5, 1.5]
    assert row["n_moves"] == 0
    assert row["first_seen"] == _BASE
    assert row["last_seen"] == _BASE + pd.Timedelta(minutes=7)


def test_alt_rungs_coming_and_going_move_neither_line():
    # Far rungs appear and vanish between polls; the balanced 1.5 rung and its 2.5
    # partner hold, so the fair line holds at 1.5 + 0.02 / 0.07.
    ladder = _ladder(
        (0, [(0.5, 0.80), (1.5, 0.52), (2.5, 0.45), (3.5, 0.20)]),
        (10, [(1.5, 0.52), (2.5, 0.45)]),
        (20, [(0.5, 0.80), (1.5, 0.52), (2.5, 0.45), (4.5, 0.10)]),
        book="Sleeper",
    )
    row = build_line_movement(ladder, _offers(Platform="Sleeper")).iloc[0]

    assert row["n_moves"] == row["n_price_moves"] == 0
    assert json.loads(row["series"]) == [1.5, 1.5, 1.5]
    assert json.loads(row["fair_series"]) == [1.79, 1.79, 1.79]


def test_straddle_flip_moves_the_main_line_but_barely_the_fair_line():
    # A 0.002 price nudge flips which of two straddling rungs is balanced; interpolating
    # between them keeps the fair line near 21.
    ladder = _ladder(
        (0, [(20.5, 0.549), (21.5, 0.449)]),
        (10, [(20.5, 0.551), (21.5, 0.451)]),
        market="receiving yards",
    )
    row = build_line_movement(ladder, _offers(Market="receiving yards")).iloc[0]

    assert row["n_moves"] == 1
    assert row["n_price_moves"] == 0
    assert json.loads(row["series"]) == [20.5, 21.5]
    assert json.loads(row["fair_series"]) == [20.99, 21.01]
    assert abs(row["fair_move"]) < 0.05
    # The flip is also the last poll, and is listed once.
    assert [change["line"] for change in json.loads(row["changes"])] == [20.5, 21.5]


def test_price_only_move_registers_in_the_fair_line():
    # Underdog reprices a touchdown prop by its multiplier and never moves the 0.5 line.
    sigma = 0.62
    ladder = _ladder((0, [(0.5, 0.5)]), (60, [(0.5, 0.6)]), market="tds")
    row = build_line_movement(ladder, _offers(Market="tds", **{"Projection STD": sigma})).iloc[0]

    assert row["n_moves"] == 0
    assert row["n_price_moves"] == 1
    assert row["fair_move"] == round(sigma * norm.ppf(0.6), 2)
    assert json.loads(row["series"]) == [0.5, 0.5]


def test_fair_line_is_the_posted_line_at_even_price():
    row = build_line_movement(_observations([3.5, 4.5, 4.0], [0, 60, 120]), _offers()).iloc[0]

    assert json.loads(row["fair_series"]) == json.loads(row["series"]) == [3.5, 4.5, 4.0]
    assert row["fair_move"] == row["move"] == 0.5
    assert row["n_price_moves"] == 0


def test_missing_projection_std_falls_back_to_projection_times_cv():
    # Book-fallback offers carry no model spread; resolve_std scales Projection by CV.
    offers = _offers(Projection=4.0, CV=0.5, **{"Projection STD": np.nan})
    ladder = _ladder((0, [(3.5, 0.5)]), (60, [(3.5, 0.6)]))

    row = build_line_movement(ladder, offers).iloc[0]
    assert row["fair_move"] == round(4.0 * 0.5 * norm.ppf(0.6), 2)


def test_unpartnered_fair_line_is_not_floored_at_zero_but_stays_finite():
    # A first-TD longshot sits well below its line; a one-sided payout under 1.0x
    # implies p_over past 1, which the clip keeps off an infinite quantile.
    offers = _offers(Market="tds", **{"Projection STD": 0.62})
    longshot = build_line_movement(_ladder((0, [(0.5, 0.15)]), market="tds"), offers).iloc[0]
    one_sided = build_line_movement(_ladder((0, [(0.5, 1.25)]), market="tds"), offers).iloc[0]

    [longshot_fair] = json.loads(longshot["fair_series"])
    assert longshot_fair == round(0.5 + 0.62 * norm.ppf(0.15), 2)
    assert longshot_fair < 0
    assert json.loads(one_sided["fair_series"]) == [round(0.5 + 0.62 * norm.ppf(0.99), 2)]


def test_out_of_order_rows_still_find_the_main_rung_and_its_partner():
    ladder = _ladder(
        (0, [(19.5, 0.62), (20.5, 0.549), (21.5, 0.449), (22.5, 0.33)]),
        (30, [(19.5, 0.64), (20.5, 0.571), (21.5, 0.47), (22.5, 0.35)]),
        market="receiving yards",
    )
    shuffled = ladder.iloc[[6, 1, 3, 4, 0, 7, 2, 5]]

    row = build_line_movement(shuffled, _offers(Market="receiving yards")).iloc[0]
    assert json.loads(row["series"]) == [20.5, 21.5]
    # 20.5 pairs with 21.5, not the farther 22.5; 21.5 pairs with 20.5, not 19.5.
    assert json.loads(row["fair_series"]) == [20.99, 21.2]
    assert row["first_seen"] == _BASE


def test_a_line_posted_twice_in_one_poll_keeps_the_price_nearest_even():
    # Staged so that neither the first nor the last copy of a line is the one to keep.
    ladder = _ladder(
        (0, [(0.5, 0.90), (1.5, 0.47), (0.5, 0.70), (1.5, 0.40), (2.5, 0.30)]),
        book="Sleeper",
    )
    row = build_line_movement(ladder, _offers(Platform="Sleeper")).iloc[0]

    [change] = json.loads(row["changes"])
    assert (change["line"], change["p_over"]) == (1.5, 0.47)
    # Interpolated against the 0.5 rung at 0.70, not 0.90: 1.5 - 0.03 / 0.23.
    assert change["fair"] == 1.37


def test_changes_list_the_first_poll_every_change_and_the_last():
    ladder = _ladder(
        (0, [(3.5, 0.5)]),
        (10, [(3.5, 0.5)]),
        (20, [(4.5, 0.5)]),
        (30, [(4.5, 0.6)]),
        (40, [(4.5, 0.6)]),
        (50, [(4.5, 0.6)]),
    )
    changes = json.loads(build_line_movement(ladder, _offers()).iloc[0]["changes"])

    repriced = round(4.5 + _SIGMA * norm.ppf(0.6), 2)
    assert changes == [
        {"t": "2026-09-05T08:00:00+00:00", "line": 3.5, "p_over": 0.5, "fair": 3.5},
        {"t": "2026-09-05T08:20:00+00:00", "line": 4.5, "p_over": 0.5, "fair": 4.5},
        {"t": "2026-09-05T08:30:00+00:00", "line": 4.5, "p_over": 0.6, "fair": repriced},
        {"t": "2026-09-05T08:50:00+00:00", "line": 4.5, "p_over": 0.6, "fair": repriced},
    ]
