"""Pin the per-offer line-movement contract the Board's move column reads.

``build_line_movement`` is the only producer of
``current_line_movement.parquet``; the dashboard left-joins it onto
``current_offers`` on ``(League, Platform, Market, Player, Date)``, so the key
set, the column set, and the JSON ``series`` shape are all external contract.
"""

from __future__ import annotations

import datetime
import json

import pandas as pd

from sportstradamus.helpers.io import LINE_MOVEMENT_COLS
from sportstradamus.prediction.line_movement import _MOVEMENT_POINTS, build_line_movement

_BASE = pd.Timestamp("2026-09-05 08:00:00")


def _offers(**overrides) -> pd.DataFrame:
    row = {
        "League": "NFL",
        "Platform": "Underdog",
        "Market": "receptions",
        "Player": "Travis Etienne",
        "Date": "2026-09-13",
        "Line": 3.5,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _observations(lines, offsets_minutes, **overrides) -> pd.DataFrame:
    key = {
        "league": "NFL",
        "market": "receptions",
        "game_date": datetime.date(2026, 9, 13),
        "entity": "Travis Etienne",
        "book": "Underdog",
    }
    key.update(overrides)
    return pd.DataFrame(
        [
            {**key, "observed_at": _BASE + pd.Timedelta(minutes=offset), "line": line}
            for line, offset in zip(lines, offsets_minutes, strict=True)
        ]
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

    assert json.loads(row["series"]) == [3.5]
    assert row["move"] == 0.0
    assert row["n_moves"] == 0
    assert row["open_line"] == row["close_line"] == 3.5
    assert row["first_seen"] == row["last_seen"] == _BASE


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
