"""Weather collector building blocks: venue lookup, kickoff times, reply flattening, windows."""

from __future__ import annotations

import logging

import pandas as pd

from sportstradamus.collectors import weather


def _schedule(*games: tuple) -> pd.DataFrame:
    """An nflverse schedule frame of the columns ``nfl_games`` reads."""
    return pd.DataFrame(games, columns=["game_id", "gameday", "gametime", "roof", "stadium"])


def _mlb_game(
    pk: int,
    venue: int,
    start: str,
    *,
    state: str = "Scheduled",
    tbd: bool = False,
    condition: str | None = None,
) -> dict:
    """A statsapi schedule game; ``weather`` appears only once statsapi posts a condition."""
    game = {
        "gamePk": pk,
        "gameType": "R",
        "gameDate": start,
        "status": {"detailedState": state, "startTimeTBD": tbd},
        "venue": {"id": venue, "name": f"Venue {venue}"},
    }
    if condition:
        game["weather"] = {"condition": condition, "temp": "64", "wind": "8 mph, Out To CF"}
    return game


def _payload(*days: tuple[str, list[dict]]) -> dict:
    """A statsapi schedule payload, one entry per (date, games)."""
    return {"dates": [{"date": date, "games": games} for date, games in days]}


def _location(times: list[str], **hourly: list[float]) -> dict:
    """One location of an Open-Meteo reply asked with ``timezone=UTC``."""
    return {"latitude": 44.5, "longitude": -88.0, "hourly": {"time": times, **hourly}}


def test_an_nflverse_spelling_resolves_to_its_table_entry():
    """A former or short name aliases the current entry, coordinates included."""
    games = weather.nfl_games(
        _schedule(
            ("2025_01_BAL_BUF", "2025-09-07", "20:20", "outdoors", "New Era Field"),
            ("2025_01_CIN_CLE", "2025-09-07", "13:00", "outdoors", "FirstEnergy Stadium"),
            ("2024_06_JAX_CHI", "2024-10-13", "09:30", "outdoors", "Tottenham Stadium"),
        )
    )

    expected = ["Highmark Stadium", "Huntington Bank Field", "Tottenham Hotspur Stadium"]
    table = weather.VENUES["NFL"]
    assert games["venue"].tolist() == expected
    assert games[["lat", "lon"]].values.tolist() == [
        [table[name]["lat"], table[name]["lon"]] for name in expected
    ]


def test_the_table_roof_decides_over_the_nflverse_label():
    """nflverse calls Munich a dome and leaves a retractable blank until it is played."""
    games = weather.nfl_games(
        _schedule(
            ("2026_10_NE_DET", "2026-11-15", "09:30", "dome", "FC Bayern Munich Stadium"),
            ("2026_13_TEN_HOU", "2026-12-06", "13:00", None, "Reliant Stadium"),
        )
    )

    assert games[["venue", "roof"]].values.tolist() == [
        ["Allianz Arena", "open"],
        ["NRG Stadium", "retractable"],
    ]
    assert games["schedule_label"].tolist() == ["dome", None]


def test_eastern_kickoffs_convert_to_naive_utc_across_daylight_saving():
    """The same 9:30 ET London and Munich kickoffs land an hour apart in UTC once EDT ends."""
    games = weather.nfl_games(
        _schedule(
            ("2026_05_PHI_JAX", "2026-10-11", "09:30", "outdoors", "Tottenham Hotspur Stadium"),
            ("2026_10_NE_DET", "2026-11-15", "09:30", "dome", "FC Bayern Munich Stadium"),
            ("2025_01_BAL_BUF", "2025-09-07", "20:20", "outdoors", "New Era Field"),
        )
    )

    assert games["kickoff"].tolist() == [
        pd.Timestamp("2026-10-11 13:30"),
        pd.Timestamp("2026-11-15 14:30"),
        pd.Timestamp("2025-09-08 00:20"),
    ]


def test_a_game_without_a_kickoff_time_gets_no_row():
    games = weather.nfl_games(
        _schedule(
            ("2026_18_GB_CHI", "2027-01-10", None, None, "Soldier Field"),
            ("2026_18_NYG_GB", "2027-01-10", "13:00", "outdoors", "Lambeau Field"),
        )
    )

    assert games["game_id"].tolist() == ["2026_18_NYG_GB"]


def test_the_failed_fetch_fallback_frame_gives_no_games_typed_like_a_full_schedule():
    """StatsNFL's fallback has no gametime or stadium; the windows still build from its games."""
    fallback = pd.DataFrame(columns=["away_team", "home_team", "gameday", "week", "game_id"])
    full = weather.nfl_games(
        _schedule(("2026_18_NYG_GB", "2027-01-10", "13:00", "outdoors", "Lambeau Field"))
    )

    games = weather.nfl_games(fallback)

    assert games.empty
    assert games.dtypes.equals(full.dtypes)
    assert weather.window_rows(games, weather.hourly_frame(_location([]), ["Lambeau Field"])).empty


def test_mlb_keeps_only_games_with_a_start_and_labels_them_from_statsapi():
    """Postponed, cancelled and TBD-start games drop; a postponed game's make-up stays."""
    payload = _payload(
        (
            "2026-04-03",
            [
                _mlb_game(1, 17, "2026-04-03T18:20:00Z", condition="Sunny"),
                _mlb_game(2, 3313, "2026-04-03T23:05:00Z", state="Postponed"),
                _mlb_game(3, 4, "2026-04-03T23:10:00Z", state="Cancelled"),
                _mlb_game(4, 2681, "2026-04-03T22:10:00Z", tbd=True),
            ],
        ),
        (
            "2026-04-04",
            [
                _mlb_game(2, 3313, "2026-04-04T17:05:00Z"),
                _mlb_game(5, 12, "2026-04-04T20:10:00Z"),
            ],
        ),
    )

    games = weather.mlb_games(payload)

    assert games[["game_id", "venue", "roof"]].values.tolist() == [
        ["1", "17", "open"],
        ["2", "3313", "open"],
        ["5", "12", "dome"],
    ]
    assert games["kickoff"].tolist() == [
        pd.Timestamp("2026-04-03 18:20"),
        pd.Timestamp("2026-04-04 17:05"),
        pd.Timestamp("2026-04-04 20:10"),
    ]
    assert games["schedule_label"].tolist() == ["Sunny", None, None]


def test_an_mlb_game_listed_twice_keeps_its_latest_start():
    """A suspended game is listed on the date it began and the date it resumed, both Final."""
    # The resumption comes first so only the latest start, not the last listing, can win.
    payload = _payload(
        ("2026-06-11", [_mlb_game(7, 2681, "2026-06-11T17:05:00Z", state="Final")]),
        ("2026-06-10", [_mlb_game(7, 2681, "2026-06-10T23:05:00Z", state="Final")]),
    )

    assert weather.mlb_games(payload)["kickoff"].tolist() == [pd.Timestamp("2026-06-11 17:05")]


def test_an_unknown_venue_warns_once_however_many_games_it_holds(caplog):
    schedule = _schedule(
        ("2026_09_KC_MIA", "2026-11-01", "09:30", "outdoors", "Estadio Nuevo"),
        ("2026_14_LV_DEN", "2026-12-13", "09:30", "outdoors", "Estadio Nuevo"),
        ("2026_09_NYG_GB", "2026-11-01", "13:00", "outdoors", "Lambeau Field"),
    )
    payload = _payload(
        (
            "2026-04-03",
            [
                _mlb_game(8, 9999, "2026-04-03T18:20:00Z"),
                _mlb_game(9, 9999, "2026-04-03T23:05:00Z"),
            ],
        )
    )

    with caplog.at_level(logging.WARNING, logger="log"):
        nfl = weather.nfl_games(schedule)
        mlb = weather.mlb_games(payload)

    assert nfl["venue"].tolist() == ["Lambeau Field"]
    assert mlb.empty
    assert [record.getMessage() for record in caplog.records] == [
        "NFL weather: no venue entry for 'Estadio Nuevo'; its games get no row",
        "MLB weather: no venue entry for '9999'; its games get no row",
    ]


def test_request_params_list_each_location_in_row_order_on_the_pinned_model():
    venues = pd.DataFrame(
        {"venue": ["Lambeau Field", "17"], "lat": [44.501, 41.9482], "lon": [-88.062, -87.6555]}
    )

    assert weather.request_params(venues, ["temperature_2m", "wind_speed_10m"]) == {
        "latitude": "44.501,41.9482",
        "longitude": "-88.062,-87.6555",
        "hourly": "temperature_2m,wind_speed_10m",
        "models": "gfs_global",
        "timezone": "UTC",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
    }


def test_hourly_frame_flattens_a_multi_location_array_in_request_order():
    times = ["2026-11-15T18:00", "2026-11-15T19:00"]
    reply = [
        _location(times, temperature_2m=[31.2, 30.8]),
        _location(times, temperature_2m=[44.0, 43.1]),
    ]

    frame = weather.hourly_frame(reply, ["Lambeau Field", "Soldier Field"])

    assert frame.columns.tolist() == ["venue", "forecast_hour", "temperature_2m"]
    assert frame["venue"].tolist() == ["Lambeau Field"] * 2 + ["Soldier Field"] * 2
    assert frame["forecast_hour"].tolist() == [pd.Timestamp(time) for time in times] * 2
    assert frame["temperature_2m"].tolist() == [31.2, 30.8, 44.0, 43.1]


def test_hourly_frame_reads_a_single_location_object_and_keeps_variable_names():
    """A Previous Runs reply's variables keep their lead suffix."""
    reply = _location(["2024-10-06T17:00"], wind_speed_10m_previous_day4=[12.5])

    assert weather.hourly_frame(reply, ["17"]).to_dict("records") == [
        {
            "venue": "17",
            "forecast_hour": pd.Timestamp("2024-10-06 17:00"),
            "wind_speed_10m_previous_day4": 12.5,
        }
    ]


def test_an_open_game_gets_six_hours_from_the_hour_before_its_kickoff_hour():
    """A 23:40 UTC first pitch floors to 23:00; the hour past the forecast horizon stays, NaN."""
    games = weather.mlb_games(_payload(("2026-07-04", [_mlb_game(10, 17, "2026-07-04T23:40:00Z")])))
    times = pd.date_range("2026-07-04 21:00", periods=6, freq="h").strftime("%Y-%m-%dT%H:%M")
    temperatures = [70.0, 71.0, 72.0, 73.0, 74.0, 75.0]
    hourly = weather.hourly_frame(
        _location(list(times), **dict.fromkeys(weather.HOURLY_VARS, temperatures)), ["17"]
    )

    rows = weather.window_rows(games, hourly)

    assert rows.columns.tolist() == list(weather.WEATHER_DTYPES)
    assert rows["forecast_hour"].tolist() == list(
        pd.date_range("2026-07-04 22:00", periods=6, freq="h")
    )
    assert rows["temperature_2m"].tolist()[:5] == temperatures[1:]
    assert pd.isna(rows["temperature_2m"].iloc[5])
    assert (rows["model"] == weather.MODEL).all()


def test_a_dome_game_gets_one_empty_row_and_a_retractable_its_full_window():
    games = weather.nfl_games(
        _schedule(
            ("2026_13_CHI_DET", "2026-12-06", "13:00", "dome", "Ford Field"),
            ("2026_13_TEN_HOU", "2026-12-06", "13:00", None, "Reliant Stadium"),
        )
    )
    times = pd.date_range("2026-12-06 17:00", periods=6, freq="h").strftime("%Y-%m-%dT%H:%M")
    winds = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    hourly = weather.hourly_frame(_location(list(times), wind_speed_10m=winds), ["NRG Stadium"])

    rows = weather.window_rows(games, hourly)

    dome = rows[rows["venue"] == "Ford Field"]
    assert len(dome) == 1
    assert pd.isna(dome["forecast_hour"].iloc[0])
    assert pd.isna(dome["wind_speed_10m"].iloc[0])
    retractable = rows[rows["venue"] == "NRG Stadium"]
    assert (retractable["roof"] == "retractable").all()
    assert retractable["wind_speed_10m"].tolist() == winds


def test_every_committed_venue_has_coordinates_and_a_roof_class():
    for entry in [*weather.VENUES["NFL"].values(), *weather.VENUES["MLB"].values()]:
        assert isinstance(entry["lat"], float | int) and -90 <= entry["lat"] <= 90, entry
        assert isinstance(entry["lon"], float | int) and -180 <= entry["lon"] <= 180, entry
        assert entry["roof"] in {"open", "retractable", "dome"}, entry
    aliases = [alias for entry in weather.VENUES["NFL"].values() for alias in entry["aliases"]]
    assert len(aliases) == len(set(aliases))
    assert not set(aliases) & set(weather.VENUES["NFL"])
    assert all(key.isdigit() for key in weather.VENUES["MLB"])
    assert all(
        entry["azimuth"] is None or 0 <= entry["azimuth"] < 360
        for entry in weather.VENUES["MLB"].values()
    )
