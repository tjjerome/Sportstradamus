"""The training read of book quotes: each row at its game's start minus ``TRAINING_LOOKBACK``.

``stats.game_starts`` keeps each league's starts in a store; ``Stats.get_training_matrix`` and
``scripts/inject_backfilled_odds`` read every row through its ``read_times`` (owner,
2026-11-20). The store sits in a temporary leagues dir, and each league's fetch answers a
payload in the live response's shape, trimmed to the keys the parse reads, so nothing here
touches the network.
"""

import logging
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from nba_api.stats.endpoints import scheduleleaguev2

from sportstradamus.scripts.inject_backfilled_odds import resolve_cached_quotes
from sportstradamus.stats import game_starts, nba_client
from sportstradamus.stats.base import Stats

_DAY = date(2026, 1, 15)
_LOG_STRINGS = {
    "game": "game",
    "player": "player",
    "date": "date",
    "team": "team",
    "opponent": "opponent",
    "home": "home",
    "usage": "usage",
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An empty store: every league's file lands under ``tmp_path``."""
    monkeypatch.setattr(game_starts, "LEAGUES_DIR", tmp_path)
    return tmp_path


def _stored(store, league: str, starts: dict[str, str]) -> None:
    frame = pd.DataFrame({"game_id": list(starts), "start": pd.to_datetime(list(starts.values()))})
    (store / league.lower()).mkdir(parents=True, exist_ok=True)
    frame.to_parquet(store / league.lower() / "game_starts.parquet", index=False)


def _schedule_lists(monkeypatch, league: str, starts: dict[str, str]) -> list[list[str]]:
    """Stand the league's schedule in with one listing ``starts``; record each fetch's games."""
    fetches = []

    def fetch(_league, missing):
        fetches.append(sorted(missing["game_id"]))
        return pd.Series(starts, dtype=object)

    monkeypatch.setitem(game_starts._SOURCES, league, fetch)
    return fetches


def _games(*ids: str) -> pd.DataFrame:
    return pd.DataFrame({"game_id": list(ids), "date": [_DAY] * len(ids)})


def _training_stats(gamelog: pd.DataFrame, reads: list) -> SimpleNamespace:
    """A Stats stand-in whose quote resolver records each batch's players and read time."""

    def resolve(stats, _market, _date, at):
        reads.append((sorted(stats.index), at))
        return pd.DataFrame({"Archived": True}, index=stats.index)

    def get_stats(_market, offers, _game_date):
        players = [offer["Player"] for offer in offers]
        return pd.DataFrame({"Home": [True] * len(players), "Avg10": 1.0}, index=players)

    return SimpleNamespace(
        league="NBA",
        gamelog=gamelog,
        usage_stat="usage",
        volume_stats=[],
        log_strings=_LOG_STRINGS,
        preflight_training_dependencies=lambda _market: None,
        _market_position_filter=lambda frame, _market: frame,
        _dispatch_volume_stats=lambda _offers, _game_date, _market: None,
        get_stats=get_stats,
        resolve_player_market_odds=resolve,
    )


def _gamelog(*rows: tuple[str, str]) -> pd.DataFrame:
    """One gameday's gamelog: a ``(game, player)`` per row."""
    return pd.DataFrame(
        {
            "game": [game for game, _ in rows],
            "player": [player for _, player in rows],
            "date": [_DAY] * len(rows),
            "team": "AAA",
            "opponent": "BBB",
            "home": True,
            "usage": 30,
            "market": 10,
        }
    )


def test_two_games_on_one_date_read_each_at_its_own_start(store):
    """A 7:00 PM ET tip reads at 23:00 UTC, a 10:00 PM ET one at 02:00 the next day."""
    _stored(store, "NBA", {"0022500601": "2026-01-16 00:00", "0022500602": "2026-01-16 03:00"})
    gamelog = _gamelog(
        ("0022500601", "Early One"),
        ("0022500601", "Early Two"),
        ("0022500602", "Late One"),
    )
    reads = []

    matrix = Stats.get_training_matrix(
        _training_stats(gamelog, reads), "market", cutoff_date=date(2025, 1, 1)
    )

    assert reads == [
        (["Early One", "Early Two"], pd.Timestamp("2026-01-15 23:00")),
        (["Late One"], pd.Timestamp("2026-01-16 02:00")),
    ]
    assert matrix.set_index("Player")["GameTime"].to_dict() == {
        "Early One": pd.Timestamp("2026-01-16 00:00"),
        "Early Two": pd.Timestamp("2026-01-16 00:00"),
        "Late One": pd.Timestamp("2026-01-16 03:00"),
    }


def test_a_game_with_no_start_reads_at_noon_utc_and_is_counted(store, monkeypatch, caplog):
    """The schedule lists no start for the game (a TBD doubleheader second game, say)."""
    _stored(store, "NBA", {"0022500601": "2026-01-16 00:00"})
    _schedule_lists(monkeypatch, "NBA", {})
    gamelog = _gamelog(
        ("0022500601", "Timed One"),
        ("0022500603", "Untimed One"),
        ("0022500603", "Untimed Two"),
    )
    reads = []

    with caplog.at_level(logging.WARNING, logger="log"):
        matrix = Stats.get_training_matrix(
            _training_stats(gamelog, reads), "market", cutoff_date=date(2025, 1, 1)
        )

    assert reads == [
        (["Untimed One", "Untimed Two"], pd.Timestamp("2026-01-15 12:00")),
        (["Timed One"], pd.Timestamp("2026-01-15 23:00")),
    ]
    assert not matrix["GameTime"].isna().any()
    assert "NBA: 2 of 3 training rows have no stored game start" in caplog.text


def test_the_store_fetches_only_the_games_it_lacks(store, monkeypatch):
    """A game the schedule lists no start for is stored NaT, so it is not fetched again either."""
    fetches = _schedule_lists(
        monkeypatch,
        "NBA",
        {"g1": "2026-01-16T00:00:00Z", "g2": "2026-01-16T03:00:00Z", "g9": "2026-01-16T01:00:00Z"},
    )

    game_starts.game_starts("NBA", _games("g1", "g3"))
    game_starts.game_starts("NBA", _games("g1", "g2", "g3"))
    starts = game_starts.game_starts("NBA", _games("g2", "g3", "g1"))

    assert fetches == [["g1", "g3"], ["g2"]]
    # Only the games asked for are kept: a future game the season lists could still move.
    assert starts.to_dict() == {
        "g1": pd.Timestamp("2026-01-16 00:00"),
        "g3": pd.NaT,
        "g2": pd.Timestamp("2026-01-16 03:00"),
    }


def test_a_stored_store_answers_without_a_fetch(store, monkeypatch):
    _stored(store, "NHL", {"2025020601": "2026-01-16 00:00"})
    fetches = _schedule_lists(monkeypatch, "NHL", {})

    reads = game_starts.read_times(
        "NHL", pd.Series([2025020601, 2025020601]), pd.Series([_DAY, _DAY])
    )

    assert fetches == []
    assert reads.tolist() == [pd.Timestamp("2026-01-15 23:00")] * 2


@pytest.mark.parametrize(
    ("league", "game_id", "season", "league_id", "host"),
    [
        ("NBA", "0022500601", "2025-26", "00", nba_client.NBA_HOST),
        ("WNBA", "1022500101", 2025, "10", nba_client.WNBA_HOST),
    ],
)
def test_nba_and_wnba_starts_come_from_each_seasons_schedule(
    store, monkeypatch, league, game_id, season, league_id, host
):
    calls = []
    payload = {
        "leagueSchedule": {
            "gameDates": [
                {"games": [{"gameId": game_id, "gameDateTimeUTC": "2026-01-16T00:30:00Z"}]}
            ]
        }
    }

    def fetch(endpoint_cls, *, host, **params):
        calls.append((endpoint_cls, host, params))
        return SimpleNamespace(get_dict=lambda: payload)

    monkeypatch.setattr(nba_client, "fetch", fetch)

    starts = game_starts.game_starts(league, _games(game_id))

    assert calls == [
        (scheduleleaguev2.ScheduleLeagueV2, host, {"league_id": league_id, "season": season})
    ]
    assert starts[game_id] == pd.Timestamp("2026-01-16 00:30")


def _nhl_week(first: str, games: dict[str, list[tuple[int, str]]]) -> dict:
    """api-web's ``schedule/{date}``: the seven days from ``first``, each with its games."""
    days = pd.date_range(first, periods=7).strftime("%Y-%m-%d")
    return {
        "gameWeek": [
            {
                "date": day,
                "games": [{"id": pk, "startTimeUTC": start} for pk, start in games.get(day, [])],
            }
            for day in days
        ]
    }


def test_nhl_starts_walk_only_the_weeks_holding_a_missing_game(store, monkeypatch):
    """Two 4 Nations games share a week; the final, a week on, takes a second request."""
    weeks = {
        "2025-02-12": _nhl_week(
            "2025-02-12",
            {
                "2025-02-12": [(2024190001, "2025-02-13T01:00:00Z")],
                "2025-02-15": [(2024190003, "2025-02-15T18:00:00Z")],
            },
        ),
        "2025-02-20": _nhl_week(
            "2025-02-20", {"2025-02-20": [(2024200007, "2025-02-21T01:00:00Z")]}
        ),
    }
    asked = []

    def get(url):
        asked.append(url)
        return weeks[url.rsplit("/", 1)[1]]

    monkeypatch.setattr(game_starts, "scraper", SimpleNamespace(get=get))
    games = pd.DataFrame(
        {
            "game_id": ["2024190001", "2024190003", "2024200007"],
            "date": [date(2025, 2, 12), date(2025, 2, 15), date(2025, 2, 20)],
        }
    )

    starts = game_starts.game_starts("NHL", games)

    assert asked == [
        "https://api-web.nhle.com/v1/schedule/2025-02-12",
        "https://api-web.nhle.com/v1/schedule/2025-02-20",
    ]
    assert starts.to_dict() == {
        "2024190001": pd.Timestamp("2025-02-13 01:00"),
        "2024190003": pd.Timestamp("2025-02-15 18:00"),
        "2024200007": pd.Timestamp("2025-02-21 01:00"),
    }


def test_an_nhl_week_the_scraper_could_not_read_raises_and_stores_nothing(store, monkeypatch):
    """``Scrape.get`` answers ``{}`` after its retries; storing NaT then would pin the stand-in."""
    monkeypatch.setattr(game_starts, "scraper", SimpleNamespace(get=lambda _url: {}))

    with pytest.raises(RuntimeError, match="NHL schedule for the week of 2025-02-12"):
        game_starts.game_starts(
            "NHL", pd.DataFrame({"game_id": ["2024190001"], "date": [date(2025, 2, 12)]})
        )

    assert not (store / "nhl" / "game_starts.parquet").exists()


def _mlb_game(pk: int, start: str, state: str = "Final", tbd: bool = False) -> dict:
    return {
        "gamePk": pk,
        "gameDate": start,
        "status": {"detailedState": state, "startTimeTBD": tbd},
        "venue": {"id": 17},
    }


def test_mlb_starts_keep_a_postponed_games_played_instance(store, monkeypatch):
    """One request per season spans its missing dates (statsapi cuts a span off a year past
    its start); a TBD start (a doubleheader's game 2) stays NaT."""
    payloads = {
        "2025-09-28": {
            "dates": [{"date": "2025-09-28", "games": [_mlb_game(5, "2025-09-28T19:10:00Z")]}]
        },
        "2026-06-10": {
            "dates": [
                {
                    "date": "2026-06-10",
                    "games": [
                        _mlb_game(7, "2026-06-10T23:05:00Z", state="Postponed"),
                        _mlb_game(8, "2026-06-10T17:05:00Z"),
                        _mlb_game(9, "2026-06-10T20:33:00Z", tbd=True),
                    ],
                },
                {"date": "2026-06-11", "games": [_mlb_game(7, "2026-06-11T17:10:00Z")]},
            ]
        },
    }
    requests_made = []

    def get(url, params, timeout):
        requests_made.append((url, params["startDate"], params["endDate"]))
        payload = payloads[params["startDate"]]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setattr(game_starts.requests, "get", get)
    games = pd.DataFrame(
        {
            "game_id": ["7", "8", "9", "5"],
            "date": [date(2026, 6, 11), date(2026, 6, 10), date(2026, 6, 10), date(2025, 9, 28)],
        }
    )

    starts = game_starts.game_starts("MLB", games)

    assert requests_made == [
        (game_starts.MLB_SCHEDULE_URL, "2025-09-28", "2025-09-28"),
        (game_starts.MLB_SCHEDULE_URL, "2026-06-10", "2026-06-11"),
    ]
    assert starts.to_dict() == {
        "7": pd.Timestamp("2026-06-11 17:10"),
        "8": pd.Timestamp("2026-06-10 17:05"),
        "9": pd.NaT,
        "5": pd.Timestamp("2025-09-28 19:10"),
    }


def test_nfl_starts_convert_eastern_kickoffs_to_utc_across_dst(store, monkeypatch):
    """8:20 PM EDT in September is 00:20 UTC the next day; 1:00 PM EST in December is 18:00."""
    seasons = []

    def import_schedules(years):
        seasons.append(years)
        return pd.DataFrame(
            {
                "game_id": ["2025_01_DAL_PHI", "2025_14_CHI_GB"],
                "gameday": ["2025-09-04", "2025-12-07"],
                "gametime": ["20:20", "13:00"],
            }
        )

    monkeypatch.setattr(game_starts.nfl, "import_schedules", import_schedules)
    games = pd.DataFrame(
        {
            "game_id": ["2025_01_DAL_PHI", "2025_14_CHI_GB"],
            "date": [date(2025, 9, 4), date(2025, 12, 7)],
        }
    )

    starts = game_starts.game_starts("NFL", games)

    assert seasons == [[2025]]
    assert starts.to_dict() == {
        "2025_01_DAL_PHI": pd.Timestamp("2025-09-05 00:20"),
        "2025_14_CHI_GB": pd.Timestamp("2025-12-07 18:00"),
    }


def test_the_repair_reads_each_cached_row_at_its_own_games_start(store, monkeypatch):
    """Cached rows carry Player and Date; the loaded gamelog names each row's game."""
    _stored(store, "NBA", {"0022500601": "2026-01-16 00:00", "0022500602": "2026-01-16 03:00"})
    _schedule_lists(monkeypatch, "NBA", {})
    reads = []

    def resolve(stats, _market, _date, at):
        reads.append((sorted(stats.index), at))
        return pd.DataFrame({"Odds": 0.5}, index=stats.index)

    stat_data = SimpleNamespace(
        league="NBA",
        log_strings={"game": "GAME_ID", "date": "GAME_DATE", "player": "PLAYER_NAME"},
        # The gamelog keeps an older row's raw accent; the matrix keys the row without it.
        gamelog=pd.DataFrame(
            {
                "GAME_ID": ["0022500601", "0022500602"],
                "GAME_DATE": ["2026-01-15T00:00:00", "2026-01-15T00:00:00"],
                "PLAYER_NAME": ["Nikola Jokić", "Late Player"],
            }
        ),
        depth_quoted_markets=frozenset(),
        window_short_logs=lambda _date: None,
        resolve_player_market_odds=resolve,
    )
    cached = pd.DataFrame(
        {
            "Player": ["Nikola Jokic", "Late Player", "Not Logged"],
            "Date": ["2026-01-15"] * 3,
            "Avg10": [20.0, 10.0, 5.0],
        }
    )

    resolve_cached_quotes(stat_data, "PTS", cached)

    assert reads == [
        (["Not Logged"], pd.Timestamp("2026-01-15 12:00")),
        (["Nikola Jokic"], pd.Timestamp("2026-01-15 23:00")),
        (["Late Player"], pd.Timestamp("2026-01-16 02:00")),
    ]
