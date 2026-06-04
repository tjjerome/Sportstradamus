"""Odds API ingest for game-level and player-prop markets.

Entry point: ``confer`` (wired to ``poetry run confer`` via ``pyproject.toml``).
Two internal workhorses:

* :func:`get_moneylines` — fetches h2h / totals / spreads for every active
  league of interest and writes book-level EVs to the archive's
  ``Moneyline`` / ``Totals`` buckets.
* :func:`get_props` — fetches per-event player prop markets, computes a
  no-vig EV per book, and writes them to the per-market archive buckets.

Both support a historical mode (``date != today``) that routes the request
through ``the-odds-api.com``'s historical endpoints; that path uses the
paid ``odds_api_plus`` key because the free tier doesn't include history.

The ``--close-lines`` flag swaps the broad pipeline for a cheap targeted
pass: it reads ``data/upcoming_events.json`` (refreshed by every broad
``confer`` run) and only fires per-event endpoint calls for games starting
within the closing window. If the window is empty, it exits before
touching the archive or the network. The intended cron schedule —
American-sports start hours only — is::

    */5 11-23,0-1 * * * cd <repo> && poetry run confer --close-lines
"""

import importlib.resources as pkg_resources
import json
from datetime import datetime, timedelta
from http import HTTPStatus
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from time import sleep

import click
import numpy as np
import pytz
import requests
from tqdm import tqdm

from sportstradamus import creds, data
from sportstradamus.helpers import (
    Archive,
    abbreviations,
    get_ev,
    get_logger,
    no_vig_odds,
    remove_accents,
    stat_cv,
    stat_dist,
)
from sportstradamus.helpers.io import read_upcoming_events, write_upcoming_events
from sportstradamus.spiderLogger import logger

# Closing-line capture window. Run `confer --close-lines` every 5 minutes
# during American-sports hours; only events commencing inside this window
# get a per-event endpoint hit, so the worst-case token cost per tick is
# bounded by `len(upcoming_events.json) * markets_per_event` and most ticks
# are no-ops.
CLOSING_LEAD_MIN = 5
CLOSING_LEAD_MAX = 25

# Leagues with live odds we care about. The Odds API also surfaces NHL / MLB
# but their prop coverage is thin enough that we get those from the direct
# book scrapers in sportstradamus.books instead.
LEAGUES_OF_INTEREST = ("NBA", "NFL", "WNBA")

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_SPORTS_URL = f"{_ODDS_API_BASE}/sports/"
ODDS_API_EVENTS_URL = f"{_ODDS_API_BASE}/sports/{{sport}}/events"
ODDS_API_EVENT_ODDS_URL = f"{_ODDS_API_BASE}/sports/{{sport}}/events/{{eventId}}/odds"
ODDS_API_ODDS_URL = f"{_ODDS_API_BASE}/sports/{{sport}}/odds/"
ODDS_API_HISTORICAL_EVENTS_URL = f"{_ODDS_API_BASE}/historical/sports/{{sport}}/events"
ODDS_API_HISTORICAL_EVENT_ODDS_URL = (
    f"{_ODDS_API_BASE}/historical/sports/{{sport}}/events/{{eventId}}/odds"
)
ODDS_API_HISTORICAL_ODDS_URL = f"{_ODDS_API_BASE}/sports/{{sport}}/odds-history/"

# Warn when the Odds API account drops below this many remaining requests so the
# season-long token budget doesn't silently run dry mid-slate.
_LOW_API_CREDITS_THRESHOLD = 50


class OddsAPIAuthError(RuntimeError):
    """Raised on an Odds API ``401`` — out of usage credits or a bad key.

    Both are unrecoverable for the current run, so we fail loud rather than
    silently writing an empty archive. The historical backfill catches this
    to checkpoint progress and exit cleanly for a later resume.
    """


def _get_with_retry(url, params=None):
    """GET ``url`` with one 429-retry. Returns the ``requests.Response``.

    The Odds API hands back 429s under bursty load; a single 1-second
    retry clears them in practice. A ``401`` (out of credits / bad key)
    raises :class:`OddsAPIAuthError`; other non-200 statuses propagate back
    so callers can decide whether to ``continue`` or bail.
    """
    res = requests.get(url, params=params)
    if res.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        sleep(1)
        res = requests.get(url, params=params)
    if res.status_code == HTTPStatus.UNAUTHORIZED:
        raise OddsAPIAuthError(f"Odds API 401 at {url}: {res.text[:200]}")
    return res


@click.command()
@click.option(
    "--close-lines",
    "close_lines",
    is_flag=True,
    default=False,
    help=(
        "Targeted close-line scrape only. Reads data/upcoming_events.json and hits "
        "per-event endpoints for games starting within the closing window. "
        "Exits without touching archive or API when no events are due."
    ),
)
@click.option(
    "--fixture-dir",
    "fixture_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Read canned Odds API responses from this directory instead of hitting the "
        "live API. Used by the integration test suite. Skips game-level moneyline "
        "fetches; player props are loaded from <fixture_dir>/sports.json, "
        "<fixture_dir>/events_<sport_key>.json, and "
        "<fixture_dir>/odds_<sport_key>_<event_id>.json."
    ),
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Verbosity for the structured JSONL log.",
)
def confer(close_lines: bool, fixture_dir: Path | None, log_level: str):
    """Fetch current odds and player props into the archive."""
    cli_log = get_logger("confer")
    cli_log.setLevel(log_level)
    cli_log.info(
        "confer invoked",
        extra={"close_lines": close_lines, "fixture_dir": str(fixture_dir)},
    )
    filepath = pkg_resources.files(creds) / "keys.json"
    with open(filepath) as infile:
        keys = json.load(infile)

    filepath = pkg_resources.files(data) / "config" / "stat_map.json"
    with open(filepath) as infile:
        stat_map = json.load(infile)

    if close_lines:
        _close_lines_pass(keys["odds_api_plus"], stat_map["Odds API"])
        return

    archive = Archive()
    logger.info("Archive loaded")

    if fixture_dir is None:
        archive = get_moneylines(archive, keys)
        logger.info("Game data complete")

    archive = get_props(
        archive, keys["odds_api_plus"], stat_map["Odds API"], fixture_dir=fixture_dir
    )
    logger.info("Player data complete, writing to file...")

    archive.write()
    logger.info("Success!")


def _moneyline_sports(apikey, sport, key, historical):
    """Resolve ``[(odds_api_sport_key, league), ...]`` and the low-credit flag.

    Returns ``(None, False)`` to signal the caller to bail without touching the
    archive. ``sport="All"`` enumerates active leagues from the Odds API sports
    index filtered to ``LEAGUES_OF_INTEREST``; an explicit ``sport`` requires the
    caller to pass the Odds API ``key`` directly (used by the backfill script).
    """
    if sport != "All":
        if key is None:
            logger.warning("Key needed for sports other than All")
            return None, False
        return [(key, sport)], False
    if historical:
        logger.warning("All sports only supported if date is today")
        return None, False
    res = _get_with_retry(ODDS_API_SPORTS_URL, params={"apiKey": apikey["odds_api"]})
    if res.status_code != HTTPStatus.OK:
        return None, False
    low_on_credits = int(res.headers.get("X-Requests-Remaining")) < _LOW_API_CREDITS_THRESHOLD
    sports = [
        (s["key"], s["title"])
        for s in res.json()
        if s["title"] in LEAGUES_OF_INTEREST and s["active"]
    ]
    return sports, low_on_credits


def _moneyline_request(apikey, date, historical, low_on_credits):
    markets = ["h2h", "totals", "spreads"]
    if historical:
        return ODDS_API_HISTORICAL_ODDS_URL, 1, {
            "apiKey": apikey["odds_api_plus"],
            "regions": "us",
            "date": date.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "markets": ",".join(markets),
        }
    return ODDS_API_ODDS_URL, 6, {
        "apiKey": apikey["odds_api_plus"] if low_on_credits else apikey["odds_api"],
        "regions": "us",
        "markets": ",".join(markets),
    }


def _parse_market_books(game):
    """Parse one game's bookmakers into per-book no-vig EV dicts.

    Returns ``(moneyline_home, moneyline_away, totals, spread_home,
    spread_away)``, each keyed by book; the caller blends ``totals`` with the
    spreads into the archive's ``Totals`` bucket.
    """
    moneyline_home, moneyline_away = {}, {}
    totals = {}
    spread_home, spread_away = {}, {}
    for book in game["bookmakers"]:
        for market in book["markets"]:
            if market["key"] == "h2h":
                odds = no_vig_odds(
                    market["outcomes"][0]["price"], market["outcomes"][1]["price"]
                )
                if market["outcomes"][0]["name"] == game["home_team"]:
                    moneyline_home[book["key"]] = odds[0]
                    moneyline_away[book["key"]] = odds[1]
                else:
                    moneyline_home[book["key"]] = odds[1]
                    moneyline_away[book["key"]] = odds[0]
            elif market["key"] == "totals":
                outcomes = sorted(market["outcomes"], key=itemgetter("name"))
                odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
                totals[book["key"]] = get_ev(outcomes[1]["point"], odds[1])
            elif market["key"] == "spreads" and market["outcomes"][0].get("point"):
                outcomes = sorted(market["outcomes"], key=itemgetter("point"))
                odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
                spread = get_ev(outcomes[1]["point"], odds[1])
                if outcomes[0]["name"] == game["home_team"]:
                    spread_home[book["key"]] = spread
                    spread_away[book["key"]] = -spread
                else:
                    spread_home[book["key"]] = -spread
                    spread_away[book["key"]] = spread
    return moneyline_home, moneyline_away, totals, spread_home, spread_away


def _store_game_moneylines(archive, game, league, date, dayDelta):
    """Resolve one game's date + teams and write its moneyline/totals books."""
    gameDate = datetime.fromisoformat(game["commence_time"]).astimezone(
        pytz.timezone("America/Chicago")
    )
    if gameDate > date + timedelta(days=dayDelta):
        return
    gameDate = gameDate.strftime("%Y-%m-%d")

    homeTeam = abbreviations[league].get(remove_accents(game["home_team"]))
    awayTeam = abbreviations[league].get(remove_accents(game["away_team"]))
    if homeTeam is None or awayTeam is None:
        return

    moneyline_home, moneyline_away, totals, spread_home, spread_away = _parse_market_books(game)
    archive.set_team_books(league, "Moneyline", gameDate, awayTeam, moneyline_away)
    archive.set_team_books(league, "Moneyline", gameDate, homeTeam, moneyline_home)
    archive.set_team_books(
        league,
        "Totals",
        gameDate,
        awayTeam,
        {k: (v + spread_away.get(k, 0)) / 2 for k, v in totals.items()},
    )
    archive.set_team_books(
        league,
        "Totals",
        gameDate,
        homeTeam,
        {k: (v + spread_home.get(k, 0)) / 2 for k, v in totals.items()},
    )


def get_moneylines(
    archive,
    apikey,
    date=datetime.now().astimezone(pytz.timezone("America/Chicago")),
    sport="All",
    key=None,
):
    """Fetch h2h / totals / spreads into archive's Moneyline & Totals buckets.

    When ``sport="All"`` (the ``confer`` default), enumerates active leagues
    from the Odds API sports index and filters to ``LEAGUES_OF_INTEREST``.
    When called with an explicit ``sport`` + ``key`` the caller supplies
    the Odds API sport key directly (used by ``scripts/moneylines_hist.py``
    for backfills).
    """
    historical = date.date() != datetime.today().date()
    sports, low_on_credits = _moneyline_sports(apikey, sport, key, historical)
    if sports is None:
        return archive

    url_template, dayDelta, params = _moneyline_request(apikey, date, historical, low_on_credits)
    for sport, league in sports:
        res = _get_with_retry(url_template.format(sport=sport), params=params)
        if res.status_code != HTTPStatus.OK:
            continue
        games = res.json()["data"] if historical else res.json()
        for game in tqdm(games, desc=f"Getting {league} Game Data", unit="game"):
            _store_game_moneylines(archive, game, league, date, dayDelta)

    return archive


def get_props(
    archive,
    apikey,
    props,
    date=datetime.now().astimezone(pytz.timezone("America/Chicago")),
    sport="All",
    key=None,
    fixture_dir: Path | None = None,
):
    """Fetch per-event player-prop markets and store book-level EVs."""
    stat_cv["NCAAB"] = stat_cv["NBA"]
    stat_cv["NCAAF"] = stat_cv["NFL"]
    historical = date.date() != datetime.today().date()

    if fixture_dir is not None:
        return _get_props_from_fixtures(archive, props, Path(fixture_dir), date)

    if sport == "All":
        if historical:
            logger.warning("All sports only supported if date is today")
            return archive
        res = _get_with_retry(ODDS_API_SPORTS_URL, params={"apiKey": apikey})
        if res.status_code != HTTPStatus.OK:
            return archive

        res = res.json()
        sports = [
            (s["key"], s["title"]) for s in res if s["title"] in LEAGUES_OF_INTEREST and s["active"]
        ]
    elif key is None:
        logger.warning("Key needed for sports other than All")
        return archive
    else:
        sports = [(key, sport)]

    if historical:
        event_url_template = ODDS_API_HISTORICAL_EVENTS_URL
        odds_url_template = ODDS_API_HISTORICAL_EVENT_ODDS_URL
        dayDelta = 1
        params = {
            "apiKey": apikey,
            "date": date.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    else:
        event_url_template = ODDS_API_EVENTS_URL
        odds_url_template = ODDS_API_EVENT_ODDS_URL
        dayDelta = 6
        params = {"apiKey": apikey}

    ledger = {(e["sport_key"], e["event_id"]): e for e in read_upcoming_events()}

    for sport, league in sports:
        params.update({"markets": ",".join(props[league].keys())})
        if league == "MLB":
            params["markets"] = params["markets"] + ",totals_1st_1_innings,spreads_1st_1_innings"
        events = _get_with_retry(event_url_template.format(sport=sport), params=params)
        if events.status_code != HTTPStatus.OK:
            continue

        events = events.json()["data"] if historical else events.json()

        for event in events:
            gameDate = datetime.fromisoformat(event["commence_time"]).astimezone(
                pytz.timezone("America/Chicago")
            )
            if gameDate > date + timedelta(days=dayDelta):
                continue
            gameDate_str = gameDate.strftime("%Y-%m-%d")
            event_params = {**params, "regions": "us"}
            res = _get_with_retry(
                odds_url_template.format(sport=sport, eventId=event["id"]), params=event_params
            )
            if res.status_code == HTTPStatus.NOT_FOUND:
                return archive
            if res.status_code != HTTPStatus.OK:
                continue

            game = res.json()["data"] if historical else res.json()
            observed_at = _historical_observed_at(historical, gameDate)
            _archive_event_props(archive, game, league, props, gameDate_str, observed_at)

            if not historical:
                ledger[(sport, event["id"])] = {
                    "sport_key": sport,
                    "event_id": event["id"],
                    "league": league,
                    "commence_time": event["commence_time"],
                    "markets": list(props[league].keys()),
                }

    if not historical:
        write_upcoming_events(_prune_upcoming_events(list(ledger.values())))

    return archive


def _historical_observed_at(historical, gameDate):
    """Snapshot stamp for backfilled rows, or ``None`` for live runs.

    Live writes default to ``utcnow()``. Backfilled rows stamp game-day
    01:00 so they beat the migrated midnight (``00:00``) rows on recency
    while staying before any ``kickoff - 8h`` point-in-time training read.
    """
    if not historical:
        return None
    return datetime(gameDate.year, gameDate.month, gameDate.day, 1, 0, 0)


def _resolve_prop_lines(lines):
    """Normalize one player's raw option rows to the one or two lines we price.

    Defaults a missing point to 0.5, renames ``Yes``/``No`` to ``Over``/
    ``Under``, collapses a multi-point quote to the line whose price is closest
    to even, trims to a single Over/Under pair, and flips a lone ``Under`` to an
    ``Over`` with the inverted price.
    """
    lines = list(lines)
    for line in lines:
        line.setdefault("point", 0.5)
        line["name"] = {"Yes": "Over", "No": "Under"}.get(line["name"], line["name"])

    lines = sorted(lines, key=itemgetter("name"))
    if len({line["point"] for line in lines}) > 1:
        trueline = sorted(lines, key=(lambda x: np.abs(x["price"] - 2)))[0]["point"]
        lines = [line for line in lines if line["point"] == trueline]
    if len(lines) > 2:
        lines = [
            next(line for line in lines if line["name"] == "Over"),
            next(line for line in lines if line["name"] == "Under"),
        ]
    if len(lines) == 1 and lines[0]["name"] == "Under":
        lines[0]["name"] = "Over"
        under = lines[0]["price"]
        lines[0]["price"] = under / (under - 1)
    return lines


def _event_player_props(book, market, league, props, odds):
    """Accumulate one book's player-prop market into ``odds`` (mutated)."""
    market_name = props[league].get(market["key"])
    odds.setdefault(market_name, {})

    outcomes = [
        o for o in market["outcomes"] if "description" in o and "name" in o and o["price"] > 1
    ]
    outcomes = sorted(outcomes, key=itemgetter("description", "name"))

    for player, lines in groupby(outcomes, itemgetter("description")):
        player = remove_accents(player).replace(" Total", "")
        odds[market_name].setdefault(player, {"EV": {}, "Lines": []})
        lines = _resolve_prop_lines(lines)
        line = lines[0].get("point", 0.5)
        odds[market_name][player]["Lines"].append(line)
        price = no_vig_odds(*[x["price"] for x in lines])
        ev = get_ev(
            line,
            price[1],
            stat_cv[league].get(market_name, 1),
            dist=stat_dist.get(league, {}).get(market_name, "SkewNormal"),
        )
        odds[market_name][player]["EV"][book["key"]] = ev


def _event_totals_book(market, book, totals):
    """Accumulate one book's totals sub-market into ``totals`` (mutated)."""
    spread_name = " ".join(market["key"].split("_")[1:])
    outcomes = sorted(market["outcomes"], key=itemgetter("name"))
    sub_odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
    totals.setdefault(spread_name, {})
    totals[spread_name][book["key"]] = get_ev(outcomes[1]["point"], sub_odds[1])


def _event_spread_book(market, book, game, spread_home, spread_away):
    """Accumulate one book's spread sub-market into the spread dicts (mutated)."""
    spread_name = " ".join(market["key"].split("_")[1:])
    outcomes = sorted(market["outcomes"], key=itemgetter("point"))
    sub_odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
    spread = get_ev(outcomes[1]["point"], sub_odds[1])
    spread_home.setdefault(spread_name, {})
    spread_away.setdefault(spread_name, {})
    if outcomes[0]["name"] == game["home_team"]:
        spread_home[spread_name][book["key"]] = spread
        spread_away[spread_name][book["key"]] = -spread
    else:
        spread_home[spread_name][book["key"]] = -spread
        spread_away[spread_name][book["key"]] = spread


def _write_event_totals(archive, game, league, gameDate, totals, spread_home, spread_away):
    """Write the totals team buckets, blending in the parsed spreads."""
    for market in totals:
        home_team = abbreviations[league][remove_accents(game["home_team"])]
        away_team = abbreviations[league][remove_accents(game["away_team"])]
        # spread_home/away are keyed by spread-name; totals is keyed by book.
        # The .get(k, 0) always falls through to 0, so the written value is
        # simply total / 2. Intentional: preserve the archived behavior.
        archive.set_team_books(
            league,
            market,
            gameDate,
            home_team,
            {k: (v + spread_home.get(k, 0)) / 2 for k, v in totals[market].items()},
        )
        archive.set_team_books(
            league,
            market,
            gameDate,
            away_team,
            {k: (v + spread_away.get(k, 0)) / 2 for k, v in totals[market].items()},
        )


def _archive_event_props(archive, game, league, props, gameDate, observed_at=None):
    """Parse one Odds API event response and write its odds into ``archive``.

    Splits markets into player props (per-player EV and consensus lines),
    totals (game-total team buckets), and spreads (used to fold home/away
    team-total adjustments back into the totals write). Mirrors the inline
    logic that ``get_props`` previously ran, hoisted out so the
    ``--close-lines`` pass can reuse it without duplicating the parser.
    """
    odds = {}
    totals = {}
    spread_home = {}
    spread_away = {}

    for book in game["bookmakers"]:
        for market in book["markets"]:
            if "totals" in market["key"]:
                _event_totals_book(market, book, totals)
            elif "spread" in market["key"]:
                _event_spread_book(market, book, game, spread_home, spread_away)
            else:
                _event_player_props(book, market, league, props, odds)

    for market in odds:
        for player, entry in odds[market].items():
            line = np.median(entry["Lines"])
            archive.merge_player_books(
                league, market, gameDate, player, entry["EV"], [line], observed_at=observed_at
            )

    _write_event_totals(archive, game, league, gameDate, totals, spread_home, spread_away)


def _get_props_from_fixtures(archive, props, fixture_dir: Path, date):
    """Replay canned Odds API responses from ``fixture_dir`` into ``archive``.

    Mirrors the live ``get_props`` flow but swaps the three HTTP touchpoints
    (sports index, per-sport events, per-event odds) for JSON files on disk.
    The downstream pipeline — archive writes, ledger updates, prop parsing
    via :func:`_archive_event_props` — is path-identical to a live run so
    the integration test exercises the same code paths.

    Expected fixture layout::

        <fixture_dir>/sports.json
        <fixture_dir>/events_<sport_key>.json
        <fixture_dir>/odds_<sport_key>_<event_id>.json

    ``sports.json`` filters which sport keys are processed; only entries
    whose ``title`` is in :data:`LEAGUES_OF_INTEREST` and whose ``active``
    flag is true are considered.
    """
    sports_path = fixture_dir / "sports.json"
    with open(sports_path) as infile:
        sports_payload = json.load(infile)
    sports = [
        (s["key"], s["title"])
        for s in sports_payload
        if s["title"] in LEAGUES_OF_INTEREST and s["active"]
    ]

    ledger = {(e["sport_key"], e["event_id"]): e for e in read_upcoming_events()}

    for sport, league in sports:
        events_path = fixture_dir / f"events_{sport}.json"
        if not events_path.is_file():
            continue
        with open(events_path) as infile:
            events = json.load(infile)

        for event in events:
            gameDate = datetime.fromisoformat(event["commence_time"]).astimezone(
                pytz.timezone("America/Chicago")
            )
            if gameDate > date + timedelta(days=6):
                continue
            gameDate_str = gameDate.strftime("%Y-%m-%d")

            odds_path = fixture_dir / f"odds_{sport}_{event['id']}.json"
            if not odds_path.is_file():
                continue
            with open(odds_path) as infile:
                game = json.load(infile)

            _archive_event_props(archive, game, league, props, gameDate_str)
            ledger[(sport, event["id"])] = {
                "sport_key": sport,
                "event_id": event["id"],
                "league": league,
                "commence_time": event["commence_time"],
                "markets": list(props[league].keys()),
            }

    write_upcoming_events(_prune_upcoming_events(list(ledger.values())))
    return archive


def _prune_upcoming_events(events):
    """Drop events whose commence_time is in the past (UTC)."""
    now = datetime.now(pytz.utc)
    keep = []
    for e in events:
        try:
            ts = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        except (ValueError, KeyError, AttributeError):
            continue
        if ts.astimezone(pytz.utc) > now:
            keep.append(e)
    return keep


def _close_lines_pass(apikey, props):
    """Per-event close-line scrape. Exits early when the window is empty.

    Loads ``data/upcoming_events.json``, filters to events with a
    ``commence_time`` between ``CLOSING_LEAD_MIN`` and ``CLOSING_LEAD_MAX``
    minutes from now, and only then opens ``Archive`` and hits the per-event
    Odds API endpoint for each. Past-commence entries are pruned on the way
    out so the ledger stays small. The five-minute cron tick yields no
    archive read, no API call, and no log noise on empty windows — the
    intended common case.
    """
    ledger = read_upcoming_events()
    ledger = _prune_upcoming_events(ledger)

    now = datetime.now(pytz.utc)
    window_start = now + timedelta(minutes=CLOSING_LEAD_MIN)
    window_end = now + timedelta(minutes=CLOSING_LEAD_MAX)

    due = []
    for e in ledger:
        try:
            ts = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        except (ValueError, KeyError, AttributeError):
            continue
        ts_utc = ts.astimezone(pytz.utc)
        if window_start <= ts_utc <= window_end:
            due.append(e)

    if not due:
        write_upcoming_events(ledger)
        return

    logger.info(f"close-lines: {len(due)} event(s) due")
    archive = Archive()
    params = {"apiKey": apikey, "regions": "us"}

    for e in due:
        sport_key = e["sport_key"]
        event_id = e["event_id"]
        league = e["league"]
        markets = e.get("markets") or list(props.get(league, {}).keys())
        if not markets:
            continue
        event_params = {**params, "markets": ",".join(markets)}
        res = _get_with_retry(
            ODDS_API_EVENT_ODDS_URL.format(sport=sport_key, eventId=event_id),
            params=event_params,
        )
        if res.status_code != HTTPStatus.OK:
            logger.warning(f"close-lines: {league} {event_id} returned status {res.status_code}")
            continue
        game = res.json()
        gameDate = (
            datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            .astimezone(pytz.timezone("America/Chicago"))
            .strftime("%Y-%m-%d")
        )
        _archive_event_props(archive, game, league, props, gameDate)

    archive.write()
    write_upcoming_events(_prune_upcoming_events(ledger))
    logger.info("close-lines: archive updated")


if __name__ == "__main__":
    confer()
