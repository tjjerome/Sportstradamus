"""Odds API ingest for game-level and player-prop markets.

Entry point: ``confer`` (wired to ``poetry run confer`` via ``pyproject.toml``).
Two internal workhorses:

* :func:`get_moneylines` — fetches h2h / totals / spreads for every active
  league of interest and writes book-level EVs to the archive's
  ``Moneyline`` / ``Totals`` buckets.
* :func:`get_props` — fetches per-event player prop markets, computes a
  no-vig EV per book, and writes them to the per-market archive buckets.

Both support a historical mode (``date != today``) that routes the request
through ``the-odds-api.com``'s historical endpoints; that path spends from
the dedicated high-limit ``odds_api_max`` backfill key (the free tier has
no history), while live confer/close-lines stay on ``odds_api``/``odds_api_plus``.

Credit budgeting — the usage ledger and the broad-run pacing governor —
lives in ``sportstradamus.helpers.odds_budget``.

The ``--close-lines`` flag swaps the broad pipeline for a cheap targeted
pass: it reads ``data/upcoming_events.json`` (refreshed by every broad
``confer`` run) and only fires per-event endpoint calls for games starting
within the closing window. If the window is empty, it exits before
touching the archive or the network. The intended cron schedule —
American-sports start hours only — is::

    */10 11-23,0-1 * * * cd <repo> && poetry run confer --close-lines
"""

import importlib.resources as pkg_resources
import json
from datetime import datetime, timedelta
from http import HTTPStatus
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from time import sleep
from typing import NamedTuple

import click
import numpy as np
import pytz
import requests
from tqdm import tqdm

from sportstradamus import creds, data
from sportstradamus.helpers import (
    Archive,
    abbreviations,
    book_gate,
    get_ev,
    get_logger,
    no_vig_odds,
    odds_budget,
    remove_accents,
    stat_cv,
    stat_dist,
)
from sportstradamus.helpers.distributions import SN_MAX_MEAN_FACTOR
from sportstradamus.helpers.io import (
    read_league_activity,
    read_upcoming_events,
    write_league_activity,
    write_upcoming_events,
)
from sportstradamus.spiderLogger import logger

# Closing-line capture window. Run `confer --close-lines` every 10 minutes
# during American-sports hours; only events commencing inside this window
# get a per-event endpoint hit, so the worst-case token cost per tick is
# bounded by `len(upcoming_events.json) * markets_per_event` and most ticks
# are no-ops.
CLOSING_LEAD_MIN = 5
CLOSING_LEAD_MAX = 25

# Leagues with live odds we care about. All five tracked leagues since the
# 2026-07 MLB/NHL activation; the budget governor decides which get fetched
# on a given broad run, so an expensive slate throttles itself.
LEAGUES_OF_INTEREST = ("NBA", "NFL", "WNBA", "MLB", "NHL")

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

# How many days ahead of the run date to ingest games for. Live runs sweep a
# multi-day slate; historical backfill is anchored to a single day.
_LIVE_DAY_WINDOW = 6
_HIST_DAY_WINDOW = 1

# Markets whose main line has left the `us` region entirely: the only books still
# quoting them (fliff, ballybet, betparx, hardrockbet, espnbet) sit in `us2`, so a
# us-only request archives nothing and the cell silently loses its book. Billing is
# per market per region, so routing these to a second us2 request instead of folding
# them into the us one costs exactly what a us-only request cost. NBA TOV went dark on
# 2025-10-29 and was found this way; re-probe when a market goes quiet. Markets that
# merely have *more* books in us2 stay here deliberately -- buying that depth would add
# a region to every market, and the us pool is deep enough.
US2_ONLY_MARKETS = {
    "NBA": ("player_turnovers",),
}

# Game lines (totals, spreads) are symmetric: the no-vig median price IS the
# implied value, so their EV must invert under a symmetric distribution. Pin the
# family explicitly rather than lean on get_ev's default — a 2026-03 default flip
# to "Gamma" silently inflated every archived NBA total by 1/ln(2) (Gamma at cv=1
# is exponential, mean = median/ln(2)) before it was caught. "Normal" forces
# alpha=0 and is immune to any future change of get_ev's default.
_GAME_LINE_DIST = "Normal"


class OddsAPIAuthError(RuntimeError):
    """Raised on an Odds API ``401`` — out of usage credits or a bad key.

    Both are unrecoverable for the current run, so we fail loud rather than
    silently writing an empty archive. The historical backfill catches this
    to checkpoint progress and exit cleanly for a later resume.
    """


def _active_in_scope(sport, leagues=None):
    """True if an Odds API ``/sports`` entry is active and league-of-interest.

    Shared by the budget probe and the two ``*_sports`` resolvers so the
    ``LEAGUES_OF_INTEREST`` / ``leagues`` filter logic lives in one place.
    """
    allowed = LEAGUES_OF_INTEREST if leagues is None else leagues
    return sport["title"] in allowed and sport["active"]


def _get_with_retry(url, params=None):
    """GET ``url`` with one 429-retry. Returns the ``requests.Response``.

    The Odds API hands back 429s under bursty load; a single 1-second
    retry clears them in practice. As the single chokepoint for all Odds API
    HTTP, every response is fed to the :mod:`odds_budget` usage ledger. A
    ``401`` (out of credits / bad key) raises :class:`OddsAPIAuthError`;
    other non-200 statuses propagate back so callers can decide whether to
    ``continue`` or bail.
    """
    res = requests.get(url, params=params)
    if res.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        sleep(1)
        res = requests.get(url, params=params)
    # Before the 401 raise so exhausted-key responses still hit the ledger.
    odds_budget.record_response(url, params, res)
    if res.status_code == HTTPStatus.UNAUTHORIZED:
        raise OddsAPIAuthError(f"Odds API 401 at {url}: {res.text[:200]}")
    return res


def _league_activity_record(prev, commences, now):
    """Build one league's snapshot record from its feed commence times.

    ``last_game`` is the most recent commence seen slip into the past —
    rolled forward from the previous record — and is what keeps
    ``Stats.update`` running through the post-season ingest tail.
    ``opener`` is the season's first game: tracked live off the feed while
    the league is still preseason (the earliest upcoming game IS opening
    night), then frozen for the season so ``Stats`` can adopt it as
    ``season_start``. Returns ``None`` for a league with nothing scheduled
    and nothing to remember.
    """
    past = [c for c in (prev.get("next_game"), prev.get("last_game")) if c is not None]
    past += commences
    past = [c for c in past if datetime.fromisoformat(c.replace("Z", "+00:00")) <= now]
    last_game = max(past, default=None)
    days_to_next = None
    if commences:
        next_game = datetime.fromisoformat(commences[0].replace("Z", "+00:00"))
        days_to_next = (next_game - now).total_seconds() / 86400
    tier = odds_budget.classify_tier(days_to_next, prev.get("tier"), odds_budget.BUDGET_CFG)
    if tier is not None:
        record = {"tier": tier, "next_game": commences[0]}
        opener = commences[0] if tier == "preseason" else prev.get("opener")
        if opener is not None:
            record["opener"] = opener
    elif prev or last_game:
        record = {"tier": "idle"}
    else:
        return None
    if last_game is not None:
        record["last_game"] = last_game
    return record


def _classify_league_activity(key, sports):
    """Tier each index-active league of interest via its free events feed.

    Fetches ``/events`` (cost 0) per league, tiers by days to the earliest
    commence time (:func:`odds_budget.classify_tier` — sticky through
    mid-season gaps), and persists the snapshot the prophecize/meditate/
    pickem-build gates read. Leagues that go idle keep an ``idle`` record
    (season-end memory for the ``Stats.update`` window), including leagues
    the sports index no longer lists. A non-OK events response carries the
    league's previous record forward so one bad response can't silently
    black out a league.
    """
    previous = read_league_activity().get("leagues", {})
    now = datetime.now(pytz.utc)
    leagues = {}
    for sport in sports:
        league = sport["title"]
        res = _get_with_retry(
            ODDS_API_EVENTS_URL.format(sport=sport["key"]), params={"apiKey": key}
        )
        if res.status_code != HTTPStatus.OK:
            if league in previous:
                leagues[league] = previous[league]
            continue
        record = _league_activity_record(
            previous.get(league, {}), sorted(e["commence_time"] for e in res.json()), now
        )
        if record is not None:
            leagues[league] = record
    for league in previous.keys() - {s["title"] for s in sports}:
        leagues[league] = _league_activity_record(previous[league], [], now)
    write_league_activity({"updated": now.isoformat(), "leagues": leagues})
    return {lg: rec["tier"] for lg, rec in leagues.items() if rec["tier"] != "idle"}


def _resolve_broad_run_leagues(keys, cli_log):
    """Probe the sports index and decide which leagues a broad run may fetch.

    Returns ``(leagues, skip_run)``: ``leagues`` is ``None`` for an
    unrestricted run (probe failure, or the governor not enforcing) or the
    governor's allowed-leagues tuple; ``skip_run`` is ``True`` only when
    enforcement is on and nothing is admitted (floor reserve exhausted, or
    every league idle).
    """
    # The sports index is a free endpoint; its quota headers feed the governor.
    res = _get_with_retry(ODDS_API_SPORTS_URL, params={"apiKey": keys["odds_api_plus"]})
    if res.status_code != HTTPStatus.OK:
        cli_log.warning(
            "budget probe failed, proceeding ungoverned", extra={"status": res.status_code}
        )
        return None, False

    active = [s for s in res.json() if _active_in_scope(s)]
    remaining = int(res.headers["X-Requests-Remaining"])
    cfg = odds_budget.BUDGET_CFG
    tiers = _classify_league_activity(keys["odds_api_plus"], active)
    now = datetime.now(pytz.utc)
    floor_per_day, league_costs = odds_budget.estimate_costs(now, cfg)
    decision = odds_budget.broad_run_allowance(
        now, remaining, tiers, floor_per_day, league_costs, cfg
    )
    cli_log.info(
        "budget decision",
        extra={
            "enforce": cfg["enforce"],
            "tiers": tiers,
            "allowed": list(decision.allowed_leagues),
            "reason": decision.reason,
            "reserve": round(decision.reserve),
            "per_slot": round(decision.per_slot),
            "spendable": round(decision.spendable),
            "remaining": remaining,
            "floor_per_day": round(floor_per_day),
        },
    )
    if cfg["enforce"] and not decision.allowed_leagues:
        # Exit 0: a governor skip is success, not a healthchecks FAIL.
        cli_log.info("budget skip: nothing admitted", extra={"reason": decision.reason})
        return None, True
    return (decision.allowed_leagues if cfg["enforce"] else None), False


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
        odds_budget.set_run_context("close_lines")
        _close_lines_pass(keys, stat_map["Odds API"])
        cli_log.info("run usage", extra=odds_budget.run_summary())
        return

    leagues = None
    if fixture_dir is None:
        odds_budget.set_run_context("broad")
        leagues, skip_run = _resolve_broad_run_leagues(keys, cli_log)
        if skip_run:
            return

    archive = Archive()
    logger.info("Archive loaded")

    if fixture_dir is None:
        archive = get_moneylines(archive, keys, leagues=leagues)
        logger.info("Game data complete")

    archive = get_props(
        archive,
        keys["odds_api_plus"],
        stat_map["Odds API"],
        fixture_dir=fixture_dir,
        leagues=leagues,
    )
    logger.info("Player data complete, writing to file...")

    archive.write()
    logger.info("Success!")
    cli_log.info("run usage", extra=odds_budget.run_summary())


def _moneyline_sports(apikey, sport, key, historical, leagues=None):
    """Resolve ``[(odds_api_sport_key, league), ...]`` for the game-line fetch.

    Returns ``None`` to signal the caller to bail without touching the
    archive. ``sport="All"`` enumerates active leagues from the Odds API sports
    index filtered to ``LEAGUES_OF_INTEREST`` (or ``leagues`` when the budget
    governor narrows the slate); an explicit ``sport`` requires the caller to
    pass the Odds API ``key`` directly (used by the backfill script).
    """
    if sport != "All":
        if key is None:
            logger.warning("Key needed for sports other than All")
            return None
        return [(key, sport)]
    if historical:
        logger.warning("All sports only supported if date is today")
        return None
    res = _get_with_retry(ODDS_API_SPORTS_URL, params={"apiKey": apikey["odds_api_plus"]})
    if res.status_code != HTTPStatus.OK:
        return None
    return [(s["key"], s["title"]) for s in res.json() if _active_in_scope(s, leagues)]


def _moneyline_request(apikey, date, historical):
    markets = ["h2h", "totals", "spreads"]
    if historical:
        # Historical fetches are the backfill path — funded by the dedicated
        # high-limit key, never the live confer/close-lines keys.
        return (
            ODDS_API_HISTORICAL_ODDS_URL,
            _HIST_DAY_WINDOW,
            {
                "apiKey": apikey["odds_api_max"],
                "regions": "us",
                "date": date.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "markets": ",".join(markets),
            },
        )
    return (
        ODDS_API_ODDS_URL,
        _LIVE_DAY_WINDOW,
        {
            "apiKey": apikey["odds_api_plus"],
            "regions": "us",
            "markets": ",".join(markets),
        },
    )


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
                odds = no_vig_odds(market["outcomes"][0]["price"], market["outcomes"][1]["price"])
                if market["outcomes"][0]["name"] == game["home_team"]:
                    moneyline_home[book["key"]] = odds[0]
                    moneyline_away[book["key"]] = odds[1]
                else:
                    moneyline_home[book["key"]] = odds[1]
                    moneyline_away[book["key"]] = odds[0]
            elif market["key"] == "totals":
                outcomes = sorted(market["outcomes"], key=itemgetter("name"))
                odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
                totals[book["key"]] = get_ev(outcomes[1]["point"], odds[1], dist=_GAME_LINE_DIST)
            elif market["key"] == "spreads" and market["outcomes"][0].get("point"):
                outcomes = sorted(market["outcomes"], key=itemgetter("point"))
                odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
                spread = get_ev(outcomes[1]["point"], odds[1], dist=_GAME_LINE_DIST)
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
    leagues=None,
):
    """Fetch h2h / totals / spreads into archive's Moneyline & Totals buckets.

    When ``sport="All"`` (the ``confer`` default), enumerates active leagues
    from the Odds API sports index and filters to ``LEAGUES_OF_INTEREST`` —
    or to ``leagues`` when the broad-run budget governor passes an allowance.
    When called with an explicit ``sport`` + ``key`` the caller supplies
    the Odds API sport key directly (used by ``scripts/moneylines_hist.py``
    for backfills).
    """
    historical = date.date() != datetime.today().date()
    sports = _moneyline_sports(apikey, sport, key, historical, leagues)
    if sports is None:
        return archive

    url_template, dayDelta, params = _moneyline_request(apikey, date, historical)
    for sport, league in sports:
        res = _get_with_retry(url_template.format(sport=sport), params=params)
        if res.status_code != HTTPStatus.OK:
            continue
        games = res.json()["data"] if historical else res.json()
        for game in tqdm(games, desc=f"Getting {league} Game Data", unit="game"):
            _store_game_moneylines(archive, game, league, date, dayDelta)

    return archive


def _props_sports(apikey, sport, key, historical, leagues=None):
    """Resolve ``[(odds_api_sport_key, league), ...]`` for the prop fetch.

    Returns ``None`` to signal the caller to bail without touching the archive.
    ``sport="All"`` enumerates active leagues from the Odds API sports index
    filtered to ``LEAGUES_OF_INTEREST`` (or ``leagues`` when the budget governor
    narrows the slate); an explicit ``sport`` requires the caller to pass the
    Odds API ``key`` directly.
    """
    if sport != "All":
        if key is None:
            logger.warning("Key needed for sports other than All")
            return None
        return [(key, sport)]
    if historical:
        logger.warning("All sports only supported if date is today")
        return None
    res = _get_with_retry(ODDS_API_SPORTS_URL, params={"apiKey": apikey})
    if res.status_code != HTTPStatus.OK:
        return None
    return [(s["key"], s["title"]) for s in res.json() if _active_in_scope(s, leagues)]


class _PropsRequest(NamedTuple):
    event_url_template: str
    odds_url_template: str
    day_delta: int
    params: dict
    historical: bool


def _props_request(apikey, date, historical):
    if historical:
        return _PropsRequest(
            ODDS_API_HISTORICAL_EVENTS_URL,
            ODDS_API_HISTORICAL_EVENT_ODDS_URL,
            _HIST_DAY_WINDOW,
            {"apiKey": apikey, "date": date.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
            historical,
        )
    return _PropsRequest(
        ODDS_API_EVENTS_URL,
        ODDS_API_EVENT_ODDS_URL,
        _LIVE_DAY_WINDOW,
        {"apiKey": apikey},
        historical,
    )


def _region_markets(league, props):
    """Split one league's market list into the ``(region, markets)`` calls to make.

    Every market lands in exactly one region, so the ``markets x regions`` bill matches
    a single us-only request while :data:`US2_ONLY_MARKETS` still get archived.
    """
    markets = list(props[league])
    if league == "MLB":
        markets += ["totals_1st_1_innings", "spreads_1st_1_innings"]
    us2 = US2_ONLY_MARKETS.get(league, ())
    return (
        ("us", [m for m in markets if m not in us2]),
        ("us2", [m for m in markets if m in us2]),
    )


def _fetch_event_odds(req, sport, event_id, region_markets):
    """One event's odds, one request per region, merged into a single payload.

    Returns ``False`` when the endpoint 404s (the caller aborts the whole run), ``None``
    when no region returned anything usable, else the payload with every region's
    bookmakers concatenated. The regions carry disjoint market lists, so no book can
    contribute the same market twice.
    """
    merged, bookmakers = None, []
    for region, markets in region_markets:
        if not markets:
            continue
        res = _get_with_retry(
            req.odds_url_template.format(sport=sport, eventId=event_id),
            params={**req.params, "markets": ",".join(markets), "regions": region},
        )
        if res.status_code == HTTPStatus.NOT_FOUND:
            return False
        if res.status_code != HTTPStatus.OK:
            continue
        payload = res.json()["data"] if req.historical else res.json()
        merged = merged or payload
        bookmakers.extend(payload.get("bookmakers", []))
    if merged is None:
        return None
    merged["bookmakers"] = bookmakers
    return merged


def _store_sport_props(archive, ledger, sport, league, props, date, req, observed_at_hour=None):
    """Fetch and archive one sport's events.

    Returns ``False`` when a per-event odds endpoint 404s (the caller aborts the
    whole run, matching the original early ``return``); ``True`` otherwise.
    """
    region_markets = _region_markets(league, props)
    fetch_params = {**req.params, "markets": ",".join(props[league])}
    events = _get_with_retry(req.event_url_template.format(sport=sport), params=fetch_params)
    if events.status_code != HTTPStatus.OK:
        return True

    for event in events.json()["data"] if req.historical else events.json():
        gameDate = datetime.fromisoformat(event["commence_time"]).astimezone(
            pytz.timezone("America/Chicago")
        )
        if gameDate > date + timedelta(days=req.day_delta):
            continue
        payload = _fetch_event_odds(req, sport, event["id"], region_markets)
        if payload is False:
            return False
        if payload is None:
            continue

        observed_at = _historical_observed_at(req.historical, gameDate, observed_at_hour)
        _archive_event_props(
            archive,
            payload,
            league,
            props,
            gameDate.strftime("%Y-%m-%d"),
            observed_at,
        )
        if not req.historical:
            _record_upcoming_event(ledger, sport, event, league, props)
    return True


def _record_upcoming_event(ledger, sport, event, league, props):
    """Refresh an event's ledger entry, carrying the close-lines dedupe stamp.

    A broad-run rebuild must not wipe ``close_fetched``, or a broad slot
    landing inside an event's closing window would re-arm a second paid
    close fetch.
    """
    entry = {
        "sport_key": sport,
        "event_id": event["id"],
        "league": league,
        "commence_time": event["commence_time"],
        "markets": list(props[league].keys()),
    }
    stamp = ledger.get((sport, event["id"]), {}).get("close_fetched")
    if stamp:
        entry["close_fetched"] = stamp
    ledger[(sport, event["id"])] = entry


def get_props(
    archive,
    apikey,
    props,
    date=datetime.now().astimezone(pytz.timezone("America/Chicago")),
    sport="All",
    key=None,
    fixture_dir: Path | None = None,
    observed_at_hour=None,
    leagues=None,
):
    """Fetch per-event player-prop markets and store book-level EVs.

    ``observed_at_hour`` overrides the historical snapshot stamp (default
    game-day 01:00): close-layer backfills stamp a post-read hour so their
    rows are evaluation-only by construction. ``leagues`` narrows the
    ``sport="All"`` slate (the broad-run budget governor passes its allowance).
    """
    stat_cv["NCAAB"] = stat_cv["NBA"]
    stat_cv["NCAAF"] = stat_cv["NFL"]
    historical = date.date() != datetime.today().date()

    if fixture_dir is not None:
        return _get_props_from_fixtures(archive, props, Path(fixture_dir), date)

    sports = _props_sports(apikey, sport, key, historical, leagues)
    if sports is None:
        return archive

    req = _props_request(apikey, date, historical)
    ledger = {(e["sport_key"], e["event_id"]): e for e in read_upcoming_events()}
    for sport, league in sports:
        if not _store_sport_props(
            archive, ledger, sport, league, props, date, req, observed_at_hour
        ):
            return archive

    if not historical:
        write_upcoming_events(_prune_upcoming_events(list(ledger.values())))

    return archive


def _historical_observed_at(historical, gameDate, hour=None):
    """Snapshot stamp for backfilled rows, or ``None`` for live runs.

    Live writes default to ``utcnow()``. Backfilled rows stamp game-day
    01:00 so they beat the migrated midnight (``00:00``) rows on recency
    while staying before any ``kickoff - 8h`` point-in-time training read.
    ``hour`` overrides that for close-layer backfills, whose rows must land
    AFTER every pre-game read (evaluation-only layer).
    """
    if not historical:
        return None
    return datetime(gameDate.year, gameDate.month, gameDate.day, hour or 1, 0, 0)


_PROP_SIDE_RENAME = {"Yes": "Over", "No": "Under"}


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
        line["name"] = _PROP_SIDE_RENAME.get(line["name"], line["name"])

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


def _devig_ladder(rows):
    """De-vig every offered point in one ``(player, book)`` quote.

    Returns ``[(line, p_over), ...]`` across all alt-line rungs — the full ladder
    the consensus collapse discards. A lone ``Under`` is flipped to an ``Over``
    price (matching :func:`_resolve_prop_lines`); a lone ``Over`` de-vigs one-sided.
    Rungs keep first-appearance order.
    """
    by_point: dict[float, dict[str, float]] = {}
    for row in rows:
        name = _PROP_SIDE_RENAME.get(row["name"], row["name"])
        by_point.setdefault(row.get("point", 0.5), {})[name] = row["price"]
    rungs = []
    for point, sides in by_point.items():
        over, under = sides.get("Over"), sides.get("Under")
        if over is None and under is None:
            continue
        if over is None:
            over, under = under / (under - 1), None
        rungs.append((point, no_vig_odds(over, under)[0]))
    return rungs


def _event_player_props(book, market, league, props, odds):
    """Accumulate one book's player-prop market into ``odds`` (mutated).

    ``*_alternate`` market keys (requested only by close-layer backfills; they
    map onto the same market name as their primary key) contribute their rungs
    to the ladder and nothing else — pricing an alt rung as the book's EV would
    corrupt the consensus.
    """
    market_name = props[league].get(market["key"])
    is_alternate = market["key"].endswith("_alternate")
    odds.setdefault(market_name, {})

    outcomes = [
        o for o in market["outcomes"] if "description" in o and "name" in o and o["price"] > 1
    ]
    outcomes = sorted(outcomes, key=itemgetter("description", "name"))

    for player, lines in groupby(outcomes, itemgetter("description")):
        player = remove_accents(player).replace(" Total", "")
        entry = odds[market_name].setdefault(
            player, {"EV": {}, "Lines": [], "Ladder": {}, "Quotes": {}}
        )
        lines = list(lines)
        rungs = _devig_ladder(lines)
        if rungs:
            entry["Ladder"].setdefault(book["key"], []).extend(rungs)
        if is_alternate:
            continue
        lines = _resolve_prop_lines(lines)
        line = lines[0].get("point", 0.5)
        entry["Lines"].append(line)
        price = no_vig_odds(*[x["price"] for x in lines])
        dist = stat_dist.get(league, {}).get(market_name, "SkewNormal")
        gate = book_gate(league, market_name, dist)
        ev = get_ev(line, price[1], stat_cv[league].get(market_name, 1), dist=dist, gate=gate)
        # get_ev returns its ceiling when no mean in [0, SN_MAX_MEAN_FACTOR × line]
        # reproduces the price, so the value is a bound, not a mean — persisting it is how
        # every 2026-06+ NFL tds row went bad. A gate refuted by a sharp quote is one cause
        # (population zi 0.78 vs quoted-star unders ~0.42); an ungated family too tight for
        # an extreme price is another, and both poison training identically. Store a NULL ev
        # with the shape-free quote instead; the consensus reader None-skips the book. The
        # DFS path (add_dfs) keeps its pinned clamp semantics — guarded downstream.
        clamped = ev >= max(SN_MAX_MEAN_FACTOR * line, 1.0)
        entry["EV"][book["key"]] = None if clamped else ev
        entry["Quotes"][book["key"]] = (line, price[1])


def _event_totals_book(market, book, totals):
    """Accumulate one book's totals sub-market into ``totals`` (mutated)."""
    spread_name = " ".join(market["key"].split("_")[1:])
    outcomes = sorted(market["outcomes"], key=itemgetter("name"))
    sub_odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
    totals.setdefault(spread_name, {})
    totals[spread_name][book["key"]] = get_ev(
        outcomes[1]["point"], sub_odds[1], dist=_GAME_LINE_DIST
    )


def _event_spread_book(market, book, game, spread_home, spread_away):
    """Accumulate one book's spread sub-market into the spread dicts (mutated)."""
    spread_name = " ".join(market["key"].split("_")[1:])
    outcomes = sorted(market["outcomes"], key=itemgetter("point"))
    sub_odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
    spread = get_ev(outcomes[1]["point"], sub_odds[1], dist=_GAME_LINE_DIST)
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
            if entry["EV"]:
                line = np.median(entry["Lines"])
                archive.merge_player_books(
                    league,
                    market,
                    gameDate,
                    player,
                    entry["EV"],
                    [line],
                    observed_at=observed_at,
                    book_quotes=entry["Quotes"],
                )
            for book, rungs in entry["Ladder"].items():
                archive.add_ladder(
                    league, market, gameDate, player, book, rungs, observed_at=observed_at
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
            if gameDate > date + timedelta(days=_LIVE_DAY_WINDOW):
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


def _close_lines_pass(keys, props):
    """Per-event close-line scrape. Exits early when the window is empty.

    Loads ``data/upcoming_events.json``, filters to events with a
    ``commence_time`` between ``CLOSING_LEAD_MIN`` and ``CLOSING_LEAD_MAX``
    minutes from now, and only then opens ``Archive`` and hits the per-event
    Odds API endpoint for each — spending from ``keys["odds_api_plus"]``,
    with a one-shot fallback to the ``odds_api`` margin key if the plus key
    exhausts mid-run. Each event is fetched once: a successful parse stamps
    ``close_fetched`` on its ledger entry, so later ticks inside the same
    20-minute window skip it (a failed fetch stays unstamped and retries next
    tick). Past-commence entries are pruned on the way out so the ledger
    stays small. The ten-minute cron tick yields no archive read, no API
    call, and no log noise on empty windows — the intended common case.
    """
    # style: allow-complexity — flat close-lines pass (CC 11): window filter,
    # then a per-event fetch loop whose one-shot margin-key fallback adds the
    # two branches over the limit; extracting it would scatter the key-flip state.
    ledger = read_upcoming_events()
    ledger = _prune_upcoming_events(ledger)

    now = datetime.now(pytz.utc)
    window_start = now + timedelta(minutes=CLOSING_LEAD_MIN)
    window_end = now + timedelta(minutes=CLOSING_LEAD_MAX)

    due = []
    for e in ledger:
        if "close_fetched" in e:
            continue
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
    params = {"apiKey": keys["odds_api_plus"], "regions": "us"}

    for e in due:
        sport_key = e["sport_key"]
        event_id = e["event_id"]
        league = e["league"]
        markets = e.get("markets") or list(props.get(league, {}).keys())
        if not markets:
            continue
        event_params = {**params, "markets": ",".join(markets)}
        url = ODDS_API_EVENT_ODDS_URL.format(sport=sport_key, eventId=event_id)
        try:
            res = _get_with_retry(url, params=event_params)
        except OddsAPIAuthError:
            # One fallback per run: after the flip every later event builds its
            # params from the mutated dict, and a margin-key 401 propagates.
            if params["apiKey"] == keys["odds_api"]:
                raise
            logger.warning("close-lines: plus key exhausted, falling back to margin key")
            params["apiKey"] = keys["odds_api"]
            event_params = {**params, "markets": ",".join(markets)}
            res = _get_with_retry(url, params=event_params)
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
        e["close_fetched"] = now.isoformat()

    archive.write()
    write_upcoming_events(_prune_upcoming_events(ledger))
    logger.info("close-lines: archive updated")


if __name__ == "__main__":
    confer()
