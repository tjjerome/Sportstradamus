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
"""

import importlib.resources as pkg_resources
import json
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter
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
    no_vig_odds,
    remove_accents,
    stat_cv,
)
from sportstradamus.spiderLogger import logger

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


def _get_with_retry(url, params=None):
    """GET ``url`` with one 429-retry. Returns the ``requests.Response``.

    The Odds API hands back 429s under bursty load; a single 1-second
    retry clears them in practice. Other non-200 statuses propagate back
    so callers can decide whether to ``continue`` or bail.
    """
    res = requests.get(url, params=params)
    if res.status_code == 429:
        sleep(1)
        res = requests.get(url, params=params)
    return res


@click.command()
def confer():
    """Fetch current odds and player props into the archive."""
    filepath = pkg_resources.files(creds) / "keys.json"
    with open(filepath) as infile:
        keys = json.load(infile)

    archive = Archive()
    logger.info("Archive loaded")

    archive = get_moneylines(archive, keys)
    logger.info("Game data complete")

    filepath = pkg_resources.files(data) / "stat_map.json"
    with open(filepath) as infile:
        stat_map = json.load(infile)

    archive = get_props(archive, keys["odds_api_plus"], stat_map["Odds API"])
    logger.info("Player data complete, writing to file...")

    archive.write()
    logger.info("Success!")


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
    low_on_credits = 0

    if sport == "All":
        if historical:
            logger.warning("All sports only supported if date is today")
            return archive
        res = _get_with_retry(ODDS_API_SPORTS_URL, params={"apiKey": apikey["odds_api"]})
        if res.status_code != 200:
            return archive

        low_on_credits = int(res.headers.get("X-Requests-Remaining")) < 50
        res = res.json()

        sports = [
            (s["key"], s["title"]) for s in res if s["title"] in LEAGUES_OF_INTEREST and s["active"]
        ]
    elif key is None:
        logger.warning("Key needed for sports other than All")
        return archive
    else:
        sports = [(key, sport)]

    markets = ["h2h", "totals", "spreads"]

    if historical:
        url_template = ODDS_API_HISTORICAL_ODDS_URL
        dayDelta = 1
        params = {
            "apiKey": apikey["odds_api_plus"],
            "regions": "us",
            "date": date.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "markets": ",".join(markets),
        }
    else:
        url_template = ODDS_API_ODDS_URL
        dayDelta = 6
        params = {
            "apiKey": apikey["odds_api_plus"] if low_on_credits else apikey["odds_api"],
            "regions": "us",
            "markets": ",".join(markets),
        }

    for sport, league in sports:
        res = _get_with_retry(url_template.format(sport=sport), params=params)
        if res.status_code != 200:
            continue

        res = res.json()["data"] if historical else res.json()

        for game in tqdm(res, desc=f"Getting {league} Game Data", unit="game"):
            gameDate = datetime.fromisoformat(game["commence_time"]).astimezone(
                pytz.timezone("America/Chicago")
            )
            if gameDate > date + timedelta(days=dayDelta):
                continue
            gameDate = gameDate.strftime("%Y-%m-%d")

            homeTeam = abbreviations[league].get(remove_accents(game["home_team"]))
            awayTeam = abbreviations[league].get(remove_accents(game["away_team"]))
            if homeTeam is None or awayTeam is None:
                continue

            moneyline_home = {}
            moneyline_away = {}
            totals = {}
            spread_home = {}
            spread_away = {}

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

            archive._mark_changed(league, "Moneyline")
            archive._mark_changed(league, "Totals")
            archive[league].setdefault("Moneyline", {}).setdefault(gameDate, {})
            archive[league].setdefault("Totals", {}).setdefault(gameDate, {})

            archive[league]["Moneyline"][gameDate][awayTeam] = moneyline_away
            archive[league]["Moneyline"][gameDate][homeTeam] = moneyline_home

            archive[league]["Totals"][gameDate][awayTeam] = {
                k: (v + spread_away.get(k, 0)) / 2 for k, v in totals.items()
            }
            archive[league]["Totals"][gameDate][homeTeam] = {
                k: (v + spread_home.get(k, 0)) / 2 for k, v in totals.items()
            }

    return archive


def get_props(
    archive,
    apikey,
    props,
    date=datetime.now().astimezone(pytz.timezone("America/Chicago")),
    sport="All",
    key=None,
):
    """Fetch per-event player-prop markets and store book-level EVs."""
    stat_cv["NCAAB"] = stat_cv["NBA"]
    stat_cv["NCAAF"] = stat_cv["NFL"]
    historical = date.date() != datetime.today().date()

    if sport == "All":
        if historical:
            logger.warning("All sports only supported if date is today")
            return archive
        res = _get_with_retry(ODDS_API_SPORTS_URL, params={"apiKey": apikey})
        if res.status_code != 200:
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

    for sport, league in sports:
        params.update({"markets": ",".join(props[league].keys())})
        if league == "MLB":
            params["markets"] = params["markets"] + ",totals_1st_1_innings,spreads_1st_1_innings"
        events = _get_with_retry(event_url_template.format(sport=sport), params=params)
        if events.status_code != 200:
            continue

        events = events.json()["data"] if historical else events.json()

        for event in events:
            gameDate = datetime.fromisoformat(event["commence_time"]).astimezone(
                pytz.timezone("America/Chicago")
            )
            if gameDate > date + timedelta(days=dayDelta):
                continue
            gameDate = gameDate.strftime("%Y-%m-%d")
            event_params = {**params, "regions": "us"}
            res = _get_with_retry(
                odds_url_template.format(sport=sport, eventId=event["id"]), params=event_params
            )
            if res.status_code == 404:
                return archive
            elif res.status_code != 200:
                continue

            game = res.json()["data"] if historical else res.json()

            odds = {}
            totals = {}
            spread_home = {}
            spread_away = {}

            for book in game["bookmakers"]:
                for market in book["markets"]:
                    if "totals" in market["key"]:
                        spread_name = " ".join(market["key"].split("_")[1:])
                        outcomes = sorted(market["outcomes"], key=itemgetter("name"))
                        sub_odds = no_vig_odds(outcomes[0]["price"], outcomes[1]["price"])
                        totals.setdefault(spread_name, {})
                        totals[spread_name][book["key"]] = get_ev(outcomes[1]["point"], sub_odds[1])
                        continue
                    elif "spread" in market["key"]:
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
                        continue

                    market_name = props[league].get(market["key"])

                    odds.setdefault(market_name, {})

                    outcomes = [
                        o
                        for o in market["outcomes"]
                        if "description" in o and "name" in o and o["price"] > 1
                    ]
                    outcomes = sorted(outcomes, key=itemgetter("description", "name"))

                    for player, lines in groupby(outcomes, itemgetter("description")):
                        player = remove_accents(player).replace(" Total", "")
                        odds[market_name].setdefault(player, {"EV": {}, "Lines": []})
                        lines = list(lines)
                        for line in lines:
                            line.setdefault("point", 0.5)
                            line["name"] = {"Yes": "Over", "No": "Under"}.get(
                                line["name"], line["name"]
                            )

                        lines = sorted(lines, key=itemgetter("name"))
                        if len({line["point"] for line in lines}) > 1:
                            trueline = sorted(lines, key=(lambda x: np.abs(x["price"] - 2)))[0][
                                "point"
                            ]
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

                        line = lines[0].get("point", 0.5)
                        odds[market_name][player]["Lines"].append(line)
                        price = no_vig_odds(*[x["price"] for x in lines])
                        ev = get_ev(line, price[1], stat_cv[league].get(market_name, 1))

                        odds[market_name][player]["EV"][book["key"]] = ev

            for market in odds:
                archive._mark_changed(league, market)
                archive[league].setdefault(market, {}).setdefault(gameDate, {})
                for player in odds[market]:
                    archive[league][market][gameDate].setdefault(player, {"EV": {}, "Lines": []})
                    archive[league][market][gameDate][player]["EV"].update(
                        odds[market][player]["EV"]
                    )

                    line = np.median(odds[market][player]["Lines"])
                    if line not in archive[league][market][gameDate][player]["Lines"]:
                        archive[league][market][gameDate][player]["Lines"].append(line)

            for market in totals:
                archive._mark_changed(league, market)
                archive[league].setdefault(market, {}).setdefault(gameDate, {})

                archive[league][market][gameDate][
                    abbreviations[league][remove_accents(game["home_team"])]
                ] = {k: (v + spread_home.get(k, 0)) / 2 for k, v in totals[market].items()}
                archive[league][market][gameDate][
                    abbreviations[league][remove_accents(game["away_team"])]
                ] = {k: (v + spread_away.get(k, 0)) / 2 for k, v in totals[market].items()}

    return archive


if __name__ == "__main__":
    confer()
