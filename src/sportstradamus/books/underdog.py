"""Underdog Fantasy scraper: player props, team and game markets, alternate lines.

Endpoints, payloads and caching are documented in ``docs/underdog_api.md``; the
request plan per run is its §2. Every request here is public: nothing carries
the ten-minute auth token.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from http import HTTPStatus

from tqdm import tqdm

from sportstradamus.helpers import (
    ABBR_MAP,
    Scrape,
    get_mlb_pitchers,
    nhl_goalies,
    remove_accents,
    requests,
    stat_map,
)
from sportstradamus.spiderLogger import logger

_UD_BASE = "https://api.underdogfantasy.com"
# The legacy bulk feed: every player prop across all sports in one request. A
# retired version answers 426 upgrade_required, which Scrape.get logs only at
# DEBUG, so get_ud warns whenever the feed comes back empty.
UD_LINES_URL = f"{_UD_BASE}/v1/over_under_lines"
UD_LOBBY_URL = f"{_UD_BASE}/v1/lobbies/content"
UD_ALT_LINES_URL = f"{_UD_BASE}/v3/over_unders/{{over_under_id}}/alternate_projections"
# Lobby requests 400 without the state config; the product experience id is what
# makes prediction-market lines visible (docs/underdog_api.md §5).
_UD_QUERY = (
    "product=fantasy"
    "&product_experience_id=b34dfd93-d0e8-4da3-8bf4-45c15c548dec"
    "&state_config_id=16fa6ed3-ea21-4654-bcee-fb32d2f31357"
)

# Leagues with a Stats model: team markets and alternate lines are fetched only
# where prophecize can price them.
UD_MODELED_LEAGUES = ("NFL", "MLB", "NBA", "WNBA", "NHL")

# Lobby line categories (docs/underdog_api.md §7.6). The feed already carries
# player_prop, so one lobby read per league asks for the other four: moneyline /
# spread / total, period lines, team totals and props, yes/no game props. No pill
# id or token is involved, so every modeled league gets the same coverage.
UD_TEAM_CATEGORIES = ("core", "partial_core", "team_prop", "misc")

# Alternate lines come one market per request with no batch form. Three workers
# on one keep-alive session read ~17 markets/s: ~2,000 markets per two-minute
# budget, nearest kickoffs first, the rest left to a later hourly run. Four
# workers ran at 36/s and the origin started answering ``projections: []`` after
# ~1,850 requests in a minute (2026-09-11); the empty-streak stop below catches
# that.
UD_ALT_LINES_BUDGET_S = 120
UD_ALT_LINES_WORKERS = 3
UD_ALT_LINES_TIMEOUT_S = 10
# Markets answering without rungs in a row before the pass stops: a game about to
# lock answers empty one market at a time, a rate limit answers empty for all.
UD_ALT_LINES_EMPTY_STREAK = 25

# Moneylines and yes/no markets carry no stat_value; "over 0.5 occurrences" keeps
# them on the two-sided offer shape (add_dfs skips a 0 line).
_BINARY_LINE = 0.5
# Option choices by payout slot: over is higher / yes / the home side of a
# moneyline or spread, under the other side.
_CHOICE_SLOT = {"higher": 0, "yes": 0, "home": 0, "lower": 1, "no": 1, "away": 1}
# A line is priced only when each of its options is one of these, once. Mass-option
# markets (race to X, winning margin, halftime/fulltime, first team to score) repeat
# yes / no once per outcome inside one line and have no two-sided reading.
_LINE_CHOICES = frozenset(_CHOICE_SLOT) | {"draw"}

_scraper: Scrape | None = None


def _get_scraper() -> Scrape:
    """Return the shared Scrape singleton, creating it on first call."""
    global _scraper
    if _scraper is None:
        _scraper = Scrape()
    return _scraper


def _ud_date(scheduled_at: str) -> str:
    """Underdog UTC timestamp → date string shifted to UTC-5."""
    return (datetime.strptime(scheduled_at, "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=5)).strftime(
        "%Y-%m-%d"
    )


def _ud_league(sport_id: str) -> str:
    """Strip Underdog's "COMBOS" suffix from a sport_id to get the league code."""
    return sport_id.replace("COMBOS", "")


def _by_id(container) -> dict:
    """Index a feed list or a lobby dict by ``str(id)``: JSON keys are strings, match ids ints."""
    items = container.values() if isinstance(container, dict) else container
    return {str(item["id"]): item for item in items}


def _ud_sides(title: str) -> tuple[str, str] | None:
    """``(home, away)`` from a game title: ``AWAY @ HOME``, or ``HOME vs AWAY`` for soccer and solo games."""
    if " @ " in title:
        away, home = title.split(" @ ", 1)
        return home, away
    if " vs " in title:
        home, away = title.split(" vs ", 1)
        return home, away
    return None


def _ud_matches(games: dict, solo_games: dict, team_abbr: dict) -> dict:
    """Map ``str(match_id)`` → home / away / league / date / commence for team and solo games."""
    matches = {}
    for match_id, game in games.items():
        matches[match_id] = {
            "Home": team_abbr.get(game["home_team_id"], ""),
            "Away": team_abbr.get(game["away_team_id"], ""),
            "League": _ud_league(game["sport_id"]),
            "Date": _ud_date(game["scheduled_at"]),
            "Commence": game["scheduled_at"],
        }
    for match_id, game in solo_games.items():
        sides = _ud_sides(game["title"])
        if sides is None:
            continue
        matches[match_id] = {
            "Home": sides[0],
            "Away": sides[1],
            "League": _ud_league(game["sport_id"]),
            "Date": _ud_date(game["scheduled_at"]),
            "Commence": game["scheduled_at"],
        }
    return matches


def _team_abbrs(payload: dict) -> dict:
    """``{team_id: abbr}`` from a lobby payload's ``teams`` dict; the feed carries none."""
    return {team_id: team["abbr"] for team_id, team in _by_id(payload.get("teams", [])).items()}


def _ud_entities(payload: dict, team_abbr: dict | None = None) -> tuple[dict, dict, dict, dict]:
    """Index players, appearances, matches and team abbreviations from either container style.

    The feed sends lists and no ``teams``; the lobby sends id-keyed dicts. Team
    abbreviations come from the payload's ``teams``, then from ``team_abbr`` (the
    modeled leagues' lobby dicts, read before the feed is parsed), and last from
    each game's ``abbreviated_title``. The title is the weak source: ``AWAY @ HOME``
    holds for US sports, but ``vs`` titles put the home side first in soccer and
    the away side first in esports, so it only serves leagues without a model.
    """
    players = _by_id(payload.get("players", []))
    games = _by_id(payload.get("games", []))
    appearances = _by_id(payload.get("appearances", []))
    team_abbr = {**(team_abbr or {}), **_team_abbrs(payload)}
    for game in games.values():
        sides = _ud_sides(game.get("abbreviated_title") or game["title"])
        if sides is not None:
            team_abbr.setdefault(game["home_team_id"], sides[0])
            team_abbr.setdefault(game["away_team_id"], sides[1])
    matches = _ud_matches(games, _by_id(payload.get("solo_games", [])), team_abbr)
    return players, appearances, matches, team_abbr


def _ud_fantasy_market(player: str, league: str, market: str) -> str:
    """Split Underdog 'Fantasy Points' into position-specific MLB/NHL markets."""
    if "Fantasy" in market and league == "MLB":
        pitchers = list(get_mlb_pitchers().values())
        return "Pitcher Fantasy Points" if player in pitchers else "Hitter Fantasy Points"
    if "Fantasy" in market and league == "NHL":
        return "Goalie Fantasy Points" if player in list(nhl_goalies) else "Skater Fantasy Points"
    return market


def _ud_boosts(options: list[dict]) -> list[float]:
    """``[over, under]`` payout multipliers; a missing or suspended side stays 0.

    A three-way period moneyline (``away`` / ``draw`` / ``home``) is priced from
    the home side only: its ``away`` option is not the complement of ``home``.
    """
    boosts = [0.0, 0.0]
    for option in options:
        slot = _CHOICE_SLOT.get(option["choice"])
        if slot is not None and option.get("status", "active") == "active":
            boosts[slot] = float(option["payout_multiplier"])
    if any(option["choice"] == "draw" for option in options):
        boosts[1] = 0.0
    return boosts


def _ud_subject(
    appearance: dict, game: dict, players: dict, team_abbr: dict
) -> tuple[str, str, str] | None:
    """``(entity, league, team)`` a line is keyed on: the player, the team, or a match's home side.

    ``None`` for combo players ("A + B"), which the pipeline cannot score.
    """
    if appearance["type"] == "Player":
        player = players[appearance["player_id"]]
        name = str(player["first_name"] or "") + " " + str(player["last_name"] or "")
        if "+" in name:
            return None
        league = player["sport_id"].replace("COMBOS", "").replace("COMBO", "")
        return remove_accents(name), league, team_abbr.get(appearance["team_id"], "")
    team = (
        game["Home"] if appearance["type"] == "Match" else team_abbr.get(appearance["team_id"], "")
    )
    return ABBR_MAP.get(team, team), game["League"], team


def _ud_offer(
    line: dict, players: dict, appearances: dict, matches: dict, team_abbr: dict
) -> dict | None:
    """Parse one over/under line into an offer, or ``None`` for lines the pipeline cannot use.

    Player lines keep the legacy shape. Team-appearance lines (team totals, team
    TDs) are keyed on the team and match-appearance lines (moneyline, spread, game
    total, period lines) on the home team, whose handicap ``stat_value`` carries.
    Their market is the appearance type plus the API ``stat`` slug (``team runs``,
    ``match moneyline``): the display name "Moneyline" is the sportsbook team market
    in the archive (``archive._TEAM_ONLY_MARKETS``) and the bare slug (``runs``,
    ``points``) is a player market's internal name, and the archive enumerates a
    market's entities as players when it builds training data.
    """
    stat = line["over_under"]["appearance_stat"]
    appearance = appearances[stat["appearance_id"]]
    game = matches.get(str(appearance["match_id"]))
    if game is None:
        return None
    subject = _ud_subject(appearance, game, players, team_abbr)
    if subject is None:
        return None
    entity, league, team = subject
    if appearance["type"] == "Player":
        market = _ud_fantasy_market(entity, league, stat["display_stat"])
    else:
        market = f"{appearance['type'].lower()} {stat['stat']}"
    opponent = game["Away"] if game["Home"] == team else game["Home"]
    boosts = _ud_boosts(line["options"])
    return {
        "Player": entity,
        "League": league,
        "Team": ABBR_MAP.get(team, team),
        "Opponent": ABBR_MAP.get(opponent, opponent),
        "Date": game["Date"],
        "Commence": game["Commence"],
        "Market": market,
        "Line": _BINARY_LINE if line["stat_value"] is None else float(line["stat_value"]),
        "Boost_Over": boosts[0],
        "Boost_Under": boosts[1],
    }


def _ud_offers(payload: dict, team_abbr: dict | None = None) -> list[tuple[dict, dict]]:
    """``(offer, line)`` pairs for every active pre-game line in one feed or lobby payload."""
    players, appearances, matches, team_abbr = _ud_entities(payload, team_abbr)
    lines = payload["over_under_lines"]
    priced = []
    for line in lines.values() if isinstance(lines, dict) else lines:
        choices = [option["choice"] for option in line["options"]]
        if (
            line["status"] != "active"
            or line["live_event"]
            or len(set(choices)) < len(choices)
            or not set(choices) <= _LINE_CHOICES
        ):
            continue
        offer = _ud_offer(line, players, appearances, matches, team_abbr)
        if offer is not None:
            priced.append((offer, line))
    return priced


def _ud_team_offers(scraper: Scrape, sports: set[str]) -> tuple[list[tuple[dict, dict]], dict]:
    """Team and game markets per modeled sport, plus the team abbreviations their lobbies carry.

    One ``match_grouped_lines`` request per sport carrying every category in
    ``UD_TEAM_CATEGORIES``; the combined answer is the union of the single ones.
    """
    categories = "&".join(f"market_categories%5B%5D={category}" for category in UD_TEAM_CATEGORIES)
    urls = [
        f"{UD_LOBBY_URL}/match_grouped_lines?sport_id={sport}&include_live=true&{categories}&{_UD_QUERY}"
        for sport in sorted(sports & set(UD_MODELED_LEAGUES))
    ]
    priced, team_abbr = [], {}
    for url in tqdm(urls, desc="Getting Underdog team markets", unit="request"):
        payload = scraper.get(url)
        if payload:
            team_abbr.update(_team_abbrs(payload))
            priced.extend(_ud_offers(payload))
    return priced, team_abbr


def _ud_rungs(
    offer: dict,
    over_under_id: str,
    deadline: float,
    stop: threading.Event,
    session: requests.Session,
) -> list[dict] | None:
    """One market's alternate lines as offers; ``None`` once the pass has been stopped.

    A request error or a non-200 answer costs that market its rungs, not the run:
    the main lines are already in hand.
    """
    if stop.is_set() or time.monotonic() > deadline:
        return None
    try:
        response = session.get(
            UD_ALT_LINES_URL.format(over_under_id=over_under_id), timeout=UD_ALT_LINES_TIMEOUT_S
        )
    except requests.RequestException as exc:
        logger.debug(f"Alternate lines {over_under_id}: {exc}")
        return []
    if response.status_code != HTTPStatus.OK:
        logger.debug(f"Alternate lines {over_under_id}: {response.status_code}")
        return []
    rungs = []
    for projection in response.json()["projections"]:
        if projection["is_main"]:
            continue
        over, under = _ud_boosts(projection["options"])
        rungs.append(
            {
                **offer,
                "Line": float(projection["stat_value"]),
                "Boost_Over": over,
                "Boost_Under": under,
            }
        )
    return rungs


def _ud_alt_lines(candidates: list[tuple[dict, str]]) -> list[dict]:
    """Alternate-line offers for ``(offer, over_under_id)`` pairs, games about to lock and deep ladders first.

    One session shared by the workers keeps their connections alive: a fresh
    connection per request also meant a DNS lookup per request, which fell over
    under the burst.
    """
    deadline = time.monotonic() + UD_ALT_LINES_BUDGET_S
    stop = threading.Event()
    order = sorted(candidates, key=lambda item: (item[0]["Commence"], -item[0]["Line"]))
    rungs, fetched, empty, streak = [], 0, 0, 0
    with requests.Session() as session, ThreadPoolExecutor(UD_ALT_LINES_WORKERS) as pool:
        results = pool.map(lambda item: _ud_rungs(*item, deadline, stop, session), order)
        for result in tqdm(
            results, total=len(order), desc="Getting Underdog alternate lines", unit="market"
        ):
            if result is None:
                continue
            fetched += 1
            rungs.extend(result)
            streak = 0 if result else streak + 1
            empty += not result
            if streak == UD_ALT_LINES_EMPTY_STREAK:
                stop.set()
                logger.warning(
                    f"{streak} alternate-line markets in a row answered without rungs; "
                    "stopping the pass (rate limit?)"
                )
    logger.info(
        f"{fetched} alternate-line markets fetched ({empty} without rungs), "
        f"{len(order) - fetched} not fetched"
    )
    return rungs


def get_ud():
    """Retrieve Underdog offers: player props, team and game markets, alternate lines.

    Returns:
        dict: ``{league: {market: [offer, ...]}}`` where each offer carries
            ``Player``, ``League``, ``Team``, ``Opponent``, ``Date``, ``Commence``
            (full tip-off ISO timestamp, for the dashboard's "locks in"
            countdown), ``Market``, ``Line``, ``Boost_Over`` and ``Boost_Under``.
            Team and game markets are keyed on the team (the home team for
            match-level lines) under ``"team <slug>"`` / ``"match <slug>"``;
            alternate lines are extra offers on the same market with their own
            ``Line``. ``{}`` when the feed answers nothing.

    Requests per run: the feed, one lobby read per modeled league with games in
    it (all of its team and game categories), then one alternate-line read per
    modeled, mapped player prop until the budget runs out.
    """
    scraper = _get_scraper()
    logger.info("Getting Underdog Lines")
    feed = scraper.get(UD_LINES_URL)
    if not feed:
        logger.warning(
            "Underdog feed returned nothing; check for a retired endpoint (docs/underdog_api.md §2)"
        )
        return {}
    sports = {_ud_league(game["sport_id"]) for game in feed["games"]}
    team_offers, team_abbr = _ud_team_offers(scraper, sports)
    priced = _ud_offers(feed, team_abbr) + team_offers
    candidates = [
        (offer, line["over_under_id"])
        for offer, line in priced
        if line["over_under"]["has_alternates"]
        and offer["League"] in UD_MODELED_LEAGUES
        and offer["Market"] in stat_map["Underdog"]
    ]
    offers: dict = {}
    for offer in [offer for offer, _ in priced] + _ud_alt_lines(candidates):
        offers.setdefault(offer["League"], {}).setdefault(offer["Market"], []).append(offer)
    logger.info(
        f"{sum(len(rows) for markets in offers.values() for rows in markets.values())} Underdog offers found"
    )
    return offers
