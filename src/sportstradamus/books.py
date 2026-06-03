"""Live sportsbook scrapers for Underdog Fantasy and Sleeper.

These are the two DFS platforms still scraped directly. Sportsbook odds
(DraftKings, FanDuel, Pinnacle, Caesars, etc.) are ingested via the Odds
API in :mod:`sportstradamus.moneylines`; the deprecated per-book scrapers
live in ``src/deprecated/books_deprecated.py``.
"""

from datetime import datetime, timedelta
from http import HTTPStatus
from operator import itemgetter

from tqdm import tqdm

from sportstradamus.helpers import (
    Scrape,
    get_mlb_pitchers,
    nhl_goalies,
    remove_accents,
    requests,
)
from sportstradamus.spiderLogger import logger

# Underdog Fantasy API endpoints
_UD_BASE = "https://api.underdogfantasy.com"
UD_TEAMS_URL = "https://stats.underdogfantasy.com/v1/teams"
UD_LINES_URL = f"{_UD_BASE}/beta/v6/over_under_lines"
UD_RIVALS_URL = f"{_UD_BASE}/beta/v3/rival_lines"

# Sleeper API endpoints
SLEEPER_AVAILABLE_URL = "https://api.sleeper.app/lines/available"
SLEEPER_ALT_URL = "https://api.sleeper.app/lines/available_alt"
SLEEPER_GAMES_URL = "https://api.sleeper.app/scores/lines_game_picker"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/players/{league}?exclude_injury=false"

# Team abbreviation corrections applied across both scrapers
ABBR_MAP = {"WSH": "WAS", "GS": "GSW", "PHO": "PHX", "NOP": "NO", "AZ": "ARI"}

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


def _ud_player_ids(players_payload: list[dict]) -> dict:
    """Map Underdog player id → name + league (COMBO suffixes stripped)."""
    return {
        i["id"]: {
            "Name": str(i["first_name"] or "") + " " + str(i["last_name"] or ""),
            "League": i["sport_id"].replace("COMBOS", "").replace("COMBO", ""),
        }
        for i in players_payload
    }


def _ud_match_ids(games: list[dict], solo_games: list[dict], team_ids: dict) -> dict:
    """Map match id → home/away/league/date from team games and solo (combo) games."""
    match_ids = {}
    for i in games:
        match_ids[i["id"]] = {
            "Home": team_ids.get(i["home_team_id"]),
            "Away": team_ids.get(i["away_team_id"]),
            "League": i["sport_id"].replace("COMBOS", ""),
            "Date": _ud_date(i["scheduled_at"]),
        }
    for i in solo_games:
        if " vs " not in i["title"] and " @ " not in i["title"]:
            continue
        if " vs " in i["title"]:
            i["title"] = " @ ".join(i["title"].split(" vs ")[::-1])
        match_ids[i["id"]] = {
            "Home": i["title"].split(" @ ")[1],
            "Away": i["title"].split(" @ ")[0],
            "League": i["sport_id"].replace("COMBOS", ""),
            "Date": _ud_date(i["scheduled_at"]),
        }
    return match_ids


def _ud_appearances(
    appearances: list[dict], player_ids: dict, team_ids: dict, match_ids: dict
) -> tuple[dict, dict]:
    """Resolve each appearance id to its player and match context."""
    players, matches = {}, {}
    for i in appearances:
        players[i["id"]] = {
            "Name": player_ids.get(i["player_id"], {"Name": ""})["Name"],
            "Team": team_ids.get(i["team_id"], ""),
            "League": player_ids.get(i["player_id"], {"League": ""})["League"],
        }
        matches[i["id"]] = {
            "Home": match_ids.get(i["match_id"], {"Home": ""})["Home"],
            "Away": match_ids.get(i["match_id"], {"Away": ""})["Away"],
            "Date": match_ids.get(i["match_id"], {"Date": ""})["Date"],
        }
    return players, matches


def _ud_augment_from_rivals(
    rivals: dict, team_ids: dict, match_ids: dict, player_ids: dict, players: dict, matches: dict
) -> None:
    """Add rival-only players / games / appearances onto the lines-built maps."""
    for i in rivals["players"]:
        if i["id"] not in player_ids:
            player_ids[i["id"]] = {
                "Name": str(i["first_name"] or "") + " " + str(i["last_name"] or ""),
                "League": i["sport_id"],
            }
    for i in rivals["games"]:
        if i["id"] not in match_ids:
            match_ids[i["id"]] = {
                "Home": team_ids[i["home_team_id"]],
                "Away": team_ids[i["away_team_id"]],
                "League": i["sport_id"],
                "Date": _ud_date(i["scheduled_at"]),
            }
    for i in rivals["appearances"]:
        if i["id"] not in players:
            players[i["id"]] = {
                "Name": player_ids.get(i["player_id"], {"Name": ""})["Name"],
                "Team": team_ids.get(i["team_id"], ""),
                "League": player_ids.get(i["player_id"], {"League": ""})["League"],
            }
        if i["id"] not in matches:
            matches[i["id"]] = {
                "Home": match_ids.get(i["match_id"], {"Home": ""})["Home"],
                "Away": match_ids.get(i["match_id"], {"Away": ""})["Away"],
                "Date": match_ids.get(i["match_id"], {"Date": ""})["Date"],
            }


def _ud_fantasy_market(player: str, league: str, market: str) -> str:
    """Split Underdog 'Fantasy Points' into position-specific MLB/NHL markets."""
    if "Fantasy" in market and league == "MLB":
        pitchers = list(get_mlb_pitchers().values())
        return "Pitcher Fantasy Points" if player in pitchers else "Hitter Fantasy Points"
    if "Fantasy" in market and league == "NHL":
        return "Goalie Fantasy Points" if player in list(nhl_goalies) else "Skater Fantasy Points"
    return market


def _ud_over_under_offer(o: dict, players: dict, matches: dict) -> tuple[dict | None, str]:
    """Parse one over/under line into an offer; ``(None, market)`` for combo legs.

    The second tuple element is the raw display stat — the caller threads the
    last one into the rivals pass (see :func:`get_ud`).
    """
    player = players[o["over_under"]["appearance_stat"]["appearance_id"]]
    market = o["over_under"]["appearance_stat"]["display_stat"]
    if "+" in player["Name"]:
        return None, market
    game = matches.get(
        o["over_under"]["appearance_stat"]["appearance_id"], {"Home": "", "Away": ""}
    )
    opponent = game["Home"]
    if opponent == player["Team"]:
        opponent = game["Away"]
    boosts = [0, 0]
    for option in o["options"]:
        if option["choice"] == "higher":
            boosts[0] = float(option["payout_multiplier"])
        elif option["choice"] == "lower":
            boosts[1] = float(option["payout_multiplier"])
    n = {
        "Player": remove_accents(player["Name"]),
        "League": player["League"],
        "Team": ABBR_MAP.get(player["Team"], player["Team"]),
        "Opponent": ABBR_MAP.get(opponent, opponent),
        "Date": game["Date"],
        "Market": market,
        "Line": float(o["stat_value"]),
        "Boost_Over": boosts[0],
        "Boost_Under": boosts[1],
    }
    n["Market"] = _ud_fantasy_market(n["Player"], n["League"], market)
    return n, market


def _ud_rival_offer(o: dict, players: dict, matches: dict, ou_market: str) -> tuple[dict, str]:
    """Parse one rival (head-to-head) line into an offer; returns ``(offer, bet)``."""
    player1 = players[o["options"][0]["appearance_stat"]["appearance_id"]]
    player2 = players[o["options"][1]["appearance_stat"]["appearance_id"]]
    game1 = matches[o["options"][0]["appearance_stat"]["appearance_id"]]
    game2 = matches[o["options"][1]["appearance_stat"]["appearance_id"]]
    opponent1 = game1.get("Home", "NA")
    if opponent1 == player1["Team"]:
        opponent1 = game1.get("Away", "NA")
    opponent2 = game2.get("Home", "NA")
    if opponent2 == player2["Team"]:
        opponent2 = game2.get("Away", "NA")
    bet = o["options"][0]["appearance_stat"]["display_stat"].replace("Fewer ", "")
    n = {
        "Player": remove_accents(player1["Name"]) + " vs. " + remove_accents(player2["Name"]),
        "League": player1["League"],
        "Team": ABBR_MAP.get(player1["Team"], player1["Team"])
        + "/"
        + ABBR_MAP.get(player2["Team"], player2["Team"]),
        "Opponent": ABBR_MAP.get(opponent1, opponent1)
        + "/"
        + ABBR_MAP.get(opponent2, opponent2),
        "Date": game1["Date"],
        "Market": "H2H " + bet,
        "Line": float(o["options"][0]["spread"]) - float(o["options"][1]["spread"]),
        "Boost": 1,
    }
    # Legacy quirk preserved: the rivals fantasy reclassification keys off
    # ``ou_market`` — the last over/under line's market, leaked from get_ud's
    # prior loop — not this rival's own bet.
    if "Fantasy" in ou_market and n["League"] == "MLB":
        pitchers = list(get_mlb_pitchers().values())
        n["Market"] = (
            "H2H Pitcher Fantasy Points"
            if n["Player"] in pitchers
            else "H2H Hitter Fantasy Points"
        )
    return n, bet


def get_ud():
    """Retrieve player prop offers from the Underdog Fantasy API.

    Fetches over/under lines and rival (head-to-head) lines, normalizes
    player names and team abbreviations, and splits MLB/NHL fantasy points
    into position-specific markets.

    Returns:
        dict: ``{league: {market: [offer, ...]}}`` where each offer contains
            ``Player``, ``League``, ``Team``, ``Opponent``, ``Date``,
            ``Market``, ``Line``, ``Boost_Over``, and ``Boost_Under``.
            Returns ``[]`` if the teams endpoint fails, ``{}`` if the lines
            endpoint fails.
    """
    scraper = _get_scraper()
    logger.info("Getting Underdog Lines")
    teams = scraper.get(UD_TEAMS_URL)
    if not teams:
        logger.warning("Could not receive offer data")
        return []
    team_ids = {i["id"]: i["abbr"] for i in teams["teams"]}

    api = scraper.get(UD_LINES_URL)
    if not api:
        logger.info("No offers found")
        return {}
    player_ids = _ud_player_ids(api["players"])
    match_ids = _ud_match_ids(api["games"], api["solo_games"], team_ids)
    players, matches = _ud_appearances(api["appearances"], player_ids, team_ids, match_ids)

    offers = {}
    market = ""
    for o in tqdm(api["over_under_lines"], desc="Getting Underdog Over/Unders", unit="offer"):
        n, line_market = _ud_over_under_offer(o, players, matches)
        if n is not None:
            market = line_market
            offers.setdefault(n["League"], {}).setdefault(n["Market"], []).append(n)

    rivals = scraper.get(UD_RIVALS_URL)
    if not rivals:
        logger.info(str(len(offers)) + " offers found")
        return offers
    _ud_augment_from_rivals(rivals, team_ids, match_ids, player_ids, players, matches)

    for o in tqdm(rivals["rival_lines"], desc="Getting Underdog Rivals", unit="offer"):
        n, bet = _ud_rival_offer(o, players, matches, market)
        offers.setdefault(n["League"], {}).setdefault(bet, []).append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers


def get_sleeper():
    """Retrieve player prop offers from the Sleeper API.

    Fetches standard and alternate lines, resolves player names and
    team abbreviations, and structures offers by league and market.

    Returns:
        dict: ``{league: {market: [offer, ...]}}`` where each offer contains
            ``Player``, ``League``, ``Team``, ``Opponent``, ``Date``,
            ``Market``, ``Line``, ``Boost_Over``, and ``Boost_Under``.
            Returns ``{}`` on API failure.
    """
    offers = {}
    res = requests.get(SLEEPER_AVAILABLE_URL)

    if res.status_code != HTTPStatus.OK:
        return offers

    res = res.json()
    leagues = {x["sport"] for x in res}

    alt = requests.get(SLEEPER_ALT_URL)
    if alt.status_code == HTTPStatus.OK:
        res.extend(alt.json())

    game_res = requests.get(SLEEPER_GAMES_URL)
    if game_res.status_code != HTTPStatus.OK:
        return offers

    games = {}
    for game in game_res.json():
        games.setdefault(game["sport"], {})
        home_team = game.get("metadata", {}).get("home_team")
        if isinstance(home_team, dict):
            home_team = home_team.get("team")
        away_team = game.get("metadata", {}).get("away_team")
        if isinstance(away_team, dict):
            away_team = away_team.get("team")
        games[game["sport"]][game["game_id"]] = {
            "teams": [ABBR_MAP.get(home_team, home_team), ABBR_MAP.get(away_team, away_team)],
            "date": game.get("date"),
        }

    for league in tqdm(leagues, desc="Getting Sleeper lines...", leave=False):
        players = requests.get(SLEEPER_PLAYERS_URL.format(league=league))
        if players.status_code != HTTPStatus.OK:
            continue

        players = players.json()
        props = [x for x in res if x["sport"] == league]

        for prop in props:
            player = players.get(prop["subject_id"])
            if player is None:
                continue

            player_name = remove_accents(" ".join([player["first_name"], player["last_name"]]))
            player_team = player["team"]
            teams = games.get(league, {}).get(prop["game_id"], {}).get("teams")
            if len(teams) != 2:
                continue

            game_date = games.get(league, {}).get(prop["game_id"], {}).get("date")

            all_outcomes = sorted(prop["options"], key=itemgetter("outcome", "outcome_value"))
            lines = {x["outcome_value"] for x in all_outcomes}
            for line in lines:
                outcomes = [x for x in all_outcomes if x["outcome_value"] == line]
                if len(outcomes) < 2:
                    if outcomes[0]["outcome"] == "over":
                        outcomes = [*outcomes, {}]
                    else:
                        outcomes = [{}, *outcomes]

                opp = next(team for team in teams if team != player_team) if player_team else None

                n = {
                    "Player": player_name,
                    "League": league.upper(),
                    "Team": ABBR_MAP.get(player_team, player_team),
                    "Opponent": ABBR_MAP.get(opp, opp),
                    "Date": game_date,
                    "Market": prop["wager_type"],
                    "Line": line,
                    "Boost_Over": float(outcomes[0].get("payout_multiplier", 0)),
                    "Boost_Under": float(outcomes[1].get("payout_multiplier", 0)),
                }

                offers.setdefault(n["League"], {}).setdefault(n["Market"], [])
                offers[n["League"]][n["Market"]].append(n)

    return offers
