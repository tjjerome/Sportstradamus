"""Sleeper scraper: player props from the public ``lines`` feeds."""

from http import HTTPStatus
from operator import itemgetter

from tqdm import tqdm

from sportstradamus.helpers import ABBR_MAP, remove_accents, requests
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S

SLEEPER_AVAILABLE_URL = "https://api.sleeper.app/lines/available"
SLEEPER_ALT_URL = "https://api.sleeper.app/lines/available_alt"
SLEEPER_GAMES_URL = "https://api.sleeper.app/scores/lines_game_picker"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/players/{league}?exclude_injury=false"


def _sleeper_games_map(game_payload: list[dict]) -> dict:
    """Map ``{league: {game_id: {teams, date}}}`` from the Sleeper games feed.

    Team metadata arrives either as a bare abbreviation or as a nested
    ``{"team": abbr}`` dict; both forms are flattened to the abbreviation.
    """
    games = {}
    for game in game_payload:
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
    return games


def _sleeper_prop_offers(prop: dict, players: dict, league: str, games: dict) -> list[dict]:
    """Parse one Sleeper prop into per-line offers.

    Returns ``[]`` when the player is unknown or the game lacks two teams. A
    line carrying a single outcome is padded with an empty option so the
    over boost always lands in slot 0 and the under boost in slot 1.
    """
    player = players.get(prop["subject_id"])
    if player is None:
        return []

    player_name = remove_accents(" ".join([player["first_name"], player["last_name"]]))
    player_team = player["team"]
    teams = games.get(league, {}).get(prop["game_id"], {}).get("teams")
    if len(teams) != 2:
        return []

    game_date = games.get(league, {}).get(prop["game_id"], {}).get("date")
    opp = next(team for team in teams if team != player_team) if player_team else None

    all_outcomes = sorted(prop["options"], key=itemgetter("outcome", "outcome_value"))
    lines = {x["outcome_value"] for x in all_outcomes}
    offers = []
    for line in lines:
        outcomes = [x for x in all_outcomes if x["outcome_value"] == line]
        if len(outcomes) < 2:
            outcomes = [*outcomes, {}] if outcomes[0]["outcome"] == "over" else [{}, *outcomes]
        offers.append(
            {
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
        )
    return offers


def _sleeper_fetch_json(url: str) -> dict | list | None:
    """GET a Sleeper endpoint; return the parsed JSON body, or ``None`` on a non-200."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    if response.status_code != HTTPStatus.OK:
        return None
    return response.json()


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
    res = _sleeper_fetch_json(SLEEPER_AVAILABLE_URL)
    if res is None:
        return offers
    leagues = {x["sport"] for x in res}

    alt = _sleeper_fetch_json(SLEEPER_ALT_URL)
    if alt is not None:
        res.extend(alt)

    games_payload = _sleeper_fetch_json(SLEEPER_GAMES_URL)
    if games_payload is None:
        return offers
    games = _sleeper_games_map(games_payload)

    for league in tqdm(leagues, desc="Getting Sleeper lines...", leave=False):
        players = _sleeper_fetch_json(SLEEPER_PLAYERS_URL.format(league=league))
        if players is None:
            continue
        props = [x for x in res if x["sport"] == league]
        for prop in props:
            for n in _sleeper_prop_offers(prop, players, league, games):
                offers.setdefault(n["League"], {}).setdefault(n["Market"], []).append(n)

    return offers
