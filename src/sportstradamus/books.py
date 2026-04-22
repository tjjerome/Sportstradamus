from datetime import datetime, timedelta
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

scraper = Scrape()


def get_ud():
    """Retrieves player offers data from the Underdog Fantasy API.

    Returns:
        dict: A dictionary containing player offers data in the following format:
              {
                  'League Name': {
                      'Market Name': [
                          {
                              'Player': player_name,
                              'League': league_name,
                              'Team': team_name,
                              'Date': start_date,
                              'Market': market_type,
                              'Line': line_score,
                              'Opponent': opponent_description
                          },
                          ...
                      ],
                      ...
                  },
                  ...
              }
    """
    logger.info("Getting Underdog Lines")
    teams = scraper.get("https://stats.underdogfantasy.com/v1/teams")

    if not teams:
        logger.warning("Could not receive offer data")
        return []

    team_ids = {}
    for i in teams["teams"]:
        team_ids[i["id"]] = i["abbr"]

    api = scraper.get("https://api.underdogfantasy.com/beta/v6/over_under_lines")
    if not api:
        logger.info("No offers found")
        return {}

    player_ids = {}
    for i in api["players"]:
        player_ids[i["id"]] = {
            "Name": str(i["first_name"] or "") + " " + str(i["last_name"] or ""),
            "League": i["sport_id"].replace("COMBOS", "").replace("COMBO", ""),
        }

    match_ids = {}
    for i in api["games"]:
        match_ids[i["id"]] = {
            "Home": team_ids.get(i["home_team_id"]),
            "Away": team_ids.get(i["away_team_id"]),
            "League": i["sport_id"].replace("COMBOS", ""),
            "Date": (
                datetime.strptime(i["scheduled_at"], "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=5)
            ).strftime("%Y-%m-%d"),
        }
    for i in api["solo_games"]:
        if " vs " not in i["title"] and " @ " not in i["title"]:
            continue
        if " vs " in i["title"]:
            i["title"] = " @ ".join(i["title"].split(" vs ")[::-1])
        match_ids[i["id"]] = {
            "Home": i["title"].split(" @ ")[1],
            "Away": i["title"].split(" @ ")[0],
            "League": i["sport_id"].replace("COMBOS", ""),
            "Date": (
                datetime.strptime(i["scheduled_at"], "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=5)
            ).strftime("%Y-%m-%d"),
        }

    players = {}
    matches = {}
    for i in api["appearances"]:
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

    abbr_map = {"WSH": "WAS", "GS": "GSW", "PHO": "PHX", "NOP": "NO", "AZ": "ARI"}

    offers = {}
    for o in tqdm(api["over_under_lines"], desc="Getting Underdog Over/Unders", unit="offer"):
        player = players[o["over_under"]["appearance_stat"]["appearance_id"]]
        if "+" in player["Name"]:
            continue
        game = matches.get(
            o["over_under"]["appearance_stat"]["appearance_id"],
            {"Home": "", "Away": ""},
        )
        opponent = game["Home"]
        if opponent == player["Team"]:
            opponent = game["Away"]
        market = o["over_under"]["appearance_stat"]["display_stat"]
        boosts = [0, 0]
        for option in o["options"]:
            if option["choice"] == "higher":
                boosts[0] = float(option["payout_multiplier"])
            elif option["choice"] == "lower":
                boosts[1] = float(option["payout_multiplier"])
        n = {
            "Player": remove_accents(player["Name"]),
            "League": player["League"],
            "Team": abbr_map.get(player["Team"], player["Team"]),
            "Opponent": abbr_map.get(opponent, opponent),
            "Date": game["Date"],
            "Market": market,
            "Line": float(o["stat_value"]),
            "Boost_Over": boosts[0],
            "Boost_Under": boosts[1],
        }
        if "Fantasy" in market and n["League"] == "MLB":
            if n["Player"] in list(get_mlb_pitchers().values()):
                n["Market"] = "Pitcher Fantasy Points"
            else:
                n["Market"] = "Hitter Fantasy Points"
        if "Fantasy" in market and n["League"] == "NHL":
            if n["Player"] in list(nhl_goalies):
                n["Market"] = "Goalie Fantasy Points"
            else:
                n["Market"] = "Skater Fantasy Points"

        if n["League"] not in offers:
            offers[n["League"]] = {}
        if n["Market"] not in offers[n["League"]]:
            offers[n["League"]][n["Market"]] = []

        offers[n["League"]][n["Market"]].append(n)

        if False:
            # TODO figure out why this keeps freezing
            id = o["over_under"].get("id", 0)
            alts = scraper.get(
                f"https://api.underdogfantasy.com/v3/over_unders/{id}/alternate_projections"
            )
            if alts:
                for a in alts["projections"]:
                    if not a["is_main"]:
                        boosts = [0, 0]
                        for option in a["options"]:
                            if option["choice"] == "higher":
                                boosts[0] = float(option["payout_multiplier"])
                            elif option["choice"] == "lower":
                                boosts[1] = float(option["payout_multiplier"])
                        n = {
                            "Player": remove_accents(player["Name"]),
                            "League": player["League"],
                            "Team": abbr_map.get(player["Team"], player["Team"]),
                            "Opponent": abbr_map.get(opponent, opponent),
                            "Date": game["Date"],
                            "Market": market,
                            "Line": float(a["stat_value"]),
                            "Boost_Over": boosts[0],
                            "Boost_Under": boosts[1],
                        }
                        offers[n["League"]][n["Market"]].append(n)

    rivals = scraper.get("https://api.underdogfantasy.com/beta/v3/rival_lines")

    if not rivals:
        logger.info(str(len(offers)) + " offers found")
        return offers

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
                "Date": (
                    datetime.strptime(i["scheduled_at"], "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=5)
                ).strftime("%Y-%m-%d"),
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

    for o in tqdm(rivals["rival_lines"], desc="Getting Underdog Rivals", unit="offer"):
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
            "Team": abbr_map.get(player1["Team"], player1["Team"])
            + "/"
            + abbr_map.get(player2["Team"], player2["Team"]),
            "Opponent": abbr_map.get(opponent1, opponent1)
            + "/"
            + abbr_map.get(opponent2, opponent2),
            "Date": game1["Date"],
            "Market": "H2H " + bet,
            "Line": float(o["options"][0]["spread"]) - float(o["options"][1]["spread"]),
            "Boost": 1,
        }
        if "Fantasy" in market and n["League"] == "MLB":
            if n["Player"] in list(get_mlb_pitchers().values()):
                n["Market"] = "H2H Pitcher Fantasy Points"
            else:
                n["Market"] = "H2H Hitter Fantasy Points"

        if n["League"] not in offers:
            offers[n["League"]] = {}
        if bet not in offers[n["League"]]:
            offers[n["League"]][bet] = []

        offers[n["League"]][bet].append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers


def get_sleeper():
    offers = {}
    url = "https://api.sleeper.app/lines/available"
    res = requests.get(url)

    if res.status_code != 200:
        return offers

    res = res.json()
    leagues = set([x["sport"] for x in res])

    url = "https://api.sleeper.app/lines/available_alt"
    alt = requests.get(url)

    if alt.status_code == 200:
        res.extend(alt.json())

    abbr_map = {"WSH": "WAS", "GS": "GSW", "PHO": "PHX", "NOP": "NO", "AZ": "ARI"}

    url = "https://api.sleeper.app/scores/lines_game_picker"
    game_res = requests.get(url)

    if game_res.status_code != 200:
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
            "teams": [abbr_map.get(home_team, home_team), abbr_map.get(away_team, away_team)],
            "date": game.get("date"),
        }

    for league in tqdm(leagues, desc="Getting Sleeper lines...", leave=False):
        url = f"https://api.sleeper.app/players/{league}?exclude_injury=false"
        players = requests.get(url)

        if players.status_code != 200:
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
            lines = set([x["outcome_value"] for x in all_outcomes])
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
                    "Team": abbr_map.get(player_team, player_team),
                    "Opponent": abbr_map.get(opp, opp),
                    "Date": game_date,
                    "Market": prop["wager_type"],
                    "Line": line,
                    "Boost_Over": float(outcomes[0].get("payout_multiplier", 0)),
                    "Boost_Under": float(outcomes[1].get("payout_multiplier", 0)),
                }

                offers.setdefault(n["League"], {}).setdefault(n["Market"], [])
                offers[n["League"]][n["Market"]].append(n)

    return offers


