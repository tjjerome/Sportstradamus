from sportstradamus.spiderLogger import logger
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
import random
from tqdm import tqdm
import numpy as np
import statsapi as mlb
from time import sleep
from operator import itemgetter
from sportstradamus.helpers import (
    apikey,
    requests,
    scraper,
    remove_accents,
    no_vig_odds,
    get_ev,
    mlb_pitchers,
    nhl_goalies
)


def get_dk(events, categories, league):
    """
    Retrieve DraftKings data for given events and categories.

    Args:
        events (str): The event ID.
        categories (list): The list of categories to retrieve data for.

    Returns:
        dict: The parsed DraftKings data.
    """
    players = []
    markets = []
    games = {}

    # Iterate over categories
    for cat in tqdm(categories):
        # Get data from DraftKings API
        dk_api = scraper.get(
            f"https://sportsbook.draftkings.com//sites/US-SB/api/v5/eventgroups/{events}/categories/{cat}?format=json"
        )

        if not dk_api:
            continue

        if "errorStatus" in dk_api:
            logger.warning(dk_api["errorStatus"]["developerMessage"])
            continue

        # Store game IDs and names
        for i in dk_api["eventGroup"]["events"]:
            games[i["eventId"]] = {"Title": i["name"], "Date": datetime.fromisoformat(
                i["startDate"]).astimezone().strftime("%Y-%m-%d")}

        # Get offer subcategory descriptors
        for i in dk_api["eventGroup"]["offerCategories"]:
            if "offerSubcategoryDescriptors" in i:
                markets = i["offerSubcategoryDescriptors"]

        subcategoryIds = []  # Need subcategoryIds first

        # Retrieve subcategory IDs
        for i in markets:
            subcategoryIds.append(i["subcategoryId"])

        # Iterate over subcategories
        for ids in subcategoryIds:
            # Get data from DraftKings API
            dk_api = scraper.get(
                f"https://sportsbook.draftkings.com//sites/US-SB/api/v5/eventgroups/{events}/categories/{cat}/subcategories/{ids}?format=json"
            )
            if not dk_api:
                continue

            # Get offer subcategory descriptors
            for i in dk_api["eventGroup"]["offerCategories"]:
                if "offerSubcategoryDescriptors" in i:
                    markets = i["offerSubcategoryDescriptors"]

            # Iterate over markets
            for i in markets:
                if "offerSubcategory" in i:
                    market = i["name"]
                    for j in i["offerSubcategory"]["offers"]:
                        for k in j:
                            if market in ["TD Scorer", "Goalscorer"]:
                                for o in k["outcomes"]:
                                    if o.get("criterionName") == "Anytime Scorer":
                                        player = remove_accents(
                                            o["participant"])
                                        p = no_vig_odds(o["oddsDecimal"])
                                        newline = {
                                            "Player": player,
                                            "League": league,
                                            "Market": market,
                                            "Date": games[k["eventId"]]["Date"],
                                            "Line": '0.5',
                                            "Over": p[0],
                                            "Under": p[1]
                                        }
                                        players.append(newline)

                            if "line" not in k["outcomes"][0]:
                                continue
                            if "participant" in k["outcomes"][0]:
                                player = remove_accents(
                                    k["outcomes"][0]["participant"])
                            elif market == "1st Inning Total Runs":
                                for e in dk_api["eventGroup"]["events"]:
                                    if e["eventId"] == k["eventId"]:
                                        player = (
                                            remove_accents(e["eventMetadata"][
                                                "participantMetadata"
                                            ].get("awayTeamStartingPitcherName", ""))
                                            + " + "
                                            + remove_accents(e["eventMetadata"][
                                                "participantMetadata"
                                            ].get("homeTeamStartingPitcherName", ""))
                                        )
                            elif ": " in k["label"]:
                                player = k["label"][: k["label"].find(": ")]
                            elif k["eventId"] in games:
                                player = games[k["eventId"]]["Title"]
                            else:
                                continue

                            try:
                                outcomes = sorted(
                                    k["outcomes"][:2], key=lambda x: x["label"]
                                )
                                line = outcomes[0]["line"]
                                if len(outcomes) == 1:
                                    p = no_vig_odds(outcomes[0]["oddsDecimal"])
                                else:
                                    p = no_vig_odds(
                                        outcomes[0]["oddsDecimal"],
                                        outcomes[1]["oddsDecimal"],
                                    )
                                newline = {
                                    "Player": player,
                                    "League": league,
                                    "Market": market,
                                    "Date": games[k["eventId"]]["Date"],
                                    "Line": line,
                                    "Over": p[0],
                                    "Under": p[1]
                                }
                                players.append(newline)
                            except:
                                continue
    return players


def get_fd(sport, tabs):
    """
    Retrieve FanDuel data for a given sport and tabs.

    Args:
        sport (str): The sport ID.
        tabs (list): The list of tabs to retrieve data for.

    Returns:
        dict: The parsed FanDuel data.
    """
    api_url = "https://sbapi.tn.sportsbook.fanduel.com/api/{}?"
    params = {
        "betexRegion": "GBR",
        "capiJurisdiction": "intl",
        "currencyCode": "USD",
        "exchangeLocale": "en_US",
        "includePrices": "true",
        "language": "en",
        "regionCode": "NAMERICA",
        "timezone": "America%2FChicago",
        "_ak": "FhMFpcPWXMeyZxOx",
        "page": "CUSTOM",
        "customPageId": sport
    }

    data = scraper.get_proxy(
        api_url.format("content-managed-page") + urlencode(params)
    )

    if not data:
        logger.warning(f"No lines found for {sport}.")
        return {}

    attachments = data.get("attachments")
    if not attachments or not attachments.get("events"):
        logger.warning(f"No events found for {sport}.")
        return {}

    events = attachments["events"]

    # Filter event IDs based on open date
    event_ids = [(str(event['eventId']), datetime.fromisoformat(event["openDate"]).astimezone().strftime("%Y-%m-%d")) for event in events.values() if (datetime.fromisoformat(
        event["openDate"]) > datetime.now(timezone.utc)) and (datetime.fromisoformat(event["openDate"]) - timedelta(days=10) < datetime.now(timezone.utc))]

    players = []

    # Iterate over event IDs and tabs
    for event_id, date in tqdm(event_ids, desc="Processing Events", unit="event"):
        skip = False
        for tab in tabs:
            if skip:
                continue
            new_params = params | {
                "usePlayerPropsVirtualMarket": "true",
                "eventId": event_id,
                "tab": tab,
            }

            data = scraper.get_proxy(
                api_url.format("event-page") + urlencode(new_params)
            )

            if not data:
                skip = True
                continue

            attachments = data.get("attachments")
            events = attachments.get("events")
            offers = attachments.get("markets")
            if not events or not offers:
                skip = True
                logger.warning(f"No offers found for {event_id}, {tab}.")
                continue

            # Filter offers based on market name
            offers = [
                offer
                for offer in offers.values()
                if offer["bettingType"] == "MOVING_HANDICAP"
                or offer["bettingType"] == "ODDS"
            ]
            offers = [
                offer
                for offer in offers
                if not any(
                    substring in offer["marketName"]
                    for substring in [
                        "Total Points",
                        "Spread Betting",
                        "Total Match Points",
                        "Spread",
                        "Run Line",
                        "Total Runs",
                        "Puck Line",
                        "Total Goals",
                        "Qtr",
                        "Moneyline",
                        "Result",
                        "Odd/Even",
                        "Alt ",
                    ]
                )
            ]

            if len(offers) == 0:
                skip = True

            # Iterate over offers
            for offer in offers:

                if offer["marketName"] in ["Any Time Touchdown Scorer", "To Record A Hit", "To Hit A Home Run",
                                           "To Record a Run", "To Record an RBI", "To Hit a Single",
                                           "To Hit a Double", "Any Time Goal Scorer", "Player to Record 1+ Assists",
                                           "Player to Record 1+ Points", "Player to Record 1+ Powerplay Points"]:
                    for o in offer["runners"]:
                        player = remove_accents(o["runnerName"])
                        market = offer["marketName"]

                        p = no_vig_odds(
                            o["winRunnerOdds"]["trueOdds"]["decimalOdds"]["decimalOdds"])

                        newline = {
                            "Player": player,
                            "Market": market,
                            "League": sport.upper(),
                            "Date": date,
                            "Line": "0.5",
                            "Over": p[0],
                            "Under": p[1],
                        }
                        players.append(newline)

                if len(offer["runners"]) != 2:
                    continue

                try:
                    if " - " in offer["marketName"]:
                        player = remove_accents(
                            offer["marketName"].split(" - ")[0])
                        market = offer["marketName"].split(" - ")[1]
                    elif sport == 'mlb' and "@" in events[event_id]["name"]:
                        teams = (
                            events[event_id]["name"]
                            .replace(" @ ", ")")
                            .replace(" (", ")")
                            .split(")")
                        )
                        pitcher1 = mlb_pitchers.get(
                            mlb.lookup_team(teams[0].strip())[
                                0]["fileCode"].upper(), ""
                        )
                        pitcher2 = mlb_pitchers.get(
                            mlb.lookup_team(teams[3].strip())[
                                0]["fileCode"].upper(), ""
                        )
                        player = pitcher1 + " + " + pitcher2
                        market = offer["marketName"]
                    else:
                        continue

                    outcomes = sorted(
                        offer["runners"][:2], key=lambda x: x["runnerName"])
                    if outcomes[1]["handicap"] == 0:
                        line = [
                            float(s)
                            for s in market.split(" ")
                            if s.replace(".", "").isdigit()
                        ]
                        if line:
                            line = line[-1]
                        else:
                            continue
                    else:
                        line = outcomes[1]["handicap"]

                    p = no_vig_odds(
                        outcomes[0]["winRunnerOdds"]["trueOdds"]["decimalOdds"]["decimalOdds"],
                        outcomes[1]["winRunnerOdds"]["trueOdds"]["decimalOdds"]["decimalOdds"],
                    )

                    newline = {
                        "Player": player,
                        "Market": market,
                        "League": sport.upper(),
                        "Date": date,
                        "Line": line,
                        "Over": p[0],
                        "Under": p[1],
                    }
                    players.append(newline)
                except:
                    continue

    return players


def get_pinnacle(league):
    """
    Retrieve Pinnacle data for a given league.

    Args:
        league (int): The league ID.

    Returns:
        dict: The parsed Pinnacle data.
    """
    header = {"X-API-KEY": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"}
    url = f"https://guest.api.arcadia.pinnacle.com/0.1/leagues/{league}/markets/straight",

    leagues = {246: "MLB", 487: "NBA", 889: "NFL", 1456: "NHL"}

    try:
        odds = scraper.get_proxy(url, header)
    except:
        logger.warning("No lines found for league: " + str(league))
        return {}

    if not type(odds) == list or not type(odds[0]) == dict:
        logger.warning("No lines found for league: " + str(league))
        return {}

    lines = {}
    for line in odds:
        if (
            not line["prices"]
            or "points" not in line["prices"][0]
            or line["prices"][0]["points"] == 0
        ):
            continue

        if line["matchupId"] not in lines:
            lines[line["matchupId"]] = {}

        if line["key"] not in lines[line["matchupId"]]:
            lines[line["matchupId"]][line["key"]] = {}

        for price in line["prices"]:
            if price.get("participantId"):
                lines[line["matchupId"]][line["key"]][price["participantId"]] = {
                    "Price": price["price"],
                    "Line": price["points"],
                }
            elif price.get("designation"):
                lines[line["matchupId"]][line["key"]][
                    price["designation"].capitalize()
                ] = {"Price": price["price"], "Line": price["points"]}

    url = f"https://guest.api.arcadia.pinnacle.com/0.1/leagues/{league}/matchups"

    try:
        sleep(random.uniform(3, 5))
        api = scraper.get_proxy(url, header)
        markets = [
            line
            for line in api
            if line.get("special", {"category": ""}).get("category") == "Player Props"
        ]
    except:
        logger.warning("No lines found for league: " + str(league))
        return {}

    players = []
    for market in tqdm(markets):
        player = market["special"]["description"].replace(
            " (BAL)", "").replace(" (BUF)", "").replace(" (CAR)", "")
        player = player.replace(" (", ")").split(")")
        bet = player[1]
        player = remove_accents(player[0])
        try:
            outcomes = sorted(market["participants"]
                              [:2], key=lambda x: x["name"])
            if outcomes[0]["name"] == "No":
                outcomes.reverse()

            code = "s;0;ou" if "s;0;ou" in lines[market["id"]] else "s;0;m"
            line = lines[market["id"]][code][outcomes[1]
                                             ["id"]].get("Line", 0.5)
            prices = [
                lines[market["id"]][code][participant["id"]]["Price"]
                for participant in outcomes
            ]
            p = no_vig_odds(prices[0], prices[1])
            newline = {
                "Player": player,
                "Market": bet,
                "League": leagues.get(league),
                "Date": datetime.fromisoformat(market["startTime"]).astimezone().strftime("%Y-%m-%d"),
                "EV": get_ev(line, p[1]),
                "Line": str(line),
                "Over": p[0],
                "Under": p[1],
            }
            players.append(newline)
        except:
            continue

    if league == 246:
        games = [
            line
            for line in api
            if 3 in [period["period"] for period in line.get("periods")]
        ]
        for game in tqdm(games):
            if lines.get(game["id"], {"s;3;ou;0.5": False}).get("s;3;ou;0.5"):
                try:
                    game_num = 1
                    pitchers = []
                    for team in game["participants"]:
                        if team["name"][0] == "G":
                            game_num = int(team["name"][1])
                            team_name = team["name"][3:]
                        else:
                            team_name = team["name"]

                        team_abbr = mlb.lookup_team(
                            team_name)[0]["fileCode"].upper()
                        if game_num > 1:
                            team_abbr = team_abbr + str(game_num)

                        pitchers.append(mlb_pitchers.get(team_abbr))

                    player = " + ".join(pitchers)
                    bet = "1st Inning Runs Allowed"

                    line = lines[game["id"]]["s;3;ou;0.5"]["Under"]["Line"]
                    prices = [
                        lines[game["id"]]["s;3;ou;0.5"]["Over"]["Price"],
                        lines[game["id"]]["s;3;ou;0.5"]["Under"]["Price"],
                    ]
                    p = no_vig_odds(prices[0], prices[1])
                    newline = {
                        "Player": player,
                        "Market": bet,
                        "League": leagues.get(league),
                        "Date": datetime.fromisoformat(market["startTime"]).astimezone().strftime("%Y-%m-%d"),
                        "EV": get_ev(line, p[1]),
                        "Line": str(line),
                        "Over": p[0],
                        "Under": p[1],
                    }
                    players.append(newline)

                except:
                    continue

    return players


def get_caesars(sport, league):
    """
    Retrieves player props data from the Caesars API for the specified sport and league.

    Args:
        sport (str): The sport for which to fetch player props data.
        league (int): The league ID for which to fetch player props data.

    Returns:
        dict: A dictionary containing player props data.

    """
    # Define the Caesars API URL
    caesars = f"https://api.americanwagering.com/regions/us/locations/mi/brands/czr/sb/v3/sports/{sport}/events/schedule/?competitionIds={league}&content-type=json"

    # Set request parameters
    params = {"api_key": apikey, "url": caesars, "optimize_request": True}

    # Mapping to swap certain market display names
    marketSwap = {
        "Points + Assists + Rebounds": "Pts + Rebs + Asts",
        "3pt Field Goals": "3-PT Made",
        "Bases": "Total Bases",
    }

    try:
        # Make a GET request to the Caesars API
        api = scraper.get_proxy(caesars)

        # Get the game IDs for the specified sport and league
        gameIds = [
            game["id"]
            for game in api["competitions"][0]["events"]
            if game["type"] == "MATCH"
            and not game["started"]
            and game["marketCountActivePreMatch"] > 20
        ]

    except Exception as exc:
        logger.exception("Caesars API request failed")
        return {}

    players = []
    for id in tqdm(gameIds):
        caesars = f"https://api.americanwagering.com/regions/us/locations/mi/brands/czr/sb/v3/events/{id}?content-type=json"
        params["url"] = caesars

        # Sleep for a random interval to avoid overloading the API
        sleep(random.uniform(1, 5))

        try:
            # Make a GET request to the Caesars API for each game ID
            api = scraper.get_proxy(caesars)

            # Filter and retrieve the desired markets
            markets = [
                market
                for market in api["markets"]
                if market.get("active")
                and (
                    market.get("metadata", {}).get("marketType", {})
                    == "PLAYERLINEBASED" or
                    market.get("metadata", {}).get("marketType", {})
                    == "PLAYEROUTCOME"
                    or market.get("displayName") == "Run In 1st Inning?"
                )
            ]

            date = datetime.fromisoformat(
                api["startTime"]).astimezone().strftime("%Y-%m-%d")

        except:
            logger.exception(f"Unable to parse game {id}")
            continue

        for market in markets:
            if market.get("metadata", {}).get("marketType", {}) == "PLAYEROUTCOME":
                marketName = market.get("displayName")
                line = 0.5
                for o in market.get("selections", []):
                    player = remove_accents(
                        o.get('name', '').replace('|', '').strip())
                    if not o.get("price"):
                        continue
                    p = no_vig_odds(o["price"]["d"])
                    newline = {
                        "Player": player,
                        "Market": marketName,
                        "League": api["competitionName"],
                        "Date": date,
                        "Line": line,
                        "Over": p[0],
                        "Under": p[1],
                    }
                    players.append(newline)

                continue
            if market.get("displayName") == "Run In 1st Inning?":
                # Handle special case for "Run In 1st Inning?" market
                marketName = "1st Inning Runs Allowed"
                line = 0.5
                player = " + ".join(
                    [
                        mlb_pitchers.get(
                            team["teamData"].get("teamAbbreviation", ""), ""
                        )
                        for team in api["markets"][0]["selections"]
                    ]
                )
            else:
                # Get player and market names
                player = remove_accents(market["metadata"]["player"])
                marketName = (
                    market["displayName"]
                    .replace("|", "")
                    .replace("Total", "")
                    .replace("Player", "")
                    .replace("Batter", "")
                    .strip()
                )
                marketName = marketSwap.get(marketName, marketName)

                if marketName == "Props":
                    marketName = market.get("metadata", {}).get(
                        "marketCategoryName")

                line = market["line"]

            # Calculate odds and expected value
            p = no_vig_odds(
                market["selections"][0]["price"]["d"],
                market["selections"][1]["price"]["d"],
            )
            newline = {
                "Player": player,
                "Market": marketName,
                "League": api["competitionName"],
                "Date": date,
                "Line": line,
                "Over": p[0],
                "Under": p[1],
            }
            players.append(newline)

    return players


def get_pp(books=False):
    """
    Retrieves player offers data from the PrizePicks API.

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
                              'Opponent': opponent_name
                          },
                          ...
                      ],
                      ...
                  },
                  ...
              }
    """
    logger.info("Getting PrizePicks Lines")
    offers = {}
    live_bets = [
        "1H",
        "2H",
        "1P",
        "2P",
        "3P",
        "1Q",
        "2Q",
        "3Q",
        "4Q",
        "LIVE",
        "SZN",
        "SZN2",
        "SERIES",
    ]

    try:
        # Retrieve the available leagues
        leagues = scraper.get_proxy("https://api.prizepicks.com/leagues")
        leagues = [
            i["id"]
            for i in leagues["data"]
            if i["attributes"]["projections_count"] > 0
            and not any([string in i["attributes"]["name"] for string in live_bets])
        ]
        if not books:
            leagues = [league for league in leagues if int(league) in [
                2, 3, 7, 8, 9]]
    except:
        logger.exception("Retrieving leagues failed")
        leagues = [2, 3, 7, 8, 9]

    for l in tqdm(leagues, desc="Processing PrizePicks offers"):
        try:
            # Retrieve players and lines for each league
            api = scraper.get_proxy(
                f"https://api.prizepicks.com/projections?league_id={l}")
            players = api["included"]
            lines = api["data"]
        except:
            logger.exception(f"League {l} failed")
            continue

        player_ids = {}
        league = None

        abbr_map = {
            "WSH": "WAS",
            "GS": "GSW",
            "PHO": "PHX",
            "NOP": "NO",
            "JAC": "JAX",
            "LAV": "LV",
            "NJY": "NYJ",
            "NJD": "NJ",
            "AZ": "ARI"
        }

        for p in players:
            if p["type"] == "new_player":
                player_ids[p["id"]] = {
                    "Name": p["attributes"]["name"].replace("\t", "").strip(),
                    "Team": abbr_map.get(p["attributes"]["team"], p["attributes"]["team"])
                }
                if "position" in p["attributes"]:
                    player_ids[p["id"]].update(
                        {"Position": p["attributes"]["position"]})
            elif p["type"] == "league":
                league = p["attributes"]["name"].replace("CMB", "")

        for o in tqdm(lines, desc="Getting offers for " + league, unit="offer"):
            if o["attributes"]["adjusted_odds"] or o["attributes"]["is_promo"]:
                continue
            
            players = player_ids[o["relationships"]
                                 ["new_player"]["data"]["id"]]["Name"].split(" + ")
            teams = player_ids[o["relationships"]["new_player"]
                               ["data"]["id"]]["Team"].upper().split("/")
            opponents = o["attributes"]["description"].upper().split("/")
            n = {
                "Player": " + ".join([remove_accents(p) for p in players]),
                "League": league,
                "Team": "/".join([abbr_map.get(t, t) for t in teams]),
                "Opponent": "/".join([abbr_map.get(t, t) for t in opponents]),
                "Date": o["attributes"]["start_time"].split("T")[0],
                "Market": o["attributes"]["stat_type"].replace(" (Combo)", ""),
                "Line": o["attributes"]["line_score"],
            }

            if league == "NFL" and n["Market"] == "Pass+Rush+Rec TDs":
                if player_ids[o["relationships"]["new_player"]["data"]["id"]].get("Position") == "QB":
                    n["Market"] = "Pass+Rush TDs"
                else:
                    n["Market"] = "Rush+Rec TDs"

            if league not in offers:
                offers[league] = {}
            if n["Market"] not in offers[league]:
                offers[league][n["Market"]] = []

            offers[league][n["Market"]].append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers


def get_ud():
    """
    Retrieves player offers data from the Underdog Fantasy API.

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

    api = scraper.get(
        "https://api.underdogfantasy.com/beta/v5/over_under_lines")
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
            "Home": team_ids[i["home_team_id"]],
            "Away": team_ids[i["away_team_id"]],
            "League": i["sport_id"].replace("COMBOS", ""),
            "Date": (
                datetime.strptime(i["scheduled_at"], "%Y-%m-%dT%H:%M:%SZ")
                - timedelta(hours=5)
            ).strftime("%Y-%m-%d"),
        }
    for i in api["solo_games"]:
        if not " vs " in i["title"] and not " @ " in i["title"]:
            continue
        if " vs " in i["title"]:
            i["title"] = " @ ".join(i["title"].split(" vs ")[::-1])
        match_ids[i["id"]] = {
            "Home": i["title"].split(" @ ")[1],
            "Away": i["title"].split(" @ ")[0],
            "League": i["sport_id"].replace("COMBOS", ""),
            "Date": (
                datetime.strptime(i["scheduled_at"], "%Y-%m-%dT%H:%M:%SZ")
                - timedelta(hours=5)
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

    abbr_map = {
        "WSH": "WAS",
        "GS": "GSW",
        "PHO": "PHX",
        "NOP": "NO",
        "AZ": "ARI"
    }

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
            if n["Player"] in list(mlb_pitchers.values()):
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

    rivals = scraper.get("https://api.underdogfantasy.com/beta/v3/rival_lines")

    if not rivals:
        logger.info(str(len(offers)) + " offers found")
        return offers

    for i in rivals["players"]:
        if not i["id"] in player_ids:
            player_ids[i["id"]] = {
                "Name": str(i["first_name"] or "") + " " + str(i["last_name"] or ""),
                "League": i["sport_id"],
            }

    for i in rivals["games"]:
        if not i["id"] in match_ids:
            match_ids[i["id"]] = {
                "Home": team_ids[i["home_team_id"]],
                "Away": team_ids[i["away_team_id"]],
                "League": i["sport_id"],
                "Date": (
                    datetime.strptime(i["scheduled_at"], "%Y-%m-%dT%H:%M:%SZ")
                    - timedelta(hours=5)
                ).strftime("%Y-%m-%d"),
            }

    for i in rivals["appearances"]:
        if not i["id"] in players:
            players[i["id"]] = {
                "Name": player_ids.get(i["player_id"], {"Name": ""})["Name"],
                "Team": team_ids.get(i["team_id"], ""),
                "League": player_ids.get(i["player_id"], {"League": ""})["League"],
            }

        if not i["id"] in matches:
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
        opponent1 = game1["Home"]
        if opponent1 == player1["Team"]:
            opponent1 = game1["Away"]
        opponent2 = game2["Home"]
        if opponent2 == player2["Team"]:
            opponent2 = game2["Away"]
        bet = o["options"][0]["appearance_stat"]["display_stat"].replace(
            "Fewer ", "")
        n = {
            "Player": remove_accents(player1["Name"])
            + " vs. "
            + remove_accents(player2["Name"]),
            "League": player1["League"],
            "Team": abbr_map.get(player1["Team"], player1["Team"]) + "/" + abbr_map.get(player2["Team"], player2["Team"]),
            "Opponent": abbr_map.get(opponent1, opponent1) + "/" + abbr_map.get(opponent2, opponent2),
            "Date": game1["Date"],
            "Market": "H2H " + bet,
            "Line": float(o["options"][0]["spread"]) - float(o["options"][1]["spread"]),
            "Boost": 1
        }
        if "Fantasy" in market and n["League"] == "MLB":
            if n["Player"] in list(mlb_pitchers.values()):
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


    abbr_map = {
        "WSH": "WAS",
        "GS": "GSW",
        "PHO": "PHX",
        "NOP": "NO",
        "AZ": "ARI"
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

            game_id = prop["game_id"].split("_")
            teams = [x for x in game_id if not x.isnumeric()]
            if len(teams) != 2:
                continue

            game_date = game_id[0]
            game_date = game_date[:4]+'-'+game_date[4:6]+'-'+game_date[6:]

            all_outcomes = sorted(prop["options"], key=itemgetter('outcome', 'outcome_value'))
            lines = set([x["outcome_value"] for x in all_outcomes])
            for line in lines:
                outcomes = [x for x in all_outcomes if x["outcome_value"] == line]
                if len(outcomes) < 2:
                    if outcomes[0]["outcome"] == "over":
                        outcomes = outcomes + [{}]
                    else:
                        outcomes = [{}] + outcomes
                player_name = remove_accents(" ".join([player["first_name"], player["last_name"]]))
                player_team = player["team"]
                opp = [team for team in teams if team != player_team][0]

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

def get_thrive():
    """
    Retrieves upcoming house prop data from the Thrive Fantasy API.

    Returns:
        dict: A dictionary containing house prop data in the following format:
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
    params = {
        "api_key": apikey,
        "url": "https://api.thrivefantasy.com/houseProp/upcomingHouseProps",
        "optimize_request": True,
        "keep_headers": True,
    }
    payload = {
        "currentPage": 1,
        "currentSize": 100,
        "half": 0,
        "Latitude": "29.5908265",
        "Longitude": "-95.1381594",
    }
    header = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Token": "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0amplcm9tZSIsImF1ZGllbmNlIjoiSU9TIiwicGFzc3dvcmRMYXN0Q2hhbmdlZEF0IjpudWxsLCJjcmVhdGVkIjoxNjg1NjQ1MDM1MTcyLCJleHAiOjE3MjgwMDAwMDAsImlhdCI6MTY4NTY0NTAzNX0.rkilAs3Jl5cidWGDYXD6nEUxNATdDIxq_ZkJi8MNo-56hcDq_jWoIU9f0Mz6XZ7RAN_tQawU3uErF8Jql7N0eA",
    }
    logger.info("Getting Thrive Lines")
    try:
        i = 0
        api = {}
        while "success" not in api and i < 3:
            i += 1
            api = requests.post(
                "https://proxy.scrapeops.io/v1/",
                params=params,
                headers=header | scraper.header,
                json=payload,
            ).json()
    except:
        logger.exception(id)
        return []

    if "success" not in api:
        logger.error(api["message"])
        return []

    abbr_map = {
        "WSH": "WAS",
        "GS": "GSW",
        "PHO": "PHX",
        "NOP": "NO"
    }

    lines = api["response"]["data"]

    offers = {}
    for line in tqdm(lines, desc="Getting Thrive Offers", unit="offer"):
        o = line.get("contestProp")
        team = o["player1"]["teamAbbr"].upper()
        opponent = str(o.get("team2Abbr", "")
                       or "").upper()
        if team == opponent:
            opponent = str(o.get("team1Abbr", "")
                           or "").upper()
        n = {
            "Player": remove_accents(
                " ".join([o["player1"]["firstName"], o["player1"]["lastName"]])
            ),
            "League": o["player1"]["leagueType"],
            "Team": abbr_map.get(team, team),
            "Opponent": abbr_map.get(opponent, opponent),
            "Date": (
                datetime.strptime(
                    o["startTime"], "%Y/%m/%d %H:%M") - timedelta(hours=5)
            ).strftime("%Y-%m-%d"),
            "Market": " + ".join(o["player1"]["propParameters"]),
            "Line": float(o["propValue"]),
        }
        if n["League"] == "HOCKEY":
            n["League"] = "NHL"

        if n["League"] not in offers:
            offers[n["League"]] = {}
        if n["Market"] not in offers[n["League"]]:
            offers[n["League"]][n["Market"]] = []

        offers[n["League"]][n["Market"]].append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers


def get_parp():
    """
    Retrieves cross-game search data from the ParlayPlay API.

    Returns:
        dict: A dictionary containing cross-game search data in the following format:
              {
                  'League Name': {
                      'Market Name': [
                          {
                              'Player': player_name,
                              'League': league_name,
                              'Team': team_name,
                              'Date': match_date,
                              'Market': market_type,
                              'Line': line_score,
                              'Slide': slide_indicator,
                              'Opponent': opponent_description
                          },
                          ...
                      ],
                      ...
                  },
                  ...
              }
    """
    logger.info("Getting ParlayPlay Lines")
    header = {
        "X-CSRFToken": "1",
        "X-Parlay-Request": "1",
        "X-ParlayPlay-Platform": "web",
        "X-Requested-With": "XMLHttpRequest"
    }
    try:
        api = scraper.get_proxy(
            "https://parlayplay.io/api/v1/crossgame/search/?sport=All&league=&includeAlt=true",
            headers=header
        )
    except:
        logger.exception(id)
        return {}

    if "players" not in api:
        logger.error("Error getting ParlayPlay Offers")
        return {}
    
    abbr_map = {
        "WSH": "WAS",
        "GS": "GSW",
        "PHO": "PHX",
        "NOP": "NO",
        "JAC": "JAX",
        "LAV": "LV",
        "NJY": "NYJ",
        "NJD": "NJ",
        "AZ": "ARI",
        "CHW": "CWS"
    }
    offers = {}
    for player in tqdm(api["players"], desc="Getting ParlayPlay Offers", unit="offer"):
        league = player["match"]["league"]["leagueNameShort"]
        offers.setdefault(league, {})
        teams = [
            player["match"]["homeTeam"]["teamAbbreviation"],
            player["match"]["awayTeam"]["teamAbbreviation"],
        ]

        player_team = player["player"]["team"]["teamAbbreviation"]
        opponent_team = [team for team in
                         [player["match"]["homeTeam"]["teamAbbreviation"],
                             player["match"]["awayTeam"]["teamAbbreviation"]]
                         if team != player_team][0]

        if player_team is not None:
            player_team = abbr_map.get(player_team, player_team)
        if opponent_team is not None:
            opponent_team = abbr_map.get(opponent_team, opponent_team)

        for stat in player["stats"]:
            market = stat["challengeName"]
            if "Fantasy" in market:
                if player["match"]["league"]["leagueNameShort"] == "MLB":
                    if player.get("position") == "SP":
                        market = "Pitcher Fantasy Points"
                    else:
                        market = "Hitter Fantasy Points"

            offers[league].setdefault(market, [])
            n = {
                "Player": remove_accents(player["player"]["fullName"]),
                "League": league,
                "Team": player_team,
                "Opponent": opponent_team,
                "Date": player["match"]["matchDate"].split("T")[0],
                "Market": market
            }
            for altLine in stat["altLines"]["values"]:
                m = n | {
                    "Line": float(altLine["selectionPoints"]),
                    "Boost_Over": float(altLine.get("decimalPriceOver", 0.0)),
                    "Boost_Under": float(altLine.get("decimalPriceUnder", 0.0))
                }
                
                offers[n["League"]][n["Market"]].append(m)

    for combo in tqdm(api["comboPackages"], desc="Getting ParlayPlay Combos", unit="offer"):
        teams = []
        opponents = []
        players = []
        for player in combo['packageLegs']:
            players.append(remove_accents(player["player"]["fullName"]))

            player_team = player["player"]["team"]["teamAbbreviation"]

            opponent_team = [team for team in
                             [player["match"]["homeTeam"]["teamAbbreviation"],
                              player["match"]["awayTeam"]["teamAbbreviation"]]
                             if team != player_team][0]
            
            if player_team is not None:
                player_team = abbr_map.get(player_team, player_team)
                teams.append(player_team)
            if opponent_team is not None:
                opponent_team = abbr_map.get(opponent_team, opponent_team)
                opponents.append(opponent_team)

        market = combo['pickType']['challengeName']
        if "Fantasy" in market:
            if combo['league']['leagueNameShort'] == "MLB":
                if player.get("position") == "SP":
                    market = "Pitcher Fantasy Points"
                else:
                    market = "Hitter Fantasy Points"

        n = {
            "Player": " + ".join(players),
            "League": player["match"]["league"]["leagueNameShort"],
            "Team": "/".join(teams),
            "Opponent": "/".join(opponents),
            "Date": player["match"]["matchDate"].split("T")[0],
            "Market": market,
            "Line": float(combo["pickValue"]),
            "Boost_Over": combo.get("defaultMultiplier", 1.77),
            "Boost_Under": combo.get("defaultMultiplier", 1.77)
        }

        if n["League"] not in offers:
            offers[n["League"]] = {}
        if n["Market"] not in offers[n["League"]]:
            offers[n["League"]][n["Market"]] = []

        offers[n["League"]][n["Market"]].append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers
