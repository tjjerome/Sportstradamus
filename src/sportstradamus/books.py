from sportstradamus.spiderLogger import logger
from datetime import datetime, timedelta
import random
from tqdm import tqdm
import numpy as np
import statsapi as mlb
from time import sleep
from sportstradamus.helpers import (
    apikey,
    requests,
    scraper,
    remove_accents,
    no_vig_odds,
    get_ev,
    prob_to_odds,
    mlb_pitchers,
    nhl_goalies
)


def get_dk(events, categories):
    """
    Retrieve DraftKings data for given events and categories.

    Args:
        events (str): The event ID.
        categories (list): The list of categories to retrieve data for.

    Returns:
        dict: The parsed DraftKings data.
    """
    players = {}
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
            games[i["eventId"]] = i["name"]

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
            for i in tqdm(markets):
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
                                            "EV": get_ev(.5, p[1]),
                                            "Line": '0.5',
                                            "Over": str(prob_to_odds(p[0])),
                                            "Under": str(prob_to_odds(p[1]))
                                        }
                                        if player not in players:
                                            players[player] = {}

                                        players[player][market] = newline

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
                                player = games[k["eventId"]]
                            else:
                                continue

                            if player not in players:
                                players[player] = {}

                            try:
                                outcomes = sorted(
                                    k["outcomes"][:2], key=lambda x: x["label"]
                                )
                                line = outcomes[1]["line"]
                                p = no_vig_odds(
                                    outcomes[0]["oddsDecimal"],
                                    outcomes[1]["oddsDecimal"],
                                )
                                newline = {
                                    "EV": get_ev(line, p[1]),
                                    "Line": str(outcomes[1]["line"]),
                                    "Over": str(prob_to_odds(p[0])),
                                    "Under": str(prob_to_odds(p[1])),
                                }
                                players[player][market] = newline
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
    api_url = "https://sbapi.tn.sportsbook.fanduel.com/api/{}"
    params = [
        ("betexRegion", "GBR"),
        ("capiJurisdiction", "intl"),
        ("currencyCode", "USD"),
        ("exchangeLocale", "en_US"),
        ("includePrices", "true"),
        ("language", "en"),
        ("regionCode", "NAMERICA"),
        ("timezone", "America%2FChicago"),
        ("_ak", "FhMFpcPWXMeyZxOx"),
        ("page", "CUSTOM"),
        ("customPageId", sport),
    ]

    data = scraper.get(
        api_url.format("content-managed-page"),
        params={key: value for key, value in params},
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
    event_ids = [
        event['eventId']
        for event in events.values()
        if datetime.strptime(event["openDate"], "%Y-%m-%dT%H:%M:%S.%fZ")
        > datetime.now()
        and datetime.strptime(event["openDate"], "%Y-%m-%dT%H:%M:%S.%fZ")
        - timedelta(days=10)
        < datetime.today()
    ]

    players = {}

    # Iterate over event IDs and tabs
    for event_id in tqdm(event_ids, desc="Processing Events", unit="event"):
        for tab in tabs:
            new_params = [
                ("usePlayerPropsVirtualMarket", "true"),
                ("eventId", event_id),
                ("tab", tab),
            ]

            data = scraper.get(
                api_url.format("event-page"),
                params={key: value for key, value in params + new_params},
            )

            if not data:
                continue

            attachments = data.get("attachments")
            events = attachments.get("events")
            offers = attachments.get("markets")
            if not events or not offers:
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

            # Iterate over offers
            for offer in tqdm(offers, desc="Processing Offers", unit="offer"):

                if offer["marketName"] in ["Any Time Touchdown Scorer", "To Record A Hit", "To Hit A Home Run",
                                           "To Record a Run", "To Record an RBI", "To Hit a Single",
                                           "To Hit a Double", "Any Time Goal Scorer", "Player to Record 1+ Assists",
                                           "Player to Record 1+ Points"]:
                    for o in offer["runners"]:
                        player = remove_accents(o["runnerName"])
                        market = offer["marketName"]
                        if player not in players:
                            players[player] = {}

                        p = no_vig_odds(
                            o["winRunnerOdds"]["trueOdds"]["decimalOdds"]["decimalOdds"])

                        newline = {
                            "EV": get_ev(0.5, p[1]),
                            "Line": "0.5",
                            "Over": str(prob_to_odds(p[0])),
                            "Under": str(prob_to_odds(p[1])),
                        }
                        players[player][market] = newline

                if len(offer["runners"]) != 2:
                    continue

                try:
                    if " - " in offer["marketName"]:
                        player = remove_accents(
                            offer["marketName"].split(" - ")[0])
                        market = offer["marketName"].split(" - ")[1]
                    elif "@" in events[event_id]["name"]:
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

                    if player not in players:
                        players[player] = {}

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
                        "EV": get_ev(line, p[1]),
                        "Line": str(line),
                        "Over": str(prob_to_odds(p[0])),
                        "Under": str(prob_to_odds(p[1])),
                    }
                    players[player][market] = newline
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
    params = {
        "api_key": apikey,
        "url": f"https://guest.api.arcadia.pinnacle.com/0.1/leagues/{league}/markets/straight",
        "optimize_request": True,
        "keep_headers": True,
    }

    try:
        odds = requests.get(
            "https://proxy.scrapeops.io/v1/",
            headers=header | scraper.header,
            params=params,
        ).json()
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

    params[
        "url"
    ] = f"https://guest.api.arcadia.pinnacle.com/0.1/leagues/{league}/matchups"

    try:
        sleep(random.uniform(3, 5))
        api = requests.get(
            "https://proxy.scrapeops.io/v1/",
            headers=header | scraper.header,
            params=params,
        ).json()
        markets = [
            line
            for line in api
            if line.get("special", {"category": ""}).get("category") == "Player Props"
        ]
    except:
        logger.warning("No lines found for league: " + str(league))
        return {}

    players = {}
    for market in tqdm(markets):
        player = market["special"]["description"]
        player = player.replace(" (", ")").split(")")
        bet = player[1]
        player = remove_accents(player[0])
        if player not in players:
            players[player] = {}
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
                "EV": get_ev(line, p[1]),
                "Line": str(line),
                "Over": str(prob_to_odds(p[0])),
                "Under": str(prob_to_odds(p[1])),
            }
            players[player][bet] = newline
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
                    if player not in players:
                        players[player] = {}

                    line = lines[game["id"]]["s;3;ou;0.5"]["Under"]["Line"]
                    prices = [
                        lines[game["id"]]["s;3;ou;0.5"]["Over"]["Price"],
                        lines[game["id"]]["s;3;ou;0.5"]["Under"]["Price"],
                    ]
                    p = no_vig_odds(prices[0], prices[1])
                    newline = {
                        "EV": get_ev(line, p[1]),
                        "Line": str(line),
                        "Over": str(prob_to_odds(p[0])),
                        "Under": str(prob_to_odds(p[1])),
                    }
                    players[player][bet] = newline

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
        api = requests.get("https://proxy.scrapeops.io/v1/",
                           params=params).json()

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

    players = {}
    for id in tqdm(gameIds):
        caesars = f"https://api.americanwagering.com/regions/us/locations/mi/brands/czr/sb/v3/events/{id}?content-type=json"
        params["url"] = caesars

        # Sleep for a random interval to avoid overloading the API
        sleep(random.uniform(1, 5))

        try:
            # Make a GET request to the Caesars API for each game ID
            api = requests.get(
                "https://proxy.scrapeops.io/v1/", params=params).json()

            # Filter and retrieve the desired markets
            markets = [
                market
                for market in api["markets"]
                if market.get("active")
                and (
                    market.get("metadata", {}).get("marketType", {})
                    == "PLAYERLINEBASED"
                    or market.get("displayName") == "Run In 1st Inning?"
                )
            ]

        except:
            logger.exception(f"Unable to parse game {id}")
            continue

        for market in tqdm(markets):
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

            if player not in players:
                players[player] = {}

            # Calculate odds and expected value
            p = no_vig_odds(
                market["selections"][0]["price"]["d"],
                market["selections"][1]["price"]["d"],
            )
            newline = {
                "EV": get_ev(line, p[1]),
                "Line": line,
                "Over": str(prob_to_odds(p[0])),
                "Under": str(prob_to_odds(p[1])),
            }
            players[player][marketName] = newline

    return players


def get_pp():
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
    params = {
        "api_key": apikey,
        "url": f"https://api.prizepicks.com/leagues",
        "optimize_request": True,
    }
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
        leagues = requests.get(
            "https://proxy.scrapeops.io/v1/", params=params).json()
        leagues = [
            i["id"]
            for i in leagues["data"]
            if i["attributes"]["projections_count"] > 0
            and not any([string in i["attributes"]["name"] for string in live_bets])
        ]
    except:
        logger.exception("Retrieving leagues failed")
        leagues = [2, 7, 8, 9]

    for l in tqdm(leagues, desc="Processing PrizePicks offers"):
        params["url"] = f"https://api.prizepicks.com/projections?league_id={l}"

        try:
            # Retrieve players and lines for each league
            api = requests.get(
                "https://proxy.scrapeops.io/v1/", params=params).json()
            players = api["included"]
            lines = api["data"]
        except:
            logger.exception(f"League {l} failed")
            continue

        player_ids = {}
        league = None

        for p in players:
            if p["type"] == "new_player":
                player_ids[p["id"]] = {
                    "Name": p["attributes"]["name"].replace("\t", "").strip(),
                    "Team": p["attributes"]["team"].replace("JAC", "JAX").replace("WSH", "WAS").replace("LAV", "LV"),
                }
                if "position" in p["attributes"]:
                    player_ids[p["id"]].update(
                        {"Position": p["attributes"]["position"]})
            elif p["type"] == "league":
                league = p["attributes"]["name"].replace("CMB", "")

        for o in tqdm(lines, desc="Getting offers for " + league, unit="offer"):
            n = {
                "Player": remove_accents(
                    player_ids[o["relationships"]
                               ["new_player"]["data"]["id"]]["Name"]
                ),
                "League": league,
                "Team": player_ids[o["relationships"]["new_player"]["data"]["id"]]["Team"].upper(),
                "Date": o["attributes"]["start_time"].split("T")[0],
                "Market": o["attributes"]["stat_type"].replace(" (Combo)", ""),
                "Line": o["attributes"]["line_score"],
                "Opponent": o["attributes"]["description"].upper().replace("JAC", "JAX").replace("WSH", "WAS").replace("LAV", "LV"),
            }

            if o["attributes"]["is_promo"]:
                n["Line"] = o["attributes"]["flash_sale_line_score"]

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
        "https://api.underdogfantasy.com/beta/v3/over_under_lines")
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

    offers = {}
    for o in tqdm(
        api["over_under_lines"], desc="Getting Underdog Over/Unders", unit="offer"
    ):
        player = players[o["over_under"]["appearance_stat"]["appearance_id"]]
        game = matches.get(
            o["over_under"]["appearance_stat"]["appearance_id"],
            {"Home": "", "Away": ""},
        )
        opponent = game["Home"]
        if opponent == player["Team"]:
            opponent = game["Away"]
        market = o["over_under"]["appearance_stat"]["display_stat"]
        n = {
            "Player": remove_accents(player["Name"]),
            "League": player["League"],
            "Team": player["Team"].replace("WSH", "WAS").replace("NOP", "NO"),
            "Date": game["Date"],
            "Market": market,
            "Line": float(o["stat_value"]),
            "Boost": float(o["options"][0]["payout_multiplier"]),
            "Opponent": opponent.replace("WSH", "WAS").replace("NOP", "NO"),
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
            "Team": player1["Team"].replace("WSH", "WAS").replace("NOP", "NO") + "/" + player2["Team"].replace("WSH", "WAS").replace("NOP", "NO"),
            "Date": game1["Date"],
            "Market": "H2H " + bet,
            "Line": float(o["options"][0]["spread"]) - float(o["options"][1]["spread"]),
            "Opponent": opponent1.replace("WSH", "WAS").replace("NOP", "NO") + "/" + opponent2.replace("WSH", "WAS").replace("NOP", "NO"),
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
            "Team": team.replace("WSH", "WAS"),
            "Date": (
                datetime.strptime(
                    o["startTime"], "%Y/%m/%d %H:%M") - timedelta(hours=5)
            ).strftime("%Y-%m-%d"),
            "Market": " + ".join(o["player1"]["propParameters"]),
            "Line": float(o["propValue"]),
            "Opponent": opponent.replace("WSH", "WAS"),
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
    params = {
        "api_key": apikey,
        "url": "https://parlayplay.io/api/v1/crossgame/search/?format=json&league=&sport=All",
        "optimize_request": True,
        "keep_headers": True
    }
    header = {
        "Accept": "application/json",
        "X-CSRFToken": "FoEEn6o8fwxrKIrSzyphlphpVjBAEVZQANmhb2xeMmmRZwvbaDDZt5zGKoXNzrc2",
        "X-Parlay-Request": "1",
        "X-Parlayplay-Platform": "web",
    }
    try:
        api = requests.get(
            "https://proxy.scrapeops.io/v1/",
            params=params,
            headers=header,
        ).json()
    except:
        logger.exception(id)
        return {}

    if "players" not in api:
        logger.error("Error getting ParlayPlay Offers")
        return {}

    offers = {}
    for player in tqdm(api["players"], desc="Getting ParlayPlay Offers", unit="offer"):
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
            player_team = player_team.replace(
                "CHW", "CWS").replace("WSH", "WAS")
        if opponent_team is not None:
            opponent_team = opponent_team.replace(
                "CHW", "CWS").replace("WSH", "WAS")

        for stat in player["stats"]:
            market = stat["challengeName"]
            if "Fantasy" in market:
                if player["match"]["league"]["leagueNameShort"] == "MLB":
                    if player.get("position") == "SP":
                        market = "Pitcher Fantasy Points"
                    else:
                        market = "Hitter Fantasy Points"
            n = {
                "Player": remove_accents(player["player"]["fullName"]),
                "League": player["match"]["league"]["leagueNameShort"],
                "Team": player_team,
                "Date": player["match"]["matchDate"].split("T")[0],
                "Market": market,
                "Line": float(stat["statValue"]),
                "Slide": "N",
                "Opponent": opponent_team,
            }

            if n["League"] not in offers:
                offers[n["League"]] = {}
            if n["Market"] not in offers[n["League"]]:
                offers[n["League"]][n["Market"]] = []

            offers[n["League"]][n["Market"]].append(n)
            if stat["slideRange"]:
                m = n | {"Slide": "Y", "Line": np.min(stat["slideRange"])}
                offers[n["League"]][n["Market"]].append(m)
                m = n | {"Slide": "Y", "Line": np.max(stat["slideRange"])}
                offers[n["League"]][n["Market"]].append(m)

    for combo in tqdm(api["comboPackages"], desc="Getting ParlayPlay Combos", unit="offer"):
        teams = []
        opponents = []
        players = []
        for player in combo['packageLegs']:
            players.append(remove_accents(player["player"]["fullName"]))

            player_team = player["player"]["team"]["teamAbbreviation"]
            if player_team is not None:
                teams.append(player_team.replace(
                    "CHW", "CWS").replace("WSH", "WAS"))

            opponent_team = [team for team in
                             [player["match"]["homeTeam"]["teamAbbreviation"],
                              player["match"]["awayTeam"]["teamAbbreviation"]]
                             if team != player_team][0]
            if opponent_team is not None:
                opponents.append(opponent_team.replace(
                    "CHW", "CWS").replace("WSH", "WAS"))

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
            "Date": player["match"]["matchDate"].split("T")[0],
            "Market": market,
            "Line": float(combo["pickValue"]),
            "Slide": "N",
            "Opponent": "/".join(opponents),
        }

        if n["League"] not in offers:
            offers[n["League"]] = {}
        if n["Market"] not in offers[n["League"]]:
            offers[n["League"]][n["Market"]] = []

        offers[n["League"]][n["Market"]].append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers
