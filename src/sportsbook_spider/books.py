import datetime
import random
from tqdm import tqdm
import numpy as np
import statsapi as mlb
import logging
from time import sleep
from sportsbook_spider.helpers import apikey, requests, scraper, remove_accents, no_vig_odds, get_ev, prob_to_odds, mlb_pitchers

logger = logging.getLogger(__name__)

# Get DraftKings Odds


def get_dk(events, categories):
    players = {}
    markets = []
    games = {}
    for cat in tqdm(categories):
        dk_api = scraper.get(
            f"https://sportsbook.draftkings.com//sites/US-SB/api/v5/eventgroups/{events}/categories/{cat}?format=json")

        if not dk_api:
            continue

        if 'errorStatus' in dk_api:
            logger.warning(dk_api['errorStatus']['developerMessage'])
            continue

        for i in dk_api['eventGroup']['events']:
            games[i['eventId']] = i['name']

        for i in dk_api['eventGroup']['offerCategories']:
            if 'offerSubcategoryDescriptors' in i:
                markets = i['offerSubcategoryDescriptors']

        subcategoryIds = []  # Need subcategoryIds first
        for i in markets:
            subcategoryIds.append(i['subcategoryId'])

        for ids in subcategoryIds:
            dk_api = scraper.get(
                f"https://sportsbook.draftkings.com//sites/US-SB/api/v5/eventgroups/{events}/categories/{cat}/subcategories/{ids}?format=json")
            if not dk_api:
                # logger.info(str(len(players)) + " lines found")
                continue

            for i in dk_api['eventGroup']['offerCategories']:
                if 'offerSubcategoryDescriptors' in i:
                    markets = i['offerSubcategoryDescriptors']

            for i in tqdm(markets):
                if 'offerSubcategory' in i:
                    market = i['name']
                    for j in i['offerSubcategory']['offers']:
                        for k in j:
                            if 'line' not in k['outcomes'][0]:
                                continue
                            if 'participant' in k['outcomes'][0]:
                                player = k['outcomes'][0]['participant']
                            elif market == '1st Inning Total Runs':
                                for e in dk_api['eventGroup']['events']:
                                    if e['eventId'] == k['eventId']:
                                        player = e['eventMetadata']['participantMetadata'].get(
                                            'awayTeamStartingPitcherName', '') + " + " + e['eventMetadata']['participantMetadata'].get('homeTeamStartingPitcherName', '')
                            elif ': ' in k['label']:
                                player = k['label'][:k['label'].find(': ')]
                            elif k['eventId'] in games:
                                player = games[k['eventId']]
                            else:
                                continue

                            if player not in players:
                                players[player] = {}

                            try:
                                outcomes = sorted(
                                    k['outcomes'][:2], key=lambda x: x['label'])
                                line = outcomes[1]['line']
                                p = no_vig_odds(
                                    outcomes[0]['oddsDecimal'], outcomes[1]['oddsDecimal'])
                                newline = {
                                    "EV": get_ev(line, p[1]),
                                    "Line": str(outcomes[1]['line']),
                                    "Over": str(prob_to_odds(p[0])),
                                    "Under": str(prob_to_odds(p[1]))
                                }
                                players[player][market] = newline
                            except:
                                continue
    return players

# Get FanDuel Odds


def get_fd(sport, tabs):
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
        ("customPageId", sport)
    ]

    response = scraper.get(api_url.format(
        "content-managed-page"), params={key: value for key, value in params})
    if not response:
        logger.warning("No lines found")
        return {}

    attachments = response.get('attachments')
    events = attachments['events']
    event_ids = [event for event in events if datetime.datetime.strptime(events[event]['openDate'], "%Y-%m-%dT%H:%M:%S.%fZ") > datetime.datetime.today()
                 and datetime.datetime.strptime(events[event]['openDate'], "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.timedelta(days=5) < datetime.datetime.today()]

    players = {}

    for event_id in event_ids:
        for tab in tqdm(tabs):
            new_params = [
                ("usePlayerPropsVirtualMarket", "true"),
                ("eventId", event_id),
                ("tab", tab)
            ]

            response = scraper.get(api_url.format(
                "event-page"), params={key: value for key, value in params+new_params})

            if not response:
                logger.info(str(len(players)) + " lines found")
                return players

            attachments = response.get('attachments')
            events = attachments['events']
            offers = attachments['markets']

            offers = [offer for offer in offers.values(
            ) if offer['bettingType'] == 'MOVING_HANDICAP' or offer['bettingType'] == 'ODDS']
            offers = [offer for offer in offers if not any(substring in offer['marketName'] for substring in [
                                                           "Total Points", "Spread Betting", "Run Line", "Total Runs", "Puck Line", "Total Goals", "Qtr", "Moneyline", "Result", "Odd/Even", "To ", "Alt "])]

            for k in tqdm(offers):
                if len(k['runners']) != 2:
                    continue
                try:
                    if " - " in k['marketName']:
                        player = remove_accents(
                            k['marketName'].split(' - ')[0])
                        market = k['marketName'].split(' - ')[1]

                    elif "@" in events[event_id]['name']:
                        teams = events[event_id]['name'].replace(
                            " @ ", ")").replace(" (", ")").split(")")
                        pitcher1 = mlb_pitchers.get(mlb.lookup_team(
                            teams[0].strip())[0]['fileCode'].upper(), '')
                        pitcher2 = mlb_pitchers.get(mlb.lookup_team(
                            teams[3].strip())[0]['fileCode'].upper(), '')
                        player = pitcher1 + " + " + pitcher2
                        market = k['marketName']

                    else:
                        continue

                    if player not in players:
                        players[player] = {}

                    outcomes = sorted(k['runners'][:2],
                                      key=lambda x: x['runnerName'])
                    if outcomes[1]['handicap'] == 0:
                        line = [float(s) for s in market.split(
                            " ") if s.replace('.', '').isdigit()]
                        if line:
                            line = line[-1]
                        else:
                            continue
                    else:
                        line = outcomes[1]['handicap']

                    p = no_vig_odds(outcomes[0]['winRunnerOdds']['trueOdds']['decimalOdds']['decimalOdds'],
                                    outcomes[1]['winRunnerOdds']['trueOdds']['decimalOdds']['decimalOdds'])
                    newline = {
                        "EV": get_ev(line, p[1]),
                        "Line": str(line),
                        "Over": str(prob_to_odds(p[0])),
                        "Under": str(prob_to_odds(p[1]))
                    }
                    players[player][market] = newline
                except:
                    continue

    return players

# Get Pinnacle Odds


def get_pinnacle(league):
    header = {
        "X-API-KEY": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"
    }
    params = {
        'api_key': apikey,
        'url': f"https://guest.api.arcadia.pinnacle.com/0.1/leagues/{league}/markets/straight",
        'optimize_request': True,
        'keep_headers': True
    }

    try:
        odds = requests.get("https://proxy.scrapeops.io/v1/",
                            headers=header | scraper.header, params=params).json()
    except:
        logger.warning("No lines found for league: " + str(league))
        return {}

    if not type(odds) == list or not type(odds[0]) == dict:
        logger.warning("No lines found for league: " + str(league))
        return {}

    lines = {}
    for line in odds:
        if not line["prices"] or "points" not in line["prices"][0] or line["prices"][0]["points"] == 0:
            continue

        if line['matchupId'] not in lines:
            lines[line['matchupId']] = {}

        if line['key'] not in lines[line['matchupId']]:
            lines[line['matchupId']][line['key']] = {}

        for price in line['prices']:
            if price.get('participantId'):
                lines[line['matchupId']][line['key']][price['participantId']] = {
                    'Price': price['price'],
                    'Line': price['points']
                }
            elif price.get('designation'):
                lines[line['matchupId']][line['key']][price['designation'].capitalize()] = {
                    'Price': price['price'],
                    'Line': price['points']
                }

    params['url'] = f"https://guest.api.arcadia.pinnacle.com/0.1/leagues/{league}/matchups"

    try:
        sleep(random.uniform(3, 5))
        api = requests.get("https://proxy.scrapeops.io/v1/",
                           headers=header | scraper.header, params=params).json()
        markets = [line for line in api if line.get(
            'special', {'category': ''}).get('category') == 'Player Props']
    except:
        logger.warning("No lines found for league: " + str(league))
        return {}

    players = {}
    for market in tqdm(markets):
        player = market['special']['description']
        player = player.replace(' (', ')').split(')')
        bet = player[1]
        player = remove_accents(player[0])
        if player not in players:
            players[player] = {}
        try:
            outcomes = sorted(market['participants']
                              [:2], key=lambda x: x['name'])
            line = lines[market['id']]['s;0;ou'][outcomes[1]['id']]['Line']
            prices = [lines[market['id']]['s;0;ou'][participant['id']]
                      ['Price'] for participant in outcomes]
            p = no_vig_odds(prices[0], prices[1])
            newline = {
                "EV": get_ev(line, p[1]),
                "Line": str(line),
                "Over": str(prob_to_odds(p[0])),
                "Under": str(prob_to_odds(p[1]))
            }
            players[player][bet] = newline
        except:
            continue

    if league == 246:
        games = [line for line in api if 3 in [period['period']
                                               for period in line.get('periods')]]
        for game in tqdm(games):
            if lines.get(game['id'], {'s;3;ou;0.5': False}).get('s;3;ou;0.5'):
                try:
                    game_num = 1
                    pitchers = []
                    for team in game['participants']:
                        if team['name'][0] == "G":
                            game_num = int(team['name'][1])
                            team_name = team['name'][3:]
                        else:
                            team_name = team['name']

                        team_abbr = mlb.lookup_team(
                            team_name)[0]['fileCode'].upper()
                        if game_num > 1:
                            team_abbr = team_abbr + str(game_num)

                        pitchers.append(mlb_pitchers.get(team_abbr))

                    player = " + ".join(pitchers)
                    bet = "1st Inning Runs Allowed"
                    if player not in players:
                        players[player] = {}

                    line = lines[game['id']]['s;3;ou;0.5']['Under']['Line']
                    prices = [lines[game['id']]['s;3;ou;0.5']['Over']['Price'],
                              lines[game['id']]['s;3;ou;0.5']['Under']['Price']]
                    p = no_vig_odds(prices[0], prices[1])
                    newline = {
                        "EV": get_ev(line, p[1]),
                        "Line": str(line),
                        "Over": str(prob_to_odds(p[0])),
                        "Under": str(prob_to_odds(p[1]))
                    }
                    players[player][bet] = newline

                except:
                    continue

    return players

# Get Caesar's lines


def get_caesars(sport, league):
    caesars = f"https://api.americanwagering.com/regions/us/locations/mi/brands/czr/sb/v3/sports/{sport}/events/schedule/?competitionIds={league}&content-type=json"
    params = {
        'api_key': apikey,
        'url': caesars,
        'optimize_request': True
    }

    marketSwap = {
        'Points + Assists + Rebounds': 'Pts + Rebs + Asts',
        '3pt Field Goals': '3-PT Made',
        'Bases': 'Total Bases'
    }

    try:
        api = requests.get("https://proxy.scrapeops.io/v1/",
                           params=params).json()

        gameIds = [game['id'] for game in api['competitions'][0]
                   ['events'] if game['type'] == 'MATCH' and not game['started']
                   and game['marketCountActivePreMatch'] > 100]

    except Exception as exc:
        logger.exception("Caesars API request failed")
        return {}

    players = {}
    for id in tqdm(gameIds):
        caesars = f"https://api.americanwagering.com/regions/us/locations/mi/brands/czr/sb/v3/events/{id}?content-type=json"
        params['url'] = caesars
        sleep(random.uniform(5, 10))
        try:
            api = requests.get(
                "https://proxy.scrapeops.io/v1/", params=params).json()
            markets = [market for market in api['markets'] if market.get('active') and (market.get('metadata', {}).get(
                'marketType', {}) == 'PLAYERLINEBASED' or market.get('displayName') == 'Run In 1st Inning?')]

        except:
            logger.exception("Unable to parse game")
            continue

        for market in tqdm(markets):
            if market.get('displayName') == 'Run In 1st Inning?':
                marketName = "1st Inning Runs Allowed"
                line = 0.5
                player = " + ".join([mlb_pitchers.get(team['teamData'].get('teamAbbreviation', ''), '')
                                     for team in api['markets'][0]['selections']])
            else:
                player = remove_accents(market['metadata']['player'])
                marketName = market['displayName'].replace(
                    '|', '').replace("Total", "").replace("Player", "").replace("Batter", "").strip()
                marketName = marketSwap.get(marketName, marketName)
                if marketName == 'Props':
                    marketName = market.get('metadata', {}).get(
                        'marketCategoryName')
                line = market['line']

            if not player in players:
                players[player] = {}

            p = no_vig_odds(market['selections'][0]['price']
                            ['d'], market['selections'][1]['price']['d'])
            newline = {
                'EV': get_ev(line, p[1]),
                'Line': line,
                'Over': str(prob_to_odds(p[0])),
                'Under': str(prob_to_odds(p[1]))
            }
            players[player][marketName] = newline

    return players

# Get current PrizePicks lines


def get_pp():
    offers = []
    """
    leagues = [i['id'] for i in leagues['data']
               if i['attributes']['projections_count'] > 0]
   """
    leagues = [2, 7, 8]

    logger.info("Processing PrizePicks offers")
    for l in tqdm(leagues):
        params = {
            'api_key': '82ccbf28-ddd6-4e37-b3a1-0097b10fd412',
            'url': f"https://api.prizepicks.com/projections?league_id={l}",
            'optimize_request': True
        }

        try:
            api = requests.get("https://proxy.scrapeops.io/v1/",
                               params=params).json()
            players = api['included']
            lines = api['data']
        except:
            continue

        player_ids = {}
        for p in players:
            if p['type'] == 'new_player':
                player_ids[p['id']] = {
                    'Name': p['attributes']['name'].replace('\t', ''),
                    'Team': p['attributes']['team']
                }
            elif p['type'] == 'league':
                league = p['attributes']['name']

        logger.info("Getting offers for " + league)
        for o in tqdm(lines):
            n = {
                'Player': remove_accents(player_ids[o['relationships']['new_player']['data']['id']]['Name']),
                'League': league,
                'Team': player_ids[o['relationships']['new_player']['data']['id']]['Team'],
                'Market': o['attributes']['stat_type'],
                'Line': o['attributes']['line_score'],
                'Opponent': o['attributes']['description']
            }
            if o['attributes']['is_promo']:
                n['Line'] = o['attributes']['flash_sale_line_score']
            offers.append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers

# Get current Underdog lines:


def get_ud():
    teams = scraper.get(
        'https://stats.underdogfantasy.com/v1/teams')

    if not teams:
        logger.warning("Could not receive offer data")
        return []

    offers = []
    team_ids = {}
    for i in teams['teams']:
        team_ids[i['id']] = i['abbr']

    api = scraper.get(
        'https://api.underdogfantasy.com/beta/v3/over_under_lines')
    if not api:
        logger.info(str(len(offers)) + " offers found")
        return offers

    player_ids = {}
    for i in api['players']:
        player_ids[i['id']] = {
            'Name': str(i['first_name'] or "") + " " + str(i['last_name'] or ""),
            'League': i['sport_id']
        }

    match_ids = {}
    for i in api['games']:
        match_ids[i['id']] = {
            'Home': team_ids[i['home_team_id']],
            'Away': team_ids[i['away_team_id']],
            'League': i['sport_id']
        }

    players = {}
    matches = {}
    for i in api['appearances']:
        players[i['id']] = {
            'Name': player_ids.get(i['player_id'], {'Name': ""})['Name'],
            'Team': team_ids.get(i['team_id'], ""),
            'League': player_ids.get(i['player_id'], {'League': ""})['League']
        }
        matches[i['id']] = {
            'Home': match_ids.get(i['match_id'], {'Home': ''})['Home'],
            'Away': match_ids.get(i['match_id'], {'Away': ''})['Away']
        }

    offers = []
    logger.info("Getting Underdog Over/Unders")
    for o in tqdm(api['over_under_lines']):
        player = players[o['over_under']['appearance_stat']['appearance_id']]
        game = matches.get(o['over_under']['appearance_stat']
                           ['appearance_id'], {'Home': '', 'Away': ''})
        opponent = game['Home']
        if opponent == player['Team']:
            opponent = game['Away']
        n = {
            'Player': remove_accents(player['Name']),
            'League': player['League'],
            'Team': player['Team'],
            'Market': o['over_under']['appearance_stat']['display_stat'],
            'Line': float(o['stat_value']),
            'Opponent': opponent
        }
        offers.append(n)

    rivals = scraper.get(
        'https://api.underdogfantasy.com/beta/v3/rival_lines')

    if not rivals:
        logger.info(str(len(offers)) + " offers found")
        return offers

    for i in rivals['players']:
        if not i['id'] in player_ids:
            player_ids[i['id']] = {
                'Name': str(i['first_name'] or "") + " " + str(i['last_name'] or ""),
                'League': i['sport_id']
            }

    for i in rivals['games']:
        if not i['id'] in match_ids:
            match_ids[i['id']] = {
                'Home': team_ids[i['home_team_id']],
                'Away': team_ids[i['away_team_id']],
                'League': i['sport_id']
            }

    for i in rivals['appearances']:
        if not i['id'] in players:
            players[i['id']] = {
                'Name': player_ids.get(i['player_id'], {'Name': ""})['Name'],
                'Team': team_ids.get(i['team_id'], ""),
                'League': player_ids.get(i['player_id'], {'League': ""})['League']
            }

        if not i['id'] in matches:
            matches[i['id']] = {
                'Home': match_ids.get(i['match_id'], {'Home': ''})['Home'],
                'Away': match_ids.get(i['match_id'], {'Away': ''})['Away']
            }

    logger.info("Getting Underdog Rivals")
    for o in tqdm(rivals['rival_lines']):
        player1 = players[o['options'][0]['appearance_stat']['appearance_id']]
        player2 = players[o['options'][1]['appearance_stat']['appearance_id']]
        game1 = matches[o['options'][0]['appearance_stat']['appearance_id']]
        game2 = matches[o['options'][1]['appearance_stat']['appearance_id']]
        opponent1 = game1['Home']
        if opponent1 == player1['Team']:
            opponent1 = game1['Away']
        opponent2 = game2['Home']
        if opponent2 == player2['Team']:
            opponent2 = game2['Away']
        bet = o['options'][0]['appearance_stat']['display_stat']
        n = {
            'Player': player1['Name'] + " vs. " + player2['Name'],
            'League': player1['League'],
            'Team': player1['Team'] + "/" + player2['Team'],
            'Market': "H2H " + bet,
            'Line': float(o['options'][0]['spread'])-float(o['options'][1]['spread']),
            'Opponent': opponent1 + " / " + opponent2
        }
        offers.append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers

# Get Thrive Lines


def get_thrive():
    params = {
        'api_key': apikey,
        'url': "https://api.thrivefantasy.com/houseProp/upcomingHouseProps",
        'optimize_request': True,
        'keep_headers': True
    }
    payload = {"currentPage": 1, "currentSize": 100, "half": 0,
               "Latitude": "29.5908265", "Longitude": "-95.1381594"}
    header = {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*',
              'Token': 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0amplcm9tZSIsImF1ZGllbmNlIjoiSU9TIiwicGFzcyI6IiQyYSQxMCRMOGxMaTlUR2REVXZvdE9OTWhSaGxPbWZJc1ptaWRTdGFlZkxiU1ZZTkF4TVBQQTB1Q0ZQLiIsImNyZWF0ZWQiOjE2ODQwNzkwNzAyODgsImV4cCI6MTY4NDY4Mzg3MH0.GLlJuz0fdh0MsrF1IOQsAW47JflfoSSlSRgyo2bQPe-u6b7zZ7AJBLPeKJZCPKUPsrYWsjgyA3fIKgs2bOQtpA'}
    logger.info("Getting Thrive Lines")
    try:
        api = requests.post("https://proxy.scrapeops.io/v1/", params=params,
                            headers=header | scraper.header, json=payload).json()
    except Exception as exc:
        logger.exception(id)

    if api['success']:
        lines = api['response']['data']
    else:
        return []

    offers = []
    for line in tqdm(lines):
        o = line.get('contestProp')
        n = {
            'Player': remove_accents(" ".join([o['player1']['firstName'], o['player1']['lastName']])),
            'League': o['player1']['leagueType'],
            'Team': o['player1']['teamAbbr'],
            'Market': " + ".join(o['player1']['propParameters']),
            'Line': float(o['propValue']),
            'Opponent': o['team2Abbr']
        }
        if n['League'] == 'HOCKEY':
            n['League'] = 'NHL'

        offers.append(n)

    logger.info(str(len(offers)) + " offers found")
    return offers


def get_parp():
    params = {
        'api_key': '82ccbf28-ddd6-4e37-b3a1-0097b10fd412',
        'url': "https://parlayplay.io/api/v1/crossgame/search/?format=json&sport=All&league=",
        'optimize_request': True,
        'keep_headers': True
    }
    header = {'Accept': 'application/json', 'X-Parlay-Request': '1', 'X-Parlayplay-Platform': 'web',
              'X-Csrftoken': 'FoEEn6o8fwxrKIrSzyphlphpVjBAEVZQANmhb2xeMmmRZwvbaDDZt5zGKoXNzrc2'}
    try:
        api = requests.get("https://proxy.scrapeops.io/v1/",
                           params=params, headers=header | scraper.header).json()
    except Exception as exc:
        logger.exception(id)
        return []

    offers = []
    for player in tqdm(api['players']):
        teams = [player['match']['homeTeam']['teamAbbreviation'],
                 player['match']['awayTeam']['teamAbbreviation']]
        for stat in player['stats']:
            n = {
                'Player': remove_accents(player['player']['fullName']),
                'League': player['match']['league']['leagueNameShort'],
                'Team': player['player']['team']['teamAbbreviation'],
                'Market': stat['challengeName'],
                'Line': float(stat['statValue']),
                'Slide': 'N',
                'Opponent': [team for team in teams if team != player['player']['team']['teamAbbreviation']][0]
            }
            offers.append(n)
            if stat['slideRange']:
                m = n | {'Slide': 'Y', 'Line': np.min(stat['slideRange'])}
                offers.append(m)
                m = n | {'Slide': 'Y', 'Line': np.max(stat['slideRange'])}
                offers.append(m)

    return offers
