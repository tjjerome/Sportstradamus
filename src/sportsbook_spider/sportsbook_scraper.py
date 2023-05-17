# -*- coding: utf-8 -*-
"""Sportsbook Scraper

TODO:

*   Integrate Feedback
*   No line movement odds
*   Expand combo stats
*   1H, 2H, and Live Bets
*   Tennis/Golf/Racing
*   Add eSports (maybe from GGBET?)
*   Move to Google Apps Script
"""

import os.path
import pandas as pd
import numpy as np
import datetime
import traceback
import pickle
import random
from tqdm import tqdm
import statsapi as mlb
import nba_api.stats.endpoints as nba
from nba_api.stats.static import players as nba_static
import nfl_data_py as nfl
from time import sleep
from scipy.stats import poisson, skellam
import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from helpers import apikey, requests, scraper, remove_accents, odds_to_prob, get_ev, mlb_pitchers, get_loc


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
            print(dk_api['errorStatus']['developerMessage'])
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
                # print(str(len(players)) + " lines found")
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
                                over = 1 / outcomes[0]['oddsDecimal']
                                under = 1 / outcomes[1]['oddsDecimal']
                                newline = {
                                    "EV": get_ev(line, over, under),
                                    "Line": str(outcomes[1]['line']),
                                    "Over": str(outcomes[0]['oddsAmerican']),
                                    "Under": str(outcomes[1]['oddsAmerican'])
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
        print("No lines found")
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
                print(str(len(players)) + " lines found")
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

                    over = 1 / \
                        outcomes[0]['winRunnerOdds']['trueOdds']['decimalOdds']['decimalOdds']
                    under = 1 / \
                        outcomes[1]['winRunnerOdds']['trueOdds']['decimalOdds']['decimalOdds']
                    newline = {
                        "EV": get_ev(line, over, under),
                        "Line": str(line),
                        "Over": str(outcomes[0]['winRunnerOdds']['americanDisplayOdds']['americanOdds']),
                        "Under": str(outcomes[1]['winRunnerOdds']['americanDisplayOdds']['americanOdds'])
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
        print("No lines found for league: " + str(league))
        return {}

    if not type(odds) == list or not type(odds[0]) == dict:
        print("No lines found for league: " + str(league))
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
        print("No lines found for league: " + str(league))
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
            prices = [odds_to_prob(
                lines[market['id']]['s;0;ou'][participant['id']]['Price']) for participant in outcomes]
            newline = {
                "EV": get_ev(line, prices[0], prices[1]),
                "Line": str(line),
                "Over": str(lines[market['id']]['s;0;ou'][outcomes[0]['id']]['Price']),
                "Under": str(lines[market['id']]['s;0;ou'][outcomes[1]['id']]['Price'])
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
                    prices = [odds_to_prob(lines[game['id']]['s;3;ou;0.5']['Over']['Price']), odds_to_prob(
                        lines[game['id']]['s;3;ou;0.5']['Under']['Price'])]
                    newline = {
                        "EV": get_ev(line, prices[0], prices[1]),
                        "Line": str(line),
                        "Over": str(lines[game['id']]['s;3;ou;0.5']['Over']['Price']),
                        "Under": str(lines[game['id']]['s;3;ou;0.5']['Under']['Price'])
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
        print(exc)
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
            print("Unable to parse game")
            continue

        for market in tqdm(markets):
            if market.get('displayName') == 'Run In 1st Inning?':
                marketName = "1st Inning Runs Allowed"
                line = 0.5
                player = " + ".join([mlb_pitchers.get(team['teamData']['teamAbbreviation'], '')
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

            over = 1/market['selections'][0]['price']['d']
            under = 1/market['selections'][1]['price']['d']
            ev = get_ev(line, over, under)
            newline = {
                'EV': ev,
                'Line': line,
                'Over': market['selections'][0]['price']['a'],
                'Under': market['selections'][1]['price']['a'],
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

    print("Processing PrizePicks offers")
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

        print("Getting offers for " + league)
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

    print(str(len(offers)) + " offers found")
    return offers

# Get current Underdog lines:


def get_ud():
    teams = scraper.get(
        'https://stats.underdogfantasy.com/v1/teams')

    if not teams:
        print("Could not receive offer data")
        return []

    offers = []
    team_ids = {}
    for i in teams['teams']:
        team_ids[i['id']] = i['abbr']

    api = scraper.get(
        'https://api.underdogfantasy.com/beta/v3/over_under_lines')
    if not api:
        print(str(len(offers)) + " offers found")
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
    print("Getting Underdog Over/Unders")
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
        print(str(len(offers)) + " offers found")
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

    print("Getting Underdog Rivals")
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

    print(str(len(offers)) + " offers found")
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
    print("Getting Thrive Lines")
    try:
        api = requests.post("https://proxy.scrapeops.io/v1/", params=params,
                            headers=header | scraper.header, json=payload).json()
    except Exception as exc:
        print(id)
        print(exc)

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

    print(str(len(offers)) + " offers found")
    return offers


# Authorize the gspread API
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
]
creds = None
# The file token.json stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
if os.path.exists('./creds/token.json'):
    creds = Credentials.from_authorized_user_file('./creds/token.json', SCOPES)
# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            './creds/credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('./creds/token.json', 'w') as token:
        token.write(creds.to_json())
gc = gspread.authorize(creds)


""""
Start gathering sportsbook data
"""
print("Getting DraftKings MLB lines")
dk_data = get_dk(84240, [743, 1024, 1031])  # MLB
print("Getting DraftKings NBA lines")
dk_data.update(get_dk(42648, [583, 1215, 1216, 1217, 1218, 1219, 1220]))  # NBA
print("Getting DraftKings NHL lines")
dk_data.update(get_dk(42133, [550, 1064, 1189]))  # NHL
# dk_data.update(get_dk(92893, [488, 633])) # Tennis
# dk_data.update(get_dk(91581, [488, 633])) # Tennis
print(str(len(dk_data)) + " offers found")

print("Getting FanDuel MLB lines")
fd_data = get_fd('mlb', ['batter-props', 'pitcher-props', 'innings'])
print("Getting FanDuel NBA lines")
fd_data.update(get_fd('nba', ['player-points', 'player-rebounds',
                              'player-assists', 'player-threes', 'player-combos', 'player-defense']))
print("Getting FanDuel NHL lines")
fd_data.update(
    get_fd('nhl', ['goal-scorer', 'shots', 'points-assists', 'goalie-props']))
print(str(len(fd_data)) + " offers found")

print("Getting Pinnacle MLB lines")
pin_data = get_pinnacle(246)  # MLB
print("Getting Pinnacle NBA lines")
pin_data.update(get_pinnacle(487))  # NBA
print("Getting Pinnacle NHL lines")
pin_data.update(get_pinnacle(1456))  # NHL
print(str(len(pin_data)) + " offers found")

csb_data = {}
print("Getting Caesars MLB Lines")
sport = "baseball"
league = "04f90892-3afa-4e84-acce-5b89f151063d"
csb_data.update(get_caesars(sport, league))
print("Getting Caesars NBA Lines")
sport = "basketball"
league = "5806c896-4eec-4de1-874f-afed93114b8c"  # NBA
csb_data.update(get_caesars(sport, league))
print("Getting Caesars NHL Lines")
sport = "icehockey"
league = "b7b715a9-c7e8-4c47-af0a-77385b525e09"
csb_data.update(get_caesars(sport, league))
print("Getting Caesars NFL Lines")
# sport = "americanfootball"
# league = "007d7c61-07a7-4e18-bb40-15104b6eac92"
# csb_data.update(get_caesars(sport, league))
print(str(len(csb_data)) + " offers found")

"""
Start gathering player stats
"""
print("Getting NBA stats")
nba_gamelog = nba.playergamelogs.PlayerGameLogs(
    season_nullable='2022-23').get_normalized_dict()['PlayerGameLogs']
nba_playoffs = nba.playergamelogs.PlayerGameLogs(
    season_nullable='2022-23', season_type_nullable='Playoffs').get_normalized_dict()['PlayerGameLogs']
nba_gamelog = nba_playoffs + nba_gamelog
with open("./data/nba_players.dat", "rb") as infile:
    nba_players = pickle.load(infile)
    nba_players = {int(k): v for k, v in nba_players.items()}

for game in tqdm(nba_gamelog):
    if game['PLAYER_ID'] not in nba_players:
        nba_players[game['PLAYER_ID']] = nba.commonplayerinfo.CommonPlayerInfo(
            player_id=game['PLAYER_ID']).get_normalized_dict()['CommonPlayerInfo'][0]
        sleep(0.5)

    game['POS'] = nba_players[game['PLAYER_ID']].get('POSITION')
    teams = game['MATCHUP'].replace("vs.", "@").split(" @ ")
    for team in teams:
        if team != game['TEAM_ABBREVIATION']:
            game['OPP'] = team

    game["PRA"] = game['PTS'] + game['REB'] + game['AST']
    game["PR"] = game['PTS'] + game['REB']
    game["PA"] = game['PTS'] + game['AST']
    game["RA"] = game['REB'] + game['AST']
    game["BLST"] = game['BLK'] + game['STL']

with open("./data/nba_players.dat", "wb") as outfile:
    pickle.dump(nba_players, outfile)


def get_nba_stats(player, opponent, market, line):
    if " + " in player:
        players = player.split(" + ")
        try:
            player_ids = [nba_static.find_players_by_full_name(
                player)[0]['id'] for player in players]
        except:
            return np.ones(5)*-1000

        player1_games = [
            game for game in nba_gamelog if game['PLAYER_ID'] == player_ids[0]]
        player2_games = [
            game for game in nba_gamelog if game['PLAYER_ID'] == player_ids[1]]

        if not player1_games or not player2_games:
            return np.ones(5) * -1000
        last10avg = (np.mean([game[market] for game in player1_games][:10]) +
                     np.mean([game[market] for game in player2_games][:10]) - line)/line
        last5 = -1000
        seasontodate = -1000
        headtohead = -1000
        dvpoa = -1000
    elif " vs. " in player:
        players = player.split(" vs. ")
        try:
            player_ids = [nba_static.find_players_by_full_name(
                player)[0]['id'] for player in players]
        except:
            return np.ones(5)*-1000

        player1_games = [
            game for game in nba_gamelog if game['PLAYER_ID'] == player_ids[0]]
        player2_games = [
            game for game in nba_gamelog if game['PLAYER_ID'] == player_ids[1]]

        if not player1_games or not player2_games:
            return np.ones(5) * -1000
        last10avg = (np.mean([game[market] for game in player1_games][:10]) + line -
                     np.mean([game[market] for game in player2_games][:10]))
        last5 = -1000
        seasontodate = -1000
        headtohead = -1000
        dvpoa = -1000
    else:
        player_id = nba_static.find_players_by_full_name(player)
        if player_id:
            player_id = player_id[0]['id']
            positions = nba_players[player_id]['POSITION'].split('-')
        else:
            return np.ones(5)*-1000

        player_games = [
            game for game in nba_gamelog if game['PLAYER_ID'] == player_id]

        if not player_games:
            return np.ones(5) * -1000
        last10avg = (np.mean([game[market]
                              for game in player_games][:10]) - line)/line
        made_line = [int(game[market] > line) for game in player_games]
        last5 = np.mean(made_line[:5])
        seasontodate = np.mean(made_line)
        headtohead = [int(game[market] > line)
                      for game in player_games if game['OPP'] == opponent]
        if np.isnan(last5):
            last5 = -1000

        if np.isnan(seasontodate):
            seasontodate = -1000

        if not headtohead:
            headtohead = -1000
        else:
            headtohead = np.mean(headtohead+[1, 0])

        dvp = {}
        leagueavg = {}
        for position in positions:
            for game in nba_gamelog:
                if position in game['POS']:
                    if game['GAME_ID']+position not in leagueavg:
                        leagueavg[game['GAME_ID']+position] = 0

                    leagueavg[game['GAME_ID'] +
                              position] = leagueavg[game['GAME_ID']+position] + game[market]

                    if game['OPP'] == opponent:
                        if game['GAME_ID']+position not in dvp:
                            dvp[game['GAME_ID']+position] = 0

                        dvp[game['GAME_ID']+position] = dvp[game['GAME_ID'] +
                                                            position] + game[market]

        if not dvp:
            dvpoa = -1000
        else:
            dvp = np.mean(list(dvp.values()))
            leagueavg = np.mean(list(leagueavg.values()))/2
            dvpoa = (dvp-leagueavg)/leagueavg

    return np.array([last10avg, last5, seasontodate, headtohead, dvpoa])


def mlb_parse_game(gameId):
    mlb_gamelog = []
    game = mlb.boxscore_data(gameId)
    linescore = mlb.get('game_linescore', {'gamePk': str(gameId)})
    awayTeam = game['teamInfo']['away']['abbreviation']
    homeTeam = game['teamInfo']['home']['abbreviation']
    awayPitcher = game['awayPitchers'][1]['personId']
    awayPitcher = game['away']['players']['ID' +
                                          str(awayPitcher)]['person']['fullName']
    homePitcher = game['homePitchers'][1]['personId']
    homePitcher = game['home']['players']['ID' +
                                          str(homePitcher)]['person']['fullName']
    awayInning1Runs = linescore['innings'][0]['away']['runs']
    homeInning1Runs = linescore['innings'][0]['home']['runs']
    for v in game['away']['players'].values():
        if v['person']['id'] == game['awayPitchers'][1]['personId'] or v['person']['id'] in game['away']['batters']:
            n = {
                'gameId': game['gameId'],
                'playerId': v['person']['id'],
                'playerName': v['person']['fullName'],
                'position': v.get('position', {'abbreviation': ''})['abbreviation'],
                'team': awayTeam,
                'opponent': homeTeam,
                'opponent pitcher': homePitcher,
                'starting pitcher': v['person']['id'] == game['awayPitchers'][1]['personId'],
                'starting batter': v['person']['id'] in game['away']['batters'],
                'hits': v['stats']['batting'].get('hits', 0),
                'total bases': v['stats']['batting'].get('hits', 0) + v['stats']['batting'].get('doubles', 0) + 2*v['stats']['batting'].get('triples', 0) + 3*v['stats']['batting'].get('homeRuns', 0),
                'singles': v['stats']['batting'].get('hits', 0)-v['stats']['batting'].get('doubles', 0)-v['stats']['batting'].get('triples', 0)-v['stats']['batting'].get('homeRuns', 0),
                'batter strikeouts': v['stats']['batting'].get('strikeOuts', 0),
                'runs': v['stats']['batting'].get('runs', 0),
                'rbi': v['stats']['batting'].get('rbi', 0),
                'hits+runs+rbi': v['stats']['batting'].get('hits', 0) + v['stats']['batting'].get('runs', 0) + v['stats']['batting'].get('rbi', 0),
                'walks': v['stats']['batting'].get('baseOnBalls', 0),
                'pitcher strikeouts': v['stats']['pitching'].get('strikeOuts', 0),
                'walks allowed': v['stats']['pitching'].get('baseOnBalls', 0),
                'runs allowed': v['stats']['pitching'].get('runs', 0),
                'hits allowed': v['stats']['pitching'].get('hits', 0),
                'pitching outs': 3*int(v['stats']['pitching'].get('inningsPitched', '0.0').split('.')[0]) + int(v['stats']['pitching'].get('inningsPitched', '0.0').split('.')[1]),
                '1st inning runs allowed': homeInning1Runs if v['person']['id'] == game['awayPitchers'][1]['personId'] else 0
            }
            mlb_gamelog.append(n)

    for v in game['home']['players'].values():
        if v['person']['id'] == game['homePitchers'][1]['personId'] or v['person']['id'] in game['home']['batters']:
            n = {
                'gameId': game['gameId'],
                'playerId': v['person']['id'],
                'playerName': v['person']['fullName'],
                'position': v.get('position', {'abbreviation': ''})['abbreviation'],
                'team': homeTeam,
                'opponent': awayTeam,
                'opponent pitcher': awayPitcher,
                'starting pitcher': v['person']['id'] == game['homePitchers'][1]['personId'],
                'starting batter': v['person']['id'] in game['home']['batters'],
                'hits': v['stats']['batting'].get('hits', 0),
                'total bases': v['stats']['batting'].get('hits', 0) + v['stats']['batting'].get('doubles', 0) + 2*v['stats']['batting'].get('triples', 0) + 3*v['stats']['batting'].get('homeRuns', 0),
                'singles': v['stats']['batting'].get('hits', 0)-v['stats']['batting'].get('doubles', 0)-v['stats']['batting'].get('triples', 0)-v['stats']['batting'].get('homeRuns', 0),
                'batter strikeouts': v['stats']['batting'].get('strikeOuts', 0),
                'runs': v['stats']['batting'].get('runs', 0),
                'rbi': v['stats']['batting'].get('rbi', 0),
                'hits+runs+rbi': v['stats']['batting'].get('hits', 0) + v['stats']['batting'].get('runs', 0) + v['stats']['batting'].get('rbi', 0),
                'walks': v['stats']['batting'].get('baseOnBalls', 0),
                'pitcher strikeouts': v['stats']['pitching'].get('strikeOuts', 0),
                'walks allowed': v['stats']['pitching'].get('baseOnBalls', 0),
                'runs allowed': v['stats']['pitching'].get('runs', 0),
                'hits allowed': v['stats']['pitching'].get('hits', 0),
                'pitching outs': 3*int(v['stats']['pitching'].get('inningsPitched', '0.0').split('.')[0]) + int(v['stats']['pitching'].get('inningsPitched', '0.0').split('.')[1]),
                '1st inning runs allowed': awayInning1Runs if v['person']['id'] == game['homePitchers'][1]['personId'] else 0
            }
            mlb_gamelog.append(n)

    return mlb_gamelog


print("Getting MLB Stats")
mlb_game_ids = mlb.schedule(start_date=mlb.latest_season(
)['regularSeasonStartDate'], end_date=datetime.date.today())
mlb_game_ids = [game['game_id'] for game in mlb_game_ids if game['status']
                == 'Final' and game['game_type'] != 'E' and game['game_type'] != 'S']

with open("./data/mlb_data.dat", "rb") as infile:
    mlb_data = pickle.load(infile)

for id in tqdm(mlb_game_ids):
    if id not in mlb_data['games']:
        mlb_data['games'].append(id)
        mlb_data['gamelog'] = mlb_data['gamelog'] + mlb_parse_game(id)

for game in mlb_data['gamelog']:
    if datetime.datetime.strptime(game['gameId'][:10], '%Y/%m/%d') < datetime.datetime.today() - datetime.timedelta(days=365):
        mlb_data['gamelog'].remove(game)

with open("./data/mlb_data.dat", "wb") as outfile:
    pickle.dump(mlb_data, outfile)


def get_mlb_stats(player, opponent, market, line):
    opponent = opponent.replace('ARI', 'AZ')

    if mlb_data['gamelog'][0].get(market) is None:
        return np.ones(5) * -1000

    if " + " in player:
        players = player.split(" + ")

        if any([string in market for string in ['allowed', 'pitch']]):
            player1_games = [game for game in mlb_data['gamelog']
                             if game['playerName'] == players[0] and game['starting pitcher']]
            player2_games = [game for game in mlb_data['gamelog']
                             if game['playerName'] == players[1] and game['starting pitcher']]
        else:
            player1_games = [game for game in mlb_data['gamelog']
                             if game['playerName'] == players[0] and game['starting batter']]
            player2_games = [game for game in mlb_data['gamelog']
                             if game['playerName'] == players[1] and game['starting batter']]

        if not player1_games or not player2_games:
            return np.ones(5) * -1000
        last10avg = (np.mean([game[market] for game in player1_games][-10:]) +
                     np.mean([game[market] for game in player2_games][-10:]) - line)/line
        last5 = -1000
        seasontodate = -1000
        headtohead = -1000
        dvpoa = -1000
    elif " vs. " in player:
        players = player.split(" vs. ")

        if any([string in market for string in ['allowed', 'pitch']]):
            player1_games = [game for game in mlb_data['gamelog']
                             if game['playerName'] == players[0] and game['starting pitcher']]
            player2_games = [game for game in mlb_data['gamelog']
                             if game['playerName'] == players[1] and game['starting pitcher']]
        else:
            player1_games = [game for game in mlb_data['gamelog']
                             if game['playerName'] == players[0] and game['starting batter']]
            player2_games = [game for game in mlb_data['gamelog']
                             if game['playerName'] == players[1] and game['starting batter']]

        if not player1_games or not player2_games:
            return np.ones(5) * -1000
        last10avg = (np.mean([game[market] for game in player2_games][-10:]) + line -
                     np.mean([game[market] for game in player1_games][-10:]))
        last5 = -1000
        seasontodate = -1000
        headtohead = -1000
        dvpoa = -1000
    else:
        season_start = datetime.datetime.strptime(
            mlb.latest_season()['regularSeasonStartDate'], "%Y-%m-%d")
        pitcher = mlb_pitchers.get(opponent, '')
        if any([string in market for string in ['allowed', 'pitch']]):
            player_games = [game for game in mlb_data['gamelog']
                            if game['playerName'] == player and game['starting pitcher']]
        else:
            player_games = [game for game in mlb_data['gamelog']
                            if game['playerName'] == player and game['starting batter']]
        if not player_games:
            return np.ones(5) * -1000
        last10avg = (np.mean([game[market]
                              for game in player_games][-10:])-line)/line
        made_line = [int(game[market] > line) for game in player_games if datetime.datetime.strptime(
            game['gameId'][:10], "%Y/%m/%d") >= season_start]
        last5 = np.mean(made_line[-5:])
        seasontodate = np.mean(made_line)
        if np.isnan(last5):
            last5 = -1000

        if np.isnan(seasontodate):
            seasontodate = -1000

        if any([string in market for string in ['allowed', 'pitch']]):
            headtohead = [int(game[market] > line)
                          for game in player_games if game['opponent'] == opponent]
            dvp = np.mean([game[market] for game in mlb_data['gamelog']
                           if game['starting pitcher'] and game['opponent'] == opponent])
            leagueavg = np.mean(
                [game[market] for game in mlb_data['gamelog'] if game['starting pitcher']])

        else:
            headtohead = [int(game[market] > line)
                          for game in player_games if game['opponent pitcher'] == pitcher]
            dvp = np.mean([game[market] for game in mlb_data['gamelog']
                           if game['starting batter'] and game['opponent pitcher'] == pitcher])
            leagueavg = np.mean(
                [game[market] for game in mlb_data['gamelog'] if game['starting batter']])

        if not headtohead:
            headtohead = -1000
        else:
            headtohead = np.mean(headtohead+[1, 0])
        if not dvp:
            dvpoa = -1000
        else:
            dvpoa = (dvp-leagueavg)/leagueavg

    return np.array([last10avg, last5, seasontodate, headtohead, dvpoa])


print("Getting NHL data")
with open("./data/nhl_skater_data.dat", "rb") as infile:
    nhl_skater_data = pickle.load(infile)

params = {
    'isAggregate': 'false',
    'isGame': 'true',
    'sort': '[{"property":"gameDate","direction":"ASC"},{"property":"playerId","direction":"ASC"}]',
    'start': 0,
    'limit': 100,
    'factCayenneExp': 'gamesPlayed>=1',
    'cayenneExp': 'gameTypeId>=2 and gameDate>=' + '"'+nhl_skater_data[-1]['gameDate']+'"'
}
nhl_request_data = scraper.get(
    'https://api.nhle.com/stats/rest/en/skater/summary', params=params)['data']
if nhl_request_data:
    for x in nhl_skater_data[-300:]:
        if x['gameDate'] == nhl_skater_data[-1]['gameDate']:
            nhl_skater_data.remove(x)
nhl_skater_data = nhl_skater_data + [{k: v for k, v in skater.items() if k in ['assists', 'gameDate', 'gameId', 'goals', 'opponentTeamAbbrev',
                                                                               'playerId', 'points', 'positionCode', 'shots', 'skaterFullName', 'teamAbbrev']} for skater in nhl_request_data]

while nhl_request_data:
    params['start'] += 100
    nhl_request_data = scraper.get(
        'https://api.nhle.com/stats/rest/en/skater/summary', params=params)['data']
    if not nhl_request_data and params['start'] == 10000:
        params['start'] = 0
        params['cayenneExp'] = params['cayenneExp'][:-12] + \
            '"'+nhl_skater_data[-1]['gameDate']+'"'
        nhl_request_data = scraper.get(
            'https://api.nhle.com/stats/rest/en/skater/summary', params=params)['data']
        for x in nhl_skater_data[-300:]:
            if x['gameDate'] == nhl_skater_data[-1]['gameDate']:
                nhl_skater_data.remove(x)
    nhl_skater_data = nhl_skater_data + [{k: v for k, v in skater.items() if k in ['assists', 'gameDate', 'gameId', 'goals', 'opponentTeamAbbrev',
                                                                                   'playerId', 'points', 'positionCode', 'shots', 'skaterFullName', 'teamAbbrev']} for skater in nhl_request_data]

for game in nhl_skater_data:
    if datetime.datetime.strptime(game['gameDate'], '%Y-%m-%d') < datetime.datetime.today() - datetime.timedelta(days=365):
        nhl_skater_data.remove(game)

with open("./data/nhl_skater_data.dat", "wb") as outfile:
    pickle.dump(nhl_skater_data, outfile)
print('Skater data complete')
with open("./data/nhl_goalie_data.dat", "rb") as infile:
    nhl_goalie_data = pickle.load(infile)

params = {
    'isAggregate': 'false',
    'isGame': 'true',
    'sort': '[{"property":"gameDate","direction":"ASC"},{"property":"playerId","direction":"ASC"}]',
    'start': 0,
    'limit': 100,
    'factCayenneExp': 'gamesPlayed>=1',
    'cayenneExp': 'gameTypeId>=2 and gameDate>=' + '"'+nhl_goalie_data[-1]['gameDate']+'"'
}
nhl_request_data = scraper.get(
    'https://api.nhle.com/stats/rest/en/goalie/summary', params=params)['data']
if nhl_request_data:
    for x in nhl_goalie_data[-50:]:
        if x['gameDate'] == nhl_goalie_data[-1]['gameDate']:
            nhl_goalie_data.remove(x)
nhl_goalie_data = nhl_goalie_data + [{k: v for k, v in goalie.items() if k in ['gameDate', 'gameId', 'goalsAgainst',
                                                                               'opponentTeamAbbrev', 'playerId', 'positionCode', 'saves', 'goalieFullName', 'teamAbbrev']} for goalie in nhl_request_data]

while nhl_request_data:
    params['start'] += 100
    nhl_request_data = scraper.get(
        'https://api.nhle.com/stats/rest/en/goalie/summary', params=params)['data']
    if not nhl_request_data and params['start'] == 10000:
        params['start'] = 0
        params['cayenneExp'] = params['cayenneExp'][:-12] + \
            '"'+nhl_goalie_data[-1]['gameDate']+'"'
        nhl_request_data = scraper.get(
            'https://api.nhle.com/stats/rest/en/goalie/summary', params=params)['data']
        if nhl_request_data:
            for x in nhl_goalie_data[-50:]:
                if x['gameDate'] == nhl_goalie_data[-1]['gameDate']:
                    nhl_goalie_data.remove(x)
    nhl_goalie_data = nhl_goalie_data + [{k: v for k, v in goalie.items() if k in ['gameDate', 'gameId', 'goalsAgainst',
                                                                                   'opponentTeamAbbrev', 'playerId', 'positionCode', 'saves', 'goalieFullName', 'teamAbbrev']} for goalie in nhl_request_data]
for game in nhl_goalie_data:
    if datetime.datetime.strptime(game['gameDate'], '%Y-%m-%d') < datetime.datetime.today() - datetime.timedelta(days=365):
        nhl_goalie_data.remove(game)

with open("./data/nhl_goalie_data.dat", "wb") as outfile:
    pickle.dump(nhl_goalie_data, outfile)
print('Goalie data complete')


def get_nhl_stats(player, opponent, market, line):

    opponent = opponent.replace('NJ', 'NJD').replace(
        'TB', 'TBL').replace('LA', 'LAK')

    if market == 'PTS':
        market = 'points'
    elif market == 'AST':
        market = 'assists'
    elif market == 'BLK':
        return np.ones(5) * -1000

    if " + " in player:
        players = player.split(" + ")

        if market in ['goalsAgainst', 'saves']:
            player1_games = [
                game for game in nhl_goalie_data if players[0] == game['goalieFullName']]
            player2_games = [
                game for game in nhl_goalie_data if players[1] == game['goalieFullName']]
        else:
            player1_games = [
                game for game in nhl_skater_data if players[0] == game['skaterFullName']]
            player2_games = [
                game for game in nhl_skater_data if players[1] == game['skaterFullName']]

        if not player1_games or not player2_games:
            return np.ones(5) * -1000

        last10avg = (np.mean([game[market] for game in player1_games][-10:]) +
                     np.mean([game[market] for game in player2_games][-10:]) - line)/line
        last5 = -1000
        seasontodate = -1000
        headtohead = -1000
        dvpoa = -1000
    elif " vs. " in player:
        players = player.split(" vs. ")

        if market in ['goalsAgainst', 'saves']:
            player1_games = [
                game for game in nhl_goalie_data if players[0] == game['goalieFullName']]
            player2_games = [
                game for game in nhl_goalie_data if players[1] == game['goalieFullName']]
        else:
            player1_games = [
                game for game in nhl_skater_data if players[0] == game['skaterFullName']]
            player2_games = [
                game for game in nhl_skater_data if players[1] == game['skaterFullName']]

        if not player1_games or not player2_games:
            return np.ones(5) * -1000

        last10avg = (np.mean([game[market] for game in player2_games][-10:]) + line -
                     np.mean([game[market] for game in player1_games][-10:]))
        last5 = -1000
        seasontodate = -1000
        headtohead = -1000
        dvpoa = -1000
    else:
        if market in ['goalsAgainst', 'saves']:
            player_games = [
                game for game in nhl_goalie_data if player == game['goalieFullName']]
        else:
            player_games = [
                game for game in nhl_skater_data if player == game['skaterFullName']]

        if not player_games:
            return np.ones(5) * -1000

        last10avg = (np.mean([game[market]
                              for game in player_games][-10:])-line)/line
        made_line = [int(game[market] > line) for game in player_games if datetime.datetime.strptime(
            game['gameDate'], "%Y-%m-%d") >= datetime.datetime.strptime("2022-10-07", "%Y-%m-%d")]
        headtohead = [int(game[market] > line)
                      for game in player_games if opponent == game['opponentTeamAbbrev']]
        last5 = np.mean(made_line[-5:])
        seasontodate = np.mean(made_line)
        if np.isnan(last5):
            last5 = -1000

        if np.isnan(seasontodate):
            seasontodate = -1000

        if not headtohead:
            headtohead = -1000
        else:
            headtohead = np.mean(headtohead+[1, 0])

        dvp = {}
        leagueavg = {}
        if market in ['goalsAgainst', 'saves']:
            for game in nhl_goalie_data:
                id = game['gameId']
                if id not in leagueavg:
                    leagueavg[id] = 0
                leagueavg[id] += game[market]
                if opponent == game['opponentTeamAbbrev']:
                    if id not in dvp:
                        dvp[id] = 0
                    dvp[id] += game[market]

        else:
            position = player_games[0]['positionCode']
            for game in nhl_skater_data:
                if game['positionCode'] == position:
                    id = game['gameId']
                    if id not in leagueavg:
                        leagueavg[id] = 0
                    leagueavg[id] += game[market]
                    if opponent == game['opponentTeamAbbrev']:
                        if id not in dvp:
                            dvp[id] = 0
                        dvp[id] += game[market]

        if not dvp:
            dvpoa = -1000
        else:
            dvp = np.mean(list(dvp.values()))
            leagueavg = np.mean(list(leagueavg.values()))/2
            dvpoa = (dvp-leagueavg)/leagueavg
    return np.array([last10avg, last5, seasontodate, headtohead, dvpoa])


untapped_markets = []
tapped_markets = []

pp_offers = get_pp()

pp2dk = {
    'Runs': 'Runs Scored',
    'Pitcher Strikeouts': 'Strikeouts',
    'Walks Allowed': 'Walks',
    '1st Inning Runs Allowed': '1st Inning Total Runs',
    'Hitter Strikeouts': 'Strikeouts',
    'Hits+Runs+RBIS': 'Hits + Runs + RBIs',
    'Earned Runs Allowed': 'Earned Runs',
    'Blks+Stls': 'Steals + Blocks',
    'Blocked Shots': 'Blocks',
    'Pts+Asts': 'Pts + Ast',
    'Pts+Rebs': 'Pts + Reb',
    'Pts+Rebs+Asts': 'Pts + Reb + Ast',
    'Rebs+Asts': 'Ast + Reb',
    '3-PT Made': 'Threes',
    'Goalie Saves': 'Saves',
    'Shots On Goal': 'Player Shots on Goal'
}

pp2fd = {
    'Blocked Shots': 'Blocks',
    'Pts+Asts': 'Pts + Ast',
    'Pts+Rebs': 'Pts + Reb',
    'Pts+Rebs+Asts': 'Pts + Reb + Ast',
    'Rebs+Asts': 'Reb + Ast',
    '3-PT Made': 'Made Threes',
    'Pitcher Strikeouts': 'Strikeouts',
    '1st Inning Runs Allowed': '1st Inning Over/Under 0.5 Runs',
    'Goalie Saves': 'Saves',
    'Shots On Goal': 'Shots on Goal'
}

pp2pin = {
    'Pitcher Strikeouts': 'Total Strikeouts',
    '3-PT Made': '3 Point FG',
    'Goalie Saves': 'Saves'
}

pp2csb = {
    'Shots On Goal': 'Shots',
    'Pts+Rebs+Asts': 'Pts + Rebs + Asts',
    'Hitter Strikeouts': 'Strikeouts',
    'Pitcher Strikeouts': 'Pitching Strikeouts',
    'Blks+Stls': 'Blocks + Steals',
    'Pts+Asts': 'Points + Assists',
    'Pts+Rebs': 'Points + Rebounds',
    'Rebs+Asts': 'Rebounds + Assists'
}

pp2stats = {
    '3-PT Made': 'FG3M',
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Pts+Rebs': 'PR',
    'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA',
    'Pts+Rebs+Asts': 'PRA',
    'Blocked Shots': 'BLK',
    'Steals': 'STL',
    'Assists': 'AST',
    'Blks+Stls': 'BLST',
    'Turnovers': 'TOV',
    'Hits+Runs+RBIS': 'hits+runs+rbi',
    'Total Bases': 'total bases',
    'Hitter Strikeouts': 'batter strikeouts',
    'Pitcher Strikeouts': 'pitcher strikeouts',
    'Runs': 'runs',
    'RBIs': 'rbi',
    'Pitching Outs': 'pitching outs',
    'Hits Allowed': 'hits allowed',
    'Walks Allowed': 'walks allowed',
    'Earned Runs Allowed': 'runs allowed',
    '1st Inning Runs Allowed': '1st inning runs allowed',
    'Shots On Goal': 'shots',
    'Goals': 'goals',
    'Goalie Saves': 'saves'
}

dk_data['PP Key'] = pp2dk
fd_data['PP Key'] = pp2fd
pin_data['PP Key'] = pp2pin
csb_data['PP Key'] = pp2csb

live_bets = ["1H", "2H", "1P", "2P", "3P", "1Q",
             "2Q", "3Q", "4Q", "LIVE", "SZN", "SZN2", "SERIES"]
if len(pp_offers) > 0:
    print("Matching PrizePicks offers")
    for o in tqdm(pp_offers):
        opponent = o.get('Opponent')
        if any(substring in o['League'] for substring in live_bets):
            o['Market'] = o['Market'] + " " + \
                [live for live in live_bets if live in o['League']][0]
        try:
            v = []
            lines = []
            newline = {"Platform": "PrizePicks",
                       "League": o['League'], "Market": o['Market']}
            for dataset in [dk_data, fd_data, pin_data, csb_data]:
                codex = dataset['PP Key']
                offer = dataset.get(o['Player'], {o['Player']: None}).get(
                    codex.get(o['Market'], o['Market']))
                if offer is not None:
                    v.append(offer['EV'])
                elif " + " in o['Player'] or " vs. " in o['Player']:
                    players = o['Player'].replace(" vs. ", " + ").split(" + ")
                    player = players[1] + " + " + players[0]
                    offer = dataset.get(player, {player: None}).get(
                        codex.get(o['Market'], o['Market']))
                    if offer is not None:
                        v.append(offer['EV'])
                    else:
                        ev1 = dataset.get(players[0], {players[0]: None}).get(
                            codex.get(o['Market'], o['Market']))
                        ev2 = dataset.get(players[1], {players[1]: None}).get(
                            codex.get(o['Market'], o['Market']))
                        if ev1 is not None and ev2 is not None:
                            v.append(ev1['EV']+ev2['EV'])
                            offer = {
                                'Line': str(float(ev1['Line']) + float(ev2['Line'])),
                                'Over': str((float(ev1['Over']) + float(ev2['Over']))/2),
                                'Under': str((float(ev1['Under']) + float(ev2['Under']))/2),
                            }

                lines.append(offer)

            if v:
                tapped_markets.append(newline)
                if newline in untapped_markets:
                    untapped_markets.remove(newline)

                v = np.mean(v)
                line = (np.ceil(o['Line']-1), np.floor(o['Line']))
                p = [poisson.cdf(line[0], v), poisson.sf(line[1], v)]
                push = 1-p[1]-p[0]

                stats = np.ones(5) * -1000
                if o['League'] == 'NBA':
                    stats = get_nba_stats(
                        o['Player'], opponent, pp2stats[o['Market']], o['Line'])
                elif o['League'] == 'MLB':
                    stats = get_mlb_stats(
                        o['Player'], opponent, pp2stats[o['Market']], o['Line'])
                elif o['League'] == 'NHL':
                    stats = get_nhl_stats(
                        o['Player'], opponent, pp2stats[o['Market']], o['Line'])

                if p[1] > p[0]:
                    o['Bet'] = 'Over'
                    o['Prob'] = p[1]+push/2

                else:
                    o['Bet'] = 'Under'
                    o['Prob'] = p[0]+push/2

                o['Last 10 Avg'] = stats[0] if stats[0] != -1000 else 'N/A'
                o['Last 5'] = stats[1] if stats[1] != -1000 else 'N/A'
                o['Season'] = stats[2] if stats[2] != -1000 else 'N/A'
                o['H2H'] = stats[3] if stats[3] != -1000 else 'N/A'
                o['OvP'] = stats[4] if stats[4] != -1000 else 'N/A'

                if o['Bet'] == 'Over':
                    loc = get_loc(stats)
                else:
                    loc = -get_loc(stats)

                if loc+o['Prob'] > .555:
                    o['Good Bet'] = 'Y'
                else:
                    o['Good Bet'] = 'N'

                o['DraftKings'] = lines[0]['Line'] + "/" + \
                    lines[0][o['Bet']] if lines[0] else 'N/A'
                o['FanDuel'] = lines[1]['Line'] + "/" + \
                    lines[1][o['Bet']] if lines[1] else 'N/A'
                o['Pinnacle'] = lines[2]['Line'] + "/" + \
                    lines[2][o['Bet']] if lines[2] else 'N/A'
                o['Caesars'] = str(lines[3]['Line']) + "/" + \
                    str(lines[3][o['Bet']]) if lines[3] else 'N/A'

            elif newline not in untapped_markets+tapped_markets:
                untapped_markets.append(newline)

        except Exception as exc:
            print(o['Player'] + ", " + o["Market"])
            print(traceback.format_exc())
            print(exc)

ud_offers = get_ud()

ud2dk = {
    'Runs': 'Runs Scored',
    'Walks Allowed': 'Walks',
    'Pts + Rebs + Asts': 'Pts + Rebs + Ast',
    'Rebounds + Assists': 'Ast + Reb',
    'Points + Assists': 'Pts + Ast',
    'Points + Rebounds': 'Pts + Reb',
    'Blocks + Steals': 'Steals + Blocks',
    '3-Pointers Made': 'Threes',
    'Shots': 'Player Shots on Goal',
    'Games won': 'Player Games Won'
}

ud2fd = {
    'Points + Assists': 'Pts + Ast',
    'Points + Rebounds': 'Pts + Reb',
    'Pts + Rebs + Asts': 'Pts + Reb + Ast',
    'Rebounds + Assists': 'Reb + Ast',
    '3-Pointers Made': 'Made Threes',
    'Shots': 'Shots on Goal'
}

ud2pin = {
    'Shots': 'Shots on Goal',
    'Strikeouts': 'Total Strikeouts',
    '3-Pointers Made': '3 Point FG',
    'Pts + Rebs + Asts': 'Pts+Rebs+Asts'
}

ud2csb = {
    'Strikeouts': 'Pitching Strikeouts',
    '3-Pointers Made': '3-Pt Made'
}

ud2stats = {
    '3-Pointers Made': 'FG3M',
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Points + Rebounds': 'PR',
    'Points + Assists': 'PA',
    'Rebounds + Assists': 'RA',
    'Pts + Rebs + Asts': 'PRA',
    'Blocks': 'BLK',
    'Steals': 'STL',
    'Assists': 'AST',
    'Blocks + Steals': 'BLST',
    'Turnovers': 'TOV',
    'Hits + Runs + RBIs': 'hits+runs+rbi',
    'Total Bases': 'total bases',
    'Strikeouts': 'pitcher strikeouts',
    'Runs': 'runs',
    'Singles': 'singles',
    'Walks': 'walks',
    'Hits': 'hits',
    'Walks Allowed': 'walks allowed',
    'Goals Against': 'goalsAgainst',
    'Goals': 'goals',
    'Shots': 'shots',
    'Saves': 'saves'
}

dk_data['UD Key'] = ud2dk
fd_data['UD Key'] = ud2fd
pin_data['UD Key'] = ud2pin
csb_data['UD Key'] = ud2csb

if len(ud_offers) > 0:

    print("Matching Underdog offers")
    for o in tqdm(ud_offers):
        opponent = o.get('Opponent')
        if any(substring in o['League'] for substring in live_bets):
            o['Market'] = o['Market'] + " " + \
                [live for live in live_bets if live in o['League']][0]
        p = []
        try:
            market = o['Market']
            newline = {"Platform": "Underdog",
                       "League": o['League'], "Market": o['Market']}
            if "H2H" in market:
                v1 = []
                v2 = []
                lines = []
                market = market[4:]
                players = o['Player'].split(" vs. ")
                for dataset in [dk_data, fd_data, pin_data, csb_data]:
                    codex = dataset['UD Key']
                    offer1 = dataset.get(players[0], {players[0]: None}).get(
                        codex.get(market, market))
                    offer2 = dataset.get(players[1], {players[1]: None}).get(
                        codex.get(market, market))
                    if offer1 is not None and offer2 is not None:
                        v1.append(offer1['EV'])
                        v2.append(offer2['EV'])
                        offer = {
                            'Line': str(float(offer2['Line']) - float(offer1['Line'])),
                            'Under': str((float(offer1['Over']) + float(offer2['Under']))/2),
                            'Over': str((float(offer1['Under']) + float(offer2['Over']))/2),
                        }
                    else:
                        offer = None
                    lines.append(offer)

                if v1 and v2:
                    v1 = np.mean(v1)
                    v2 = np.mean(v2)
                    line = (np.ceil(o['Line']-1), np.floor(o['Line']))
                    p = [skellam.cdf(line[0], v2, v1),
                         skellam.sf(line[1], v2, v1)]
                    push = 1-p[1]-p[0]

            else:
                v = []
                lines = []
                for dataset in [dk_data, fd_data, pin_data, csb_data]:
                    codex = dataset['UD Key']
                    offer = dataset.get(o['Player'], {o['Player']: None}).get(
                        codex.get(market, market))
                    if offer is not None:
                        v.append(offer['EV'])

                    lines.append(offer)

                if v:
                    v = np.mean(v)
                    line = (np.ceil(o['Line']-1), np.floor(o['Line']))
                    p = [poisson.cdf(line[0], v), poisson.sf(line[1], v)]
                    push = 1-p[1]-p[0]

            if p:
                tapped_markets.append(newline)
                if newline in untapped_markets:
                    untapped_markets.remove(newline)

                stats = np.ones(5) * -1000
                if o['League'] == 'NBA':
                    stats = get_nba_stats(
                        o['Player'], opponent, ud2stats[market], o['Line'])
                elif o['League'] == 'MLB':
                    stats = get_mlb_stats(
                        o['Player'], opponent, ud2stats[market], o['Line'])
                elif o['League'] == 'NHL':
                    stats = get_nhl_stats(
                        o['Player'], opponent, ud2stats[market], o['Line'])

                if p[1] > p[0]:
                    o['Bet'] = 'Over'
                    o['Prob'] = p[1]+push/2
                else:
                    o['Bet'] = 'Under'
                    o['Prob'] = p[0]+push/2

                o['Last 10 Avg'] = stats[0] if stats[0] != -1000 else 'N/A'
                o['Last 5'] = stats[1] if stats[1] != -1000 else 'N/A'
                o['Season'] = stats[2] if stats[2] != -1000 else 'N/A'
                o['H2H'] = stats[3] if stats[3] != -1000 else 'N/A'
                o['OvP'] = stats[4] if stats[4] != -1000 else 'N/A'

                if o['Bet'] == 'Over':
                    loc = get_loc(stats)
                else:
                    loc = -get_loc(stats)

                if loc+o['Prob'] > .555:
                    o['Good Bet'] = 'Y'
                else:
                    o['Good Bet'] = 'N'

                o['DraftKings'] = lines[0]['Line'] + "/" + \
                    lines[0][o['Bet']] if lines[0] else 'N/A'
                o['FanDuel'] = lines[1]['Line'] + "/" + \
                    lines[1][o['Bet']] if lines[1] else 'N/A'
                o['Pinnacle'] = lines[2]['Line'] + "/" + \
                    lines[2][o['Bet']] if lines[2] else 'N/A'
                o['Caesars'] = str(lines[3]['Line']) + "/" + \
                    str(lines[3][o['Bet']]) if lines[3] else 'N/A'

            elif newline not in untapped_markets+tapped_markets:
                untapped_markets.append(newline)

        except Exception as exc:
            print(o['Player'] + ", " + o["Market"])
            print(traceback.format_exc())
            print(exc)

th_offers = get_thrive()

th2dk = {
    'ASTS': 'Assists',
    'BASEs': 'Total Bases',
    'BLKS': 'Blocks',
    'GOLs + ASTs': 'Points',
    'HITs + RBIs + RUNs': 'Hits + Runs + RBIs',
    'Ks': 'Strikeouts',
    'PTS': 'Points',
    'PTS + ASTS': 'Pts + Ast',
    'PTS + REBS': 'Pts + Reb',
    'PTS + REBS + ASTS': 'Pts + Rebs + Ast',
    'REBS': 'Rebounds',
    'REBS + ASTS': 'Ast + Reb',
    'RUNs': 'Runs Scored',
    'SAVs': 'Saves',
    'STLS': 'Steals'
}

th2fd = {
    'ASTS': 'Assists',
    'BLKS': 'Blocks',
    'Ks': 'Strikeouts',
    'PTS': 'Points',
    'PTS + ASTS': 'Pts + Ast',
    'PTS + REBS': 'Pts + Reb',
    'PTS + REBS + ASTS': 'Pts + Reb + Ast',
    'REBS': 'Rebounds',
    'REBS + ASTS': 'Reb + Ast',
    'SAVs': 'Saves',
    'STLS': 'Steals'
}

th2pin = {
    'ASTS': 'Assists',
    'BASEs': 'Total Bases',
    'GOLs + ASTs': 'Points',
    'Ks': 'Total Strikeouts',
    'PTS': 'Points',
    'PTS + REBS + ASTS': 'Pts+Rebs+Asts',
    'REBS': 'Rebounds',
    'SAVs': 'Saves',
    'STLS': 'Steals'
}

th2csb = {
    'ASTS': 'Assists',
    'BASEs': 'Total Bases',
    'BLKS': 'Blocks',
    'GOLs + ASTs': 'Points',
    'Ks': 'Pitching Strikeouts',
    'PTS': 'Points',
    'PTS + ASTS': 'Points + Assists',
    'PTS + REBS': 'Points + Rebounds',
    'PTS + REBS + ASTS': 'Pts + Rebs + Asts',
    'REBS': 'Rebounds',
    'REBS + ASTS': 'Rebounds + Assists',
    'SAVs': 'Saves',
    'STLS': 'Steals'
}

th2stats = {
    'GOLs + ASTs': 'PTS',
    'REBS': 'REB',
    'PTS': 'PTS',
    'ASTS': 'AST',
    'PTS + REBS': 'PR',
    'PTS + ASTS': 'PA',
    'REBS + ASTS': 'RA',
    'PTS + REBS + ASTS': 'PRA',
    'BLKs': 'BLK',
    'ASTs': 'AST',
    'HITs + RBIs + RUNs': 'hits+runs+rbi',
    'BASEs': 'total bases',
    'Ks': 'pitcher strikeouts',
    'RUNs': 'runs',
    'SAVs': 'saves',
    'STLS': 'STL'
}

dk_data['TH Key'] = th2dk
fd_data['TH Key'] = th2fd
pin_data['TH Key'] = th2pin
csb_data['TH Key'] = th2csb

if len(th_offers) > 0:

    print("Matching Thrive offers")
    for o in tqdm(th_offers):

        opponent = o.get('Opponent')
        if any(substring in o['League'] for substring in live_bets):
            o['Market'] = o['Market'] + " " + \
                [live for live in live_bets if live in o['League']][0]
        p = []
        try:
            market = o['Market']
            newline = {"Platform": "Thrive",
                       "League": o['League'], "Market": o['Market']}

            v = []
            lines = []
            for dataset in [dk_data, fd_data, pin_data, csb_data]:
                codex = dataset['TH Key']
                offer = dataset.get(o['Player'], {o['Player']: None}).get(
                    codex.get(market, market))
                if offer is not None:
                    v.append(offer['EV'])

                lines.append(offer)

            if v:
                v = np.mean(v)
                line = (np.ceil(o['Line']-1), np.floor(o['Line']))
                p = [poisson.cdf(line[0], v), poisson.sf(line[1], v)]
                push = 1-p[1]-p[0]

            if p:
                tapped_markets.append(newline)
                if newline in untapped_markets:
                    untapped_markets.remove(newline)

                stats = np.ones(5) * -1000
                if o['League'] == 'NBA':
                    stats = get_nba_stats(
                        o['Player'], opponent, th2stats[market], o['Line'])
                elif o['League'] == 'MLB':
                    stats = get_mlb_stats(
                        o['Player'], opponent, th2stats[market], o['Line'])
                elif o['League'] == 'NHL':
                    stats = get_nhl_stats(
                        o['Player'], opponent, th2stats[market], o['Line'])

                if p[1] > p[0]:
                    o['Bet'] = 'Over'
                    o['Prob'] = p[1]+push/2
                else:
                    o['Bet'] = 'Under'
                    o['Prob'] = p[0]+push/2

                o['Last 10 Avg'] = stats[0] if stats[0] != -1000 else 'N/A'
                o['Last 5'] = stats[1] if stats[1] != -1000 else 'N/A'
                o['Season'] = stats[2] if stats[2] != -1000 else 'N/A'
                o['H2H'] = stats[3] if stats[3] != -1000 else 'N/A'
                o['OvP'] = stats[4] if stats[4] != -1000 else 'N/A'

                if o['Bet'] == 'Over':
                    loc = get_loc(stats)
                else:
                    loc = -get_loc(stats)

                if loc+o['Prob'] > .555:
                    o['Good Bet'] = 'Y'
                else:
                    o['Good Bet'] = 'N'

                o['DraftKings'] = lines[0]['Line'] + "/" + \
                    lines[0][o['Bet']] if lines[0] else 'N/A'
                o['FanDuel'] = lines[1]['Line'] + "/" + \
                    lines[1][o['Bet']] if lines[1] else 'N/A'
                o['Pinnacle'] = lines[2]['Line'] + "/" + \
                    lines[2][o['Bet']] if lines[2] else 'N/A'
                o['Caesars'] = str(lines[3]['Line']) + "/" + \
                    str(lines[3][o['Bet']]) if lines[3] else 'N/A'

            elif newline not in untapped_markets+tapped_markets:
                untapped_markets.append(newline)

        except Exception as exc:
            print(o['Player'] + ", " + o["Market"])
            print(traceback.format_exc())
            print(exc)

print("Writing to file...")
if len(pp_offers) > 0:
    pp_df = pd.DataFrame(pp_offers).dropna().drop(
        columns='Opponent').sort_values('Prob', ascending=False)
    wks = gc.open("Sports Betting").worksheet("PrizePicks")
    wks.clear()
    wks.update([pp_df.columns.values.tolist()] + pp_df.values.tolist())
    wks.set_basic_filter()
    wks.format("G:L", {"numberFormat": {
               "type": "PERCENT", "pattern": "0.00%"}})

if len(ud_offers) > 0:
    ud_df = pd.DataFrame(ud_offers).dropna().drop(
        columns='Opponent').sort_values('Prob', ascending=False)
    wks = gc.open("Sports Betting").worksheet("Underdog")
    wks.clear()
    wks.update([ud_df.columns.values.tolist()] + ud_df.values.tolist())
    wks.set_basic_filter()
    wks.format("G:L", {"numberFormat": {
               "type": "PERCENT", "pattern": "0.00%"}})

if len(th_offers) > 0:
    th_df = pd.DataFrame(th_offers).dropna().drop(
        columns='Opponent').sort_values('Prob', ascending=False)
    wks = gc.open("Sports Betting").worksheet("Thrive")
    wks.clear()
    wks.update([th_df.columns.values.tolist()] + th_df.values.tolist())
    wks.set_basic_filter()
    wks.format("G:L", {"numberFormat": {
               "type": "PERCENT", "pattern": "0.00%"}})

if len(untapped_markets) > 0:
    untapped_df = pd.DataFrame(untapped_markets).drop_duplicates()
    wks = gc.open("Sports Betting").worksheet("Untapped Markets")
    wks.clear()
    wks.update([untapped_df.columns.values.tolist()] +
               untapped_df.values.tolist())
    wks.set_basic_filter()

print("Success!")
