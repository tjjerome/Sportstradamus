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
from tqdm import tqdm
from scipy.stats import poisson, skellam
import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from sportsbook_spider.helpers import get_loc
from sportsbook_spider.books import get_caesars, get_fd, get_pinnacle, get_dk, get_pp, get_ud, get_thrive, get_parp
from sportsbook_spider.stats import statsNBA, statsMLB, statsNHL


def main():
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
        creds = Credentials.from_authorized_user_file(
            './creds/token.json', SCOPES)
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
    dk_data = {}
    print("Getting DraftKings MLB lines")
    dk_data.update(get_dk(84240, [743, 1024, 1031]))  # MLB
    print("Getting DraftKings NBA lines")
    dk_data.update(
        get_dk(42648, [583, 1215, 1216, 1217, 1218, 1219, 1220]))  # NBA
    print("Getting DraftKings NHL lines")
    dk_data.update(get_dk(42133, [550, 1064, 1189]))  # NHL
    # dk_data.update(get_dk(92893, [488, 633])) # Tennis
    # dk_data.update(get_dk(91581, [488, 633])) # Tennis
    print(str(len(dk_data)) + " offers found")

    fd_data = {}
    print("Getting FanDuel MLB lines")
    fd_data.update(get_fd('mlb', ['batter-props', 'pitcher-props', 'innings']))
    print("Getting FanDuel NBA lines")
    fd_data.update(get_fd('nba', ['player-points', 'player-rebounds',
                                  'player-assists', 'player-threes', 'player-combos', 'player-defense']))
    print("Getting FanDuel NHL lines")
    fd_data.update(
        get_fd('nhl', ['goal-scorer', 'shots', 'points-assists', 'goalie-props']))
    print(str(len(fd_data)) + " offers found")

    pin_data = {}
    print("Getting Pinnacle MLB lines")
    pin_data.update(get_pinnacle(246))  # MLB
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
    nba = statsNBA()
    nba.load()
    nba.update()
    mlb = statsMLB()
    mlb.load()
    mlb.update()
    nhl = statsNHL()
    nhl.load()
    nhl.update()

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
                        players = o['Player'].replace(
                            " vs. ", " + ").split(" + ")
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
                        stats = nba.get_stats(
                            o['Player'], opponent, pp2stats[o['Market']], o['Line'])
                    elif o['League'] == 'MLB':
                        stats = mlb.get_stats(
                            o['Player'], opponent, pp2stats[o['Market']], o['Line'])
                    elif o['League'] == 'NHL':
                        stats = nhl.get_stats(
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
                        stats = nba.get_stats(
                            o['Player'], opponent, ud2stats[market], o['Line'])
                    elif o['League'] == 'MLB':
                        stats = mlb.get_stats(
                            o['Player'], opponent, ud2stats[market], o['Line'])
                    elif o['League'] == 'NHL':
                        stats = nhl.get_stats(
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
                        stats = nba.get_stats(
                            o['Player'], opponent, th2stats[market], o['Line'])
                    elif o['League'] == 'MLB':
                        stats = mlb.get_stats(
                            o['Player'], opponent, th2stats[market], o['Line'])
                    elif o['League'] == 'NHL':
                        stats = nhl.get_stats(
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

    parp_offers = get_parp()

    parp2dk = {
        'Blocked Shots': 'Blocks',
        'Hit + Run + RBI': 'Hits + Runs + RBIs',
        'Pts + Reb + Ast': 'Pts + Rebs + Ast',
        'Reb + Ast': 'Ast + Reb',
        'Shots': 'Player Shots on Goal',
        'Shots on Goal': 'Player Shots on Goal',
        'Stl + Blk': 'Steals + Blocks',
        'Strikeouts (K)': 'Strikeouts'
    }

    parp2fd = {
        'Blocked Shots': 'Blocks',
        'Shots': 'Shots on Goal',
        'Strikeouts (K)': 'Strikeouts'
    }

    parp2pin = {
        'Blocked Shots': 'Blocks',
        'Pts + Reb + Ast': 'Pts+Rebs+Asts',
        'Shots': 'Shots on Goal',
        'Strikeouts (K)': 'Total Strikeouts'
    }

    parp2csb = {
        'Blocked Shots': 'Blocks',
        'Hit + Run + RBI': 'Hits + Runs + RBIs',
        'Pts + Ast': 'Points + Assists',
        'Pts + Reb': 'Points + Rebounds',
        'Pts + Reb + Ast': 'Pts + Rebs + Asts',
        'Reb + Ast': 'Rebounds + Assists',
        'Shots on Goal': 'Shots',
        'Stl + Blk': 'Blocks and Steals',
        'Strikeouts (K)': 'Pitching Strikeouts'
    }

    parp2stats = {
        'Assists': 'AST',
        'Blocked Shots': 'BLK',
        'Hit + Run + RBI': 'hits+runs+rbi',
        'Hits': 'hits',
        'Points': 'PTS',
        'Pts + Ast': 'PA',
        'Pts + Reb': 'PR',
        'Pts + Reb + Ast': 'PRA',
        'Reb + Ast': 'RA',
        'Rebounds': 'REB',
        'Saves': 'saves',
        'Shots': 'shots',
        'Shots on Goal': 'shots',
        'Stl + Blk': 'BLST',
        'Strikeouts (K)': 'pitcher strikeouts',
        'Turnovers': 'TOV'
    }

    dk_data['PARP Key'] = parp2dk
    fd_data['PARP Key'] = parp2fd
    pin_data['PARP Key'] = parp2pin
    csb_data['PARP Key'] = parp2csb

    if len(parp_offers) > 0:

        print("Matching ParlayPlay offers")
        for o in tqdm(parp_offers):

            opponent = o.get('Opponent')
            if any(substring in o['League'] for substring in live_bets):
                o['Market'] = o['Market'] + " " + \
                    [live for live in live_bets if live in o['League']][0]
            p = []
            try:
                market = o['Market']
                newline = {"Platform": "ParlayPlay",
                           "League": o['League'], "Market": o['Market']}

                v = []
                lines = []
                for dataset in [dk_data, fd_data, pin_data, csb_data]:
                    codex = dataset['PARP Key']
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
                        stats = nba.get_stats(
                            o['Player'], opponent, parp2stats[market], o['Line'])
                    elif o['League'] == 'MLB':
                        stats = mlb.get_stats(
                            o['Player'], opponent, parp2stats[market], o['Line'])
                    elif o['League'] == 'NHL':
                        stats = nhl.get_stats(
                            o['Player'], opponent, parp2stats[market], o['Line'])

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
    if len([o for o in pp_offers if o.get('Prob')]) > 0:
        pp_df = pd.DataFrame(pp_offers).dropna().drop(
            columns='Opponent').sort_values('Prob', ascending=False)
        wks = gc.open("Sports Betting").worksheet("PrizePicks")
        wks.clear()
        wks.update([pp_df.columns.values.tolist()] + pp_df.values.tolist())
        wks.update("R1", "Last Updated: " +
                   datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        wks.set_basic_filter()
        wks.format("G:L", {"numberFormat": {
            "type": "PERCENT", "pattern": "0.00%"}})

    if len([o for o in ud_offers if o.get('Prob')]) > 0:
        ud_df = pd.DataFrame(ud_offers).dropna().drop(
            columns='Opponent').sort_values('Prob', ascending=False)
        wks = gc.open("Sports Betting").worksheet("Underdog")
        wks.clear()
        wks.update([ud_df.columns.values.tolist()] + ud_df.values.tolist())
        wks.update("R1", "Last Updated: " +
                   datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        wks.set_basic_filter()
        wks.format("G:L", {"numberFormat": {
            "type": "PERCENT", "pattern": "0.00%"}})

    if len([o for o in th_offers if o.get('Prob')]) > 0:
        try:
            th_df = pd.DataFrame(th_offers).dropna().drop(
                columns='Opponent').sort_values('Prob', ascending=False)
            wks = gc.open("Sports Betting").worksheet("Thrive")
            wks.clear()
            wks.update([th_df.columns.values.tolist()] + th_df.values.tolist())
            wks.update("R1", "Last Updated: " +
                       datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            wks.set_basic_filter()
            wks.format("G:L", {"numberFormat": {
                "type": "PERCENT", "pattern": "0.00%"}})
        except Exception as exc:
            print(traceback.format_exc())

    if len([o for o in parp_offers if o.get('Prob')]) > 0:
        parp_df = pd.DataFrame(parp_offers).dropna().drop(
            columns='Opponent').sort_values('Prob', ascending=False)
        wks = gc.open("Sports Betting").worksheet("ParlayPlay")
        wks.clear()
        wks.update([parp_df.columns.values.tolist()] + parp_df.values.tolist())
        wks.update("S1", "Last Updated: " +
                   datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        wks.set_basic_filter()
        wks.format("G:L", {"numberFormat": {
            "type": "PERCENT", "pattern": "0.00%"}})

    if len(untapped_markets) > 0:
        untapped_df = pd.DataFrame(untapped_markets).drop_duplicates()
        wks = gc.open("Sports Betting").worksheet("Untapped Markets")
        wks.clear()
        wks.update([untapped_df.columns.values.tolist()] +
                   untapped_df.values.tolist())
        wks.update("R1", "Last Updated: " +
                   datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        wks.set_basic_filter()

    print("Success!")


if __name__ == '__main__':
    main()
