# -*- coding: utf-8 -*-
"""Sportsbook Scraper

TODO:

*   Add web app - Flask
*   Tennis/Golf/Racing/WNBA
*   Add eSports (maybe from GGBET or Betway?)
"""
from sportsbook_spider.spiderLogger import logger
from sportsbook_spider.stats import statsNBA, statsMLB, statsNHL
from sportsbook_spider.books import get_caesars, get_fd, get_pinnacle, get_dk, get_pp, get_ud, get_thrive, get_parp
from sportsbook_spider.helpers import get_pred, prob_to_odds, archive, match_offers
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import gspread
import click
from sportsbook_spider import creds, data
from scipy.stats import poisson, skellam
from functools import partialmethod
from tqdm import tqdm
import numpy as np
import pandas as pd
import os.path
import datetime
import importlib.resources as pkg_resources


@click.command()
@click.option('--progress', default=True, help='Display progress bars')
def main(progress):

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not progress))
    # Authorize the gspread API
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file'
    ]
    cred = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists((pkg_resources.files(creds) / "token.json")):
        cred = Credentials.from_authorized_user_file(
            (pkg_resources.files(creds) / "token.json"), SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                (pkg_resources.files(creds) / "credentials.json"), SCOPES)
            cred = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open((pkg_resources.files(creds) / "token.json"), 'w') as token:
            token.write(cred.to_json())
    gc = gspread.authorize(cred)

    """"
    Start gathering sportsbook data
    """
    dk_data = {}
    logger.info("Getting DraftKings MLB lines")
    dk_data.update(get_dk(84240, [743, 1024, 1031]))  # MLB
    logger.info("Getting DraftKings NBA lines")
    dk_data.update(
        get_dk(42648, [583, 1215, 1216, 1217, 1218, 1219, 1220]))  # NBA
    logger.info("Getting DraftKings NHL lines")
    dk_data.update(get_dk(42133, [550, 1064, 1189]))  # NHL
    # dk_data.update(get_dk(92893, [488, 633])) # Tennis
    # dk_data.update(get_dk(91581, [488, 633])) # Tennis
    logger.info(str(len(dk_data)) + " offers found")

    fd_data = {}
    logger.info("Getting FanDuel MLB lines")
    fd_data.update(get_fd('mlb', ['batter-props', 'pitcher-props', 'innings']))
    logger.info("Getting FanDuel NBA lines")
    fd_data.update(get_fd('nba', ['player-points', 'player-rebounds',
                                  'player-assists', 'player-threes', 'player-combos', 'player-defense']))
    logger.info("Getting FanDuel NHL lines")
    fd_data.update(
        get_fd('nhl', ['goal-scorer', 'shots', 'points-assists', 'goalie-props']))
    logger.info(str(len(fd_data)) + " offers found")

    pin_data = {}
    logger.info("Getting Pinnacle MLB lines")
    pin_data.update(get_pinnacle(246))  # MLB
    logger.info("Getting Pinnacle NBA lines")
    pin_data.update(get_pinnacle(487))  # NBA
    logger.info("Getting Pinnacle NHL lines")
    pin_data.update(get_pinnacle(1456))  # NHL
    logger.info(str(len(pin_data)) + " offers found")

    csb_data = {}
    logger.info("Getting Caesars MLB Lines")
    sport = "baseball"
    league = "04f90892-3afa-4e84-acce-5b89f151063d"
    csb_data.update(get_caesars(sport, league))
    logger.info("Getting Caesars NBA Lines")
    sport = "basketball"
    league = "5806c896-4eec-4de1-874f-afed93114b8c"  # NBA
    csb_data.update(get_caesars(sport, league))
    logger.info("Getting Caesars NHL Lines")
    sport = "icehockey"
    league = "b7b715a9-c7e8-4c47-af0a-77385b525e09"
    csb_data.update(get_caesars(sport, league))
    # logger.info("Getting Caesars NFL Lines")
    # sport = "americanfootball"
    # league = "007d7c61-07a7-4e18-bb40-15104b6eac92"
    # csb_data.update(get_caesars(sport, league))
    logger.info(str(len(csb_data)) + " offers found")

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

    pp_dict = get_pp()

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

    dk_data['PrizePicks'] = pp2dk
    fd_data['PrizePicks'] = pp2fd
    pin_data['PrizePicks'] = pp2pin
    csb_data['PrizePicks'] = pp2csb

    ud_dict = get_ud()

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
        'RBIs': 'rbi',
        'Walks Allowed': 'walks allowed',
        'Goals Against': 'goalsAgainst',
        'Goals': 'goals',
        'Shots': 'shots',
        'Saves': 'saves'
    }

    dk_data['Underdog'] = ud2dk
    fd_data['Underdog'] = ud2fd
    pin_data['Underdog'] = ud2pin
    csb_data['Underdog'] = ud2csb

    th_dict = get_thrive()

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
        'BLKS': 'BLK',
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

    dk_data['Thrive'] = th2dk
    fd_data['Thrive'] = th2fd
    pin_data['Thrive'] = th2pin
    csb_data['Thrive'] = th2csb

    parp_dict = get_parp()

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
        '3PT Made': 'FG3M',
        'Assists': 'AST',
        'Blocked Shots': 'BLK',
        'Hit + Run + RBI': 'hits+runs+rbi',
        'Runs': 'runs',
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
        'Steals': 'STL',
        'Blocks': 'BLK',
        'Strikeouts (K)': 'pitcher strikeouts',
        'Strikeouts': 'batter strikeouts',
        'Hits Allowed': 'hits allowed',
        'Walks Allowed': 'walks allowed',
        'Total Bases': 'total bases',
        'Pitching Outs': 'pitching outs',
        'Earned Runs': 'runs allowed',
        'Turnovers': 'TOV'
    }

    dk_data['ParlayPlay'] = parp2dk
    fd_data['ParlayPlay'] = parp2fd
    pin_data['ParlayPlay'] = parp2pin
    csb_data['ParlayPlay'] = parp2csb

    datasets = [dk_data, fd_data, pin_data, csb_data]

    pp_offers = []
    if len(pp_dict) > 0:
        total = sum(sum(len(i) for i in v.values()) for v in pp_dict.values())
        with tqdm(total=total, desc="Matching PrizePicks Offers", unit='offer') as pbar:
            for league, markets in pp_dict.items():
                if league == 'NBA':
                    stat_data = nba
                elif league == 'MLB':
                    stat_data = mlb
                elif league == 'NHL':
                    stat_data = nhl
                else:
                    for market, offers in markets.items():
                        untapped_markets.append({
                            'Platform': 'PrizePicks',
                            'League': league,
                            'Market': market
                        })
                        pbar.update(len(offers))
                    continue

                for market, offers in markets.items():
                    new_offers = match_offers(
                        offers, league, market, 'PrizePicks', datasets, stat_data, pp2stats, pbar)
                    if len(new_offers) == 0:
                        untapped_markets.append({
                            'Platform': 'PrizePicks',
                            'League': league,
                            'Market': market
                        })
                    else:
                        pp_offers = pp_offers + new_offers

    ud_offers = []
    if len(ud_dict) > 0:
        total = sum(sum(len(i) for i in v.values()) for v in ud_dict.values())
        with tqdm(total=total, desc="Matching Underdog Offers", unit='offer') as pbar:
            for league, markets in ud_dict.items():
                if league == 'NBA':
                    stat_data = nba
                elif league == 'MLB':
                    stat_data = mlb
                elif league == 'NHL':
                    stat_data = nhl
                else:
                    for market, offers in markets.items():
                        untapped_markets.append({
                            'Platform': 'Underdog',
                            'League': league,
                            'Market': market
                        })
                        pbar.update(len(offers))
                    continue

                for market, offers in markets.items():
                    new_offers = match_offers(
                        offers, league, market, 'Underdog', datasets, stat_data, ud2stats, pbar)
                    if len(new_offers) == 0:
                        untapped_markets.append({
                            'Platform': 'Underdog',
                            'League': league,
                            'Market': market
                        })
                    else:
                        ud_offers = ud_offers + new_offers

    th_offers = []
    if len(th_dict) > 0:
        total = sum(sum(len(i) for i in v.values()) for v in th_dict.values())
        with tqdm(total=total, desc="Matching Thrive Offers", unit='offer') as pbar:
            for league, markets in th_dict.items():
                if league == 'NBA':
                    stat_data = nba
                elif league == 'MLB':
                    stat_data = mlb
                elif league == 'NHL':
                    stat_data = nhl
                else:
                    for market, offers in markets.items():
                        untapped_markets.append({
                            'Platform': 'Thrive',
                            'League': league,
                            'Market': market
                        })
                        pbar.update(len(offers))
                    continue

                for market, offers in markets.items():
                    new_offers = match_offers(
                        offers, league, market, 'Thrive', datasets, stat_data, th2stats, pbar)
                    if len(new_offers) == 0:
                        untapped_markets.append({
                            'Platform': 'Thrive',
                            'League': league,
                            'Market': market
                        })
                    else:
                        th_offers = th_offers + new_offers

    parp_offers = []
    if len(parp_dict) > 0:
        total = sum(sum(len(i) for i in v.values())
                    for v in parp_dict.values())
        with tqdm(total=total, desc="Matching ParlayPlay Offers", unit='offer') as pbar:
            for league, markets in parp_dict.items():
                if league == 'NBA':
                    stat_data = nba
                elif league == 'MLB':
                    stat_data = mlb
                elif league == 'NHL':
                    stat_data = nhl
                else:
                    for market, offers in markets.items():
                        untapped_markets.append({
                            'Platform': 'ParlayPlay',
                            'League': league,
                            'Market': market
                        })
                        pbar.update(len(offers))
                    continue

                for market, offers in markets.items():
                    new_offers = match_offers(
                        offers, league, market, 'ParlayPlay', datasets, stat_data, parp2stats, pbar)
                    if len(new_offers) == 0:
                        untapped_markets.append({
                            'Platform': 'ParlayPlay',
                            'League': league,
                            'Market': market
                        })
                    else:
                        parp_offers = parp_offers + new_offers

    logger.info("Writing to file...")
    if len(pp_offers) > 0:
        pp_df = pd.DataFrame(pp_offers).dropna().drop(
            columns='Opponent').sort_values('Model', ascending=False)
        wks = gc.open("Sports Betting").worksheet("PrizePicks")
        wks.clear()
        wks.update([pp_df.columns.values.tolist()] + pp_df.values.tolist())
        wks.update("S1", "Last Updated: " +
                   datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        wks.set_basic_filter()
        wks.format("H:N", {"numberFormat": {
            "type": "PERCENT", "pattern": "0.00%"}})

    if len(ud_offers) > 0:
        ud_df = pd.DataFrame(ud_offers).dropna().drop(
            columns='Opponent').sort_values('Model', ascending=False)
        wks = gc.open("Sports Betting").worksheet("Underdog")
        wks.clear()
        wks.update([ud_df.columns.values.tolist()] + ud_df.values.tolist())
        wks.update("S1", "Last Updated: " +
                   datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        wks.set_basic_filter()
        wks.format("H:N", {"numberFormat": {
            "type": "PERCENT", "pattern": "0.00%"}})

    if len(th_offers) > 0:
        try:
            th_df = pd.DataFrame(th_offers).dropna().drop(
                columns='Opponent').sort_values('Model', ascending=False)
            wks = gc.open("Sports Betting").worksheet("Thrive")
            wks.clear()
            wks.update([th_df.columns.values.tolist()] + th_df.values.tolist())
            wks.update("S1", "Last Updated: " +
                       datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            wks.set_basic_filter()
            wks.format("H:N", {"numberFormat": {
                "type": "PERCENT", "pattern": "0.00%"}})
        except Exception as exc:
            logger.exception('Error writing Thrive offers')

    if len(parp_offers) > 0:
        parp_df = pd.DataFrame(parp_offers).dropna().drop(
            columns='Opponent').sort_values('Model', ascending=False)
        wks = gc.open("Sports Betting").worksheet("ParlayPlay")
        wks.clear()
        wks.update([parp_df.columns.values.tolist()] + parp_df.values.tolist())
        wks.update("T1", "Last Updated: " +
                   datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        wks.set_basic_filter()
        wks.format("I:O", {"numberFormat": {
            "type": "PERCENT", "pattern": "0.00%"}})

    if len(untapped_markets) > 0:
        untapped_df = pd.DataFrame(untapped_markets).drop_duplicates()
        wks = gc.open("Sports Betting").worksheet("Untapped Markets")
        wks.clear()
        wks.update([untapped_df.columns.values.tolist()] +
                   untapped_df.values.tolist())
        wks.set_basic_filter()

    archive.write()
    logger.info("Success!")


if __name__ == '__main__':

    main()
