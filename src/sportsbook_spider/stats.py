from sportsbook_spider.spiderLogger import logger
import os.path
import numpy as np
from datetime import datetime, timedelta, date
import pickle
import json
import importlib.resources as pkg_resources
from sportsbook_spider import data
from tqdm import tqdm
import statsapi as mlb
from scipy.optimize import minimize
import nba_api.stats.endpoints as nba
from nba_api.stats.static import players as nba_static
import nfl_data_py as nfl
from time import sleep
from sportsbook_spider.helpers import scraper, mlb_pitchers, likelihood, archive
import pandas as pd


class statsNBA:
    def __init__(self):
        self.gamelog = []
        self.players = {}
        self.season_start = datetime.strptime("2022-10-18", "%Y-%m-%d")

    def load(self):
        filepath = (pkg_resources.files(data) / "nba_players.dat")
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.players = pickle.load(infile)
                self.players = {int(k): v for k, v in self.players.items()}

        filepath = (pkg_resources.files(data) / "weights.json")
        if os.path.isfile(filepath):
            with open(filepath, "r") as infile:
                weights = json.load(infile)

            self.weights = weights

    def update(self):
        logger.info("Getting NBA stats")
        nba_gamelog = nba.playergamelogs.PlayerGameLogs(
            season_nullable='2022-23').get_normalized_dict()['PlayerGameLogs']
        nba_playoffs = nba.playergamelogs.PlayerGameLogs(
            season_nullable='2022-23', season_type_nullable='Playoffs').get_normalized_dict()['PlayerGameLogs']
        self.gamelog = nba_playoffs + nba_gamelog
        for game in tqdm(self.gamelog):
            if game['PLAYER_ID'] not in self.players:
                self.players[game['PLAYER_ID']] = nba.commonplayerinfo.CommonPlayerInfo(
                    player_id=game['PLAYER_ID']).get_normalized_dict()['CommonPlayerInfo'][0]
                sleep(0.5)

            game['POS'] = self.players[game['PLAYER_ID']].get('POSITION')
            teams = game['MATCHUP'].replace("vs.", "@").split(" @ ")
            for team in teams:
                if team != game['TEAM_ABBREVIATION']:
                    game['OPP'] = team

            game["PRA"] = game['PTS'] + game['REB'] + game['AST']
            game["PR"] = game['PTS'] + game['REB']
            game["PA"] = game['PTS'] + game['AST']
            game["RA"] = game['REB'] + game['AST']
            game["BLST"] = game['BLK'] + game['STL']

        with open((pkg_resources.files(data) / "nba_players.dat"), "wb") as outfile:
            pickle.dump(self.players, outfile)

    def get_stats(self, player, opponent, market, line):
        if " + " in player:
            players = player.split(" + ")
            if "/" in opponent:
                opponent = opponent.split("/")
            else:
                opponent = [opponent, opponent]
            try:
                player_ids = [nba_static.find_players_by_full_name(
                    player)[0]['id'] for player in players]
                positions = self.players[player_ids[0]]['POSITION'].split(
                    '-') + self.players[player_ids[1]]['POSITION'].split('-')
            except:
                return np.ones(5)*-1000

            player1_games = [
                game for game in self.gamelog if game['PLAYER_ID'] == player_ids[0]]
            player2_games = [
                game for game in self.gamelog if game['PLAYER_ID'] == player_ids[1]]

            if not player1_games or not player2_games:
                return np.ones(5) * -1000

            season1 = np.array([game[market] for game in player1_games if
                                datetime.strptime(game['GAME_DATE'].split("T")[0], "%Y-%m-%d") >= self.season_start])
            season2 = np.array([game[market] for game in player2_games if
                                datetime.strptime(game['GAME_DATE'].split("T")[0], "%Y-%m-%d") >= self.season_start])

            n = min(len(season1), len(season2))
            season1 = season1[-n:]
            season2 = season2[-n:]

            h2h1 = np.array([game[market] for game in player1_games if
                            game['OPP'] == opponent[0]])
            h2h2 = np.array([game[market] for game in player2_games if
                            game['OPP'] == opponent[1]])

            n = min(len(h2h1), len(h2h2))
            if n == 0:
                headtohead = -1000
            else:
                h2h1 = h2h1[-n:]
                h2h2 = h2h2[-n:]
                headtohead = ((h2h1+h2h2) > line).astype(int)
                headtohead = np.mean(list(headtohead)+[1, 0])

            last10avg = (np.mean([game[market] for game in player1_games[-10:]]) +
                         np.mean([game[market] for game in player2_games[-10:]]) - line)/line
            n = min(len(player1_games), len(player2_games), 5)
            last5 = np.mean(((np.array([game[market] for game in player1_games[-n:]]) +
                              np.array([game[market] for game in player2_games[-n:]])) > line).astype(int))
            seasontodate = np.mean(((season1 + season2) > line).astype(int))

            if np.isnan(last5):
                last5 = -1000

            if np.isnan(seasontodate):
                seasontodate = -1000

            dvp = {}
            leagueavg = {}
            for game in self.gamelog:
                for position in positions:
                    if position in game['POS']:
                        if game['GAME_ID']+position not in leagueavg:
                            leagueavg[game['GAME_ID']+position] = 0

                        leagueavg[game['GAME_ID'] +
                                  position] = leagueavg[game['GAME_ID']+position] + game[market]

                        if game['OPP'] in opponent:
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

        elif " vs. " in player:
            players = player.split(" vs. ")
            if "/" in opponent:
                opponent = opponent.split("/")
            else:
                opponent = [opponent, opponent]
            try:
                player_ids = [nba_static.find_players_by_full_name(
                    player)[0]['id'] for player in players]
                positions1 = self.players[player_ids[0]]['POSITION'].split('-')
                positions2 = self.players[player_ids[1]]['POSITION'].split('-')
            except:
                return np.ones(5)*-1000

            player1_games = [
                game for game in self.gamelog if game['PLAYER_ID'] == player_ids[0]]
            player2_games = [
                game for game in self.gamelog if game['PLAYER_ID'] == player_ids[1]]

            if not player1_games or not player2_games:
                return np.ones(5) * -1000

            season1 = np.array([game[market] for game in player1_games if
                                datetime.strptime(game['GAME_DATE'].split("T")[0], "%Y-%m-%d") >= self.season_start])
            season2 = np.array([game[market] for game in player2_games if
                                datetime.strptime(game['GAME_DATE'].split("T")[0], "%Y-%m-%d") >= self.season_start])

            n = min(len(season1), len(season2))
            season1 = season1[-n:]
            season2 = season2[-n:]

            h2h1 = np.array([game[market] for game in player1_games if
                            game['OPP'] == opponent[0]])
            h2h2 = np.array([game[market] for game in player2_games if
                            game['OPP'] == opponent[1]])

            n = min(len(h2h1), len(h2h2))
            if n == 0:
                headtohead = -1000
            else:
                h2h1 = h2h1[-n:]
                h2h2 = h2h2[-n:]
                headtohead = ((h2h1 + line) > h2h2).astype(int)
                headtohead = np.mean(list(headtohead)+[1, 0])

            last10avg = (np.mean([game[market] for game in player1_games][:10]) + line -
                         np.mean([game[market] for game in player2_games][:10]))
            n = min(len(player1_games), len(player2_games), 5)
            last5 = np.mean(((np.array([game[market] for game in player1_games[-n:]]) + line) >
                             np.array([game[market] for game in player2_games[-n:]])).astype(int))
            seasontodate = np.mean(((season1 + line) > season2).astype(int))

            if np.isnan(last5):
                last5 = -1000

            if np.isnan(seasontodate):
                seasontodate = -1000

            dvp1 = {}
            leagueavg1 = {}
            dvp2 = {}
            leagueavg2 = {}
            for game in self.gamelog:
                for position in positions1:
                    if position in game['POS']:
                        if game['GAME_ID']+position not in leagueavg1:
                            leagueavg1[game['GAME_ID']+position] = 0

                        leagueavg1[game['GAME_ID'] +
                                   position] = leagueavg1[game['GAME_ID']+position] + game[market]

                        if game['OPP'] in opponent:
                            if game['GAME_ID']+position not in dvp1:
                                dvp1[game['GAME_ID']+position] = 0

                            dvp1[game['GAME_ID']+position] = dvp1[game['GAME_ID'] +
                                                                  position] + game[market]

                for position in positions2:
                    if position in game['POS']:
                        if game['GAME_ID']+position not in leagueavg2:
                            leagueavg2[game['GAME_ID']+position] = 0

                        leagueavg2[game['GAME_ID'] +
                                   position] = leagueavg2[game['GAME_ID']+position] + game[market]

                        if game['OPP'] in opponent:
                            if game['GAME_ID']+position not in dvp2:
                                dvp2[game['GAME_ID']+position] = 0

                            dvp2[game['GAME_ID']+position] = dvp2[game['GAME_ID'] +
                                                                  position] + game[market]

            if not dvp1 or not dvp2:
                dvpoa = -1000
            else:
                dvp1 = np.mean(list(dvp1.values()))
                leagueavg1 = np.mean(list(leagueavg1.values()))/2
                dvp2 = np.mean(list(dvp2.values()))
                leagueavg2 = np.mean(list(leagueavg2.values()))/2
                dvpoa = (dvp1-leagueavg1)/leagueavg1 - \
                    (dvp2-leagueavg2)/leagueavg2
        else:
            player_id = nba_static.find_players_by_full_name(player)
            if player_id and player_id[0]['id'] in self.players:
                player_id = player_id[0]['id']
                positions = self.players[player_id]['POSITION'].split('-')
            else:
                return np.ones(5)*-1000

            player_games = [
                game for game in self.gamelog if game['PLAYER_ID'] == player_id]

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
            for game in self.gamelog:
                for position in positions:
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

    def get_stats_date(self, game, market, line):
        old_gamelog = self.gamelog
        try:
            i = self.gamelog.index(game)
        except:
            i = len(self.gamelog)
        if i == 0:
            return np.ones(5)*-1000
        self.gamelog = self.gamelog[i:]
        player = game['PLAYER_NAME']
        opponent = game['OPP']
        stats = self.get_stats(player, opponent, market, line)
        self.gamelog = old_gamelog
        return stats

    def bucket_stats(self, market):
        playerStats = {}
        for game in self.gamelog:

            if not game['PLAYER_NAME'] in playerStats:
                playerStats[game['PLAYER_NAME']] = {'games': []}

            playerStats[game['PLAYER_NAME']]['games'].append(game[market])

        playerStats = {k: v for k, v in playerStats.items() if len(
            v['games']) > 10 and not all([g == 0 for g in v['games']])}

        averages = []
        for player, games in playerStats.items():
            playerStats[player]['avg'] = np.mean(games['games'])
            averages.append(np.mean(games['games']))

        edges = [np.percentile(averages, p) for p in range(0, 101, 10)]
        lines = np.zeros(10)
        for i in range(1, 11):
            lines[i-1] = np.round(np.mean([v for v in averages if v <=
                                  edges[i] and v >= edges[i-1]])-.5)+.5

        for player, games in playerStats.items():
            for i in range(0, 10):
                if games['avg'] >= edges[i]:
                    playerStats[player]['bucket'] = 10-i
                    playerStats[player]['line'] = lines[i]

        return playerStats

    def get_weights(self, market):
        if market in self.weights['NBA']:
            return self.weights['NBA'].get(market)

        playerStats = self.bucket_stats(market)
        results = []
        for game in tqdm(self.gamelog):

            player = game['PLAYER_NAME']
            if not player in playerStats:
                continue
            elif playerStats[player]['bucket'] < 5:
                continue

            line = playerStats[player]['avg']
            stats = self.get_stats_date(game, market, line)
            if all(stats == np.ones(5)*-1000):
                continue
            baseline = np.array([0, 0.5, 0.5, 0.5, 0])
            stats[stats == -1000] = baseline[stats == -1000]
            stats = stats - baseline
            y = int(game[market] > line)

            results.append({'stats': stats, 'result': y,
                            'bucket': playerStats[player]['bucket']})

        res = minimize(lambda x: -likelihood(results, x),
                       np.ones(5), method='l-bfgs-b', tol=1e-8, bounds=[(0, 10)]*5)

        w = list(res.x)
        self.weights['NBA'][market] = w
        filepath = (pkg_resources.files(data) / "weights.json")
        if os.path.isfile(filepath):
            with open(filepath, "w") as outfile:
                json.dump(self.weights, outfile, indent=4)

        return w

    def get_training_matrix(self, market):
        X = pd.DataFrame()
        results = []
        for game in self.gamelog:
            if any([string in market for string in ['allowed', 'pitch']]) and not game['starting pitcher']:
                continue
            elif not any([string in market for string in ['allowed', 'pitch']]) and not game['starting batter']:
                continue

            gameDate = datetime.strptime(game['gameId'][:10], '%Y/%m/%d')
            try:
                archiveData = archive['MLB'][market][gameDate.strftime(
                    '%Y-%m-%d')][game['playerName']]
            except:
                continue

            offer = {
                'Player': game['playerName'],
                'Team': game['team'],
                'Market': market,
                'Opponent': game['opponent'],
                'Pitcher': game['opponent pitcher']
            }

            for line, stats in archiveData.items():
                if not line == 'Closing Lines':
                    if X.empty:
                        X = self.row(offer | {'Line': line}, gameDate)
                    else:
                        X = pd.concat(
                            [X, self.row(offer | {'Line': line}, gameDate)])

                    results.append(
                        {'Over': int(game[market] > line), 'Under': int(game[market] < line)})

        y = pd.DataFrame(results)

        return X, y


class statsMLB:
    def __init__(self):
        self.gamelog = []
        self.gameIds = []
        self.season_start = datetime.strptime(
            mlb.latest_season()['regularSeasonStartDate'], "%Y-%m-%d")
        self.pitchers = mlb_pitchers

    def parse_game(self, gameId):
        game = mlb.boxscore_data(gameId)
        if game:
            self.gameIds.append(gameId)
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
                        'HA': 'Away',
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
                    self.gamelog.append(n)

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
                        'HA': 'Home',
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
                    self.gamelog.append(n)

    def load(self):
        filepath = (pkg_resources.files(data) / "mlb_data.dat")
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                mlb_data = pickle.load(infile)

            self.gamelog = mlb_data['gamelog']
            self.gameIds = mlb_data['games']

        filepath = (pkg_resources.files(data) / "weights.json")
        if os.path.isfile(filepath):
            with open(filepath, "r") as infile:
                weights = json.load(infile)

            self.weights = weights

    def update(self):
        logger.info("Getting MLB Stats")
        mlb_game_ids = mlb.schedule(
            start_date=self.season_start.date(), end_date=date.today())
        mlb_game_ids = [game['game_id'] for game in mlb_game_ids if game['status']
                        == 'Final' and game['game_type'] != 'E' and game['game_type'] != 'S']

        for id in tqdm(mlb_game_ids):
            if id not in self.gameIds:
                self.parse_game(id)

        for game in self.gamelog:
            if datetime.strptime(game['gameId'][:10], '%Y/%m/%d') < datetime.today() - timedelta(days=365):
                self.gamelog.remove(game)

        with open((pkg_resources.files(data) / "mlb_data.dat"), "wb") as outfile:
            pickle.dump(
                {'games': self.gameIds, 'gamelog': self.gamelog}, outfile)

    def get_stats(self, player, opponent, market, line):
        opponent = opponent.replace('ARI', 'AZ')

        if self.gamelog[0].get(market) is None:
            return np.ones(5) * -1000

        if " + " in player:
            players = player.split(" + ")
            if "/" in opponent:
                opponent = opponent.split("/")
            else:
                opponent = [opponent, opponent]

            pitcher = [self.pitchers[o] for o in opponent]

            if any([string in market for string in ['allowed', 'pitch']]):
                player1_games = [game for game in self.gamelog
                                 if game['playerName'] == players[0] and game['starting pitcher']]
                player2_games = [game for game in self.gamelog
                                 if game['playerName'] == players[1] and game['starting pitcher']]
            else:
                player1_games = [game for game in self.gamelog
                                 if game['playerName'] == players[0] and game['starting batter']]
                player2_games = [game for game in self.gamelog
                                 if game['playerName'] == players[1] and game['starting batter']]

            if not player1_games or not player2_games:
                return np.ones(5) * -1000

            season1 = np.array([game[market] for game in player1_games if
                                datetime.strptime(game['gameId'][:10], "%Y/%m/%d") >= self.season_start])
            season2 = np.array([game[market] for game in player2_games if
                                datetime.strptime(game['gameId'][:10], "%Y/%m/%d") >= self.season_start])

            n = min(len(season1), len(season2))
            season1 = season1[-n:]
            season2 = season2[-n:]

            if any([string in market for string in ['allowed', 'pitch']]):
                h2h1 = np.array([game[market] for game in player1_games if
                                game['opponent'] == opponent[0]])
                h2h2 = np.array([game[market] for game in player2_games if
                                game['opponent'] == opponent[1]])
                dvp = np.mean([game[market] for game in self.gamelog
                               if game['starting pitcher'] and game['opponent'] in opponent])
                leagueavg = np.mean(
                    [game[market] for game in self.gamelog if game['starting pitcher']])
            else:
                h2h1 = np.array([int(game[market] > line) for game in player1_games if
                                game['opponent pitcher'] == pitcher[0]])
                h2h2 = np.array([int(game[market] > line) for game in player2_games if
                                game['opponent pitcher'] == pitcher[1]])
                dvp = np.mean([game[market] for game in self.gamelog
                               if game['starting batter'] and game['opponent pitcher'] in pitcher])
                leagueavg = np.mean(
                    [game[market] for game in self.gamelog if game['starting batter']])

            n = min(len(h2h1), len(h2h2))
            if n == 0:
                headtohead = -1000
            else:
                h2h1 = h2h1[-n:]
                h2h2 = h2h2[-n:]
                headtohead = ((h2h1+h2h2) > line).astype(int)
                headtohead = np.mean(list(headtohead)+[1, 0])

            last10avg = (np.mean([game[market] for game in player1_games[-10:]]) +
                         np.mean([game[market] for game in player2_games[-10:]]) - line)/line
            n = min(len(player1_games), len(player2_games), 5)
            last5 = np.mean(((np.array([game[market] for game in player1_games[-n:]]) +
                              np.array([game[market] for game in player2_games[-n:]])) > line).astype(int))
            seasontodate = np.mean(((season1 + season2) > line).astype(int))

            if np.isnan(last5):
                last5 = -1000

            if np.isnan(seasontodate):
                seasontodate = -1000

            if np.isnan(dvp):
                dvpoa = -1000
            else:
                dvpoa = (dvp-leagueavg)/leagueavg

        elif " vs. " in player:
            players = player.split(" vs. ")
            if "/" in opponent:
                opponent = opponent.split("/")
            else:
                opponent = [opponent, opponent]

            pitcher = [self.pitchers.get(o, '') for o in opponent]

            if any([string in market for string in ['allowed', 'pitch']]):
                player1_games = [game for game in self.gamelog
                                 if game['playerName'] == players[0] and game['starting pitcher']]
                player2_games = [game for game in self.gamelog
                                 if game['playerName'] == players[1] and game['starting pitcher']]
            else:
                player1_games = [game for game in self.gamelog
                                 if game['playerName'] == players[0] and game['starting batter']]
                player2_games = [game for game in self.gamelog
                                 if game['playerName'] == players[1] and game['starting batter']]

            if not player1_games or not player2_games:
                return np.ones(5) * -1000

            season1 = np.array([game[market] for game in player1_games if
                                datetime.strptime(game['gameId'][:10], "%Y/%m/%d") >= self.season_start])
            season2 = np.array([game[market] for game in player2_games if
                                datetime.strptime(game['gameId'][:10], "%Y/%m/%d") >= self.season_start])

            n = min(len(season1), len(season2))
            season1 = season1[-n:]
            season2 = season2[-n:]

            if any([string in market for string in ['allowed', 'pitch']]):
                h2h1 = np.array([game[market] for game in player1_games if
                                game['opponent'] == opponent[0]])
                h2h2 = np.array([game[market] for game in player2_games if
                                game['opponent'] == opponent[1]])
                dvp1 = np.mean([game[market] for game in self.gamelog
                               if game['starting pitcher'] and game['opponent'] == opponent[0]])
                dvp2 = np.mean([game[market] for game in self.gamelog
                               if game['starting pitcher'] and game['opponent'] == opponent[1]])
                leagueavg = np.mean(
                    [game[market] for game in self.gamelog if game['starting pitcher']])
            else:
                h2h1 = np.array([int(game[market] > line) for game in player1_games if
                                game['opponent pitcher'] == pitcher[0]])
                h2h2 = np.array([int(game[market] > line) for game in player2_games if
                                game['opponent pitcher'] == pitcher[1]])
                dvp1 = np.mean([game[market] for game in self.gamelog
                               if game['starting batter'] and game['opponent pitcher'] == pitcher[0]])
                dvp2 = np.mean([game[market] for game in self.gamelog
                               if game['starting batter'] and game['opponent pitcher'] == pitcher[1]])
                leagueavg = np.mean(
                    [game[market] for game in self.gamelog if game['starting batter']])

            n = min(len(h2h1), len(h2h2))
            if n == 0:
                headtohead = -1000
            else:
                h2h1 = h2h1[-n:]
                h2h2 = h2h2[-n:]
                headtohead = ((h2h1 + line) > h2h2).astype(int)
                headtohead = np.mean(list(headtohead)+[1, 0])

            last10avg = (np.mean([game[market] for game in player2_games][-10:]) + line -
                         np.mean([game[market] for game in player1_games][-10:]))
            n = min(len(player1_games), len(player2_games), 5)
            last5 = np.mean(((np.array([game[market] for game in player1_games[-n:]]) + line) >
                             np.array([game[market] for game in player2_games[-n:]])).astype(int))
            seasontodate = np.mean(((season1 + line) > season2).astype(int))

            if np.isnan(last5):
                last5 = -1000

            if np.isnan(seasontodate):
                seasontodate = -1000

            if np.isnan(dvp1) or np.isnan(dvp2):
                dvpoa = -1000
            else:
                dvpoa = (dvp1-dvp2)/leagueavg
        else:

            pitcher = self.pitchers.get(opponent, '')
            if any([string in market for string in ['allowed', 'pitch']]):
                player_games = [game for game in self.gamelog
                                if game['playerName'] == player and game['starting pitcher']]
            else:
                player_games = [game for game in self.gamelog
                                if game['playerName'] == player and game['starting batter']]
            if not player_games:
                return np.ones(5) * -1000
            last10avg = (np.mean([game[market]
                                  for game in player_games][-10:])-line)/line
            made_line = [int(game[market] > line) for game in player_games if datetime.strptime(
                game['gameId'][:10], "%Y/%m/%d") >= self.season_start]
            last5 = np.mean(made_line[-5:])
            seasontodate = np.mean(made_line)
            if np.isnan(last5):
                last5 = -1000

            if np.isnan(seasontodate):
                seasontodate = -1000

            if any([string in market for string in ['allowed', 'pitch']]):
                headtohead = [int(game[market] > line)
                              for game in player_games if game['opponent'] == opponent]
                dvp = np.mean([game[market] for game in self.gamelog
                               if game['starting pitcher'] and game['opponent'] == opponent])
                leagueavg = np.mean(
                    [game[market] for game in self.gamelog if game['starting pitcher']])

            else:
                headtohead = [int(game[market] > line)
                              for game in player_games if game['opponent pitcher'] == pitcher]
                dvp = np.mean([game[market] for game in self.gamelog
                               if game['starting batter'] and game['opponent pitcher'] == pitcher])
                leagueavg = np.mean(
                    [game[market] for game in self.gamelog if game['starting batter']])

            if not headtohead:
                headtohead = -1000
            else:
                headtohead = np.mean(headtohead+[1, 0])
            if np.isnan(dvp):
                dvpoa = -1000
            else:
                dvpoa = (dvp-leagueavg)/leagueavg

        return np.array([last10avg, last5, seasontodate, headtohead, dvpoa])

    def get_stats_date(self, game, market, line):
        old_gamelog = self.gamelog
        try:
            i = self.gamelog.index(game)
        except:
            i = len(self.gamelog)
        if i == 0:
            return np.ones(5)*-1000
        self.gamelog = self.gamelog[:i]
        self.pitchers.update({game['opponent']: game['opponent pitcher']})
        if datetime.strptime(game['gameId'][:10], "%Y/%m/%d") < self.season_start:
            self.season_start = datetime.strptime('2022-04-07', "%Y-%m-%d")
        player = game['playerName']
        opponent = game['opponent']
        stats = self.get_stats(player, opponent, market, line)
        self.gamelog = old_gamelog
        self.season_start = datetime.strptime('2023-03-30', "%Y-%m-%d")
        self.pitchers = mlb_pitchers
        return stats

    def bucket_stats(self, market):
        playerStats = {}
        for game in self.gamelog:
            if any([string in market for string in ['allowed', 'pitch']]) and not game['starting pitcher']:
                continue
            elif not any([string in market for string in ['allowed', 'pitch']]) and not game['starting batter']:
                continue

            if not game['playerName'] in playerStats:
                playerStats[game['playerName']] = {'games': []}

            playerStats[game['playerName']]['games'].append(game[market])

        playerStats = {k: v for k, v in playerStats.items() if len(
            v['games']) > 10 and not all([g == 0 for g in v['games']])}

        averages = []
        for player, games in playerStats.items():
            playerStats[player]['avg'] = np.mean(games['games'])
            averages.append(np.mean(games['games']))

        edges = [np.percentile(averages, p) for p in range(0, 101, 20)]
        lines = np.zeros(5)
        for i in range(1, 6):
            lines[i-1] = np.round(np.mean([v for v in averages if v <=
                                  edges[i] and v >= edges[i-1]])-.5)+.5

        for player, games in playerStats.items():
            for i in range(0, 5):
                if games['avg'] >= edges[i]:
                    playerStats[player]['bucket'] = 5-i
                    playerStats[player]['line'] = lines[i]

        return playerStats

    def get_weights(self, market, force=False):
        if market in self.weights['MLB'] and not force:
            return self.weights['MLB'].get(market)

        playerStats = self.bucket_stats(market)
        results = []
        for game in tqdm(self.gamelog):
            if any([string in market for string in ['allowed', 'pitch']]) and not game['starting pitcher']:
                continue
            elif not any([string in market for string in ['allowed', 'pitch']]) and not game['starting batter']:
                continue

            player = game['playerName']
            if not player in playerStats:
                continue
            elif playerStats[player]['bucket'] < 3 and not ('allowed' in market):
                continue
            elif playerStats[player]['bucket'] > 3 and ('allowed' in market):
                continue

            line = playerStats[player]['avg']
            stats = self.get_stats_date(game, market, line)
            if all(stats == np.ones(5)*-1000):
                continue
            baseline = np.array([0, 0.5, 0.5, 0.5, 0])
            stats[stats == -1000] = baseline[stats == -1000]
            stats = stats - baseline
            y = int(game[market] > line)

            results.append({'stats': stats, 'result': y,
                            'bucket': playerStats[player]['bucket']})

        res = minimize(lambda x: -likelihood(results, x),
                       np.ones(5), method='l-bfgs-b', tol=1e-8, bounds=[(0, 10)]*5)

        w = list(res.x)
        self.weights['MLB'][market] = w
        filepath = (pkg_resources.files(data) / "weights.json")
        if os.path.isfile(filepath):
            with open(filepath, "w") as outfile:
                json.dump(self.weights, outfile, indent=4)

        return w

    def row(self, offer, date=datetime.today()):
        if type(date) is datetime:
            date = date.strftime('%Y-%m-%d')

        player = offer['Player']
        team = offer['Team']
        market = offer['Market']
        line = offer['Line']
        opponent = offer['Opponent']
        if datetime.strptime(date, '%Y-%m-%d') < datetime.today():
            if offer.get('Pitcher'):
                pitcher = offer['Pitcher']
            else:
                pitcher = next((game for game in self.gamelog if game["playerName"] == player and
                                game['gameId'][:10] == date.replace('-', '/')), None)['opponent pitcher']
        else:
            pitcher = self.pitchers[opponent]
        try:
            stats = archive['MLB'][market][date][player][line]
            moneyline = archive['MLB']['Moneyline'][date].get(team, 0.5)
            total = archive['MLB']['Totals'][date].get(team, 0)
        except:
            logger.exception("Could not find archived data")

        if np.isnan(moneyline):
            moneyline = 0.5
        if np.isnan(total):
            total = 0

        date = datetime.strptime(date, '%Y-%m-%d')
        if any([string in market for string in ['allowed', 'pitch']]):
            player_games = [game for game in self.gamelog
                            if game['playerName'] == player and game['starting pitcher']
                            and datetime.strptime(game['gameId'][:10], '%Y/%m/%d') < date]
        else:
            player_games = [game for game in self.gamelog
                            if game['playerName'] == player and game['starting batter']
                            and datetime.strptime(game['gameId'][:10], '%Y/%m/%d') < date]

        if any([string in market for string in ['allowed', 'pitch']]):
            headtohead = [
                game for game in player_games if game['opponent'] == opponent]

        else:
            headtohead = [
                game for game in player_games if game['opponent pitcher'] == pitcher]
        stats[stats == -1000] = np.nan
        baseline = np.array(
            [0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5])
        stats = stats - baseline
        if np.isnan(stats[4]):
            stats[4] = 0

        game_res = [game[market] - line for game in player_games]
        h2h_res = [game[market] - line for game in headtohead]
        if len(game_res) < 10:
            i = 10 - len(game_res)
            game_res = [0]*i + game_res
        if len(h2h_res) < 5:
            i = 5 - len(h2h_res)
            h2h_res = [0]*i + h2h_res

        data = {'Games': [game_res[-10:]], 'H2H': [h2h_res[-5:]], 'DVPOA': stats[4], 'Odds': np.nanmean(stats[5:]),
                'TeamML': moneyline-0.5, 'Total': total/10}

        data = pd.DataFrame(data)
        X = data[['DVPOA', 'Odds', 'TeamML', 'Total']].fillna(0)
        X = X.join(pd.DataFrame(data.Games.to_list()
                                ).fillna(0).add_prefix('Game '))
        X = X.join(pd.DataFrame(data.H2H.to_list()).fillna(
            0).add_prefix('Meeting '))
        return X

    def get_training_matrix(self, market):
        X = pd.DataFrame()
        results = []
        for game in self.gamelog:
            if any([string in market for string in ['allowed', 'pitch']]) and not game['starting pitcher']:
                continue
            elif not any([string in market for string in ['allowed', 'pitch']]) and not game['starting batter']:
                continue

            gameDate = datetime.strptime(game['gameId'][:10], '%Y/%m/%d')
            try:
                archiveData = archive['MLB'][market][gameDate.strftime(
                    '%Y-%m-%d')][game['playerName']]
            except:
                continue

            offer = {
                'Player': game['playerName'],
                'Team': game['team'],
                'Market': market,
                'Opponent': game['opponent'],
                'Pitcher': game['opponent pitcher']
            }

            for line, stats in archiveData.items():
                if not line == 'Closing Lines':
                    if X.empty:
                        X = self.row(offer | {'Line': line}, gameDate)
                    else:
                        X = pd.concat(
                            [X, self.row(offer | {'Line': line}, gameDate)])

                    results.append(
                        {'Over': int(game[market] > line), 'Under': int(game[market] < line)})

        y = pd.DataFrame(results)

        return X, y


class statsNHL:
    def __init__(self):
        self.skater_data = {}
        self.goalie_data = {}
        self.season_start = datetime.strptime("2022-10-07", "%Y-%m-%d")

    def load(self):
        filepath = (pkg_resources.files(data) / "nhl_skater_data.dat")
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.skater_data = pickle.load(infile)

        filepath = (pkg_resources.files(data) / "nhl_goalie_data.dat")
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.goalie_data = pickle.load(infile)

        filepath = (pkg_resources.files(data) / "weights.json")
        if os.path.isfile(filepath):
            with open(filepath, "r") as infile:
                weights = json.load(infile)

            self.weights = weights

    def update(self):
        logger.info("Getting NHL data")
        params = {
            'isAggregate': 'false',
            'isGame': 'true',
            'sort': '[{"property":"gameDate","direction":"ASC"},{"property":"playerId","direction":"ASC"}]',
            'start': 0,
            'limit': 100,
            'factCayenneExp': 'gamesPlayed>=1',
            'cayenneExp': 'gameTypeId>=2 and gameDate>=' + '"'+self.skater_data[-1]['gameDate']+'"'
        }
        nhl_request_data = scraper.get(
            'https://api.nhle.com/stats/rest/en/skater/summary', params=params)['data']
        if nhl_request_data:
            for x in self.skater_data[-300:]:
                if x['gameDate'] == self.skater_data[-1]['gameDate']:
                    self.skater_data.remove(x)
        self.skater_data = self.skater_data + [{k: v for k, v in skater.items() if k in ['assists', 'gameDate', 'gameId', 'goals', 'opponentTeamAbbrev',
                                                                                         'playerId', 'points', 'positionCode', 'shots', 'skaterFullName', 'teamAbbrev']} for skater in nhl_request_data]

        while nhl_request_data:
            params['start'] += 100
            nhl_request_data = scraper.get(
                'https://api.nhle.com/stats/rest/en/skater/summary', params=params)['data']
            if not nhl_request_data and params['start'] == 10000:
                params['start'] = 0
                params['cayenneExp'] = params['cayenneExp'][:-12] + \
                    '"'+self.skater_data[-1]['gameDate']+'"'
                nhl_request_data = scraper.get(
                    'https://api.nhle.com/stats/rest/en/skater/summary', params=params)['data']
                for x in self.skater_data[-300:]:
                    if x['gameDate'] == self.skater_data[-1]['gameDate']:
                        self.skater_data.remove(x)
            self.skater_data = self.skater_data + [{k: v for k, v in skater.items() if k in ['assists', 'gameDate', 'gameId', 'goals', 'opponentTeamAbbrev',
                                                                                             'playerId', 'points', 'positionCode', 'shots', 'skaterFullName', 'teamAbbrev']} for skater in nhl_request_data]

        for game in self.skater_data:
            if datetime.strptime(game['gameDate'], '%Y-%m-%d') < datetime.today() - timedelta(days=365):
                self.skater_data.remove(game)

        with open((pkg_resources.files(data) / "nhl_skater_data.dat"), "wb") as outfile:
            pickle.dump(self.skater_data, outfile)
        logger.info('Skater data complete')

        params = {
            'isAggregate': 'false',
            'isGame': 'true',
            'sort': '[{"property":"gameDate","direction":"ASC"},{"property":"playerId","direction":"ASC"}]',
            'start': 0,
            'limit': 100,
            'factCayenneExp': 'gamesPlayed>=1',
            'cayenneExp': 'gameTypeId>=2 and gameDate>=' + '"'+self.goalie_data[-1]['gameDate']+'"'
        }
        nhl_request_data = scraper.get(
            'https://api.nhle.com/stats/rest/en/goalie/summary', params=params)['data']
        if nhl_request_data:
            for x in self.goalie_data[-50:]:
                if x['gameDate'] == self.goalie_data[-1]['gameDate']:
                    self.goalie_data.remove(x)
        self.goalie_data = self.goalie_data + [{k: v for k, v in goalie.items() if k in ['gameDate', 'gameId', 'goalsAgainst',
                                                                                         'opponentTeamAbbrev', 'playerId', 'positionCode', 'saves', 'goalieFullName', 'teamAbbrev']} for goalie in nhl_request_data]

        while nhl_request_data:
            params['start'] += 100
            nhl_request_data = scraper.get(
                'https://api.nhle.com/stats/rest/en/goalie/summary', params=params)['data']
            if not nhl_request_data and params['start'] == 10000:
                params['start'] = 0
                params['cayenneExp'] = params['cayenneExp'][:-12] + \
                    '"'+self.goalie_data[-1]['gameDate']+'"'
                nhl_request_data = scraper.get(
                    'https://api.nhle.com/stats/rest/en/goalie/summary', params=params)['data']
                if nhl_request_data:
                    for x in self.goalie_data[-50:]:
                        if x['gameDate'] == self.goalie_data[-1]['gameDate']:
                            self.goalie_data.remove(x)
            self.goalie_data = self.goalie_data + [{k: v for k, v in goalie.items() if k in ['gameDate', 'gameId', 'goalsAgainst',
                                                                                             'opponentTeamAbbrev', 'playerId', 'positionCode', 'saves', 'goalieFullName', 'teamAbbrev']} for goalie in nhl_request_data]
        for game in self.goalie_data:
            if datetime.strptime(game['gameDate'], '%Y-%m-%d') < datetime.today() - timedelta(days=365):
                self.goalie_data.remove(game)

        with open((pkg_resources.files(data) / "nhl_goalie_data.dat"), "wb") as outfile:
            pickle.dump(self.goalie_data, outfile)
        logger.info('Goalie data complete')

    def get_stats(self, player, opponent, market, line):

        opponent = opponent.replace('NJ', 'NJD').replace('TB', 'TBL')
        if opponent == 'LA':
            opponent = 'LAK'

        if market == 'PTS':
            market = 'points'
        elif market == 'AST':
            market = 'assists'
        elif market == 'BLK':
            return np.ones(5) * -1000

        if " + " in player:
            players = player.split(" + ")
            if "/" in opponent:
                opponent = opponent.split("/")
            else:
                opponent = [opponent, opponent]

            if market in ['goalsAgainst', 'saves']:
                player1_games = [
                    game for game in self.goalie_data if players[0] == game['goalieFullName']]
                player2_games = [
                    game for game in self.goalie_data if players[1] == game['goalieFullName']]
            else:
                player1_games = [
                    game for game in self.skater_data if players[0] == game['skaterFullName']]
                player2_games = [
                    game for game in self.skater_data if players[1] == game['skaterFullName']]

            if not player1_games or not player2_games:
                return np.ones(5) * -1000

            season1 = np.array([game[market] for game in player1_games if
                                datetime.strptime(game['gameDate'], "%Y-%m-%d") >= self.season_start])
            season2 = np.array([game[market] for game in player2_games if
                                datetime.strptime(game['gameDate'], "%Y-%m-%d") >= self.season_start])

            n = min(len(season1), len(season2))
            season1 = season1[-n:]
            season2 = season2[-n:]

            h2h1 = np.array([int(game[market] > line)for game in player1_games if
                             opponent[0] == game['opponentTeamAbbrev']])
            h2h2 = np.array([int(game[market] > line)for game in player2_games if
                             opponent[1] == game['opponentTeamAbbrev']])

            n = min(len(h2h1), len(h2h2))
            if n == 0:
                headtohead = -1000
            else:
                h2h1 = h2h1[-n:]
                h2h2 = h2h2[-n:]
                headtohead = ((h2h1+h2h2) > line).astype(int)
                headtohead = np.mean(list(headtohead)+[1, 0])

            last10avg = (np.mean([game[market] for game in player1_games[-10:]]) +
                         np.mean([game[market] for game in player2_games[-10:]]) - line)/line
            n = min(len(player1_games), len(player2_games), 5)
            last5 = np.mean(((np.array([game[market] for game in player1_games[-n:]]) +
                              np.array([game[market] for game in player2_games[-n:]])) > line).astype(int))
            seasontodate = np.mean(((season1 + season2) > line).astype(int))

            if np.isnan(last5):
                last5 = -1000

            if np.isnan(seasontodate):
                seasontodate = -1000

            dvp = {}
            leagueavg = {}
            if market in ['goalsAgainst', 'saves']:
                for game in self.goalie_data:
                    id = game['gameId']
                    if id not in leagueavg:
                        leagueavg[id] = 0
                    leagueavg[id] += game[market]
                    if opponent[0] == game['opponentTeamAbbrev'] or opponent[1] == game['opponentTeamAbbrev']:
                        if id not in dvp:
                            dvp[id] = 0
                        dvp[id] += game[market]

            else:
                position1 = player1_games[0]['positionCode']
                position2 = player2_games[0]['positionCode']
                for game in self.skater_data:
                    if game['positionCode'] == position1 or game['positionCode'] == position2:
                        id = game['gameId']
                        if id not in leagueavg:
                            leagueavg[id] = 0
                        leagueavg[id] += game[market]
                        if opponent[0] == game['opponentTeamAbbrev'] or opponent[1] == game['opponentTeamAbbrev']:
                            if id not in dvp:
                                dvp[id] = 0
                            dvp[id] += game[market]

            if not dvp:
                dvpoa = -1000
            else:
                dvp = np.mean(list(dvp.values()))
                leagueavg = np.mean(list(leagueavg.values()))/2
                dvpoa = (dvp-leagueavg)/leagueavg

        elif " vs. " in player:
            players = player.split(" vs. ")
            if "/" in opponent:
                opponent = opponent.split("/")
            else:
                opponent = [opponent, opponent]

            if market in ['goalsAgainst', 'saves']:
                player1_games = [
                    game for game in self.goalie_data if players[0] == game['goalieFullName']]
                player2_games = [
                    game for game in self.goalie_data if players[1] == game['goalieFullName']]
            else:
                player1_games = [
                    game for game in self.skater_data if players[0] == game['skaterFullName']]
                player2_games = [
                    game for game in self.skater_data if players[1] == game['skaterFullName']]

            if not player1_games or not player2_games:
                return np.ones(5) * -1000

            season1 = np.array([game[market] for game in player1_games if
                                datetime.strptime(game['gameDate'], "%Y-%m-%d") >= self.season_start])
            season2 = np.array([game[market] for game in player2_games if
                                datetime.strptime(game['gameDate'], "%Y-%m-%d") >= self.season_start])

            n = min(len(season1), len(season2))
            season1 = season1[-n:]
            season2 = season2[-n:]

            h2h1 = np.array([int(game[market] > line)for game in player1_games if
                             opponent[0] == game['opponentTeamAbbrev']])
            h2h2 = np.array([int(game[market] > line)for game in player2_games if
                             opponent[1] == game['opponentTeamAbbrev']])

            n = min(len(h2h1), len(h2h2))
            if n == 0:
                headtohead = -1000
            else:
                h2h1 = h2h1[-n:]
                h2h2 = h2h2[-n:]
                headtohead = ((h2h1 + line) > h2h2).astype(int)
                headtohead = np.mean(list(headtohead)+[1, 0])

            last10avg = (np.mean([game[market] for game in player2_games][-10:]) + line -
                         np.mean([game[market] for game in player1_games][-10:]))
            n = min(len(player1_games), len(player2_games), 5)
            last5 = np.mean(((np.array([game[market] for game in player1_games[-n:]]) + line) >
                             np.array([game[market] for game in player2_games[-n:]])).astype(int))
            seasontodate = np.mean(((season1 + line) > season2).astype(int))

            if np.isnan(last5):
                last5 = -1000

            if np.isnan(seasontodate):
                seasontodate = -1000

            dvp1 = {}
            dvp2 = {}
            leagueavg1 = {}
            leagueavg2 = {}
            if market in ['goalsAgainst', 'saves']:
                for game in self.goalie_data:
                    id = game['gameId']
                    if id not in leagueavg1:
                        leagueavg1[id] = 0
                    leagueavg1[id] += game[market]
                    if id not in leagueavg2:
                        leagueavg2[id] = 0
                    leagueavg2[id] += game[market]
                    if opponent[0] == game['opponentTeamAbbrev']:
                        if id not in dvp1:
                            dvp1[id] = 0
                        dvp1[id] += game[market]
                    if opponent[1] == game['opponentTeamAbbrev']:
                        if id not in dvp2:
                            dvp2[id] = 0
                        dvp2[id] += game[market]

            else:
                position1 = player1_games[0]['positionCode']
                position2 = player2_games[0]['positionCode']
                for game in self.skater_data:
                    if game['positionCode'] == position1:
                        id = game['gameId']
                        if id not in leagueavg1:
                            leagueavg1[id] = 0
                        leagueavg1[id] += game[market]
                        if opponent[0] == game['opponentTeamAbbrev']:
                            if id not in dvp1:
                                dvp1[id] = 0
                            dvp1[id] += game[market]
                    if game['positionCode'] == position2:
                        id = game['gameId']
                        if id not in leagueavg2:
                            leagueavg2[id] = 0
                        leagueavg2[id] += game[market]
                        if opponent[1] == game['opponentTeamAbbrev']:
                            if id not in dvp2:
                                dvp2[id] = 0
                            dvp2[id] += game[market]

            if not dvp1 or not dvp2:
                dvpoa = -1000
            else:
                dvp1 = np.mean(list(dvp1.values()))
                leagueavg1 = np.mean(list(leagueavg1.values()))/2
                dvp2 = np.mean(list(dvp2.values()))
                leagueavg2 = np.mean(list(leagueavg2.values()))/2
                dvpoa = (dvp1-leagueavg1)/leagueavg1 - \
                    (dvp2-leagueavg2)/leagueavg2
        else:
            if market in ['goalsAgainst', 'saves']:
                player_games = [
                    game for game in self.goalie_data if player == game['goalieFullName']]
            else:
                player_games = [
                    game for game in self.skater_data if player == game['skaterFullName']]

            if not player_games:
                return np.ones(5) * -1000

            last10avg = (np.mean([game[market]
                                  for game in player_games][-10:])-line)/line
            made_line = [int(game[market] > line) for game in player_games if
                         datetime.strptime(game['gameDate'], "%Y-%m-%d") >= self.season_start]
            headtohead = [int(game[market] > line)for game in player_games if
                          opponent == game['opponentTeamAbbrev']]
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
                for game in self.goalie_data:
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
                for game in self.skater_data:
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

    def get_stats_date(self, game, market, line):
        old_goalie_data = self.goalie_data
        old_skater_data = self.skater_data
        if market in ['goalsAgainst', 'saves']:
            try:
                i = self.goalie_data.index(game)
            except:
                i = len(self.goalie_data)
            gameDate = game['gameDate']
            sameGame = next(
                game for game in self.skater_data if game["gameDate"] == gameDate)
            try:
                j = self.skater_data.index(sameGame)
            except:
                j = len(self.skater_data)
            if i == 0 or j == 0:
                return np.ones(5)*-1000
            self.goalie_data = self.goalie_data[:i]
            self.skater_data = self.skater_data[:j]
            player = game['goalieFullName']

        else:
            try:
                i = self.skater_data.index(game)
            except:
                i = len(self.skater_data)
            gameDate = game['gameDate']
            sameGame = next(
                game for game in self.goalie_data if game["gameDate"] == gameDate)
            try:
                j = self.goalie_data.index(sameGame)
            except:
                j = len(self.goalie_data)
            if i == 0 or j == 0:
                return np.ones(5)*-1000
            self.goalie_data = self.goalie_data[:j]
            self.skater_data = self.skater_data[:i]
            player = game['skaterFullName']

        opponent = game['opponentTeamAbbrev']
        stats = self.get_stats(player, opponent, market, line)
        self.goalie_data = old_goalie_data
        self.skater_data = old_skater_data
        return stats

    def bucket_stats(self, market):
        playerStats = {}
        if market in ['goalsAgainst', 'saves']:
            gamelog = self.goalie_data
            player = 'goalieFullName'
        else:
            gamelog = self.skater_data
            player = 'skaterFullName'

        for game in gamelog:

            if not game[player] in playerStats:
                playerStats[game[player]] = {'games': []}

            playerStats[game[player]]['games'].append(game[market])

        playerStats = {k: v for k, v in playerStats.items() if len(
            v['games']) > 10 and not all([g == 0 for g in v['games']])}

        averages = []
        for player, games in playerStats.items():
            playerStats[player]['avg'] = np.mean(games['games'])
            averages.append(np.mean(games['games']))

        edges = [np.percentile(averages, p) for p in range(0, 101, 10)]
        lines = np.zeros(10)
        for i in range(1, 11):
            lines[i-1] = np.round(np.mean([v for v in averages if v <=
                                  edges[i] and v >= edges[i-1]])-.5)+.5

        for player, games in playerStats.items():
            for i in range(0, 10):
                if games['avg'] >= edges[i]:
                    playerStats[player]['bucket'] = 10-i
                    playerStats[player]['line'] = lines[i]

        return playerStats

    def get_weights(self, market):
        if market == 'PTS':
            market = 'points'
        elif market == 'AST':
            market = 'assists'
        elif market == 'BLK':
            return np.zeros(5)

        if market in self.weights['NHL']:
            return self.weights['NHL'].get(market)

        if market in ['goalsAgainst', 'saves']:
            gamelog = self.goalie_data
            playerTag = 'goalieFullName'
        else:
            gamelog = self.skater_data
            playerTag = 'skaterFullName'

        playerStats = self.bucket_stats(market)
        results = []
        for game in tqdm(gamelog):

            player = game[playerTag]
            if not player in playerStats:
                continue
            elif playerStats[player]['bucket'] < 5:
                continue

            line = playerStats[player]['avg']
            stats = self.get_stats_date(game, market, line)
            if all(stats == np.ones(5)*-1000):
                continue
            baseline = np.array([0, 0.5, 0.5, 0.5, 0])
            stats[stats == -1000] = baseline[stats == -1000]
            stats = stats - baseline
            y = int(game[market] > line)

            results.append({'stats': stats, 'result': y,
                            'bucket': playerStats[player]['bucket']})

        res = minimize(lambda x: -likelihood(results, x),
                       np.ones(5), method='l-bfgs-b', tol=1e-8, bounds=[(0, 10)]*5)

        w = list(res.x)
        self.weights['NHL'][market] = w
        filepath = (pkg_resources.files(data) / "weights.json")
        if os.path.isfile(filepath):
            with open(filepath, "w") as outfile:
                json.dump(self.weights, outfile, indent=4)

        return w

    def get_training_matrix(self, market):
        X = pd.DataFrame()
        results = []
        for game in self.gamelog:
            if any([string in market for string in ['allowed', 'pitch']]) and not game['starting pitcher']:
                continue
            elif not any([string in market for string in ['allowed', 'pitch']]) and not game['starting batter']:
                continue

            gameDate = datetime.strptime(game['gameId'][:10], '%Y/%m/%d')
            try:
                archiveData = archive['MLB'][market][gameDate.strftime(
                    '%Y-%m-%d')][game['playerName']]
            except:
                continue

            offer = {
                'Player': game['playerName'],
                'Team': game['team'],
                'Market': market,
                'Opponent': game['opponent'],
                'Pitcher': game['opponent pitcher']
            }

            for line, stats in archiveData.items():
                if not line == 'Closing Lines':
                    if X.empty:
                        X = self.row(offer | {'Line': line}, gameDate)
                    else:
                        X = pd.concat(
                            [X, self.row(offer | {'Line': line}, gameDate)])

                    results.append(
                        {'Over': int(game[market] > line), 'Under': int(game[market] < line)})

        y = pd.DataFrame(results)

        return X, y
