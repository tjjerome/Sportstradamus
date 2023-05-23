import os.path
import numpy as np
import datetime
import logging
import pickle
import json
import importlib.resources as pkg_resources
from . import data
from tqdm import tqdm
import statsapi as mlb
from scipy.stats import norm
from scipy.optimize import minimize
import nba_api.stats.endpoints as nba
from nba_api.stats.static import players as nba_static
import nfl_data_py as nfl
from time import sleep
from sportsbook_spider.helpers import scraper, mlb_pitchers, likelihood

logger = logging.getLogger(__name__)


class statsNBA:
    def __init__(self):
        self.gamelog = []
        self.players = {}

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
            try:
                player_ids = [nba_static.find_players_by_full_name(
                    player)[0]['id'] for player in players]
            except:
                return np.ones(5)*-1000

            player1_games = [
                game for game in self.gamelog if game['PLAYER_ID'] == player_ids[0]]
            player2_games = [
                game for game in self.gamelog if game['PLAYER_ID'] == player_ids[1]]

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
                game for game in self.gamelog if game['PLAYER_ID'] == player_ids[0]]
            player2_games = [
                game for game in self.gamelog if game['PLAYER_ID'] == player_ids[1]]

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
            for position in positions:
                for game in self.gamelog:
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


class statsMLB:
    def __init__(self):
        self.gamelog = []
        self.gameIds = []
        self.season_start = datetime.datetime.strptime(
            '2023-03-30', "%Y-%m-%d")
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
        mlb_game_ids = mlb.schedule(start_date=mlb.latest_season(
        )['regularSeasonStartDate'], end_date=datetime.date.today())
        mlb_game_ids = [game['game_id'] for game in mlb_game_ids if game['status']
                        == 'Final' and game['game_type'] != 'E' and game['game_type'] != 'S']

        for id in tqdm(mlb_game_ids):
            if id not in self.gameIds:
                self.parse_game(id)

        for game in self.gamelog:
            if datetime.datetime.strptime(game['gameId'][:10], '%Y/%m/%d') < datetime.datetime.today() - datetime.timedelta(days=365):
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
            last10avg = (np.mean([game[market] for game in player1_games][-10:]) +
                         np.mean([game[market] for game in player2_games][-10:]) - line)/line
            last5 = -1000
            seasontodate = -1000
            headtohead = -1000
            dvpoa = -1000
        elif " vs. " in player:
            players = player.split(" vs. ")

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
            last10avg = (np.mean([game[market] for game in player2_games][-10:]) + line -
                         np.mean([game[market] for game in player1_games][-10:]))
            last5 = -1000
            seasontodate = -1000
            headtohead = -1000
            dvpoa = -1000
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
            made_line = [int(game[market] > line) for game in player_games if datetime.datetime.strptime(
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
        if datetime.datetime.strptime(game['gameId'][:10], "%Y/%m/%d") < self.season_start:
            self.season_start = datetime.datetime.strptime(
                '2022-04-07', "%Y-%m-%d")
        player = game['playerName']
        opponent = game['opponent']
        stats = self.get_stats(player, opponent, market, line)
        self.gamelog = old_gamelog
        self.season_start = datetime.datetime.strptime(
            '2023-03-30', "%Y-%m-%d")
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

    def get_weights(self, market):
        if market in self.weights['MLB']:
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
            elif playerStats[player]['bucket'] < 3:
                continue

            line = playerStats[player]['line']
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


class statsNHL:
    def __init__(self):
        self.skater_data = {}
        self.goalie_data = {}

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
            if datetime.datetime.strptime(game['gameDate'], '%Y-%m-%d') < datetime.datetime.today() - datetime.timedelta(days=365):
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
            if datetime.datetime.strptime(game['gameDate'], '%Y-%m-%d') < datetime.datetime.today() - datetime.timedelta(days=365):
                self.goalie_data.remove(game)

        with open((pkg_resources.files(data) / "nhl_goalie_data.dat"), "wb") as outfile:
            pickle.dump(self.goalie_data, outfile)
        logger.info('Goalie data complete')

    def get_stats(self, player, opponent, market, line):

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

            last10avg = (np.mean([game[market] for game in player2_games][-10:]) + line -
                         np.mean([game[market] for game in player1_games][-10:]))
            last5 = -1000
            seasontodate = -1000
            headtohead = -1000
            dvpoa = -1000
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
