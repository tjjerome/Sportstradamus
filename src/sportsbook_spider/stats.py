import os.path
import numpy as np
import datetime
import traceback
import pickle
from tqdm import tqdm
import statsapi as mlb
import nba_api.stats.endpoints as nba
from nba_api.stats.static import players as nba_static
import nfl_data_py as nfl
from time import sleep
from sportsbook_spider.helpers import scraper, mlb_pitchers


class statsNBA:
    def __init__(self):
        self.gamelog = []
        self.players = {}

    def load(self):
        filepath = "./data/nba_players.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.players = pickle.load(infile)
                self.players = {int(k): v for k, v in self.players.items()}

    def update(self):
        print("Getting NBA stats")
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

        with open("./data/nba_players.dat", "wb") as outfile:
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
            if player_id:
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


class statsMLB:
    def __init__(self):
        self.gamelog = []
        self.gameIds = []

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
        filepath = "./data/mlb_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                mlb_data = pickle.load(infile)

            self.gamelog = mlb_data['gamelog']
            self.gameIds = mlb_data['games']

    def update(self):
        print("Getting MLB Stats")
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

        with open("./data/mlb_data.dat", "wb") as outfile:
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
            season_start = datetime.datetime.strptime(
                mlb.latest_season()['regularSeasonStartDate'], "%Y-%m-%d")
            pitcher = mlb_pitchers.get(opponent, '')
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
            if not dvp:
                dvpoa = -1000
            else:
                dvpoa = (dvp-leagueavg)/leagueavg

        return np.array([last10avg, last5, seasontodate, headtohead, dvpoa])


class statsNHL:
    def __init__(self):
        self.skater_data = {}
        self.goalie_data = {}

    def load(self):
        filepath = "./data/nhl_skater_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.skater_data = pickle.load(infile)

        filepath = "./data/nhl_goalie_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.goalie_data = pickle.load(infile)

    def update(self):
        print("Getting NHL data")
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

        with open("./data/nhl_skater_data.dat", "wb") as outfile:
            pickle.dump(self.skater_data, outfile)
        print('Skater data complete')

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

        with open("./data/nhl_goalie_data.dat", "wb") as outfile:
            pickle.dump(self.goalie_data, outfile)
        print('Goalie data complete')

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
