from sportstradamus.spiderLogger import logger
import os.path
import numpy as np
from datetime import datetime, timedelta
import pickle
import json
import importlib.resources as pkg_resources
from sportstradamus import data
from tqdm import tqdm
import statsapi as mlb
import nba_api.stats.endpoints as nba
import nfl_data_py as nfl
from scipy.stats import iqr
from time import sleep
from sportstradamus.helpers import scraper, mlb_pitchers, archive, abbreviations, remove_accents, hmean
import pandas as pd
import warnings
import requests


class Stats:
    """
    A parent class for handling and analyzing sports statistics.

    Attributes:
        gamelog (list): A list of dictionaries representing game logs.
        archive (dict): A dictionary containing archived statistics.
        players (dict): A dictionary containing player specific data.
        season_start (datetime): The start date of the  season.
        playerStats (dict): Dictionary containing player statistics.
        edges (list): List of edges.
        dvp_index (dict): Dictionary containing DVP index data.

    Methods:
        parse_game(game): Parses a game and updates the gamelog.
        load(): Loads game logs from a file.
        update(): Updates the gamelog with new game data.
        bucket_stats(market): Groups statistics into buckets based on market type.
        get_stats(offer, game_date): Retrieves statistics for a given offer and game date.
        get_training_matrix(market): Retrieves the training data matrix and target labels for a specified market.
    """

    def __init__(self):
        self.gamelog = pd.DataFrame()
        self.teamlog = pd.DataFrame()
        self.archive = {}
        self.players = {}
        self.season_start = datetime(year=1900, month=1, day=1).date()
        self.playerStats = {}
        self.edges = {}
        self.dvp_index = {}
        self.dvpoa_latest_date = datetime(year=1900, month=1, day=1).date()
        self.bucket_latest_date = datetime(year=1900, month=1, day=1).date()
        self.bucket_market = ""
        self.profile_latest_date = datetime(year=1900, month=1, day=1).date()
        self.profiled_market = ""
        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.upcoming_games = {}

    def parse_game(self, game):
        """
        Parses a game and updates the gamelog.

        Args:
            game (dict): A dictionary representing a game.

        Returns:
            None
        """
        # Implementation details...

    def load(self):
        """
        Loads game logs from a file.

        Args:
            file_path (str): The path to the file containing game logs.

        Returns:
            None
        """
        # Implementation details...

    def update(self):
        """
        Updates the gamelog with new game data.

        Args:
            None

        Returns:
            None
        """
        # Implementation details...

    def bucket_stats(self, market, date):
        """
        Groups statistics into buckets based on market type.

        Args:
            market (str): The market type.

        Returns:
            None
        """
        # Implementation details...

    def get_stats(self, offer, game_date):
        """
        Retrieves statistics for a given offer and game date.

        Args:
            offer (dict): A dictionary containing the offer details.
            game_date (datetime.datetime): The date of the game.

        Returns:
            stats (pd.DataFrame): A DataFrame containing the retrieved statistics.
        """
        # Implementation details...

    def get_training_matrix(self, market):
        """
        Retrieves the training data matrix and target labels for a specified market.

        Args:
            market (str): The market type to retrieve training data for.

        Returns:
            X (pd.DataFrame): The training data matrix.
            y (pd.DataFrame): The target labels.
        """
        # Implementation details...


class StatsNBA(Stats):
    """
    A class for handling and analyzing NBA statistics.
    Inherits from the Stats parent class.

    Additional Attributes:
        None

    Additional Methods:
        None
    """

    def __init__(self):
        """
        Initialize the StatsNBA class.
        """
        super().__init__()
        self.season_start = datetime.strptime("2023-10-24", "%Y-%m-%d").date()
        self.season = "2023-24"
        cols = ['SEASON_YEAR', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE',
                'WL', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB',
                'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                'PLUS_MINUS', 'POS', 'HOME', 'OPP', 'PRA', 'PR', 'PA', 'RA', 'BLST',
                'fantasy points prizepicks', 'fantasy points underdog', 'fantasy points parlay',
                'OFF_RATING', 'DEF_RATING', 'AST_PCT', 'OREB_PCT', 'DREB_PCT', 'REB_PCT',
                'EFG_PCT', 'TS_PCT', 'USG_PCT', 'PIE']
        self.gamelog = pd.DataFrame(columns=cols)
        team_cols = ['SEASON_YEAR', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'OPP',
                     'OFF_RATING', 'DEF_RATING', 'EFG_PCT', 'OREB_PCT', 'DREB_PCT',
                     'TM_TOV_PCT', 'PACE', 'OPP_OFF_RATING', 'OPP_DEF_RATING',
                     'OPP_EFG_PCT', 'OPP_OREB_PCT', 'OPP_DREB_PCT',
                     'OPP_TM_TOV_PCT', 'OPP_PACE']
        self.teamlog = pd.DataFrame(columns=team_cols)

    def load(self):
        """
        Load data from files.
        """
        filepath = pkg_resources.files(data) / "nba_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                nba_data = pickle.load(infile)
                self.players = nba_data['players']
                self.gamelog = nba_data['gamelog']
                self.teamlog = nba_data['teamlog']

    def update(self):
        """
        Update data from the web API.
        """
        # Fetch regular season game logs
        latest_date = self.season_start
        if not self.gamelog.empty:
            latest_date = datetime.strptime(
                self.gamelog["GAME_DATE"].max().split("T")[0], "%Y-%m-%d") + timedelta(days=1)
        today = datetime.today().date()

        self.upcoming_games = {}

        try:
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=00&Season={self.season[:4]}&RegionID=1&Date={today.strftime('%m/%d/%Y')}&EST=Y"

            ug_res = scraper.get(ug_url)['resultSets'][1]["CompleteGameList"]

            next_day = today + timedelta(days=1)
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=00&Season={self.season[:4]}&RegionID=1&Date={next_day.strftime('%m/%d/%Y')}&EST=Y"

            ug_res.extend(scraper.get(ug_url)[
                          'resultSets'][1]["CompleteGameList"])

            next_day = next_day + timedelta(days=1)
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=00&Season={self.season[:4]}&RegionID=1&Date={next_day.strftime('%m/%d/%Y')}&EST=Y"

            ug_res.extend(scraper.get(ug_url)[
                          'resultSets'][1]["CompleteGameList"])

            for game in ug_res:
                if game["vtAbbreviation"] not in self.upcoming_games:
                    self.upcoming_games[game["vtAbbreviation"]] = {
                        "Opponent": game["htAbbreviation"],
                        "Home": 0
                    }
                if game["htAbbreviation"] not in self.upcoming_games:
                    self.upcoming_games[game["htAbbreviation"]] = {
                        "Opponent": game["vtAbbreviation"],
                        "Home": 1
                    }

        except:
            pass

        params = {
            "season_nullable": self.season,
            "date_from_nullable": latest_date.strftime("%m/%d/%Y"),
            "date_to_nullable": today.strftime("%m/%d/%Y")
        }
        nba_gamelog = nba.playergamelogs.PlayerGameLogs(
            **params).get_normalized_dict()["PlayerGameLogs"]
        adv_gamelog = nba.playergamelogs.PlayerGameLogs(
            **(params | {"measure_type_player_game_logs_nullable": "Advanced"})).get_normalized_dict()["PlayerGameLogs"]
        teamlog = nba.teamgamelogs.TeamGameLogs(
            **(params | {"measure_type_player_game_logs_nullable": "Advanced"})).get_normalized_dict()["TeamGameLogs"]

        # Fetch playoffs game logs
        if today.month == 4 or True:
            params.update({'season_type_nullable': "PlayIn"})
            nba_gamelog.extend(nba.playergamelogs.PlayerGameLogs(
                **params).get_normalized_dict()["PlayerGameLogs"])
            adv_gamelog.extend(nba.playergamelogs.PlayerGameLogs(
                **(params | {"measure_type_player_game_logs_nullable": "Advanced"})).get_normalized_dict()["PlayerGameLogs"])
            teamlog.extend(nba.teamgamelogs.TeamGameLogs(
                **(params | {"measure_type_player_game_logs_nullable": "Advanced"})).get_normalized_dict()["TeamGameLogs"])
        if 4 <= today.month <= 6 or True:
            params.update({'season_type_nullable': "Playoffs"})
            nba_gamelog.extend(nba.playergamelogs.PlayerGameLogs(
                **params).get_normalized_dict()["PlayerGameLogs"])
            adv_gamelog.extend(nba.playergamelogs.PlayerGameLogs(
                **(params | {"measure_type_player_game_logs_nullable": "Advanced"})).get_normalized_dict()["PlayerGameLogs"])
            teamlog.extend(nba.teamgamelogs.TeamGameLogs(
                **(params | {"measure_type_player_game_logs_nullable": "Advanced"})).get_normalized_dict()["TeamGameLogs"])

        nba_gamelog.sort(key=lambda x: (x['GAME_ID'], x['PLAYER_ID']))
        adv_gamelog.sort(key=lambda x: (x['GAME_ID'], x['PLAYER_ID']))
        teamlog.sort(key=lambda x: (x['GAME_ID'], x['TEAM_ID']))

        team_df = []
        for team1, team2 in tqdm(zip(*[iter(teamlog)]*2), desc="Getting NBA stats", unit='team'):
            team1.update({"OPP_"+k: v for k, v in team2.items()
                         if "OPP_"+k in self.teamlog.columns})
            team1["OPP"] = team2['TEAM_ABBREVIATION']
            team2.update({"OPP_"+k: v for k, v in team1.items()
                         if "OPP_"+k in self.teamlog.columns})
            team2["OPP"] = team1['TEAM_ABBREVIATION']
            team_df.append(team1)
            team_df.append(team2)

        team_df = pd.DataFrame(team_df)

        if not team_df.empty:
            self.teamlog = pd.concat(
                [team_df[self.teamlog.columns], self.teamlog]).sort_values("GAME_DATE").reset_index(drop=True)

        # Process each game
        nba_df = []
        for i, game in enumerate(tqdm(nba_gamelog, desc="Getting NBA stats", unit='player')):
            if game["MIN"] < 12:
                continue

            player_id = game["PLAYER_ID"]

            if player_id not in self.players:
                # Fetch player information if not already present
                self.players[player_id] = nba.commonplayerinfo.CommonPlayerInfo(
                    player_id=player_id
                ).get_normalized_dict()["CommonPlayerInfo"][0]
                sleep(0.5)

            # Extract additional game information
            game["PLAYER_NAME"] = remove_accents(game["PLAYER_NAME"])
            game["POS"] = self.players[player_id].get("POSITION").replace(
                "Center-Forward", "Forward-Center").replace("Forward-Guard", "Guard-Forward")
            game["HOME"] = "vs." in game["MATCHUP"]
            teams = game["MATCHUP"].replace("vs.", "@").split(" @ ")
            for team in teams:
                if team != game["TEAM_ABBREVIATION"]:
                    game["OPP"] = team

            # Compute derived stats
            game["PRA"] = game["PTS"] + game["REB"] + game["AST"]
            game["PR"] = game["PTS"] + game["REB"]
            game["PA"] = game["PTS"] + game["AST"]
            game["RA"] = game["REB"] + game["AST"]
            game["BLST"] = game["BLK"] + game["STL"]
            game["fantasy points prizepicks"] = game["PTS"] + game["REB"] * \
                1.2 + game["AST"]*1.5 + game["BLST"]*3 - game["TOV"]
            game["fantasy points underdog"] = game["PTS"] + game["REB"] * \
                1.2 + game["AST"]*1.5 + game["BLST"]*3 - game["TOV"]
            game["fantasy points parlay"] = game["PRA"] + \
                game["BLST"]*2 - game["TOV"]

            # game.update(
            #     {k: v for k, v in adv_gamelog[i].items() if k in adv_stats})
            game.update(adv_gamelog[i])

            nba_df.append(game)

        nba_df = pd.DataFrame(nba_df)

        if not nba_df.empty:
            self.gamelog = pd.concat(
                [nba_df[self.gamelog.columns], self.gamelog]).sort_values("GAME_DATE").reset_index(drop=True)

        # Remove old games to prevent file bloat
        two_years_ago = today - timedelta(days=730)
        self.gamelog = self.gamelog[pd.to_datetime(
            self.gamelog["GAME_DATE"]).dt.date >= two_years_ago]
        self.gamelog.drop_duplicates(inplace=True)
        self.teamlog = self.teamlog[pd.to_datetime(
            self.teamlog["GAME_DATE"]).dt.date >= two_years_ago]
        self.teamlog.drop_duplicates(inplace=True)

        self.gamelog.loc[self.gamelog['TEAM_ABBREVIATION']
                         == 'UTAH', 'TEAM_ABBREVIATION'] = "UTA"
        self.gamelog.loc[self.gamelog['OPP']
                         == 'UTAH', 'OPP'] = "UTA"
        self.gamelog.loc[self.gamelog['TEAM_ABBREVIATION']
                         == 'NOP', 'TEAM_ABBREVIATION'] = "NO"
        self.gamelog.loc[self.gamelog['OPP']
                         == 'NOP', 'OPP'] = "NO"
        self.gamelog.loc[self.gamelog['TEAM_ABBREVIATION']
                         == 'GS', 'TEAM_ABBREVIATION'] = "GSW"
        self.gamelog.loc[self.gamelog['OPP']
                         == 'GS', 'OPP'] = "GSW"
        self.gamelog.loc[self.gamelog['TEAM_ABBREVIATION']
                         == 'NY', 'TEAM_ABBREVIATION'] = "NYK"
        self.gamelog.loc[self.gamelog['OPP']
                         == 'NY', 'OPP'] = "NYK"
        self.gamelog.loc[self.gamelog['TEAM_ABBREVIATION']
                         == 'SA', 'TEAM_ABBREVIATION'] = "SAS"
        self.gamelog.loc[self.gamelog['OPP']
                         == 'SA', 'OPP'] = "SAS"

        self.teamlog.loc[self.teamlog['TEAM_ABBREVIATION']
                         == 'UTAH', 'TEAM_ABBREVIATION'] = "UTA"
        self.teamlog.loc[self.teamlog['OPP']
                         == 'UTAH', 'OPP'] = "UTA"
        self.teamlog.loc[self.teamlog['TEAM_ABBREVIATION']
                         == 'NOP', 'TEAM_ABBREVIATION'] = "NO"
        self.teamlog.loc[self.teamlog['OPP']
                         == 'NOP', 'OPP'] = "NO"
        self.teamlog.loc[self.teamlog['TEAM_ABBREVIATION']
                         == 'GS', 'TEAM_ABBREVIATION'] = "GSW"
        self.teamlog.loc[self.teamlog['OPP']
                         == 'GS', 'OPP'] = "GSW"
        self.teamlog.loc[self.teamlog['TEAM_ABBREVIATION']
                         == 'NY', 'TEAM_ABBREVIATION'] = "NYK"
        self.teamlog.loc[self.teamlog['OPP']
                         == 'NY', 'OPP'] = "NYK"
        self.teamlog.loc[self.teamlog['TEAM_ABBREVIATION']
                         == 'SA', 'TEAM_ABBREVIATION'] = "SAS"
        self.teamlog.loc[self.teamlog['OPP']
                         == 'SA', 'OPP'] = "SAS"

        # Save the updated player data
        with open(pkg_resources.files(data) / "nba_data.dat", "wb") as outfile:
            pickle.dump({"players": self.players,
                         "gamelog": self.gamelog,
                         "teamlog": self.teamlog}, outfile)

    def bucket_stats(self, market, buckets=20, date=datetime.today()):
        """
        Bucket player stats based on a given market.

        Args:
            market (str): The market to bucket the player stats (e.g., 'PTS', 'REB', 'AST').
            buckets (int): The number of buckets to divide the stats into (default: 20).

        Returns:
            None.
        """
        if market == self.bucket_market and date == self.bucket_latest_date:
            return

        self.bucket_market = market
        self.bucket_latest_date = date

        # Reset playerStats and edges
        self.playerStats = {}
        self.edges = []

        # Collect stats for each player
        for game in self.gamelog:
            if datetime.strptime(game['GAME_DATE'][:10], '%Y-%m-%d').date() > date.date():
                continue
            elif datetime.strptime(game['GAME_DATE'][:10], '%Y-%m-%d').date() < (date - timedelta(days=300)).date():
                continue
            player_name = game["PLAYER_NAME"]

            if player_name not in self.playerStats:
                self.playerStats[player_name] = {"games": []}

            self.playerStats[player_name]["games"].append(game[market])

        # Filter players based on minimum games played and non-zero stats
        self.playerStats = {
            player: stats
            for player, stats in self.playerStats.items()
            if len(stats["games"]) > 10 and not all(g == 0 for g in stats["games"])
        }

        # Compute averages and percentiles
        averages = []
        for player, games in self.playerStats.items():
            self.playerStats[player]["avg"] = (
                np.mean(games["games"]) if games["games"] else 0
            )
            averages.append(self.playerStats[player]["avg"])

        if not len(averages):
            return

        # Compute edges for each bucket
        w = int(100 / buckets)
        self.edges = [np.percentile(averages, p) for p in range(0, 101, w)]
        lines = np.zeros(buckets)
        for i in range(1, buckets + 1):
            lines[i - 1] = (
                np.round(
                    np.mean(
                        [v for v in averages if self.edges[i - 1]
                            <= v <= self.edges[i]]
                    )
                    - 0.5
                )
                + 0.5
            )

        # Assign bucket and line values to each player
        for player, games in self.playerStats.items():
            for i in range(buckets):
                if games["avg"] >= self.edges[i]:
                    self.playerStats[player]["bucket"] = buckets - i
                    self.playerStats[player]["line"] = lines[i]

    def profile_market(self, market, date=datetime.today().date()):
        if type(date) is str:
            date = datetime.strptime(date, "%Y-%m-%d").date()
        if type(date) is datetime:
            date = date.date()
        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.profiled_market = market
        self.profile_latest_date = date

        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.defenseProfile = pd.DataFrame(columns=['avg', 'home', 'away'])

        one_year_ago = date - timedelta(days=300)
        gameDates = pd.to_datetime(self.gamelog["GAME_DATE"]).dt.date
        gamelog = self.gamelog[(one_year_ago <= gameDates)
                               & (gameDates < date)].copy()
        gameDates = pd.to_datetime(self.teamlog["GAME_DATE"]).dt.date
        teamlog = self.teamlog[(one_year_ago <= gameDates)
                               & (gameDates < date)].copy()

        # Retrieve moneyline and totals data from archive
        k = [[(date, team) for team in teams.keys()]
             for date, teams in archive["NBA"]["Moneyline"].items()]
        k = [item for row in k for item in row]
        flat_money = {t: archive["NBA"]["Moneyline"].get(
            t[0], {}).get(t[1], 0.5) for t in k}
        flat_total = {t: archive["NBA"]["Totals"].get(
            t[0], {}).get(t[1], 110) for t in k}
        tup_s = pd.Series(
            zip(gamelog['GAME_DATE'].str[:10], gamelog['TEAM_ABBREVIATION']), index=gamelog.index)
        gamelog.loc[:, "moneyline"] = tup_s.map(flat_money)
        gamelog.loc[:, "totals"] = tup_s.map(flat_total)

        teamstats = teamlog.groupby('TEAM_ABBREVIATION').tail(10).groupby('TEAM_ABBREVIATION')[
            teamlog.columns[6:]].mean()

        playerGroups = gamelog.\
            groupby('PLAYER_NAME').\
            filter(lambda x: (x[market].clip(0, 1).mean() > 0.3) & (x[market].count() > 1)).\
            groupby('PLAYER_NAME')

        defenseGroups = gamelog.groupby(['OPP', 'GAME_ID'])
        defenseGames = pd.DataFrame()
        defenseGames[market] = defenseGroups[market].sum()
        defenseGames['HOME'] = defenseGroups['HOME'].mean().astype(int)
        defenseGames['moneyline'] = defenseGroups['moneyline'].mean()
        defenseGames['totals'] = defenseGroups['totals'].mean()
        defenseGroups = defenseGames.groupby('OPP')

        leagueavg = playerGroups[market].mean().mean()
        if np.isnan(leagueavg):
            return

        self.playerProfile['avg'] = playerGroups[market].mean().div(
            leagueavg) - 1
        self.playerProfile['home'] = playerGroups.apply(
            lambda x: x.loc[x['HOME'], market].mean() / x[market].mean()) - 1
        self.playerProfile['away'] = playerGroups.apply(
            lambda x: x.loc[~x['HOME'].astype(bool), market].mean()/x[market].mean())-1

        leagueavg = defenseGroups[market].mean().mean()
        self.defenseProfile['avg'] = defenseGroups[market].mean().div(
            leagueavg) - 1
        self.defenseProfile['home'] = defenseGroups.apply(
            lambda x: x.loc[x['HOME'] == 1, market].mean() / x[market].mean()) - 1
        self.defenseProfile['away'] = defenseGroups.apply(
            lambda x: x.loc[x['HOME'] == 0, market].mean()/x[market].mean())-1

        stat_types = ['PLUS_MINUS', 'PFD', 'OFF_RATING', 'DEF_RATING', 'AST_PCT', 'OREB_PCT',
                      'DREB_PCT', 'REB_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT', 'PIE']
        playerstats = gamelog.fillna(0).groupby('PLAYER_NAME')[
            stat_types].mean(numeric_only=True)
        positions = ['Guard', 'Forward', 'Center',
                     'Guard-Forward', 'Forward-Center']
        for position in positions:
            positionGroups = gamelog.loc[gamelog['POS'] == position].groupby(
                ['OPP', 'GAME_ID'])
            defenseGames = pd.DataFrame()
            defenseGames[market] = positionGroups[market].sum()
            defenseGames['HOME'] = positionGroups['HOME'].mean().astype(int)
            defenseGames['moneyline'] = positionGroups['moneyline'].mean()
            defenseGames['totals'] = positionGroups['totals'].mean()
            positionGroups = defenseGames.groupby('OPP')
            leagueavg = positionGroups[market].mean().mean()
            if leagueavg == 0:
                self.defenseProfile[position] = 0
            else:
                self.defenseProfile[position] = positionGroups[market].mean().div(
                    leagueavg) - 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.playerProfile['moneyline gain'] = playerGroups.\
                apply(lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                           x[market].values.astype(float)/x[market].mean() - 1, 1)[0])

            self.playerProfile['totals gain'] = playerGroups.\
                apply(lambda x: np.polyfit(x.totals.fillna(110).values.astype(float) / 110 - x.totals.fillna(110).mean(),
                                           x[market].values.astype(float)/x[market].mean() - 1, 1)[0])

            self.defenseProfile['moneyline gain'] = defenseGroups.\
                apply(lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                           x[market].values.astype(float)/x[market].mean() - 1, 1)[0])

            self.defenseProfile['totals gain'] = defenseGroups.\
                apply(lambda x: np.polyfit(x.totals.fillna(110).values.astype(float) / 110 - x.totals.fillna(110).mean(),
                                           x[market].values.astype(float)/x[market].mean() - 1, 1)[0])

        team_stat_types = ['OFF_RATING', 'DEF_RATING', 'EFG_PCT', 'OREB_PCT', 'DREB_PCT',
                           'TM_TOV_PCT', 'PACE', 'OPP_OFF_RATING', 'OPP_DEF_RATING',
                           'OPP_EFG_PCT', 'OPP_OREB_PCT', 'OPP_DREB_PCT',
                           'OPP_TM_TOV_PCT']
        i = self.defenseProfile.index
        self.defenseProfile = self.defenseProfile.merge(
            teamstats[team_stat_types], left_on='OPP', right_on='TEAM_ABBREVIATION')
        self.defenseProfile.index = i

        self.teamProfile = teamstats[team_stat_types]

        self.playerProfile = self.playerProfile.merge(
            playerstats[stat_types], on='PLAYER_NAME')

    def dvpoa(self, team, position, market, date=datetime.today().date()):
        """
        Calculate the Defense Versus Position Above Average (DVPOA) for a specific team, position, and market.

        Args:
            team (str): The team abbreviation.
            position (str): The player's position.
            market (str): The market to calculate performance against (e.g., 'PTS', 'REB', 'AST').

        Returns:
            float: The calculated performance value.
        """
        if date != self.dvpoa_latest_date:
            self.dvp_index = {}
            self.dvpoa_latest_date = date

        if market not in self.dvp_index:
            self.dvp_index[market] = {}

        if team not in self.dvp_index[market]:
            self.dvp_index[market][team] = {}

        if position in self.dvp_index[market][team]:
            return self.dvp_index[market][team][position]

        dvp = {}
        leagueavg = {}

        for game in self.gamelog:
            if datetime.strptime(game['GAME_DATE'][:10], '%Y-%m-%d').date() > date.date():
                continue
            elif datetime.strptime(game['GAME_DATE'][:10], '%Y-%m-%d').date() < (date - timedelta(days=300)).date():
                continue
            if game["POS"] == position or game["POS"] == "-".join(
                position.split("-")[::-1]
            ):
                game_id = game["GAME_ID"]

                if game_id not in leagueavg:
                    leagueavg[game_id] = 0

                leagueavg[game_id] += game[market]

                if game["OPP"] == team:
                    if game_id not in dvp:
                        dvp[game_id] = 0

                    dvp[game_id] += game[market]

        if not dvp:
            return 0
        else:
            dvp = np.mean(list(dvp.values()))
            leagueavg = np.mean(list(leagueavg.values())) / 2
            dvpoa = (dvp - leagueavg) / leagueavg
            dvpoa = np.nan_to_num(dvpoa, nan=0.0, posinf=0.0, neginf=0.0)
            self.dvp_index[market][team][position] = dvpoa
            return dvpoa

    def get_stats(self, offer, date=datetime.today()):
        """
        Generate a pandas DataFrame with a summary of relevant stats.

        Args:
            offer (dict): The offer details containing 'Player', 'Team', 'Market', 'Line', and 'Opponent'.
            date (datetime or str): The date of the stats (default: today's date).

        Returns:
            pandas.DataFrame: The generated DataFrame with the summary of stats.
        """
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        player = offer["Player"]
        team = offer["Team"]
        market = offer["Market"]
        line = offer["Line"]
        opponent = offer["Opponent"]
        if self.defenseProfile.empty:
            logger.exception(f"{market} not profiled")
            return 0
        home = offer.get("Home")
        if home is None:
            home = self.upcoming_games.get(team, {}).get("Home", 0)

        if player not in self.playerProfile.index:
            self.playerProfile.loc[player] = np.zeros_like(
                self.playerProfile.columns)

        try:
            stats = (
                archive["NBA"]
                .get(market, {})
                .get(date, {})
                .get(player, {})
                .get(line, [0] * 4)
            )
            moneyline = archive["NBA"]["Moneyline"].get(
                date, {}).get(team, 0)
            total = archive["NBA"]["Totals"].get(date, {}).get(team, 0)

        except:
            logger.exception(f"{player}, {market}")
            return 0

        date = datetime.strptime(date, "%Y-%m-%d")

        player_games = self.gamelog.loc[(self.gamelog["PLAYER_NAME"] == player) & (
            pd.to_datetime(self.gamelog["GAME_DATE"]) < date)]

        if len(player_games) > 0:
            position = player_games.iloc[0]['POS']
        else:
            logger.warning(f"{player} not found")
            return 0

        one_year_ago = len(player_games.loc[pd.to_datetime(
            self.gamelog["GAME_DATE"]) > date-timedelta(days=300)])
        headtohead = player_games.loc[player_games["OPP"] == opponent]

        game_res = (player_games[market]).to_list()
        h2h_res = (headtohead[market]).to_list()

        dvpoa = self.defenseProfile.loc[opponent, position]

        stats = np.array(stats, dtype=np.float64)
        odds = np.nanmean(stats)
        if np.isnan(odds):
            odds = 0

        positions = ['Guard', 'Forward', 'Center',
                     'Guard-Forward', 'Forward-Center']
        data = {
            "DVPOA": dvpoa,
            "Odds": odds,
            "Line": line,
            "Avg5": np.median(game_res[-5:]) if game_res else 0,
            "Avg10": np.median(game_res[-10:]) if game_res else 0,
            "AvgYr": np.median(game_res[-one_year_ago:]) if game_res else 0,
            "AvgH2H": np.median(h2h_res[-5:]) if h2h_res else 0,
            "IQR10": iqr(game_res[-10:]) if game_res else 0,
            "IQRYr": iqr(game_res[-one_year_ago:]) if game_res else 0,
            "Mean5": np.mean(game_res[-5:]) if game_res else 0,
            "Mean10": np.mean(game_res[-10:]) if game_res else 0,
            "MeanYr": np.mean(game_res[-one_year_ago:]) if game_res else 0,
            "MeanH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
            "GamesPlayed": one_year_ago,
            "Moneyline": moneyline,
            "Total": total,
            "Home": home,
            "Position": positions.index(position)
        }

        if len(game_res) < 5:
            i = 5 - len(game_res)
            game_res = [0] * i + game_res
        if len(h2h_res) < 5:
            i = 5 - len(h2h_res)
            h2h_res = [0] * i + h2h_res

        # Update the data dictionary with additional values
        data.update(
            {"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
        data.update({"Game " + str(i + 1): game_res[-5 + i] for i in range(5)})

        player_data = self.playerProfile.loc[player]
        data.update(
            {"Player " + col: player_data[col] for col in player_data.index})

        team_data = self.teamProfile.loc[team]
        data.update(
            {"Team " + col: team_data[col] for col in team_data.index})

        defense_data = self.defenseProfile.loc[opponent]
        data.update(
            {"Defense " + col: defense_data[col] for col in defense_data.index if col not in positions})

        return data

    def get_training_matrix(self, market):
        """
        Retrieves training data in the form of a feature matrix (X) and a target vector (y) for a specified market.

        Args:
            market (str): The market for which to retrieve training data.

        Returns:
            tuple: A tuple containing the feature matrix (X) and the target vector (y).
        """
        archive.__init__("NBA")

        matrix = []

        for i, game in tqdm(self.gamelog.iterrows(), unit="game", desc="Gathering Training Data", total=len(self.gamelog)):
            gameDate = datetime.strptime(
                game["GAME_DATE"], "%Y-%m-%dT%H:%M:%S")

            if game[market] < 0:
                continue

            if gameDate < datetime(2022, 10, 1):
                continue

            data = {}
            self.profile_market(market, date=gameDate)
            name = game['PLAYER_NAME']

            if name not in self.playerProfile.index:
                continue

            data = archive["NBA"].get(market, {}).get(gameDate.strftime(
                "%Y-%m-%d"), {}).get(name, {0: [0.5]*4})

            lines = [k for k, v in data.items()]
            if "Closing Lines" in lines:
                lines.remove("Closing Lines")
                lines.append(
                    np.floor(np.mean([float(i['Line']) for i in data["Closing Lines"] if i is not None]))+0.5)

            line = lines[-1]

            offer = {
                "Player": name,
                "Team": game["TEAM_ABBREVIATION"],
                "Market": market,
                "Opponent": game["OPP"],
                "Home": int(game["HOME"])
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_get_stats = self.get_stats(
                    offer | {"Line": line}, gameDate
                )
                if type(new_get_stats) is dict:
                    if new_get_stats["Avg10"] == 0 and new_get_stats["IQR10"] == 0:
                        continue
                    new_get_stats.update(
                        {"Result": game[market]}
                    )

                    matrix.append(new_get_stats)

        M = pd.DataFrame(matrix).fillna(0.0).replace([np.inf, -np.inf], 0)

        return M


class StatsMLB(Stats):
    """
    A class for handling and analyzing MLB statistics.
    Inherits from the Stats parent class.

    Additional Attributes:
        pitchers (mlb_pitchers): Object containing MLB pitcher data.
        gameIds (list): List of game ids in gamelog.

    Additional Methods:
        None
    """

    def __init__(self):
        """
        Initialize the StatsMLB instance.
        """
        super().__init__()
        self.season_start = datetime.strptime("2023-03-30", "%Y-%m-%d"
                                              ).date()
        self.pitchers = mlb_pitchers
        self.gameIds = []
        self.gamelog = pd.DataFrame()
        self.teamlog = pd.DataFrame()
        self.park_factors = {}
        self.players = {}
        self.stat_types = {
            "batting": ["OBP", "AVG", "SLG", "PASO", "BABIP"],
            "fielding": ["DER"],
            "pitching": ["FIP", "WHIP", "ERA", "K9", "BB9", "PA9", "IP"]
        }

    def parse_game(self, gameId):
        """
        Parses the game data for a given game ID.

        Args:
            gameId (int): The ID of the game to parse.
        """
        # game = mlb.boxscore_data(gameId)
        game = scraper.get(
            f"https://baseballsavant.mlb.com/gf?game_pk={gameId}")
        new_games = []
        if game:
            linescore = game["scoreboard"]["linescore"]
            boxscore = game["boxscore"]
            awayTeam = game["away_team_data"]["abbreviation"].replace(
                "AZ", "ARI").replace("WSH", "WAS")
            homeTeam = game["home_team_data"]["abbreviation"].replace(
                "AZ", "ARI").replace("WSH", "WAS")
            bpf = self.park_factors[homeTeam]
            awayPitcherId = game["away_pitcher_lineup"][0]
            awayPitcher = boxscore["teams"]["away"]["players"]["ID" + str(awayPitcherId)]["person"][
                "fullName"
            ]
            if awayPitcherId in self.players and "throws" in self.players[awayPitcherId]:
                awayPitcherHand = self.players[awayPitcherId]["throws"]
            else:
                if str(awayPitcherId) not in game["away_pitchers"]:
                    return
                awayPitcherHand = game["away_pitchers"][str(
                    awayPitcherId)][0]["p_throws"]
                if awayPitcherId not in self.players:
                    self.players[awayPitcherId] = {
                        "name": awayPitcher, "throws": awayPitcherHand}
                else:
                    self.players[awayPitcherId]["throws"] = awayPitcherHand
            homePitcherId = game["home_pitcher_lineup"][0]
            homePitcher = boxscore["teams"]["home"]["players"]["ID" + str(homePitcherId)]["person"][
                "fullName"
            ]
            if homePitcherId in self.players and "throws" in self.players[homePitcherId]:
                homePitcherHand = self.players[homePitcherId]["throws"]
            else:
                if str(homePitcherId) not in game["home_pitchers"]:
                    return
                homePitcherHand = game["home_pitchers"][str(
                    homePitcherId)][0]["p_throws"]
                if homePitcherId not in self.players:
                    self.players[homePitcherId] = {
                        "name": homePitcher, "throws": homePitcherHand}
                else:
                    self.players[homePitcherId]["throws"] = homePitcherHand
            awayInning1Runs = linescore["innings"][0]["away"]["runs"]
            homeInning1Runs = linescore["innings"][0]["home"]["runs"]
            awayInning1Hits = linescore["innings"][0]["away"]["hits"]
            homeInning1Hits = linescore["innings"][0]["home"]["hits"]
            for v in boxscore["teams"]["away"]["players"].values():
                if (v["person"]["id"] == awayPitcherId or
                        v["person"]["id"] in boxscore["teams"]["away"]["battingOrder"]):
                    n = {
                        "gameId": gameId,
                        "gameDate": game["game_date"],
                        "playerId": v["person"]["id"],
                        "playerName": v["person"]["fullName"],
                        "position": v.get("position", {"abbreviation": ""})["abbreviation"],
                        "team": awayTeam,
                        "opponent": homeTeam,
                        "opponent pitcher": homePitcher,
                        "opponent pitcher id": homePitcherId,
                        "opponent pitcher hand": homePitcherHand,
                        "home": False,
                        "starting pitcher": v["person"]["id"] == awayPitcherId,
                        "starting batter": v["person"]["id"] in boxscore["teams"]["away"]["battingOrder"],
                        "battingOrder": boxscore["teams"]["away"]["battingOrder"].index(v["person"]["id"]) + 1
                        if v["person"]["id"] in boxscore["teams"]["away"]["battingOrder"] else 0,
                        "hits": v["stats"]["batting"].get("hits", 0),
                        "total bases": v["stats"]["batting"].get("hits", 0) + v["stats"]["batting"].get("doubles", 0) +
                        2 * v["stats"]["batting"].get("triples", 0) +
                        3 * v["stats"]["batting"].get("homeRuns", 0),
                        "singles": v["stats"]["batting"].get("hits", 0) - v["stats"]["batting"].get("doubles", 0) -
                        v["stats"]["batting"].get(
                            "triples", 0) - v["stats"]["batting"].get("homeRuns", 0),
                        "home runs": v["stats"]["batting"].get("homeRuns", 0),
                        "batter strikeouts": v["stats"]["batting"].get("strikeOuts", 0),
                        "runs": v["stats"]["batting"].get("runs", 0),
                        "rbi": v["stats"]["batting"].get("rbi", 0),
                        "hits+runs+rbi": v["stats"]["batting"].get("hits", 0) + v["stats"]["batting"].get("runs", 0) +
                        v["stats"]["batting"].get("rbi", 0),
                        "walks": v["stats"]["batting"].get("baseOnBalls", 0) + v["stats"]["batting"].get("hitByPitch", 0),
                        "stolen bases": v["stats"]["batting"].get("stolenBases", 0),
                        "atBats": v["stats"]["batting"].get("atBats", 0),
                        "pitcher strikeouts": v["stats"]["pitching"].get("strikeOuts", 0),
                        "walks allowed": v["stats"]["pitching"].get("baseOnBalls", 0) + v["stats"]["pitching"].get("hitByPitch", 0),
                        "pitches thrown": v["stats"]["pitching"].get("numberOfPitches", 0),
                        "runs allowed": v["stats"]["pitching"].get("runs", 0),
                        "hits allowed": v["stats"]["pitching"].get("hits", 0),
                        "home runs allowed": v["stats"]["pitching"].get("homeRuns", 0),
                        "pitching outs": 3 * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1]),
                        "1st inning runs allowed": homeInning1Runs if v["person"]["id"] ==
                        game["away_pitcher_lineup"][0] else 0,
                        "1st inning hits allowed": homeInning1Hits if v["person"]["id"] ==
                        game["away_pitcher_lineup"][0] else 0,
                        "hitter fantasy score": 3*v["stats"]["batting"].get("hits", 0) +
                        2*v["stats"]["batting"].get("doubles", 0) +
                        5*v["stats"]["batting"].get("triples", 0) +
                        7*v["stats"]["batting"].get("homeRuns", 0) +
                        2*v["stats"]["batting"].get("runs", 0) +
                        2*v["stats"]["batting"].get("rbi", 0) +
                        2*v["stats"]["batting"].get("baseOnBalls", 0) +
                        2*v["stats"]["batting"].get("hitByPitch", 0) +
                        5*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy score": 6*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1]) +
                        (4 if int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[
                            0]) > 5 and v["stats"]["pitching"].get("earnedRuns", 0) < 4 else 0),
                        "hitter fantasy points underdog": 3*v["stats"]["batting"].get("hits", 0) +
                        3*v["stats"]["batting"].get("doubles", 0) +
                        5*v["stats"]["batting"].get("triples", 0) +
                        7*v["stats"]["batting"].get("homeRuns", 0) +
                        2*v["stats"]["batting"].get("runs", 0) +
                        2*v["stats"]["batting"].get("rbi", 0) +
                        3*v["stats"]["batting"].get("baseOnBalls", 0) +
                        3*v["stats"]["batting"].get("hitByPitch", 0) +
                        4*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points underdog": 5*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        (5 if int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[
                            0]) > 5 and v["stats"]["pitching"].get("earnedRuns", 0) < 4 else 0),
                        "hitter fantasy points parlay": 3*v["stats"]["batting"].get("hits", 0) +
                        3*v["stats"]["batting"].get("doubles", 0) +
                        6*v["stats"]["batting"].get("triples", 0) +
                        9*v["stats"]["batting"].get("homeRuns", 0) +
                        3*v["stats"]["batting"].get("runs", 0) +
                        3*v["stats"]["batting"].get("rbi", 0) +
                        3*v["stats"]["batting"].get("baseOnBalls", 0) +
                        3*v["stats"]["batting"].get("hitByPitch", 0) +
                        6*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points parlay": 6*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1])
                    }
                    if (n["starting batter"] and n["atBats"] <= 1) or (n["starting pitcher"] and n["pitching outs"] < 6):
                        continue
                    if n["starting batter"]:
                        if n["playerId"] in self.players and "bats" in self.players[n["playerId"]]:
                            batSide = self.players[n["playerId"]]["bats"]
                        else:
                            batSide = game["away_batters"][str(
                                n["playerId"])][0]["stand"]
                            if n["playerId"] not in self.players:
                                self.players[n["playerId"]] = {
                                    "name": n["playerName"], "bats": batSide}
                            else:
                                self.players[n["playerId"]]["bats"] = batSide

                    adj = {
                        "R": n["runs"]/bpf["R"],
                        "RBI": n["rbi"]/bpf["R"],
                        "H": n["hits"]/bpf["H"],
                        "1B": n["singles"]/bpf["1B"],
                        "2B": v["stats"]["batting"].get("doubles", 0)/bpf["2B"],
                        "3B": v["stats"]["batting"].get("triples", 0)/bpf["3B"],
                        "HR": n["home runs"]/bpf["HR"],
                        "W": n["walks"]/bpf["BB"],
                        "SO": n["batter strikeouts"]/bpf["K"],
                        "RA": n["runs allowed"]/bpf["R"],
                        "HA": n["hits allowed"]/bpf["H"],
                        "HRA": n["home runs allowed"]/bpf["HR"],
                        "BB": n["walks allowed"]/bpf["BB"],
                        "K": n["pitcher strikeouts"]/bpf["K"]
                    }

                    BIP = (n["atBats"] - n["batter strikeouts"] - n["home runs"] -
                           v["stats"]["batting"].get("sacFlies", 0))
                    n.update({
                        "FIP": (3*(13*adj["HRA"] + 3*adj["BB"] - 2*adj["K"])/n["pitching outs"] + 3.2) if n["starting pitcher"] else 0,
                        "WHIP": (3*(adj["BB"] + adj["HA"])/n["pitching outs"]) if n["starting pitcher"] else 0,
                        "ERA": (9*adj["RA"]/n["pitching outs"]) if n["starting pitcher"] else 0,
                        "K9": (27*adj["K"] / n["pitching outs"]) if n["starting pitcher"] else 0,
                        "BB9": (27*adj["BB"] / n["pitching outs"]) if n["starting pitcher"] else 0,
                        "PA9": (27*v["stats"]["pitching"].get("battersFaced", 0) / n["pitching outs"]) if n["starting pitcher"] else 0,
                        "IP": (n["pitching outs"] / 3) if n["starting pitcher"] else 0,
                        "OBP": ((n["hits"] + n["walks"])/n["atBats"]/bpf["OBP"]) if n["starting batter"] else 0,
                        "AVG": (n["hits"]/n["atBats"]) if n["starting batter"] else 0,
                        "SLG": (n["total bases"]/n["atBats"]) if n["starting batter"] else 0,
                        "PASO": (v["stats"]["batting"].get("plateAppearances", 0) / adj["SO"]) if (n["starting batter"] and adj["SO"]) else v["stats"]["batting"].get("plateAppearances", 0),
                        "BABIP": ((n["hits"] - n["home runs"]) / BIP) if (n["starting batter"] and BIP) else 0,
                        "batSide": batSide if n["starting batter"] else 0
                    })

                    if (n["starting batter"] and n["atBats"] > 1) or (n["starting pitcher"]):
                        new_games.append(n)

            for v in boxscore["teams"]["home"]["players"].values():
                if (v["person"]["id"] == homePitcherId or
                        v["person"]["id"] in boxscore["teams"]["home"]["battingOrder"]):
                    n = {
                        "gameId": gameId,
                        "gameDate": game["game_date"],
                        "playerId": v["person"]["id"],
                        "playerName": v["person"]["fullName"],
                        "position": v.get("position", {"abbreviation": ""})[
                            "abbreviation"
                        ],
                        "team": homeTeam,
                        "opponent": awayTeam,
                        "opponent pitcher": awayPitcher,
                        "opponent pitcher id": awayPitcherId,
                        "opponent pitcher hand": awayPitcherHand,
                        "home": True,
                        "starting pitcher": v["person"]["id"] == homePitcherId,
                        "starting batter": v["person"]["id"] in boxscore["teams"]["home"]["battingOrder"],
                        "battingOrder": boxscore["teams"]["home"]["battingOrder"].index(v["person"]["id"]) + 1
                        if v["person"]["id"] in boxscore["teams"]["home"]["battingOrder"] else 0,
                        "hits": v["stats"]["batting"].get("hits", 0),
                        "total bases": v["stats"]["batting"].get("hits", 0)
                        + v["stats"]["batting"].get("doubles", 0)
                        + 2 * v["stats"]["batting"].get("triples", 0)
                        + 3 * v["stats"]["batting"].get("homeRuns", 0),
                        "singles": v["stats"]["batting"].get("hits", 0)
                        - v["stats"]["batting"].get("doubles", 0)
                        - v["stats"]["batting"].get("triples", 0)
                        - v["stats"]["batting"].get("homeRuns", 0),
                        "home runs": v["stats"]["batting"].get("homeRuns", 0),
                        "batter strikeouts": v["stats"]["batting"].get("strikeOuts", 0),
                        "runs": v["stats"]["batting"].get("runs", 0),
                        "rbi": v["stats"]["batting"].get("rbi", 0),
                        "hits+runs+rbi": v["stats"]["batting"].get("hits", 0)
                        + v["stats"]["batting"].get("runs", 0)
                        + v["stats"]["batting"].get("rbi", 0),
                        "walks": v["stats"]["batting"].get("baseOnBalls", 0) + v["stats"]["batting"].get("hitByPitch", 0),
                        "stolen bases": v["stats"]["batting"].get("stolenBases", 0),
                        "atBats": v["stats"]["batting"].get("atBats", 0),
                        "pitcher strikeouts": v["stats"]["pitching"].get(
                            "strikeOuts", 0
                        ),
                        "walks allowed": v["stats"]["pitching"].get("baseOnBalls", 0) + v["stats"]["pitching"].get("hitByPitch", 0),
                        "pitches thrown": v["stats"]["pitching"].get(
                            "numberOfPitches", 0
                        ),
                        "runs allowed": v["stats"]["pitching"].get("runs", 0),
                        "hits allowed": v["stats"]["pitching"].get("hits", 0),
                        "home runs allowed": v["stats"]["pitching"].get("homeRuns", 0),
                        "pitching outs": 3
                        * int(
                            v["stats"]["pitching"]
                            .get("inningsPitched", "0.0")
                            .split(".")[0]
                        )
                        + int(
                            v["stats"]["pitching"]
                            .get("inningsPitched", "0.0")
                            .split(".")[1]
                        ),
                        "1st inning runs allowed": awayInning1Runs if v["person"]["id"] ==
                        game["home_pitcher_lineup"][0] else 0,
                        "1st inning hits allowed": awayInning1Hits if v["person"]["id"] ==
                        game["home_pitcher_lineup"][0] else 0,
                        "hitter fantasy score": 3*v["stats"]["batting"].get("hits", 0) +
                        2*v["stats"]["batting"].get("doubles", 0) +
                        5*v["stats"]["batting"].get("triples", 0) +
                        7*v["stats"]["batting"].get("homeRuns", 0) +
                        2*v["stats"]["batting"].get("runs", 0) +
                        2*v["stats"]["batting"].get("rbi", 0) +
                        2*v["stats"]["batting"].get("baseOnBalls", 0) +
                        2*v["stats"]["batting"].get("hitByPitch", 0) +
                        5*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy score": 6*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1]) +
                        (4 if int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[
                            0]) > 5 and v["stats"]["pitching"].get("earnedRuns", 0) < 4 else 0),
                        "hitter fantasy points underdog": 3*v["stats"]["batting"].get("hits", 0) +
                        3*v["stats"]["batting"].get("doubles", 0) +
                        5*v["stats"]["batting"].get("triples", 0) +
                        7*v["stats"]["batting"].get("homeRuns", 0) +
                        2*v["stats"]["batting"].get("runs", 0) +
                        2*v["stats"]["batting"].get("rbi", 0) +
                        3*v["stats"]["batting"].get("baseOnBalls", 0) +
                        3*v["stats"]["batting"].get("hitByPitch", 0) +
                        4*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points underdog": 5*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        (5 if int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[
                            0]) > 5 and v["stats"]["pitching"].get("earnedRuns", 0) < 4 else 0),
                        "hitter fantasy points parlay": 3*v["stats"]["batting"].get("hits", 0) +
                        3*v["stats"]["batting"].get("doubles", 0) +
                        6*v["stats"]["batting"].get("triples", 0) +
                        9*v["stats"]["batting"].get("homeRuns", 0) +
                        3*v["stats"]["batting"].get("runs", 0) +
                        3*v["stats"]["batting"].get("rbi", 0) +
                        3*v["stats"]["batting"].get("baseOnBalls", 0) +
                        3*v["stats"]["batting"].get("hitByPitch", 0) +
                        6*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points parlay": 6*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1])
                    }
                    if (n["starting batter"] and n["atBats"] <= 1) or (n["starting pitcher"] and n["pitching outs"] < 6):
                        continue
                    if n["starting batter"]:
                        if n["playerId"] in self.players and "bats" in self.players[n["playerId"]]:
                            batSide = self.players[n["playerId"]]["bats"]
                        else:
                            batSide = game["home_batters"][str(
                                n["playerId"])][0]["stand"]
                            if n["playerId"] not in self.players:
                                self.players[n["playerId"]] = {
                                    "name": n["playerName"], "bats": batSide}
                            else:
                                self.players[n["playerId"]]["bats"] = batSide

                    adj = {
                        "R": n["runs"]/bpf["R"],
                        "RBI": n["rbi"]/bpf["R"],
                        "H": n["hits"]/bpf["H"],
                        "1B": n["singles"]/bpf["1B"],
                        "2B": v["stats"]["batting"].get("doubles", 0)/bpf["2B"],
                        "3B": v["stats"]["batting"].get("triples", 0)/bpf["3B"],
                        "HR": n["home runs"]/bpf["HR"],
                        "W": n["walks"]/bpf["BB"],
                        "SO": n["batter strikeouts"]/bpf["K"],
                        "RA": n["runs allowed"]/bpf["R"],
                        "HA": n["hits allowed"]/bpf["H"],
                        "HRA": n["home runs allowed"]/bpf["HR"],
                        "BB": n["walks allowed"]/bpf["BB"],
                        "K": n["pitcher strikeouts"]/bpf["K"]
                    }

                    BIP = (n["atBats"] - n["batter strikeouts"] - n["home runs"] -
                           v["stats"]["batting"].get("sacFlies", 0))
                    n.update({
                        "FIP": (3*(13*adj["HRA"] + 3*adj["BB"] - 2*adj["K"])/n["pitching outs"] + 3.2) if n["starting pitcher"] else 0,
                        "WHIP": (3*(adj["BB"] + adj["HA"])/n["pitching outs"]) if n["starting pitcher"] else 0,
                        "ERA": (9*adj["RA"]/n["pitching outs"]) if n["starting pitcher"] else 0,
                        "K9": (27*adj["K"] / n["pitching outs"]) if n["starting pitcher"] else 0,
                        "BB9": (27*adj["BB"] / n["pitching outs"]) if n["starting pitcher"] else 0,
                        "PA9": (27*v["stats"]["pitching"].get("battersFaced", 0) / n["pitching outs"]) if n["starting pitcher"] else 0,
                        "IP": (n["pitching outs"] / 3) if n["starting pitcher"] else 0,
                        "OBP": ((n["hits"] + n["walks"])/n["atBats"]/bpf["OBP"]) if n["starting batter"] else 0,
                        "AVG": (n["hits"]/n["atBats"]) if n["starting batter"] else 0,
                        "SLG": (n["total bases"]/n["atBats"]) if n["starting batter"] else 0,
                        "PASO": (v["stats"]["batting"].get("plateAppearances", 0) / adj["SO"]) if (n["starting batter"] and adj["SO"]) else v["stats"]["batting"].get("plateAppearances", 0),
                        "BABIP": ((n["hits"] - n["home runs"]) / BIP) if (n["starting batter"] and BIP) else 0,
                        "batSide": batSide if n["starting batter"] else 0
                    })

                    if (n["starting batter"] and n["atBats"] > 1) or (n["starting pitcher"]):
                        new_games.append(n)

        self.gamelog = pd.concat(
            [self.gamelog, pd.DataFrame.from_records(new_games)], ignore_index=True)

        teams = [
            {
                "team": homeTeam,
                "opponent": awayTeam,
                "gameId": gameId,
                "gameDate": game["game_date"],
                "OBP": float(boxscore["teams"]["home"]["teamStats"]["batting"]["obp"])/bpf["OBP"],
                "AVG": float(boxscore["teams"]["home"]["teamStats"]["batting"]["avg"]),
                "SLG": float(boxscore["teams"]["home"]["teamStats"]["batting"]["slg"]),
                "PASO": (boxscore["teams"]["home"]["teamStats"]["batting"]["plateAppearances"] /
                         boxscore["teams"]["home"]["teamStats"]["batting"]["strikeOuts"]) if
                boxscore["teams"]["home"]["teamStats"]["batting"]["strikeOuts"] else
                boxscore["teams"]["home"]["teamStats"]["batting"]["plateAppearances"],
                "BABIP": (boxscore["teams"]["home"]["teamStats"]["batting"]["hits"] -
                          boxscore["teams"]["home"]["teamStats"]["batting"]["homeRuns"]) /
                (boxscore["teams"]["home"]["teamStats"]["batting"]["atBats"] -
                 boxscore["teams"]["home"]["teamStats"]["batting"]["strikeOuts"] -
                 boxscore["teams"]["home"]["teamStats"]["batting"]["homeRuns"] -
                 boxscore["teams"]["home"]["teamStats"]["batting"]["sacFlies"]),
                "DER": 1 - ((boxscore["teams"]["away"]["teamStats"]["batting"]["hits"] +
                             boxscore["teams"]["home"]["teamStats"]["fielding"]["errors"] -
                             boxscore["teams"]["away"]["teamStats"]["batting"]["homeRuns"]) /
                            (boxscore["teams"]["away"]["teamStats"]["batting"]["plateAppearances"] -
                            boxscore["teams"]["away"]["teamStats"]["batting"]["baseOnBalls"] -
                            boxscore["teams"]["away"]["teamStats"]["batting"]["hitByPitch"] -
                            boxscore["teams"]["away"]["teamStats"]["batting"]["homeRuns"] -
                            boxscore["teams"]["away"]["teamStats"]["batting"]["strikeOuts"]))
            },
            {
                "team": awayTeam,
                "opponent": homeTeam,
                "gameId": gameId,
                "gameDate": game["game_date"],
                "OBP": float(boxscore["teams"]["away"]["teamStats"]["batting"]["obp"])/bpf["OBP"],
                "AVG": float(boxscore["teams"]["away"]["teamStats"]["batting"]["avg"]),
                "SLG": float(boxscore["teams"]["away"]["teamStats"]["batting"]["slg"]),
                "PASO": (boxscore["teams"]["away"]["teamStats"]["batting"]["plateAppearances"] /
                         boxscore["teams"]["away"]["teamStats"]["batting"]["strikeOuts"]) if
                boxscore["teams"]["away"]["teamStats"]["batting"]["strikeOuts"] else
                boxscore["teams"]["away"]["teamStats"]["batting"]["plateAppearances"],
                "BABIP": (boxscore["teams"]["away"]["teamStats"]["batting"]["hits"] -
                          boxscore["teams"]["away"]["teamStats"]["batting"]["homeRuns"]) /
                (boxscore["teams"]["away"]["teamStats"]["batting"]["atBats"] -
                 boxscore["teams"]["away"]["teamStats"]["batting"]["strikeOuts"] -
                 boxscore["teams"]["away"]["teamStats"]["batting"]["homeRuns"] -
                 boxscore["teams"]["away"]["teamStats"]["batting"]["sacFlies"]),
                "DER": 1 - ((boxscore["teams"]["home"]["teamStats"]["batting"]["hits"] +
                             boxscore["teams"]["away"]["teamStats"]["fielding"]["errors"] -
                             boxscore["teams"]["home"]["teamStats"]["batting"]["homeRuns"]) /
                            (boxscore["teams"]["home"]["teamStats"]["batting"]["plateAppearances"] -
                            boxscore["teams"]["home"]["teamStats"]["batting"]["baseOnBalls"] -
                            boxscore["teams"]["home"]["teamStats"]["batting"]["hitByPitch"] -
                            boxscore["teams"]["home"]["teamStats"]["batting"]["homeRuns"] -
                            boxscore["teams"]["home"]["teamStats"]["batting"]["strikeOuts"]))
            },
        ]

        self.teamlog = pd.concat(
            [self.teamlog, pd.DataFrame.from_records(teams)], ignore_index=True)

    def load(self):
        """
        Loads MLB player statistics from a file.
        """
        filepath = pkg_resources.files(data) / "mlb_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                mlb_data = pickle.load(infile)
                self.gamelog = mlb_data["gamelog"]
                self.teamlog = mlb_data["teamlog"]
                self.players = mlb_data["players"]
        filepath = pkg_resources.files(data) / "park_factor.json"
        if os.path.isfile(filepath):
            with open(filepath, 'r') as infile:
                self.park_factors = json.load(infile)
        filepath = pkg_resources.files(
            data) / "affinity_pitchersBySHV_matchScores.csv"
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df = df.loc[(df.key1.str[-1] == df.key2.str[-1]) &
                        (df.match_score >= 0.6)]
            df.key1 = df.key1.str[:-2].astype(int)
            df.key2 = df.key2.str[:-2].astype(int)
            self.affinity = df.groupby('key1').apply(
                lambda x: x.key2.to_list()).to_dict()

    def update(self):
        """
        Updates the MLB player statistics.
        """
        # Get the current MLB schedule
        today = datetime.today().date()
        if self.gamelog.empty:
            next_day = self.season_start
        else:
            next_day = pd.to_datetime(
                self.gamelog.gameDate).max().date()
        if next_day < self.season_start:
            next_day = self.season_start
        if next_day > today:
            next_day = today
        end_date = next_day + timedelta(days=90)
        if end_date > today:
            end_date = today
        mlb_games = mlb.schedule(
            start_date=next_day.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        mlb_teams = mlb.get("teams", {"sportId": 1})
        mlb_upcoming_games = {}
        for game in mlb_games:
            if game["status"] in ["Pre-Game", "Scheduled"]:
                awayTeam = [
                    team["abbreviation"].replace("AZ", "ARI")
                    for team in mlb_teams["teams"]
                    if team["id"] == game["away_id"]
                ][0]
                homeTeam = [
                    team["abbreviation"].replace("AZ", "ARI")
                    for team in mlb_teams["teams"]
                    if team["id"] == game["home_id"]
                ][0]
                game_bs = mlb.boxscore_data(game['game_id'])
                players = {p['id']: p['fullName']
                           for k, p in game_bs['playerInfo'].items()}
                if game["game_num"] == 1:
                    mlb_upcoming_games[awayTeam] = {
                        "Pitcher": remove_accents(game["away_probable_pitcher"]),
                        "Home": 0,
                        "Opponent": homeTeam,
                        "Batting Order": [players[i] for i in game_bs['away']['battingOrder']]
                    }
                    mlb_upcoming_games[homeTeam] = {
                        "Pitcher": remove_accents(game["home_probable_pitcher"]),
                        "Home": 1,
                        "Opponent": awayTeam,
                        "Batting Order": [players[i] for i in game_bs['home']['battingOrder']]
                    }
                elif game["game_num"] > 1:
                    mlb_upcoming_games[awayTeam + str(game["game_num"])] = {
                        "Pitcher": remove_accents(game["away_probable_pitcher"]),
                        "Home": 0,
                        "Opponent": homeTeam,
                        "Batting Order": [players[i] for i in game_bs['away']['battingOrder']]
                    }
                    mlb_upcoming_games[homeTeam + str(game["game_num"])] = {
                        "Pitcher": remove_accents(game["home_probable_pitcher"]),
                        "Home": 1,
                        "Opponent": awayTeam,
                        "Batting Order": [players[i] for i in game_bs['home']['battingOrder']]
                    }

        self.upcoming_games = mlb_upcoming_games

        if self.gamelog.empty:
            prev_game_ids = []
        else:
            prev_game_ids = self.gamelog.gameId.unique()
        mlb_game_ids = [
            game["game_id"]
            for game in mlb_games
            if game["status"] == "Final"
            and game["game_type"] != "E"
            and game["game_type"] != "S"
            and game["game_type"] != "A"
            and game["game_id"] not in prev_game_ids
        ]

        # Parse the game stats
        for id in tqdm(mlb_game_ids, desc="Getting MLB Stats"):
            self.parse_game(id)

        # Remove old games to prevent file bloat
        two_years_ago = today - timedelta(days=730)
        self.gamelog = self.gamelog[self.gamelog["gameDate"].apply(
            lambda x: two_years_ago <= datetime.strptime(x, '%Y-%m-%d').date() <= today)]
        self.gamelog = self.gamelog[~self.gamelog['opponent'].isin([
                                                                   "AL", "NL"])]
        self.gamelog.drop_duplicates(inplace=True)

        # Write to file
        with open(pkg_resources.files(data) / "mlb_data.dat", "wb") as outfile:
            pickle.dump({"players": self.players,
                         "gamelog": self.gamelog,
                         "teamlog": self.teamlog}, outfile, -1)

    def bucket_stats(self, market, buckets=20, date=datetime.today()):
        """
        Buckets the statistics of players based on a given market (e.g., 'allowed', 'pitch').

        Args:
            market (str): The market to bucket the stats for.
            buckets (int): The number of buckets to divide the stats into.

        Returns:
            None
        """
        if market == self.bucket_market and date == self.bucket_latest_date:
            return

        self.bucket_market = market
        self.bucket_latest_date = date

        # Reset playerStats and edges
        self.playerStats = pd.DataFrame()
        self.edges = []

        # Collect stats for each player
        gamelog = self.gamelog.loc[(pd.to_datetime(self.gamelog["gameDate"]) < date) &
                                   (pd.to_datetime(self.gamelog["gameDate"]) > date - timedelta(days=60))]

        if gamelog.empty:
            gamelog = self.gamelog.loc[(pd.to_datetime(self.gamelog["gameDate"]) < date) &
                                       (pd.to_datetime(self.gamelog["gameDate"]) > date - timedelta(days=240))]

        playerGroups = gamelog.\
            groupby('playerName').\
            filter(lambda x: len(x[x[market] != 0]) > 4).\
            groupby('playerName')[market]

        # Compute edges for each bucket
        w = int(100 / buckets)
        self.edges = playerGroups.mean().quantile(
            np.arange(0, 101, w)/100).to_list()

        # Assign bucket and line values to each player
        self.playerStats['avg'] = playerGroups.mean()
        self.playerStats['bucket'] = np.ceil(
            playerGroups.mean().rank(pct=True, ascending=False)*20).astype(int)
        self.playerStats['line'] = playerGroups.median()
        self.playerStats.loc[self.playerStats['line'] == 0.0, 'line'] = 0.5
        self.playerStats.loc[(np.mod(self.playerStats['line'], 1) == 0) & (
            self.playerStats['avg'] > self.playerStats['line']), 'line'] += 0.5
        self.playerStats.loc[(np.mod(self.playerStats['line'], 1) == 0) & (
            self.playerStats['avg'] < self.playerStats['line']), 'line'] -= 0.5

        self.playerStats = self.playerStats.to_dict(orient='index')

    def profile_market(self, market, date=datetime.today().date()):
        if type(date) is str:
            date = datetime.strptime(date, "%Y-%m-%d").date()
        if type(date) is datetime:
            date = date.date()

        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.profiled_market = market
        self.profile_latest_date = date

        # Initialize playerStats and edges
        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.defenseProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.pitcherProfile = pd.DataFrame(columns=['avg', 'home', 'away'])

        # Filter gamelog for games within the date range
        one_year_ago = (date - timedelta(days=300))
        gameDates = pd.to_datetime(self.gamelog["gameDate"]).dt.date
        gamelog = self.gamelog[(
            one_year_ago <= gameDates) & (gameDates < date)]
        gameDates = pd.to_datetime(self.teamlog["gameDate"]).dt.date
        teamlog = self.teamlog[(
            one_year_ago <= gameDates) & (gameDates < date)]

        # Filter non-starting pitchers or non-starting batters depending on the market
        if any([string in market for string in ["allowed", "pitch"]]):
            gamelog2 = gamelog[gamelog["starting batter"]].copy()
            gamelog = gamelog[gamelog["starting pitcher"]].copy()
        else:
            gamelog2 = gamelog[gamelog["starting pitcher"]].copy()
            gamelog = gamelog[gamelog["starting batter"]].copy()

        # Retrieve moneyline and totals data from archive
        k = [[(date, team) for team in teams.keys()]
             for date, teams in archive["MLB"]["Moneyline"].items()]
        k = [item for row in k for item in row]
        flat_money = {t: archive["MLB"]["Moneyline"].get(
            t[0], {}).get(t[1], 0.5) for t in k}
        flat_total = {t: archive["MLB"]["Totals"].get(
            t[0], {}).get(t[1], 4.5) for t in k}
        tup_s = pd.Series(
            zip(gamelog['gameDate'], gamelog['team']), index=gamelog.index)
        gamelog.loc[:, "moneyline"] = tup_s.map(flat_money)
        gamelog.loc[:, "totals"] = tup_s.map(flat_total)

        teamstats = teamlog.groupby('team').tail(10).groupby('team')[
            teamlog.columns[4:]].mean()

        # Filter players with at least 2 entries
        playerGroups = gamelog.groupby('playerName').filter(
            lambda x: (x[market].clip(0, 1).mean() > 0.3) & (x[market].count() > 1)).groupby('playerName')

        defenseGroups = gamelog.groupby('opponent')
        defenseGroups = gamelog.groupby(['opponent', 'gameId'])
        defenseGames = pd.DataFrame()
        defenseGames[market] = defenseGroups[market].sum()
        defenseGames['home'] = defenseGroups['home'].mean().astype(int)
        defenseGames['moneyline'] = defenseGroups['moneyline'].mean()
        defenseGames['totals'] = defenseGroups['totals'].mean()
        defenseGroups = defenseGames.groupby('opponent')

        pitcherGroups = gamelog.groupby(['opponent pitcher', 'gameId'])
        pitcherGames = pd.DataFrame()
        pitcherGames[market] = pitcherGroups[market].sum()
        pitcherGames['home'] = pitcherGroups['home'].mean().astype(int)
        pitcherGames['moneyline'] = pitcherGroups['moneyline'].mean()
        pitcherGames['totals'] = pitcherGroups['totals'].mean()
        pitcherGroups = pitcherGames.groupby('opponent pitcher').filter(
            lambda x: x[market].count() > 1).groupby('opponent pitcher')

        # Compute league average
        leagueavg = playerGroups[market].mean().mean()
        if np.isnan(leagueavg):
            return

        # Compute playerProfile DataFrame
        self.playerProfile['avg'] = playerGroups[market].mean().div(
            leagueavg) - 1
        self.playerProfile['home'] = playerGroups.apply(
            lambda x: x.loc[x['home'], market].mean() / x[market].mean()) - 1
        self.playerProfile['away'] = playerGroups.apply(
            lambda x: x.loc[~x['home'], market].mean() / x[market].mean()) - 1

        leagueavg = defenseGroups[market].mean().mean()
        self.defenseProfile['avg'] = defenseGroups[market].mean().div(
            leagueavg) - 1
        self.defenseProfile['home'] = defenseGroups.apply(
            lambda x: x.loc[x['home'] == 1, market].mean() / x[market].mean()) - 1
        self.defenseProfile['away'] = defenseGroups.apply(
            lambda x: x.loc[x['home'] == 0, market].mean() / x[market].mean()) - 1

        leagueavg = pitcherGroups[market].mean().mean()
        self.pitcherProfile['avg'] = pitcherGroups[market].mean().div(
            leagueavg) - 1
        self.pitcherProfile['home'] = pitcherGroups.apply(
            lambda x: x.loc[x['home'] == 1, market].mean() / x[market].mean()) - 1
        self.pitcherProfile['away'] = pitcherGroups.apply(
            lambda x: x.loc[x['home'] == 0, market].mean() / x[market].mean()) - 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.playerProfile['moneyline gain'] = playerGroups.apply(
                lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

            self.playerProfile['totals gain'] = playerGroups.apply(
                lambda x: np.polyfit(x.totals.fillna(8.3).values.astype(float) / 4.5 - x.totals.fillna(4.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

            self.defenseProfile['moneyline gain'] = defenseGroups.apply(
                lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

            self.defenseProfile['totals gain'] = defenseGroups.apply(
                lambda x: np.polyfit(x.totals.fillna(8.3).values.astype(float) / 4.5 - x.totals.fillna(4.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

            self.pitcherProfile['moneyline gain'] = pitcherGroups.apply(
                lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

            self.pitcherProfile['totals gain'] = pitcherGroups.apply(
                lambda x: np.polyfit(x.totals.fillna(8.3).values.astype(float) / 4.5 - x.totals.fillna(4.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

        if any([string in market for string in ["allowed", "pitch"]]):
            self.playerProfile['days off'] = playerGroups['gameDate'].apply(lambda x: np.diff(
                [datetime.strptime(g, "%Y-%m-%d") for g in x.iloc[-2:]])[0].days)

            playerstats = gamelog.fillna(0).groupby('playerName')[
                self.stat_types['pitching']].mean(numeric_only=True)\

            batterstats = gamelog2.groupby('playerName')[
                self.stat_types['batting']].mean(numeric_only=True)

            i = self.defenseProfile.index
            self.defenseProfile = self.defenseProfile.merge(
                teamstats[self.stat_types['batting']], left_on='opponent', right_on='team')
            self.defenseProfile.index = i

            self.teamProfile = teamstats[self.stat_types['fielding']]

            self.batterProfile = batterstats

            self.playerProfile = self.playerProfile.merge(
                playerstats, on='playerName')

        else:
            playerstats = gamelog.fillna(0).groupby('playerName')[
                self.stat_types['batting']].mean(numeric_only=True)
            pitcherstats = gamelog2.drop(columns='opponent pitcher').rename(
                columns={'playerName': 'opponent pitcher'}).groupby('opponent pitcher')[
                self.stat_types['pitching']].mean(numeric_only=True)

            i = self.defenseProfile.index
            self.defenseProfile = self.defenseProfile.merge(
                teamstats[self.stat_types['fielding']], left_on='opponent', right_on='team')
            self.defenseProfile.index = i

            self.teamProfile = teamstats[self.stat_types['batting']]

            self.playerProfile = self.playerProfile.merge(
                playerstats, on='playerName')

            self.pitcherProfile = self.pitcherProfile.merge(
                pitcherstats, on='opponent pitcher')

    def dvpoa(self, team, market, date=datetime.today().date()):
        """
        Calculates the Defense Versus Position over League-Average (DVPOA) for a given team and market.

        Args:
            team (str): The team for which to calculate the DVPOA.
            market (str): The market to calculate the DVPOA for.

        Returns:
            float: The DVPOA value for the specified team and market.
        """

        if date != self.dvpoa_latest_date:
            self.dvp_index = {}
            self.dvpoa_latest_date = date

        # Check if market exists in dvp_index dictionary
        if market not in self.dvp_index:
            self.dvp_index[market] = {}

        # Check if DVPOA value for the specified team and market is already calculated and cached
        if self.dvp_index[market].get(team):
            return self.dvp_index[market][team]

        if type(date) is datetime:
            date = date.date()

        one_year_ago = (date - timedelta(days=300))
        gamelog = self.gamelog[self.gamelog["gameDate"].apply(
            lambda x: one_year_ago <= datetime.strptime(x, '%Y/%m/%d').date() <= date)]

        # Calculate DVP (Defense Versus Position) and league average for the specified team and market
        if any([string in market for string in ["allowed", "pitch"]]):
            position_games = gamelog.loc[gamelog['starting pitcher']]
            team_games = position_games.loc[position_games['opponent'] == team]
        else:
            position_games = gamelog.loc[gamelog['starting batter']]
            team_games = position_games.loc[position_games['opponent pitcher'] == team]

        if len(team_games) == 0:
            return 0
        else:
            dvp = team_games[market].mean()
            leagueavg = position_games[market].mean()
            dvpoa = (dvp - leagueavg) / leagueavg
            dvpoa = np.nan_to_num(dvpoa, nan=0.0, posinf=0.0, neginf=0.0)
            self.dvp_index[market][team] = dvpoa
            return dvpoa

    def get_stats(self, offer, date=datetime.today()):
        """
        Calculates the relevant statistics for a given offer and date.

        Args:
            offer (dict): The offer containing player, team, market, line, and opponent information.
            date (str or datetime.datetime, optional): The date for which to calculate the statistics. Defaults to the current date.

        Returns:
            pandas.DataFrame: The calculated statistics as a DataFrame.
        """
        if type(date) is datetime:
            date = date.strftime("%Y-%m-%d")

        player = offer["Player"]
        team = offer["Team"].replace("AZ", "ARI")
        market = offer["Market"]
        if self.defenseProfile.empty:
            logger.exception(f"{market} not profiled")
            return 0
        line = offer["Line"]
        opponent = offer["Opponent"].split(" (")[0]
        home = offer.get("Home")
        if home is None:
            home = self.upcoming_games.get(team, {}).get("Home", 0)

        if player not in self.playerProfile.index:
            self.playerProfile.loc[player] = np.zeros_like(
                self.playerProfile.columns)
        if opponent not in self.defenseProfile.index:
            self.defenseProfile.loc[opponent] = np.zeros_like(
                self.defenseProfile.columns)

        try:
            if datetime.strptime(date, "%Y-%m-%d").date() < datetime.today().date():
                pitcher = offer.get("Pitcher", "")
            else:
                pitcher = self.pitchers.get(opponent, "")

            if pitcher not in self.pitcherProfile.index:
                self.pitcherProfile.loc[pitcher] = np.zeros_like(
                    self.pitcherProfile.columns)

            stats = (
                archive["MLB"]
                .get(market, {})
                .get(date, {})
                .get(player, {})
                .get(line, [0] * 4)
            )
            moneyline = archive["MLB"]["Moneyline"].get(
                date, {}).get(team, 0)
            total = archive["MLB"]["Totals"].get(date, {}).get(team, 0)

        except:
            logger.exception(f"{player}, {market}")
            return 0

        date = datetime.strptime(date, "%Y-%m-%d")

        if any([string in market for string in ["allowed", "pitch"]]):
            player_games = self.gamelog.loc[(self.gamelog["playerName"] == player) & (
                pd.to_datetime(self.gamelog.gameDate) < date) & self.gamelog["starting pitcher"]]

            headtohead = player_games.loc[player_games["opponent"] == opponent]

            pid = self.gamelog.loc[self.gamelog['playerName']
                                   == player, 'playerId'].iat[0]
        else:
            player_games = self.gamelog.loc[(self.gamelog["playerName"] == player) & (
                pd.to_datetime(self.gamelog.gameDate) < date) & self.gamelog["starting batter"]]

            headtohead = player_games.loc[player_games["opponent pitcher"] == pitcher]

            opid = self.gamelog.loc[self.gamelog['opponent pitcher']
                                    == pitcher, 'opponent pitcher id'].iat[0]

        one_year_ago = len(player_games.loc[
            pd.to_datetime(self.gamelog.gameDate) > date-timedelta(days=300)])
        game_res = (player_games[market]).to_list()
        h2h_res = (headtohead[market]).to_list()

        stats = np.array(stats, dtype=np.float64)
        odds = np.nanmean(stats)
        if np.isnan(odds):
            odds = 0

        data = {
            "DVPOA": 0,
            "Odds": odds,
            "Line": line,
            "Avg5": np.median(game_res[-5:]) if game_res else 0,
            "Avg10": np.median(game_res[-10:]) if game_res else 0,
            "AvgYr": np.median(game_res[-one_year_ago:]) if game_res else 0,
            "AvgH2H": np.median(h2h_res[-5:]) if h2h_res else 0,
            "IQR10": iqr(game_res[-10:]) if game_res else 0,
            "IQRYr": iqr(game_res[-one_year_ago:]) if game_res else 0,
            "Mean5": np.mean(game_res[-5:]) if game_res else 0,
            "Mean10": np.mean(game_res[-10:]) if game_res else 0,
            "MeanYr": np.mean(game_res[-one_year_ago:]) if game_res else 0,
            "MeanH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
            "GamesPlayed": one_year_ago,
            "Moneyline": moneyline,
            "Total": total,
            "Home": home,
        }

        if date.date() < datetime.today().date():
            game = self.gamelog.loc[(self.gamelog["playerName"] == player) & (
                pd.to_datetime(self.gamelog.gameDate) == date)]
            position = game.iloc[0]['battingOrder']
            order = self.gamelog.loc[(self.gamelog.gameId == game.iloc[0]['gameId']) & (
                self.gamelog.team == game.iloc[0]['team']) & self.gamelog['starting batter'], 'playerName'].to_list()

        else:
            order = self.upcoming_games.get(
                team, {}).get('Batting Order', [])
            if player in order:
                position = order.index(player)
            elif player_games.empty:
                position = 9
            else:
                position = int(
                    player_games['battingOrder'].iloc[-10:].median())

        if len(game_res) < 5:
            i = 5 - len(game_res)
            game_res = [0] * i + game_res
        if len(h2h_res) < 5:
            i = 5 - len(h2h_res)
            h2h_res = [0] * i + h2h_res

        # Update the data dictionary with additional values
        data.update(
            {"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
        data.update({"Game " + str(i + 1): game_res[-5 + i] for i in range(5)})

        player_data = self.playerProfile.loc[player]
        data.update(
            {"Player " + col: player_data[col] for col in player_data.index})

        if any([string in market for string in ["allowed", "pitch"]]):
            defense_data = self.defenseProfile.loc[team]

            if len(order) > 0:
                defense_data[self.stat_types['batting']] = self.batterProfile.loc[order,
                                                                                  self.stat_types['batting']].mean()

            team_data = self.teamProfile.loc[team, self.stat_types['fielding']]

            affine = self.gamelog.loc[(self.gamelog["opponent"] == opponent) & (
                pd.to_datetime(self.gamelog.gameDate) < date) & self.gamelog["starting pitcher"] & (
                self.gamelog["playerId"].isin(self.affinity[pid]))]
            aff_data = affine[self.stat_types['pitching']].mean()
        else:
            defense_data = self.pitcherProfile.loc[pitcher]
            defense_data.loc['DER'] = self.defenseProfile.loc[opponent, 'DER']

            if len(order) > 0:
                team_data = self.playerProfile.loc[order,
                                                   self.stat_types['batting']].mean()
            else:
                team_data = self.teamProfile.loc[team,
                                                 self.stat_types['batting']]

            data.update({"Position": position})

            affine = player_games.loc[player_games['opponent pitcher id'].isin(
                self.affinity[opid])]
            aff_data = affine[self.stat_types['batting']].mean()

        data.update({"H2H " + col: aff_data[col] for col in aff_data.index})

        data.update({"Team " + col: team_data[col] for col in team_data.index})

        data.update(
            {"Defense " + col: defense_data[col] for col in defense_data.index})

        park = team if home else opponent
        park_factors = self.park_factors[park]
        data.update({"PF " + col: v for col, v in park_factors.items()})

        data["DVPOA"] = data.pop("Defense avg")

        return data

    def get_training_matrix(self, market):
        """
        Retrieves the training data matrix and target labels for the specified market.

        Args:
            market (str): The market type to retrieve training data for.

        Returns:
            M (pd.DataFrame): The training data matrix.
        """
        archive.__init__("MLB")

        # Initialize an empty list for the target labels
        matrix = []

        # Iterate over the gamelog to collect training data
        for i, game in tqdm(self.gamelog.iterrows(), unit="game", desc="Gathering Training Data", total=len(self.gamelog)):
            # Skip games without starting pitcher or starting batter based on market type
            if (
                any([string in market for string in ["allowed", "pitch"]])
                and not game["starting pitcher"]
            ):
                continue
            elif (
                not any([string in market for string in ["allowed", "pitch"]])
                and not game["starting batter"]
            ):
                continue

            if game[market] < 0:
                continue

            # Retrieve data from the archive based on game date and player name
            data = {}
            gameDate = datetime.strptime(game["gameId"][:10], "%Y/%m/%d")
            if gameDate < datetime(2022, 3, 1):
                continue

            self.profile_market(market, date=gameDate.date())
            name = game['playerName']

            if name not in self.playerProfile.index:
                continue

            data = archive["MLB"][market].get(gameDate.strftime(
                "%Y-%m-%d"), {}).get(name, {0: [0.5]*4})

            lines = [k for k, v in data.items()]
            if "Closing Lines" in lines:
                lines.remove("Closing Lines")
                lines.append(
                    np.floor(np.mean([float(i['Line']) for i in data["Closing Lines"] if i is not None]))+0.5)

            line = lines[-1]

            # Construct an offer dictionary with player, team, market, opponent, and pitcher information
            offer = {
                "Player": name,
                "Team": game["team"],
                "Market": market,
                "Opponent": game["opponent"],
                "Pitcher": game["opponent pitcher"],
                "Home": int(game["home"])
            }

            # Retrieve stats using get_stats method
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_get_stats = self.get_stats(
                    offer | {"Line": line}, gameDate
                )
                if type(new_get_stats) is dict:
                    if new_get_stats["Avg10"] == 0 and new_get_stats["IQR10"] == 0:
                        continue
                    # Determine the result
                    new_get_stats.update(
                        {"Result": game[market]}
                    )

                    # Concatenate retrieved stats into the training data matrix
                    matrix.append(new_get_stats)

        # Create the target labels DataFrame
        M = pd.DataFrame(matrix).fillna(0.0).replace([np.inf, -np.inf], 0)

        return M


class StatsNFL(Stats):
    """
    A class for handling and analyzing NFL statistics.
    Inherits from the Stats parent class.

    Additional Attributes:
        None

    Additional Methods:
        None
    """

    def __init__(self):
        """
        Initialize the StatsNFL class.
        """
        super().__init__()
        self.season_start = datetime.strptime("2023-09-07", "%Y-%m-%d").date()
        cols = ['player id', 'player display name', 'position group', 'recent team', 'season', 'week', 'season type',
                'snap pct', 'completions', 'attempts', 'passing yards', 'passing tds', 'interceptions', 'sacks',
                'sack fumbles', 'sack fumbles lost', 'passing 2pt conversions', 'carries', 'rushing yards',
                'rushing tds', 'rushing fumbles', 'rushing fumbles lost', 'rushing 2pt conversions', 'receptions',
                'targets', 'receiving yards', 'receiving tds', 'receiving fumbles', 'receiving fumbles lost',
                'receiving 2pt conversions', 'fumbles', 'fumbles lost', 'yards', 'tds', 'qb yards', 'qb tds',
                'fantasy points prizepicks', 'fantasy points underdog', 'fantasy points parlayplay', 'home', 'opponent',
                'gameday', 'game id', 'target share', 'air yards share', 'wopr', 'yards per target',
                'completion percentage over expected', 'completion_percentage' 'passer rating', 'passer adot',
                'passer_adot_differential', 'time_to_throw', 'aggressiveness', 'pass_yards_per_attempt',
                'rushing yards over expected', 'rushing success rate', 'yac over expected', 'separation created',
                'targets per route run', 'first read targets per route run', 'route participation',
                'yards per route run', 'average depth of target', 'receiver cp over expected',
                'first read target share', 'redzone target share', 'redzone carry share', 'carry share']
        self.gamelog = pd.DataFrame(columns=cols)
        team_cols = ['season', 'week', 'team', 'pass_rate_over_expected', 'rush_success_rate', 'pass_success_rate',
                     'rush_success_rate_allowed', 'pass_success_rate_allowed', 'epa_per_rush', 'epa_per_pass',
                     'epa_allowed_per_rush', 'epa_allowed_per_pass', 'yards_allowed_per_rush', 'yards_allowed_per_pass',
                     'completion_percentage_allowed', 'cpoe_allowed', 'pressure_per_pass', 'stuffs_per_rush',
                     'pressure_allowed_per_pass', 'stuffs_allowed_per_rush', 'expected_yards_per_rush',
                     'blitz_rate', 'epa_per_blitz']
        self.teamlog = pd.DataFrame(columns=team_cols)
        self.stat_types = {
            'passing': ['completion percentage over expected', 'completion_percentage' 'passer rating',
                        'passer adot', 'passer_adot_differential', 'time_to_throw', 'aggressiveness',
                        'pass_yards_per_attempt'],
            'receiving': ['target share', 'air yards share', 'wopr', 'yards per target',
                          'yac over expected', 'separation created', 'targets per route run',
                          'first read targets per route run', 'route participation', 'yards per route run',
                          'average depth of target', 'receiver cp over expected',
                          'first read target share', 'redzone target share'],
            'rushing': ['snap pct', 'rushing yards over expected', 'rushing success rate',
                        'redzone carry share', 'carry share'],
            'offense': ['pass_rate_over_expected', 'rush_success_rate', 'pass_success_rate',
                        'epa_per_rush', 'epa_per_pass', 'pressure_allowed_per_pass',
                        'stuffs_allowed_per_rush', 'expected_yards_per_rush', 'epa_per_blitz'],
            'defense': ['rush_success_rate_allowed', 'pass_success_rate_allowed', 'epa_allowed_per_rush',
                        'epa_allowed_per_pass', 'yards_allowed_per_rush', 'yards_allowed_per_pass',
                        'completion_percentage_allowed', 'cpoe_allowed', 'pressure_per_pass', 'stuffs_per_rush',
                        'blitz_rate']
        }

    def load(self):
        """
        Load data from files.
        """
        filepath = pkg_resources.files(data) / "nfl_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                pdata = pickle.load(infile)
                if type(pdata) is dict:
                    self.gamelog = pdata["gamelog"]
                    self.teamlog = pdata["teamlog"]
                else:
                    self.gamelog = pdata

    def update(self):
        """
        Update data from the web API.
        """
        # Fetch game logs
        cols = ['player_id', 'player_display_name', 'position_group',
                'recent_team', 'season', 'week', 'season_type',
                'completions', 'attempts', 'passing_yards', 'passing_tds',
                'interceptions', 'sacks', 'sack_fumbles', 'sack_fumbles_lost',
                'passing_2pt_conversions', 'carries', 'rushing_yards', 'rushing_tds',
                'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_2pt_conversions',
                'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_2pt_conversions',
                'target_share', 'air_yards_share', 'wopr']
        try:
            nfl_data = nfl.import_weekly_data([self.season_start.year], cols)
        except:
            nfl_data = pd.DataFrame(columns=cols)

        snaps = nfl.import_snap_counts([self.season_start.year])
        sched = nfl.import_schedules([self.season_start.year])
        upcoming_games = sched.loc[pd.to_datetime(sched['gameday']) >= datetime.today(), [
            'gameday', 'away_team', 'home_team']]
        upcoming_games.loc[upcoming_games['away_team']
                           == 'LA', 'away_team'] = "LAR"
        upcoming_games.loc[upcoming_games['home_team']
                           == 'LA', 'home_team'] = "LAR"
        if not upcoming_games.empty:
            df1 = upcoming_games.rename(
                columns={'home_team': 'Team', 'away_team': 'Opponent'})
            df2 = upcoming_games.rename(
                columns={'away_team': 'Team', 'home_team': 'Opponent'})
            df1['Home'] = 1
            df2['Home'] = 0
            upcoming_games = pd.concat([df1, df2]).sort_values('gameday')
            self.upcoming_games = upcoming_games.groupby("Team").apply(
                lambda x: x.head(1)).droplevel(1)[['Opponent', 'Home', 'gameday']].to_dict(orient='index')

        nfl_data = nfl_data.loc[nfl_data["position_group"].isin(
            ["QB", "WR", "RB", "TE"])]
        snaps = snaps.loc[snaps["position"].isin(["QB", "WR", "RB", "TE"])]
        snaps['player_display_name'] = snaps['player'].map(remove_accents)
        snaps['snap_pct'] = snaps['offense_pct']
        snaps = snaps[['player_display_name', 'season', 'week', 'snap_pct']]

        nfl_data['player_display_name'] = nfl_data['player_display_name'].map(
            remove_accents)

        nfl_data = nfl_data.merge(
            snaps, on=['player_display_name', 'season', 'week'])
        nfl_data = nfl_data.loc[(nfl_data['position_group'] != 'QB') | (
            nfl_data['snap_pct'] > 0.8)]
        nfl_data = nfl_data.loc[(nfl_data['position_group'] != 'WR') | (
            nfl_data['snap_pct'] > 0.5)]
        nfl_data = nfl_data.loc[(nfl_data['position_group'] != 'RB') | (
            nfl_data['snap_pct'] > 0.3)]
        nfl_data = nfl_data.loc[(nfl_data['position_group'] != 'TE') | (
            nfl_data['snap_pct'] > 0.5)]

        nfl_data['fumbles'] = nfl_data['sack_fumbles'] + \
            nfl_data['rushing_fumbles'] + nfl_data['receiving_fumbles']
        nfl_data['fumbles_lost'] = nfl_data['sack_fumbles_lost'] + \
            nfl_data['rushing_fumbles_lost'] + \
            nfl_data['receiving_fumbles_lost']
        nfl_data['yards'] = nfl_data['receiving_yards'] + \
            nfl_data['rushing_yards']
        nfl_data['qb_yards'] = nfl_data['passing_yards'] + \
            nfl_data['rushing_yards']
        nfl_data['tds'] = nfl_data['rushing_tds'] + nfl_data['receiving_tds']
        nfl_data['qb_tds'] = nfl_data['rushing_tds'] + nfl_data['passing_tds']

        nfl_data['fantasy_points_prizepicks'] = nfl_data['passing_yards']/25 + nfl_data['passing_tds']*4 - \
            nfl_data['interceptions'] + nfl_data['yards']/10 + nfl_data['tds']*6 + \
            nfl_data['receptions'] - nfl_data['fumbles_lost'] + \
            nfl_data['passing_2pt_conversions']*2 + \
            nfl_data['rushing_2pt_conversions']*2 + \
            nfl_data['receiving_2pt_conversions']*2

        nfl_data['fantasy_points_underdog'] = nfl_data['passing_yards']/25 + nfl_data['passing_tds']*4 - \
            nfl_data['interceptions'] + nfl_data['yards']/10 + nfl_data['tds']*6 + \
            nfl_data['receptions']/2 - nfl_data['fumbles_lost']*2 + \
            nfl_data['passing_2pt_conversions']*2 + \
            nfl_data['rushing_2pt_conversions']*2 + \
            nfl_data['receiving_2pt_conversions']*2

        nfl_data['fantasy_points_parlayplay'] = nfl_data['passing_yards']/20 + nfl_data['passing_tds']*5 - \
            nfl_data['interceptions']*5 + nfl_data['yards']/5 + nfl_data['tds']*5 - \
            nfl_data['fumbles_lost']*3

        nfl_data['yards_per_target'] = nfl_data['receiving_yards'] / \
            nfl_data['targets']

        nfl_data.rename(
            columns=lambda x: x.replace("_", " "), inplace=True)
        self.gamelog = pd.concat(
            [self.gamelog, nfl_data], ignore_index=True).drop_duplicates(['season', 'week', 'player id'], ignore_index=True).reset_index(drop=True)

        self.players = nfl.import_ids()
        self.players = self.players.loc[self.players['position'].isin([
            'QB', 'RB', 'WR', 'TE'])]
        self.players.name = self.players.name.apply(remove_accents)
        ids = self.players[['name', 'gsis_id']].dropna()
        ids.index = ids.name
        self.ids = ids.gsis_id.to_dict()
        self.players = self.players.groupby(
            'name')['position'].apply(lambda x: x.iat[-1]).to_dict()

        need_pbp = True
        teamDataList = []
        for i, row in tqdm(self.gamelog.iterrows(), desc="Updating NFL data", unit="game", total=len(self.gamelog)):
            if row['opponent'] != row['opponent']:
                if need_pbp:
                    self.pbp = nfl.import_pbp_data([self.season_start.year])
                    self.pbp = self.pbp.loc[self.pbp['play_type'].isin(
                        ['run', 'pass'])]
                    self.pbp.loc[:, 'success'] = 0
                    self.pbp.loc[self.pbp['down'] == 1, 'success'] = (
                        self.pbp.loc[self.pbp['down'] == 1, 'ydstogo']*.4 < self.pbp.loc[self.pbp['down'] == 1, 'yards_gained']).astype(int)
                    self.pbp.loc[self.pbp['down'] == 2, 'success'] = (
                        self.pbp.loc[self.pbp['down'] == 2, 'ydstogo']*.6 < self.pbp.loc[self.pbp['down'] == 2, 'yards_gained']).astype(int)
                    self.pbp.loc[self.pbp['down'] > 2, 'success'] = (
                        self.pbp.loc[self.pbp['down'] > 2, 'ydstogo'] < self.pbp.loc[self.pbp['down'] > 2, 'yards_gained']).astype(int)
                    if self.season_start.year > 2021:
                        ftn = nfl.import_ftn_data([self.season_start.year])
                        ftn['game_id'] = ftn['nflverse_game_id']
                        ftn['play_id'] = ftn['nflverse_play_id']
                        self.pbp = self.pbp.merge(ftn)
                    else:
                        self.pbp['is_qb_out_of_pocket'] = False
                        self.pbp['is_throw_away'] = False
                        self.pbp['read_thrown'] = 0
                        self.pbp['n_blitzers'] = 0

                    self.pbp['pass'] = self.pbp['pass'].astype(bool)
                    self.pbp['rush'] = self.pbp['rush'].astype(bool)
                    self.pbp['qb_hit'] = self.pbp['qb_hit'].astype(bool)
                    self.pbp['sack'] = self.pbp['sack'].astype(bool)
                    self.pbp['qb_dropback'] = self.pbp['qb_dropback'].astype(
                        bool)
                    self.pbp['pass_attempt'] = self.pbp['pass_attempt'].astype(
                        bool)
                    self.pbp['drive_inside20'] = self.pbp['drive_inside20'].astype(
                        bool)
                    self.ngs = pd.concat([nfl.import_ngs_data('passing', [self.season_start.year]),
                                          nfl.import_ngs_data(
                                              'receiving', [self.season_start.year]),
                                          nfl.import_ngs_data('rushing', [self.season_start.year])])
                    need_pbp = False

                playerData = self.parse_pbp(
                    row['week'], row['recent team'], row['player display name'])
                if type(playerData) is int:
                    self.gamelog.drop(index=i, inplace=True)
                    continue
                if row['recent team'] not in self.teamlog.loc[(self.teamlog.season == row.season) &
                                                              (self.teamlog.week == row.week), 'team'] and \
                        (row['week'], row['recent team']) not in [(t['week'], t['team']) for t in teamDataList]:
                    teamData = {
                        "season": row.season,
                        "week": row.week,
                        "team": row['recent team']
                    }
                    teamData.update(self.parse_pbp(
                        row['week'], row['recent team']))
                    teamDataList.append(teamData)

                for k, v in playerData.items():
                    self.gamelog.at[i, k.replace("_", " ")] = v

                if row['recent team'] in sched.loc[sched['week'] == row['week'], 'home_team'].unique():
                    self.gamelog.at[i, 'home'] = True
                    self.gamelog.at[i, 'opponent'] = sched.loc[(sched['week'] == row['week']) & (sched['home_team']
                                                               == row['recent team']), 'away_team'].values[0]
                    self.gamelog.at[i, 'gameday'] = sched.loc[(sched['week'] == row['week']) & (sched['home_team']
                                                                                                == row['recent team']), 'gameday'].values[0]
                    self.gamelog.at[i, 'game id'] = sched.loc[(sched['week'] == row['week']) & (sched['home_team']
                                                                                                == row['recent team']), 'game_id'].values[0]
                else:
                    self.gamelog.at[i, 'home'] = False
                    self.gamelog.at[i, 'opponent'] = sched.loc[(sched['week'] == row['week']) & (sched['away_team']
                                                               == row['recent team']), 'home_team'].values[0]
                    self.gamelog.at[i, 'gameday'] = sched.loc[(sched['week'] == row['week']) & (sched['away_team']
                                                                                                == row['recent team']), 'gameday'].values[0]
                    self.gamelog.at[i, 'game id'] = sched.loc[(sched['week'] == row['week']) & (sched['away_team']
                                                                                                == row['recent team']), 'game_id'].values[0]

        self.teamlog = pd.concat(
            [self.teamlog, pd.DataFrame.from_records(teamDataList)], ignore_index=True)
        self.teamlog.loc[self.teamlog['team']
                         == 'LA', 'team'] = "LAR"
        self.teamlog = self.teamlog.sort_values(['season', 'week']).fillna(0)

        self.gamelog.loc[self.gamelog['recent team']
                         == 'LA', 'recent team'] = "LAR"
        self.gamelog.loc[self.gamelog['recent team']
                         == 'WSH', 'recent team'] = "WAS"
        self.gamelog.loc[self.gamelog['opponent']
                         == 'LA', 'opponent'] = "LAR"
        self.gamelog.loc[self.gamelog['opponent']
                         == 'WSH', 'opponent'] = "WAS"
        self.gamelog = self.gamelog.sort_values('gameday')

        # Remove old games to prevent file bloat
        three_years_ago = datetime.today().date() - timedelta(days=1096)
        self.gamelog = self.gamelog[self.gamelog["gameday"].apply(
            lambda x: three_years_ago <= datetime.strptime(x, '%Y-%m-%d').date() <= datetime.today().date())]
        self.gamelog = self.gamelog[~self.gamelog['opponent'].isin([
                                                                   "AFC", "NFC"])]
        self.gamelog.drop_duplicates(inplace=True)

        # Save the updated player data
        filepath = pkg_resources.files(data) / "nfl_data.dat"
        with open(filepath, 'wb') as outfile:
            pickle.dump({'gamelog': self.gamelog,
                        'teamlog': self.teamlog}, outfile, -1)

    def parse_pbp(self, week, team, playerName=""):
        pbp = self.pbp.loc[(self.pbp.week == week) & (
            (self.pbp.home_team == team) | (self.pbp.away_team == team))]
        if pbp.empty:
            return 0
        pbp_off = pbp.loc[pbp.posteam == team]
        pbp_def = pbp.loc[pbp.posteam != team]
        if playerName == "":
            proe = pbp_off['pass'].mean() - pbp_off['xpass'].mean()
            off_rush_sr = pbp_off.loc[pbp_off['rush'], 'success'].mean()
            off_pass_sr = pbp_off.loc[pbp_off['pass'], 'success'].mean()
            def_rush_sr = pbp_def.loc[pbp_def['rush'], 'success'].mean()
            def_pass_sr = pbp_def.loc[pbp_def['pass'], 'success'].mean()
            off_rush_epa = pbp_off.loc[pbp_off['rush'], 'epa'].mean()
            off_pass_epa = pbp_off.loc[pbp_off['pass'], 'epa'].mean()
            def_rush_epa = pbp_def.loc[pbp_def['rush'], 'epa'].mean()
            def_pass_epa = pbp_def.loc[pbp_def['pass'], 'epa'].mean()
            def_rush_ypa = pbp_def.loc[pbp_def['rush'], 'yards_gained'].mean()
            def_pass_ypa = pbp_def.loc[pbp_def['pass'], 'yards_gained'].mean()
            def_cpoe = pbp_def.loc[pbp_def['pass'], 'cpoe'].mean() / 100
            def_cp = pbp_def.loc[pbp_def['pass'], 'complete_pass'].mean()
            pressure_mask = pbp_def['is_qb_out_of_pocket'] | pbp_def['sack'] | pbp_def['qb_hit']
            def_press = pbp_def.loc[pbp_def['pass'] & pressure_mask, 'pass'].count(
            ) / len(pbp_def.loc[pbp_def['qb_dropback']])
            def_stuff = (pbp_def.loc[pbp_def['rush'],
                         'yards_gained'] <= 0).mean()
            pressure_mask = pbp_off['is_qb_out_of_pocket'] | pbp_off['sack'] | pbp_off['qb_hit']
            off_press = pbp_off.loc[pbp_off['pass'] & pressure_mask, 'pass'].count(
            ) / len(pbp_off.loc[pbp_off['qb_dropback']])
            off_stuff = (pbp_off.loc[pbp_off['rush'],
                         'yards_gained'] <= 0).mean()
            rush_ngs = self.ngs.loc[(self.ngs['player_position'] == 'RB') & (self.ngs['team_abbr'] == team) & (
                self.ngs['week'] == pbp.week.max()), ['expected_rush_yards', 'rush_attempts']].sum()
            off_rush_xya = rush_ngs.iloc[0]/rush_ngs.iloc[1]
            blitz_rate = pbp_def.loc[pbp_def['pass'] & (pbp_def['n_blitzers'] > 0), 'n_blitzers'].count(
            ) / len(pbp_def.loc[pbp_def['qb_dropback']])
            blitz_epa = pbp_def.loc[pbp_def['pass'] & (
                pbp_def['n_blitzers'] > 0), 'epa'].mean()

            return {
                "pass_rate_over_expected": proe,
                "rush_success_rate": off_rush_sr,
                "pass_success_rate": off_pass_sr,
                "rush_success_rate_allowed": def_rush_sr,
                "pass_success_rate_allowed": def_pass_sr,
                "epa_per_rush": off_rush_epa,
                "epa_per_pass": off_pass_epa,
                "epa_allowed_per_rush": def_rush_epa,
                "epa_allowed_per_pass": def_pass_epa,
                "yards_allowed_per_rush": def_rush_ypa,
                "yards_allowed_per_pass": def_pass_ypa,
                "completion_percentage_allowed": def_cp,
                "cpoe_allowed": def_cpoe,
                "pressure_per_pass": def_press,
                "stuffs_per_rush": def_stuff,
                "pressure_allowed_per_pass": off_press,
                "stuffs_allowed_per_rush": off_stuff,
                "expected_yards_per_rush": off_rush_xya,
                "blitz_rate": blitz_rate,
                "epa_per_blitz": blitz_epa
            }

        else:
            if self.ids.get(playerName) is None:
                return {
                    "completion_percentage_over_expected": 0,
                    "completion_percentage": 0,
                    "passer_rating": 0,
                    "passer_adot": 0,
                    "passer_adot_differential": 0,
                    "time_to_throw": 0,
                    "aggressiveness": 0,
                    "pass_yards_per_attempt": 0,
                    "rushing_yards_over_expected": 0,
                    "rushing_success_rate": 0,
                    "yac_over_expected": 0,
                    "separation_created": 0,
                    "targets_per_route_run": 0,
                    "first_read_targets_per_route_run": 0,
                    "route_participation": 0,
                    "yards_per_route_run": 0,
                    "average_depth_of_target": 0,
                    "receiver_cp_over_expected": 0,
                    "first_read_target_share": 0,
                    "redzone_target_share": 0,
                    "redzone_carry_share": 0,
                    "carry_share": 0
                }

            cpoe = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'completion_percentage_above_expectation'].mean() / 100
            cp = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'completion_percentage'].mean() / 100
            qbr = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'passer_rating'].mean()
            pass_adot = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'avg_intended_air_yards'].mean()
            pass_adot_diff = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'avg_air_yards_differential'].mean()
            time_to_throw = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'avg_time_to_throw'].mean()
            aggressiveness = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'aggressiveness'].mean()
            ryoe = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'rush_yards_over_expected'].mean()
            rush_sr = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'rush_pct_over_expected'].mean()
            yacoe = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'avg_yac_above_expectation'].mean()
            sep = self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                self.ngs['week'] == pbp.week.max()), 'avg_separation'].mean() - \
                self.ngs.loc[(self.ngs['player_display_name'] == playerName) & (
                    self.ngs['week'] == pbp.week.max()), 'avg_cushion'].mean()
            pass_ypa = pbp_off.loc[pbp_off['passer_player_id'] == self.ids.get(
                playerName), 'yards_gained'].sum() / pbp_off.loc[pbp_off['passer_player_id'] == self.ids.get(playerName), 'qb_dropback'].sum()
            routes_run = len(pbp_off.loc[pbp_off.offense_players.str.contains(
                self.ids.get(playerName)) & pbp_off['pass']])
            targets = len(
                pbp_off.loc[pbp_off['receiver_player_id'] == self.ids.get(playerName)])
            fr_targets = len(pbp_off.loc[(pbp_off['receiver_player_id'] == self.ids.get(
                playerName)) & (pbp_off['read_thrown'] == "1")])
            pass_attempts = len(pbp_off.loc[pbp_off['pass']])
            fr_pass_attempts = len(
                pbp_off.loc[(pbp_off['pass']) & (pbp_off['read_thrown'] == "1")])
            tprr = targets / routes_run if routes_run > 0 else np.nan
            frtprr = fr_targets / routes_run if routes_run > 0 else np.nan
            frt_pct = fr_targets / fr_pass_attempts if fr_pass_attempts > 0 else np.nan
            route_participation = routes_run / pass_attempts if pass_attempts > 0 else np.nan
            yprr = pbp_off.loc[pbp_off['receiver_player_id'] == self.ids.get(
                playerName), 'yards_gained'].sum() / routes_run if routes_run > 0 else np.nan
            adot = pbp_off.loc[pbp_off['receiver_player_id']
                               == self.ids.get(playerName), 'air_yards'].mean()
            rec_cpoe = pbp_off.loc[pbp_off['receiver_player_id']
                                   == self.ids.get(playerName), 'cpoe'].mean() / 100
            rz_passes = len(
                pbp_off.loc[pbp_off['pass_attempt'] & pbp_off['drive_inside20']])
            rz_target_pct = len(pbp_off.loc[(pbp_off['receiver_player_id'] == self.ids.get(
                playerName)) & pbp_off['drive_inside20']]) / rz_passes if rz_passes > 0 else np.nan
            rz_rushes = len(
                pbp_off.loc[pbp_off['rush'] & pbp_off['drive_inside20']])
            rz_attempt_pct = len(pbp_off.loc[(pbp_off['rusher_player_id'] == self.ids.get(
                playerName)) & pbp_off['drive_inside20']]) / rz_rushes if rz_rushes > 0 else np.nan
            rushes = len(pbp_off.loc[pbp_off['rush']])
            attempt_pct = len(pbp_off.loc[pbp_off['rusher_player_id'] == self.ids.get(
                playerName)]) / rushes if rushes > 0 else np.nan

            return {
                "completion_percentage_over_expected": cpoe,
                "completion_percentage": cp,
                "passer_rating": qbr,
                "passer_adot": pass_adot,
                "passer_adot_differential": pass_adot_diff,
                "time_to_throw": time_to_throw,
                "aggressiveness": aggressiveness,
                "pass_yards_per_attempt": pass_ypa,
                "rushing_yards_over_expected": ryoe,
                "rushing_success_rate": rush_sr,
                "yac_over_expected": yacoe,
                "separation_created": sep,
                "targets_per_route_run": tprr,
                "first_read_targets_per_route_run": frtprr,
                "route_participation": route_participation,
                "yards_per_route_run": yprr,
                "average_depth_of_target": adot,
                "receiver_cp_over_expected": rec_cpoe,
                "first_read_target_share": frt_pct,
                "redzone_target_share": rz_target_pct,
                "redzone_carry_share": rz_attempt_pct,
                "carry_share": attempt_pct
            }

    def bucket_stats(self, market, buckets=20, date=datetime.today()):
        """
        Bucket player stats based on a given market.

        Args:
            market (str): The market to bucket the player stats (e.g., 'PTS', 'REB', 'AST').
            buckets (int): The number of buckets to divide the stats into (default: 20).

        Returns:
            None.
        """

        if market == self.bucket_market and date == self.bucket_latest_date:
            return

        self.bucket_market = market
        self.bucket_latest_date = date

        # Reset playerStats and edges
        self.playerStats = pd.DataFrame()
        self.edges = []

        # Collect stats for each player
        one_year_ago = date - timedelta(days=300)
        gameDates = pd.to_datetime(self.gamelog["gameday"]).dt.date
        gamelog = self.gamelog[(one_year_ago <= gameDates)
                               & (gameDates < date)]

        playerGroups = gamelog.\
            groupby('player display name').\
            filter(lambda x: len(x[x[market] != 0]) > 4).\
            groupby('player display name')[market]

        # Compute edges for each bucket
        w = int(100 / buckets)
        self.edges = playerGroups.mean().quantile(
            np.arange(0, 101, w)/100).to_list()

        # Assign bucket and line values to each player
        self.playerStats['avg'] = playerGroups.mean()
        self.playerStats['bucket'] = np.ceil(
            playerGroups.mean().rank(pct=True, ascending=False)*20).astype(int)
        self.playerStats['line'] = playerGroups.median()
        self.playerStats.loc[self.playerStats['line'] == 0.0, 'line'] = 0.5
        self.playerStats.loc[(np.mod(self.playerStats['line'], 1) == 0) & (
            self.playerStats['avg'] > self.playerStats['line']), 'line'] += 0.5
        self.playerStats.loc[(np.mod(self.playerStats['line'], 1) == 0) & (
            self.playerStats['avg'] < self.playerStats['line']), 'line'] -= 0.5

        self.playerStats = self.playerStats.to_dict(orient='index')

    def profile_market(self, market, date=datetime.today().date()):
        if type(date) is str:
            date = datetime.strptime(date, "%Y-%m-%d").date()
        if type(date) is datetime:
            date = date.date()

        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.profiled_market = market
        self.profile_latest_date = date

        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.defenseProfile = pd.DataFrame(columns=['avg', 'home', 'away'])

        one_year_ago = date - timedelta(days=300)
        gameDates = pd.to_datetime(self.gamelog["gameday"]).dt.date
        gamelog = self.gamelog[(
            one_year_ago <= gameDates) & (gameDates < date)].copy()
        gameDates = pd.to_datetime(self.teamlog["gameday"]).dt.date
        teamlog = self.teamlog[(
            one_year_ago <= gameDates) & (gameDates < date)].copy()

        # Retrieve moneyline and totals data from archive
        k = [[(date, team) for team in teams.keys()]
             for date, teams in archive["NFL"]["Moneyline"].items()]
        k = [item for row in k for item in row]
        flat_money = {t: archive["NFL"]["Moneyline"].get(
            t[0], {}).get(t[1], 0.5) for t in k}
        flat_total = {t: archive["NFL"]["Totals"].get(
            t[0], {}).get(t[1], 22.5) for t in k}
        tup_s = pd.Series(
            zip(gamelog['gameday'], gamelog['recent team']), index=gamelog.index)
        gamelog.loc[:, "moneyline"] = tup_s.map(flat_money)
        gamelog.loc[:, "totals"] = tup_s.map(flat_total)

        teamstats = teamlog.groupby('team').tail(5).groupby('team')[
            teamlog.columns[3:]].mean()

        playerGroups = gamelog.\
            groupby('player display name').\
            filter(lambda x: (x[market].clip(0, 1).mean() > 0.3) & (x[market].count() > 1)).\
            groupby('player display name')

        defenseGroups = gamelog.groupby(['opponent', 'game id'])
        defenseGames = pd.DataFrame()
        defenseGames[market] = defenseGroups[market].sum()
        defenseGames['home'] = defenseGroups['home'].mean().astype(int)
        defenseGames['moneyline'] = defenseGroups['moneyline'].mean()
        defenseGames['totals'] = defenseGroups['totals'].mean()
        defenseGroups = defenseGames.groupby('opponent')

        leagueavg = playerGroups[market].mean().mean()
        if np.isnan(leagueavg):
            return

        self.playerProfile['avg'] = playerGroups[market].mean().div(
            leagueavg) - 1
        self.playerProfile['home'] = playerGroups.apply(
            lambda x: x.loc[x['home'], market].mean() / x[market].mean()) - 1
        self.playerProfile['away'] = playerGroups.apply(
            lambda x: x.loc[~x['home'].astype(bool), market].mean()/x[market].mean())-1

        leagueavg = defenseGroups[market].mean().mean()
        self.defenseProfile['avg'] = defenseGroups[market].mean().div(
            leagueavg) - 1
        self.defenseProfile['home'] = defenseGroups.apply(
            lambda x: x.loc[x['home'] == 1, market].mean() / x[market].mean()) - 1
        self.defenseProfile['away'] = defenseGroups.apply(
            lambda x: x.loc[x['home'] == 0, market].mean()/x[market].mean())-1

        if any([string in market for string in ["pass", "completions", "attempts", "interceptions", "qb"]]):
            positions = ['QB']
            stat_types = self.stat_types['passing'] + \
                self.stat_types['rushing']
        elif any([string in market for string in ["rush", "carries"]]):
            positions = ['QB', 'RB']
            stat_types = self.stat_types['rushing']
        elif any([string in market for string in ["receiving", "targets", "receptions"]]):
            positions = ['WR', 'RB', 'TE']
            stat_types = self.stat_types['receiving']
        elif market == "tds" or market == "yards":
            positions = ['WR', 'RB', 'TE']
            stat_types = self.stat_types['receiving'] + \
                self.stat_types['rushing']
        else:
            positions = ['QB', 'WR', 'RB', 'TE']
            stat_types = self.stat_types['passing'] + \
                self.stat_types['rushing'] + \
                self.stat_types['receiving']

        playerstats = gamelog.fillna(0).groupby('player display name')[
            stat_types].mean(numeric_only=True)
        for position in positions:
            positionGroups = gamelog.loc[gamelog['position group'] == position].groupby(
                ['opponent', 'game id'])
            defenseGames = pd.DataFrame()
            defenseGames[market] = positionGroups[market].sum()
            defenseGames['home'] = positionGroups['home'].mean().astype(int)
            defenseGames['moneyline'] = positionGroups['moneyline'].mean()
            defenseGames['totals'] = positionGroups['totals'].mean()
            positionGroups = defenseGames.groupby('opponent')
            leagueavg = positionGroups[market].mean().mean()
            if leagueavg == 0:
                self.defenseProfile[position] = 0
            else:
                self.defenseProfile[position] = positionGroups[market].mean().div(
                    leagueavg) - 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.playerProfile['moneyline gain'] = playerGroups.\
                apply(lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                           x[market].values.astype(float)/x[market].mean() - 1, 1)[0])

            self.playerProfile['totals gain'] = playerGroups.\
                apply(lambda x: np.polyfit(x.totals.fillna(22.5).values.astype(float) / 22.5 - x.totals.fillna(22.5).mean(),
                                           x[market].values.astype(float)/x[market].mean() - 1, 1)[0])

            self.defenseProfile['moneyline gain'] = defenseGroups.\
                apply(lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                           x[market].values.astype(float)/x[market].mean() - 1, 1)[0])

            self.defenseProfile['totals gain'] = defenseGroups.\
                apply(lambda x: np.polyfit(x.totals.fillna(22.5).values.astype(float) / 22.5 - x.totals.fillna(22.5).mean(),
                                           x[market].values.astype(float)/x[market].mean() - 1, 1)[0])

        i = self.defenseProfile.index
        self.defenseProfile = self.defenseProfile.merge(
            teamstats[self.stat_types['defense']], left_on='opponent', right_on='team')
        self.defenseProfile.index = i

        self.teamProfile = teamstats[self.stat_types['offense']]

        self.playerProfile = self.playerProfile.merge(
            playerstats[stat_types], on='player display name')

    def dvpoa(self, team, position, market, date=datetime.today().date()):
        """
        Calculate the Defense Versus Position Above Average (DVPOA) for a specific team, position, and market.

        Args:
            team (str): The team abbreviation.
            position (str): The player's position.
            market (str): The market to calculate performance against (e.g., 'PTS', 'REB', 'AST').

        Returns:
            float: The calculated performance value.
        """

        if date != self.dvpoa_latest_date:
            self.dvp_index = {}
            self.dvpoa_latest_date = date

        if market not in self.dvp_index:
            self.dvp_index[market] = {}

        if team not in self.dvp_index[market]:
            self.dvp_index[market][team] = {}

        if position in self.dvp_index[market][team]:
            return self.dvp_index[market][team][position]

        position_games = self.gamelog.loc[(self.gamelog['position group'] == position) & (
            pd.to_datetime(self.gamelog["gameday"]) < date)]
        team_games = position_games.loc[position_games['opponent'] == team]

        if len(team_games) == 0:
            return 0
        else:
            dvp = team_games[market].mean()
            leagueavg = position_games[market].mean()
            dvpoa = (dvp - leagueavg) / leagueavg
            dvpoa = np.nan_to_num(dvpoa, nan=0.0, posinf=0.0, neginf=0.0)
            self.dvp_index[market][team][position] = dvpoa
            return dvpoa

    def get_stats(self, offer, date=datetime.today()):
        """
        Generate a pandas DataFrame with a summary of relevant stats.

        Args:
            offer (dict): The offer details containing 'Player', 'Team', 'Market', 'Line', and 'Opponent'.
            date (datetime or str): The date of the stats (default: today's date).

        Returns:
            pandas.DataFrame: The generated DataFrame with the summary of stats.
        """
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        player = offer["Player"]
        team = offer["Team"]
        market = offer["Market"]
        line = offer["Line"]
        opponent = offer["Opponent"]
        if self.defenseProfile.empty:
            logger.exception(f"{market} not profiled")
            return 0
        home = offer.get("Home")
        if home is None:
            home = self.upcoming_games.get(team, {}).get("Home", 0)

        if player not in self.playerProfile.index:
            self.playerProfile.loc[player] = np.zeros_like(
                self.playerProfile.columns)

        try:
            stats = (
                archive["NFL"]
                .get(market, {})
                .get(date, {})
                .get(player, {})
                .get(line, [0] * 4)
            )
            moneyline = archive["NFL"]["Moneyline"].get(
                date, {}).get(team, 0)
            total = archive["NFL"]["Totals"].get(date, {}).get(team, 0)

        except:
            logger.exception(f"{player}, {market}")
            return 0

        date = datetime.strptime(date, "%Y-%m-%d")

        player_games = self.gamelog.loc[(self.gamelog["player display name"] == player) & (
            pd.to_datetime(self.gamelog["gameday"]) < date)]
        position = self.players.get(player, "")
        one_year_ago = len(player_games.loc[pd.to_datetime(
            self.gamelog["gameday"]) > date-timedelta(days=300)])
        if one_year_ago < 2:
            return 0

        if position == "":
            if len(player_games) > 0:
                position = player_games.iat[0, 2]
            else:
                logger.warning(f"{player} not found")
                return 0

        if position not in self.defenseProfile.columns:
            return 0

        headtohead = player_games.loc[player_games["opponent"] == opponent]

        game_res = (player_games[market]).to_list()
        h2h_res = (headtohead[market]).to_list()

        dvpoa = self.defenseProfile.loc[opponent, position]

        stats = np.array(stats, dtype=np.float64)
        odds = np.nanmean(stats)
        if np.isnan(odds):
            odds = 0

        data = {
            "DVPOA": dvpoa,
            "Odds": odds,
            "Line": line,
            "Avg5": np.median(game_res[-5:]) if game_res else 0,
            "Avg10": np.median(game_res[-10:]) if game_res else 0,
            "AvgYr": np.median(game_res[-one_year_ago:]) if game_res else 0,
            "AvgH2H": np.median(h2h_res[-5:]) if h2h_res else 0,
            "IQR10": iqr(game_res[-10:]) if game_res else 0,
            "IQRYr": iqr(game_res[-one_year_ago:]) if game_res else 0,
            "Mean5": np.mean(game_res[-5:]) if game_res else 0,
            "Mean10": np.mean(game_res[-10:]) if game_res else 0,
            "MeanYr": np.mean(game_res[-one_year_ago:]) if game_res else 0,
            "MeanH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
            "GamesPlayed": one_year_ago,
            "Moneyline": moneyline,
            "Total": total,
            "Home": home,
            "Position": ["QB", "WR", "RB", "TE"].index(position)
        }

        if len(game_res) < 5:
            i = 5 - len(game_res)
            game_res = [0] * i + game_res
        if len(h2h_res) < 5:
            i = 5 - len(h2h_res)
            h2h_res = [0] * i + h2h_res

        # Update the data dictionary with additional values
        data.update(
            {"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
        data.update({"Game " + str(i + 1): game_res[-5 + i] for i in range(5)})

        player_data = self.playerProfile.loc[player]
        data.update(
            {"Player " + col: player_data[col] for col in player_data.index})

        team_data = self.teamProfile.loc[team]
        data.update(
            {"Team " + col: team_data[col] for col in team_data.index})

        defense_data = self.defenseProfile.loc[opponent]
        data.update(
            {"Defense " + col: defense_data[col] for col in defense_data.index if col not in ["QB", "WR", "RB", "TE"]})

        return data

    def get_training_matrix(self, market):
        """
        Retrieves training data in the form of a feature matrix (X) and a target vector (y) for a specified market.

        Args:
            market (str): The market for which to retrieve training data.

        Returns:
            tuple: A tuple containing the feature matrix (X) and the target vector (y).
        """
        archive.__init__("NFL")

        # Initialize an empty list for the target labels
        matrix = []

        for i, game in tqdm(self.gamelog.iterrows(), unit="game", desc="Gathering Training Data", total=len(self.gamelog)):
            gameDate = datetime.strptime(
                game["gameday"], "%Y-%m-%d")

            if gameDate < datetime(2021, 9, 1):
                continue

            self.profile_market(market, date=gameDate)
            name = game['player display name']

            if name not in self.playerProfile.index:
                continue

            if game[market] < 0:
                continue

            data = archive["NFL"][market].get(gameDate.strftime(
                "%Y-%m-%d"), {}).get(name, {0: [0.5]*4})

            lines = [k for k, v in data.items()]
            if "Closing Lines" in lines:
                lines.remove("Closing Lines")
                lines.append(
                    np.floor(np.mean([float(i['Line']) for i in data["Closing Lines"] if i is not None]))+0.5)

            line = lines[-1]

            offer = {
                "Player": name,
                "Team": game["recent team"],
                "Market": market,
                "Opponent": game["opponent"],
                "Home": int(game["home"])
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_get_stats = self.get_stats(
                    offer | {"Line": line}, gameDate
                )
                if type(new_get_stats) is dict:

                    new_get_stats.update(
                        {"Result": game[market]}
                    )

                    matrix.append(new_get_stats)

        M = pd.DataFrame(matrix).fillna(0.0).replace([np.inf, -np.inf], 0)

        return M

    def get_fantasy(self):
        """
        Retrieves fantasy points stats.

        Args:

        Returns:
            tuple: A tuple containing the feature matrix (X) and the target vector (y).
        """

        # Initialize an empty list for the target labels
        matrix = []
        i = []

        self.profile_market('fantasy points underdog')
        depth = nfl.import_depth_charts([self.season_start.year])
        depth = depth.loc[depth.position.isin(["QB", "WR", "RB", "TE"])]
        depth = depth.loc[depth.week == depth.week.iloc[-500:].mode().iloc[0]]
        depth = depth.loc[(depth.depth_team.astype(int) == 1) | ~(
            depth.position.isin(["QB", "TE"]))]
        depth = depth.loc[depth.depth_team.astype(int) <= 2]
        depth.loc[depth['club_code'] == 'LA', 'club_code'] = "LAR"
        players = pd.Series(zip(depth['full_name'].map(
            remove_accents), depth['club_code'])).drop_duplicates().to_list()

        for player, team in tqdm(players, unit="player"):

            gameDate = self.upcoming_games.get(team, {}).get(
                'gameday', datetime.today().strftime("%Y-%m-%d"))
            opponent = self.upcoming_games.get(team, {}).get(
                'Opponent', datetime.today().strftime("%Y-%m-%d"))
            home = self.upcoming_games.get(team, {}).get(
                'Home', datetime.today().strftime("%Y-%m-%d"))
            data = archive["NFL"]['fantasy points underdog'].get(
                gameDate, {}).get(player, {0: [0.5]*4})

            lines = [k for k, v in data.items()]
            if "Closing Lines" in lines:
                lines.remove("Closing Lines")
                lines.append(
                    np.floor(np.mean([float(i['Line']) for i in data["Closing Lines"] if i is not None]))+0.5)

            line = lines[-1]

            offer = {
                "Player": player,
                "Team": team,
                "Market": 'fantasy points underdog',
                "Opponent": opponent,
                "Home": home
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_get_stats = self.get_stats(
                    offer | {"Line": line}, gameDate
                )
                if type(new_get_stats) is dict:
                    matrix.append(new_get_stats)
                    i.append(player)

        M = pd.DataFrame(matrix, index=i).fillna(
            0.0).replace([np.inf, -np.inf], 0)

        return M


class StatsNHL(Stats):
    """
    A class for handling and analyzing NHL statistics.
    Inherits from the Stats parent class.

    Additional Attributes:
        skater_data (list): A list containing skater statistics.
        goalie_data (list): A list containing goalie statistics.
        season_start (datetime.datetime): The start date of the season.

    Additional Methods:
        None
    """

    def __init__(self):
        super().__init__()
        self.season_start = datetime(2023, 10, 10)

    def load(self):
        """
        Loads NHL skater and goalie data from files.

        Args:
            None

        Returns:
            None
        """
        filepath = pkg_resources.files(data) / "nhl_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                nhl_data = pickle.load(infile)
                self.gamelog = nhl_data["gamelog"]
                self.teamlog = nhl_data["teamlog"]

    def parse_game(self, gameId, gameDate):
        game = scraper.get(
            f"https://statsapi.web.nhl.com/api/v1/game/{gameId}/linescore")
        season = str(gameId)[:4]
        season = season + str(int(season)+1)
        res = requests.get(
            f"https://moneypuck.com/moneypuck/playerData/games/{season}/{gameId}.csv").text
        res = [row.split(',') for row in res.split('\n')]
        if res[1][0] != str(gameId):
            return 0
        game_df = pd.DataFrame(res[1:-1], columns=res[0])
        pp_df = game_df.loc[game_df.situation == '5on4']
        game_df = game_df.loc[game_df.situation == 'all']
        gamelog = []
        teamlog = []
        if game and not game_df.empty:
            awayTeam = abbreviations['NHL'][remove_accents(game['teams']
                                            ['away']['team']['name'])]
            homeTeam = abbreviations['NHL'][remove_accents(game['teams']
                                            ['home']['team']['name'])]

            team_map = {
                "SJS": "SJ",
                "LAK": "LA",
                "NJD": "NJ",
                "TBL": "TB",
                "WSH": "WAS"
            }

            for i, player in game_df.iterrows():
                team = player['team']
                team = team_map.get(team, team)
                home = team == homeTeam
                opponent = awayTeam if home else homeTeam
                win = (game['teams']['home']['goals'] >
                       game['teams']['away']['goals']) == home

                if player['position'] == 'Team Level':
                    n = {
                        "gameId": gameId,
                        "gameDate": gameDate,
                        "team": team,
                        "opponent": opponent,
                        "home": home
                    }
                    stats = {
                        "Corsi": float(player["OffIce_shotAttempts_For_Percentage"]),
                        "Fenwick": float(player["OffIce_unblockedShotAttempts_For_Percentage"]),
                        "Hits": float(player["OffIce_hits_For_Percentage"]),
                        "Takeaways": float(player["OffIce_takeaways_For_Percentage"]),
                        "xGoals": float(player["OffIce_flurryScoreVenueAdjustedxGoals_For_Percentage"]),
                        "PIM": float(player["OffIce_penalityMinutes_For_Percentage"])
                    }
                    stats["GOE"] = float(
                        player["OffIce_F_goals"]) - stats["xGoals"]
                    teamlog.append(n | stats)
                else:
                    n = {
                        "gameId": gameId,
                        "gameDate": gameDate,
                        "team": team,
                        "opponent": opponent,
                        "opponent goalie": game_df.loc[(game_df.position == "G") & (game_df.team != team), 'playerName'].iat[0],
                        "home": home,
                        "playerId": player['playerId'],
                        "playerName": player['playerName'],
                        "position": player['position']
                    }
                    stats = {
                        "points": float(player["I_F_points"]),
                        "shots": float(player["I_F_shotsOnGoal"]),
                        "blocked": float(player["I_A_blockedShotAttempts"]),
                        "sogBS": float(player["I_F_shotsOnGoal"]) + float(player["I_A_blockedShotAttempts"]),
                        "goals": float(player["I_F_goals"]),
                        "assists": float(player["I_F_primaryAssists"]) + float(player["I_F_secondaryAssists"]),
                        "hits": float(player["I_F_hits"]),
                        "faceOffWins": float(player["I_F_faceOffsWon"]),
                        "timeOnIce": float(player["I_F_iceTime"])/60,
                        "saves": float(player["OnIce_A_savedShotsOnGoal"]),
                        "goalsAgainst": float(player["OnIce_A_goals"])
                    }
                    if player["playerName"] in pp_df["playerName"].to_list():
                        stats["powerPlayPoints"] = float(
                            pp_df.loc[pp_df["playerName"] == player["playerName"]]["I_F_points"].iat[0])
                    else:
                        stats["powerPlayPoints"] = 0
                    stats.update({
                        "fantasy points prizepicks": stats.get("goals", 0)*8 + stats.get("assists", 0)*5 + stats.get("sogBS", 0)*1.5,
                        "goalie fantasy points underdog": int(win)*6 + stats.get("saves", 0)*.6 - stats.get("goalsAgainst", 0)*.6,
                        "skater fantasy points underdog": stats.get("goals", 0)*6 + stats.get("assists", 0)*4 + stats.get("sogBS", 0) + stats.get("hits", 0) + stats.get("powerPlayPoints", 0)*.5,
                        "goalie fantasy points parlay": stats.get("saves", 0)*.25 - stats.get("goalsAgainst", 0),
                        "skater fantasy points parlay": stats.get("goals", 0)*3 + stats.get("assists", 0)*2 + stats.get("shots", 0)*.5 + stats.get("hits", 0) + stats.get("blocked", 0),
                    })
                    team = {v: k for k, v in team_map.items()}.get(team, team)
                    stats.update({
                        "GOE": stats["goals"] - float(player["I_F_flurryScoreVenueAdjustedxGoals"]),
                        "Fenwick": float(player["OnIce_unblockedShotAttempts_For_Percentage"]),
                        "TimeShare": stats["timeOnIce"]/(float(game_df.loc[game_df["playerName"] == team, "OffIce_F_iceTime"].iat[0])/60),
                        "ShotShare": stats["shots"]/float(game_df.loc[game_df["playerName"] == team, "OffIce_F_shotsOnGoal"].iat[0]),
                        "Shot60": stats["shots"]*60/stats["timeOnIce"],
                        "Blk60": stats["blocked"]*60/stats["timeOnIce"],
                        "Hit60": stats["hits"]*60/stats["timeOnIce"],
                        "Ast60": stats["assists"]*60/stats["timeOnIce"],
                        "SOE": float(player["OnIce_A_xGoals"]) - stats["goalsAgainst"]
                    })
                    if stats["timeOnIce"] > 6:
                        gamelog.append(n | stats)

        return gamelog, teamlog

    def update(self):
        """
        Updates the NHL skater and goalie data.

        Args:
            None

        Returns:
            None
        """

        # Get game ids
        latest_date = self.season_start
        if not self.gamelog.empty:
            latest_date = datetime.strptime(
                self.gamelog["gameDate"].max().split("T")[0], "%Y-%m-%d") + timedelta(days=1)
        today = datetime.today().date()
        start_date = latest_date.strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")
        res = scraper.get(
            f"https://statsapi.web.nhl.com/api/v1/schedule?startDate={start_date}&endDate={end_date}")

        ids = [[(game['gamePk'], date['date']) for game in date['games'] if game['gameType']
                not in ["PR", "A"]] for date in res['dates'] if datetime.strptime(date["date"], "%Y-%m-%d").date() < today]

        ids = [item for sublist in ids for item in sublist]

        # Parse the game stats
        nhl_gamelog = []
        nhl_teamlog = []
        for gameId, date in tqdm(ids, desc="Getting NHL Stats"):
            gamelog, teamlog = self.parse_game(gameId, date)
            if type(gamelog) is list:
                nhl_gamelog.extend(gamelog)
            if type(teamlog) is list:
                nhl_teamlog.extend(teamlog)

        nhl_df = pd.DataFrame(nhl_gamelog).fillna(0)
        self.gamelog = pd.concat([nhl_df, self.gamelog]).sort_values(
            "gameDate").reset_index(drop=True)
        self.teamlog = pd.concat([pd.DataFrame(nhl_teamlog).fillna(0), self.teamlog]).sort_values(
            "gameDate").reset_index(drop=True)

        self.upcoming_games = {}
        ug = [[(game['teams']['away']['team']['name'], game['teams']['home']['team']['name']) for game in date['games'] if game['gameType']
               not in ["PR", "A"]] for date in res['dates'] if today <= datetime.strptime(date["date"], "%Y-%m-%d").date()]
        ug = [item for sublist in ug for item in sublist][:20]
        for away, home in ug:
            awayTeam = abbreviations['NHL'][remove_accents(away)]
            homeTeam = abbreviations['NHL'][remove_accents(home)]
            if awayTeam not in self.upcoming_games:
                self.upcoming_games[awayTeam] = {
                    "Opponent": homeTeam,
                    "Home": 0
                }
            if homeTeam not in self.upcoming_games:
                self.upcoming_games[homeTeam] = {
                    "Opponent": awayTeam,
                    "Home": 1
                }

        # Remove old games to prevent file bloat
        two_years_ago = today - timedelta(days=730)
        self.gamelog = self.gamelog[pd.to_datetime(
            self.gamelog["gameDate"]).dt.date >= two_years_ago]
        self.gamelog.drop_duplicates(inplace=True)

        # Write to file
        with open((pkg_resources.files(data) / "nhl_data.dat"), "wb") as outfile:
            pickle.dump({
                "gamelog": self.gamelog,
                "teamlog": self.teamlog}, outfile, -1)

    def bucket_stats(self, market, date=datetime.today()):
        """
        Bucket the stats based on the specified market (e.g., 'goalsAgainst', 'saves').

        Args:
            market (str): The market to bucket the stats.

        Returns:
            None
        """

        if market == self.bucket_market and date == self.bucket_latest_date:
            return

        self.bucket_market = market
        self.bucket_latest_date = date

        # Initialize playerStats dictionary
        self.playerStats = {}
        self.edges = []

        # Iterate over each game in the gamelog
        for game in self.gamelog:
            if datetime.strptime(game['gameDate'], '%Y-%m-%d').date() > date.date():
                continue
            elif datetime.strptime(game['gameDate'], '%Y-%m-%d').date() < (date - timedelta(days=300)).date():
                continue
            if (market in ['goalsAgainst', 'saves'] or "goalie fantasy" in market) and game['position'] != "G":
                continue
            elif (market not in ['goalsAgainst', 'saves'] and "goalie fantasy" not in market) and game['position'] == "G":
                continue
            elif market not in game:
                continue

            # Check if the player is already in the playerStats dictionary
            if game['playerName'] not in self.playerStats:
                self.playerStats[game['playerName']] = {"games": []}

            # Append the market value to the player's games list
            self.playerStats[game['playerName']]["games"].append(game[market])

        # Filter playerStats based on minimum games played and non-zero games
        self.playerStats = {
            k: v
            for k, v in self.playerStats.items()
            if len(v["games"]) > 10 and not all([g == 0 for g in v["games"]])
        }

        # Calculate averages and store in playerStats dictionary
        averages = []
        for player, games in self.playerStats.items():
            self.playerStats[player]["avg"] = (
                np.mean(games["games"]) if games["games"] else 0
            )
            averages.append(self.playerStats[player]["avg"])

        if not len(averages):
            return

        # Calculate edges for bucketing based on percentiles
        self.edges = [np.percentile(averages, p) for p in range(0, 101, 10)]

        # Calculate lines for each bucket
        lines = np.zeros(10)
        for i in range(1, 11):
            lines[i - 1] = (
                np.round(
                    np.mean(
                        [
                            v
                            for v in averages
                            if v <= self.edges[i] and v >= self.edges[i - 1]
                        ]
                    )
                    - 0.5
                )
                + 0.5
            )

        # Assign bucket and line values to each player in playerStats
        for player, games in self.playerStats.items():
            for i in range(0, 10):
                if games["avg"] >= self.edges[i]:
                    self.playerStats[player]["bucket"] = 10 - i
                    line = np.median(games['games'])
                    if np.mod(line, 1) == 0:
                        line += 0.5 if self.playerStats[player]['avg'] >= line else -0.5
                    self.playerStats[player]["line"] = line

    def profile_market(self, market, date=datetime.today().date()):
        if type(date) is str:
            date = datetime.strptime(date, "%Y-%m-%d").date()
        if type(date) is datetime:
            date = date.date()

        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.profiled_market = market
        self.profile_latest_date = date

        # Initialize playerStats and edges
        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.defenseProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.pitcherProfile = pd.DataFrame(columns=['avg', 'home', 'away'])

        # Filter gamelog for games within the date range
        one_year_ago = (date - timedelta(days=300))
        gameDates = pd.to_datetime(self.gamelog["gameDate"]).dt.date
        gamelog = self.gamelog[(
            one_year_ago <= gameDates) & (gameDates < date)]
        gameDates = pd.to_datetime(self.teamlog["gameDate"]).dt.date
        teamlog = self.teamlog[(
            one_year_ago <= gameDates) & (gameDates < date)]

        # Filter non-starting goalies or non-starting skaters depending on the market
        if any([string in market for string in ["Against", "saves", "goalie"]]):
            gamelog = gamelog[gamelog["position"] == "G"].copy()
        else:
            gamelog2 = gamelog[gamelog["position"] == "G"].copy()
            gamelog = gamelog[gamelog["position"] != "G"].copy()

        # Retrieve moneyline and totals data from archive
        k = [[(date, team) for team in teams.keys()]
             for date, teams in archive["NHL"]["Moneyline"].items()]
        k = [item for row in k for item in row]
        flat_money = {t: archive["NHL"]["Moneyline"].get(
            t[0], {}).get(t[1], 0.5) for t in k}
        flat_total = {t: archive["NHL"]["Totals"].get(
            t[0], {}).get(t[1], 2.5) for t in k}
        tup_s = pd.Series(
            zip(gamelog['gameDate'], gamelog['team']), index=gamelog.index)
        gamelog.loc[:, "moneyline"] = tup_s.map(flat_money)
        gamelog.loc[:, "totals"] = tup_s.map(flat_total)

        teamstats = teamlog.groupby('team').tail(10).groupby('team')[
            teamlog.columns[5:]].mean()

        # Filter players with at least 2 entries
        playerGroups = gamelog.groupby('playerName').filter(
            lambda x: (x[market].clip(0, 1).mean() > 0.3) & (x[market].count() > 1)).groupby('playerName')

        defenseGroups = gamelog.groupby(['opponent', 'gameDate'])
        defenseGames = pd.DataFrame()
        defenseGames[market] = defenseGroups[market].sum()
        defenseGames['home'] = defenseGroups['home'].mean().astype(int)
        defenseGames['moneyline'] = defenseGroups['moneyline'].mean()
        defenseGames['totals'] = defenseGroups['totals'].mean()
        defenseGroups = defenseGames.groupby('opponent')

        # Compute league average
        leagueavg = playerGroups[market].mean().mean()
        if np.isnan(leagueavg):
            return

        # Compute playerProfile DataFrame
        self.playerProfile['avg'] = playerGroups[market].mean().div(
            leagueavg) - 1
        self.playerProfile['home'] = playerGroups.apply(
            lambda x: x.loc[x['home'], market].mean() / x[market].mean()) - 1
        self.playerProfile['away'] = playerGroups.apply(
            lambda x: x.loc[~x['home'], market].mean() / x[market].mean()) - 1

        leagueavg = defenseGroups[market].mean().mean()
        self.defenseProfile['avg'] = defenseGroups[market].mean().div(
            leagueavg) - 1
        self.defenseProfile['home'] = defenseGroups.apply(
            lambda x: x.loc[x['home'] == 0, market].mean() / x[market].mean()) - 1
        self.defenseProfile['away'] = defenseGroups.apply(
            lambda x: x.loc[x['home'] == 1, market].mean() / x[market].mean()) - 1

        positions = ["C", "R", "L", "D"]
        if not any([string in market for string in ["Against", "saves", "goalie"]]):
            for position in positions:
                positionGroups = gamelog.loc[gamelog['position'] == position].groupby(
                    'opponent')
                defenseGames = pd.DataFrame()
                defenseGames[market] = positionGroups[market].sum()
                defenseGames['home'] = positionGroups['home'].mean().astype(
                    int)
                defenseGames['moneyline'] = positionGroups['moneyline'].mean()
                defenseGames['totals'] = positionGroups['totals'].mean()
                positionGroups = defenseGames.groupby('opponent')
                leagueavg = positionGroups[market].mean().mean()
                if leagueavg == 0:
                    self.defenseProfile[position] = 0
                else:
                    self.defenseProfile[position] = positionGroups[market].mean().div(
                        leagueavg) - 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.playerProfile['moneyline gain'] = playerGroups.apply(
                lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

            self.playerProfile['totals gain'] = playerGroups.apply(
                lambda x: np.polyfit(x.totals.fillna(2.5).values.astype(float) / 8.3 - x.totals.fillna(2.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

            self.defenseProfile['moneyline gain'] = defenseGroups.apply(
                lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float) / 0.5 - x.moneyline.fillna(0.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

            self.defenseProfile['totals gain'] = defenseGroups.apply(
                lambda x: np.polyfit(x.totals.fillna(2.5).values.astype(float) / 8.3 - x.totals.fillna(2.5).mean(),
                                     x[market].values / x[market].mean() - 1, 1)[0])

        skater_stats = ["GOE", "Fenwick", "TimeShare",
                        "ShotShare", "Shot60", "Blk60", "Hit60", "Ast60"]
        if any([string in market for string in ["Against", "saves", "goalie"]]):
            playerstats = gamelog.fillna(0).groupby('playerName')[
                'SOE'].mean(numeric_only=True)

            self.playerProfile = self.playerProfile.merge(
                playerstats, on='playerName')
        else:
            playerstats = gamelog.fillna(0).groupby('playerName')[
                skater_stats].mean(numeric_only=True)

            self.playerProfile = self.playerProfile.merge(
                playerstats, on='playerName')

            self.goalieProfile = gamelog2.fillna(0).groupby('playerName')[
                'SOE'].mean(numeric_only=True)

        i = self.defenseProfile.index
        self.defenseProfile = self.defenseProfile.merge(
            teamstats, left_on='opponent', right_on='team')
        self.defenseProfile.index = i

        self.teamProfile = teamstats

    def dvpoa(self, team, position, market, date=datetime.today().date()):
        """
        Calculate the Defense Versus Position Above Average (DVPOA) for a specific team, position, and market.

        Args:
            team (str): The team abbreviation.
            position (str): The position code.
            market (str): The market to calculate DVPOA for.

        Returns:
            float: The DVPOA value.
        """

        if date != self.dvpoa_latest_date:
            self.dvp_index = {}
            self.dvpoa_latest_date = date

        # Check if the market exists in the dvp_index dictionary
        if market not in self.dvp_index:
            self.dvp_index[market] = {}

        # Check if the team exists in the dvp_index for the market
        if team not in self.dvp_index[market]:
            self.dvp_index[market][team] = {}

        # Check if the DVPOA value is already calculated and return if found
        if self.dvp_index[market][team].get(position):
            return self.dvp_index[market][team].get(position)

        # Initialize dictionaries for dvp and league averages
        dvp = {}
        leagueavg = {}

        # Calculate dvp and league averages based on the market
        for game in self.gamelog:
            if datetime.strptime(game['gameDate'], '%Y-%m-%d').date() > date.date():
                continue
            elif datetime.strptime(game['gameDate'], '%Y-%m-%d').date() < (date - timedelta(days=300)).date():
                continue
            if (market in ['goalsAgainst', 'saves'] or "goalie fantasy" in market) and game['position'] != "G":
                continue
            elif (market not in ['goalsAgainst', 'saves'] or "goalie fantasy" not in market) and game['position'] == "G":
                continue

            if game["position"] == position:
                id = game["gameId"]
                if id not in leagueavg:
                    leagueavg[id] = 0
                leagueavg[id] += game[market]
                if team == game["opponent"]:
                    if id not in dvp:
                        dvp[id] = 0
                    dvp[id] += game[market]

        # Check if dvp dictionary is empty
        if not dvp:
            return 0
        else:
            dvp = np.mean(list(dvp.values()))
            leagueavg = np.mean(list(leagueavg.values())) / 2
            dvpoa = (dvp - leagueavg) / leagueavg
            dvpoa = np.nan_to_num(dvpoa, nan=0.0, posinf=0.0, neginf=0.0)
            self.dvp_index[market][team][position] = dvpoa
            return dvpoa

    def get_stats(self, offer, date=datetime.today()):
        """
        Calculate various statistics for a given offer.

        Args:
            offer (dict): The offer details containing 'Player', 'Team', 'Market', 'Line', and 'Opponent'.
            date (str or datetime, optional): The date for which to calculate the statistics. Defaults to today's date.

        Returns:
            pandas.DataFrame: A DataFrame containing the calculated statistics.
        """

        if type(date) is datetime:
            date = date.strftime("%Y-%m-%d")

        stat_map = {
            "PTS": "points",
            "AST": "assists",
            "BLK": "blocked"
        }

        player = offer["Player"]
        team = offer["Team"]
        market = offer["Market"]
        market = stat_map.get(market, market)
        if self.defenseProfile.empty:
            logger.exception(f"{market} not profiled")
            return 0
        line = offer["Line"]
        opponent = offer["Opponent"]
        home = offer.get("Home")
        if home is None:
            home = self.upcoming_games.get(team, {}).get("Home", 0)

        if player not in self.playerProfile.index:
            self.playerProfile.loc[player] = np.zeros_like(
                self.playerProfile.columns)

        try:
            if not any([string in market for string in ["Against", "saves", "goalie"]]):
                if datetime.strptime(date, "%Y-%m-%d").date() < datetime.today().date():
                    goalie = offer.get("Goalie", "")
                else:
                    goalie = self.goalies.get(opponent, "")

                if goalie not in self.goalieProfile.index:
                    self.goalieProfile.loc[goalie] = 0

            stats = (
                archive["NHL"]
                .get(market, {})
                .get(date, {})
                .get(player, {})
                .get(line, [0] * 4)
            )
            moneyline = archive["NHL"]["Moneyline"].get(
                date, {}).get(team, 0)
            total = archive["NHL"]["Totals"].get(
                date, {}).get(team, 0)

        except:
            logger.exception(f"{player}, {market}")
            return 0

        date = datetime.strptime(date, "%Y-%m-%d")

        if any([string in market for string in ["Against", "saves", "goalie"]]):
            player_games = self.gamelog.loc[(self.gamelog["playerName"] == player) & (
                pd.to_datetime(self.gamelog.gameDate) < date) & (self.gamelog["position"] == "G")]

        else:
            player_games = self.gamelog.loc[(self.gamelog["playerName"] == player) & (
                pd.to_datetime(self.gamelog.gameDate) < date) & (self.gamelog["position"] != "G")]

        headtohead = player_games.loc[player_games["opponent"] == opponent]

        one_year_ago = len(player_games.loc[pd.to_datetime(
            self.gamelog["gameDate"]) > date-timedelta(days=300)])

        game_res = (player_games[market]).to_list()
        h2h_res = (headtohead[market]).to_list()

        stats = np.array(stats, dtype=np.float64)
        odds = np.nanmean(stats)
        if np.isnan(odds):
            odds = 0

        data = {
            "DVPOA": 0,
            "Odds": odds,
            "Line": line,
            "Avg5": np.median(game_res[-5:]) if game_res else 0,
            "Avg10": np.median(game_res[-10:]) if game_res else 0,
            "AvgYr": np.median(game_res[-one_year_ago:]) if game_res else 0,
            "AvgH2H": np.median(h2h_res[-5:]) if h2h_res else 0,
            "IQR10": iqr(game_res[-10:]) if game_res else 0,
            "IQRYr": iqr(game_res[-one_year_ago:]) if game_res else 0,
            "Mean5": np.mean(game_res[-5:]) if game_res else 0,
            "Mean10": np.mean(game_res[-10:]) if game_res else 0,
            "MeanYr": np.mean(game_res[-one_year_ago:]) if game_res else 0,
            "MeanH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
            "GamesPlayed": one_year_ago,
            "Moneyline": moneyline,
            "Total": total,
            "Home": home
        }
        positions = ["C", "R", "L", "D"]
        if not any([string in market for string in ["Against", "saves", "goalie"]]):
            if len(player_games) > 0:
                position = player_games.iloc[0]['position']
            else:
                logger.warning(f"{player} not found")
                return 0

            data.update({"Position": positions.index(position)})

        if len(game_res) < 5:
            i = 5 - len(game_res)
            game_res = [0] * i + game_res
        if len(h2h_res) < 5:
            i = 5 - len(h2h_res)
            h2h_res = [0] * i + h2h_res

        # Update the data dictionary with additional values
        data.update(
            {"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
        data.update({"Game " + str(i + 1): game_res[-5 + i] for i in range(5)})

        player_data = self.playerProfile.loc[player]
        data.update(
            {"Player " + col: player_data[col] for col in player_data.index})

        defense_data = self.defenseProfile.loc[opponent]

        data.update(
            {"Defense " + col: defense_data[col] for col in defense_data.index})

        team_data = self.teamProfile.loc[team]

        data.update(
            {"Team " + col: team_data[col] for col in team_data.index})

        if any([string in market for string in ["Against", "saves", "goalie"]]):
            data["DVPOA"] = data.pop("Defense avg")
        else:
            data["DVPOA"] = self.defenseProfile.loc[opponent, position]
            data["Goalie SOE"] = self.goalieProfile.loc[goalie]

        return data

    def get_training_matrix(self, market):
        """
        Retrieve the training matrix for the specified market.

        Args:
            market (str): The market for which to retrieve the training data.

        Returns:
            tuple: A tuple containing the training matrix (X) and the corresponding results (y).
        """

        # Initialize variables
        archive.__init__("NHL")

        matrix = []

        # Iterate over each game in the gamelog
        for i, game in tqdm(self.gamelog.iterrows(), unit="game", desc="Gathering Training Data", total=len(self.gamelog)):
            gameDate = datetime.strptime(game["gameDate"], "%Y-%m-%d")
            if gameDate < datetime(2022, 10, 1):
                continue

            if game[market] < 0:
                continue

            data = {}
            self.profile_market(market, date=gameDate)
            name = game['playerName']

            if name not in self.playerProfile.index:
                continue

            data = archive["NHL"].get(market, {}).get(gameDate.strftime(
                "%Y-%m-%d"), {}).get(name, {0: [0.5]*4})

            lines = [k for k, v in data.items()]
            if "Closing Lines" in lines:
                lines.remove("Closing Lines")
                lines.append(
                    np.floor(np.mean([float(i['Line']) for i in data["Closing Lines"] if i is not None]))+0.5)

            line = lines[-1]

            offer = {
                "Player": name,
                "Team": game["team"],
                "Market": market,
                "Opponent": game["opponent"],
                "Goalie": game["opponent goalie"],
                "Home": int(game["home"])
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_get_stats = self.get_stats(
                    offer | {"Line": line}, gameDate
                )
                if type(new_get_stats) is dict:
                    if new_get_stats["Avg10"] == 0 and new_get_stats["IQR10"] == 0:
                        continue
                    new_get_stats.update(
                        {"Result": game[market]}
                    )

                    matrix.append(new_get_stats)

        M = pd.DataFrame(matrix).fillna(0.0).replace([np.inf, -np.inf], 0)

        return M
