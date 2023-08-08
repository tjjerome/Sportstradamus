from sportstradamus.spiderLogger import logger
import os.path
import numpy as np
from datetime import datetime, timedelta
import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
from tqdm import tqdm
import statsapi as mlb
import nba_api.stats.endpoints as nba
from nba_api.stats.static import players as nba_static
import nfl_data_py as nfl
from scipy.stats import hmean
from time import sleep
from sportstradamus.helpers import scraper, mlb_pitchers, archive, abbreviations, remove_accents
import pandas as pd
import warnings


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
        self.gamelog = []
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
        self.season_start = datetime.strptime("2022-10-18", "%Y-%m-%d").date()

    def load(self):
        """
        Load data from files.
        """
        filepath = pkg_resources.files(data) / "nba_players.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.players = pickle.load(infile)
                self.players = {int(k): v for k, v in self.players.items()}

    def update(self):
        """
        Update data from the web API.
        """
        # Fetch regular season game logs
        nba_gamelog = nba.playergamelogs.PlayerGameLogs(
            season_nullable="2022-23"
        ).get_normalized_dict()["PlayerGameLogs"]

        # Fetch playoffs game logs
        nba_playoffs = nba.playergamelogs.PlayerGameLogs(
            season_nullable="2022-23", season_type_nullable="Playoffs"
        ).get_normalized_dict()["PlayerGameLogs"]

        # Combine regular season and playoffs game logs
        self.gamelog = nba_playoffs + nba_gamelog

        # Process each game
        for game in tqdm(self.gamelog, desc="Getting NBA stats"):
            player_id = game["PLAYER_ID"]

            if player_id not in self.players:
                # Fetch player information if not already present
                self.players[player_id] = nba.commonplayerinfo.CommonPlayerInfo(
                    player_id=player_id
                ).get_normalized_dict()["CommonPlayerInfo"][0]
                sleep(0.5)

            # Extract additional game information
            game["POS"] = self.players[player_id].get("POSITION")
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
            game["fantasy score"] = game["PTS"] + game["REB"] * \
                1.2 + game["AST"]*1.5 + game["BLST"]*3 - game["TOV"]
            game["fantasy points parlay"] = game["PRA"] + \
                game["BLST"]*2 - game["TOV"]

        # Save the updated player data
        with open(pkg_resources.files(data) / "nba_players.dat", "wb") as outfile:
            pickle.dump(self.players, outfile)

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
            elif datetime.strptime(game['GAME_DATE'][:10], '%Y-%m-%d').date() < (date - timedelta(days=365)).date():
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

    def profile_market(self, market, date=datetime.today()):
        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.profiled_market = market
        self.profile_latest_date = date

        # Initialize playerStats and edges
        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.playerStats = {}
        self.edges = []
        gamelog = []

        # Iterate over game log and gather stats for each player
        for game in self.gamelog:
            gameDate = datetime.strptime(
                game['GAME_DATE'][:10], '%Y-%m-%d').date()
            if gameDate > date.date():
                continue
            elif gameDate < (date - timedelta(days=365)).date():
                continue

            game["moneyline"] = float(archive["NBA"]["Moneyline"].get(
                gameDate.strftime("%Y-%m-%d"), {}).get(game["TEAM_ABBREVIATION"], 0.5))
            game["totals"] = float(archive["NBA"]["Totals"].get(
                gameDate.strftime("%Y-%m-%d"), {}).get(game["TEAM_ABBREVIATION"], 228))
            gamelog.append(game)

        gamelog = pd.DataFrame(gamelog)

        playerGroups = gamelog.\
            groupby('PLAYER_NAME').\
            filter(lambda x: len(x[x[market] != 0]) > 4).\
            groupby('PLAYER_NAME')

        leagueavg = playerGroups[market].mean().mean()
        if np.isnan(leagueavg):
            return
        self.playerProfile['avg'] = playerGroups.apply(
            lambda x: x[market].mean()/leagueavg)-1

        self.playerProfile['home'] = playerGroups.apply(
            lambda x: x.loc[x['HOME'], market].mean()/x[market].mean())-1

        self.playerProfile['away'] = playerGroups.apply(
            lambda x: x.loc[~x['HOME'].astype(bool), market].mean()/x[market].mean())-1

        self.playerProfile['moneyline gain'] = playerGroups.\
            apply(lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float),
                  x[market].values/x[market].mean(), 1)[1])

        self.playerProfile['totals gain'] = playerGroups.\
            apply(lambda x: np.polyfit(x.totals.fillna(8.3).values.astype(float)/8.3,
                  x[market].values/x[market].mean(), 1)[1])

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
            elif datetime.strptime(game['GAME_DATE'][:10], '%Y-%m-%d').date() < (date - timedelta(days=365)).date():
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
        market = offer["Market"].replace("H2H ", "")
        line = offer["Line"]
        opponent = offer["Opponent"]
        home = offer.get("Home")
        if home is None:
            home = 0
            teams = team.split("/")
            for i, t in enumerate(teams):
                home += (i+1)*self.upcoming_games.get(t, {}).get("Home", 0)

        players = player.strip().replace("vs.", "+").split(" + ")
        for p in players:
            if p not in self.playerProfile.index:
                self.playerProfile.loc[p] = np.zeros_like(
                    self.playerProfile.columns)

        try:
            stats = (
                archive["NBA"][market]
                .get(date, {})
                .get(player, {})
                .get(line, [0.5] * 4)
            )
            moneyline = 0
            total = 0
            teams = team.split("/")
            for t in teams:
                moneyline += archive["NBA"]["Moneyline"].get(
                    date, {}).get(t, np.nan)
                total += archive["NBA"]["Totals"].get(date, {}).get(t, np.nan)

            moneyline /= len(teams)
            total /= len(teams)

        except:
            return 0

        if np.isnan(moneyline):
            moneyline = 0.5
        if np.isnan(total):
            total = 228

        date = datetime.strptime(date, "%Y-%m-%d")
        if "+" in player or "vs." in player:
            players = player.strip().replace("vs.", "+").split(" + ")
            opponents = opponent.split("/")
            if len(opponents) == 1:
                opponents = opponents * 2

            player1_id = nba_static.find_players_by_full_name(players[0])
            player2_id = nba_static.find_players_by_full_name(players[1])
            if not player1_id or not player2_id:
                return 0
            player1_id = player1_id[0]["id"]
            position1 = self.players[player1_id]["POSITION"]
            player2_id = player2_id[0]["id"]
            position2 = self.players[player2_id]["POSITION"]

            if position1 is None or position2 is None:
                return 0

            player1_games = [
                game
                for game in self.gamelog
                if game["PLAYER_NAME"] == players[0]
                and datetime.strptime(game["GAME_DATE"], "%Y-%m-%dT%H:%M:%S") < date
            ]

            headtohead1 = [
                game for game in player1_games if game["OPP"] == opponents[0]
            ]

            player2_games = [
                game
                for game in self.gamelog
                if game["PLAYER_NAME"] == players[1]
                and datetime.strptime(game["GAME_DATE"], "%Y-%m-%dT%H:%M:%S") < date
            ]

            headtohead2 = [
                game for game in player2_games if game["OPP"] == opponents[1]
            ]

            n = min(len(player1_games), len(player2_games))
            m = min(len(headtohead1), len(headtohead2))

            player1_games = player1_games[:n]
            player2_games = player2_games[:n]
            headtohead1 = headtohead1[:m]
            headtohead2 = headtohead2[:m]

            dvpoa1 = self.dvpoa(opponents[0], position1, market, date=date)
            dvpoa2 = self.dvpoa(opponents[1], position2, market, date=date)

            if "+" in player:
                game_res = (
                    np.array([game[market] for game in player1_games])
                    + np.array([game[market] for game in player2_games])
                    - np.array([line] * n)
                )
                h2h_res = (
                    np.array([game[market] for game in headtohead1])
                    + np.array([game[market] for game in headtohead2])
                    - np.array([line] * m)
                )

                try:
                    dvpoa = (2 / (1 / (dvpoa1 + 1) + 1 / (dvpoa2 + 1))) - 1
                except:
                    dvpoa = 0

            else:
                game_res = (
                    np.array([game[market] for game in player1_games])
                    - np.array([game[market] for game in player2_games])
                    + np.array([line] * n)
                )
                h2h_res = (
                    np.array([game[market] for game in headtohead1])
                    - np.array([game[market] for game in headtohead2])
                    + np.array([line] * m)
                )

                try:
                    dvpoa = (2 / (1 / (dvpoa1 + 1) + 1 / (dvpoa2 + 1))) - 1
                except:
                    dvpoa = 0

            game_res = list(game_res)
            h2h_res = list(h2h_res)

        else:
            player_id = nba_static.find_players_by_full_name(player)
            if not player_id:
                return 0
            player_id = player_id[0].get("id")
            position = self.players.get(player_id, {}).get("POSITION")
            if position is None:
                return 0

            player_games = [
                game
                for game in self.gamelog
                if game["PLAYER_NAME"] == player
                and datetime.strptime(game["GAME_DATE"], "%Y-%m-%dT%H:%M:%S") < date
            ]

            headtohead = [
                game for game in player_games if game["OPP"] == opponent]

            game_res = [game[market] - line for game in player_games]
            h2h_res = [game[market] - line for game in headtohead]

            dvpoa = self.dvpoa(opponent, position, market, date=date)

        stats[stats is None] = np.nan
        odds = np.nanmean(stats)
        if np.isnan(odds):
            odds = 0.5

        data = {
            "DVPOA": dvpoa,
            "Odds": odds - 0.5,
            "Last5": np.mean([int(i > 0) for i in game_res[:5]]) - 0.5 if game_res else 0,
            "Last10": np.mean([int(i > 0) for i in game_res[:10]]) - 0.5 if game_res else 0,
            "H2H": np.mean([int(i > 0) for i in h2h_res[:5]] + [1, 0]) - 0.5 if h2h_res else 0,
            "Avg5": np.mean(game_res[:5]) if game_res else 0,
            "Avg10": np.mean(game_res[:10]) if game_res else 0,
            "AvgH2H": np.mean(h2h_res[:5]) if h2h_res else 0,
            "Moneyline": moneyline - 0.5,
            "Total": total / 228 - 1,
            "Line": line,
            "Home": home,
            "Combo": 1 if "+" in player else 0,
            "Rival": 1 if "vs." in player else 0,
        }

        if len(game_res) < 6:
            game_res.extend([0] * (6 - len(game_res)))
        if len(h2h_res) < 5:
            h2h_res.extend([0] * (5 - len(h2h_res)))

        X = pd.DataFrame(data, index=[0]).fillna(0)
        X = X.join(pd.DataFrame([h2h_res[:5]]).fillna(
            0).add_prefix("Meeting "))
        X = X.join(pd.DataFrame([game_res[:6]]).fillna(0).add_prefix("Game "))
        X = X.join(pd.DataFrame(
            [hmean(self.playerProfile.loc[players, ['avg', 'home', 'away']]+1)-1], columns=['avg', 'home', 'away']).
            add_prefix("Player "))
        X = X.join(pd.DataFrame(
            [np.mean(self.playerProfile.loc[players, self.playerProfile.columns.difference(
                ['avg', 'home', 'away'])], axis=0)],
            columns=self.playerProfile.loc[players, self.playerProfile.columns.difference(['avg', 'home', 'away'])].columns).
            add_prefix("Player "))

        return X.fillna(0.0)

    def get_training_matrix(self, market):
        """
        Retrieves training data in the form of a feature matrix (X) and a target vector (y) for a specified market.

        Args:
            market (str): The market for which to retrieve training data.

        Returns:
            tuple: A tuple containing the feature matrix (X) and the target vector (y).
        """
        archive.__init__(True)
        X = pd.DataFrame()
        results = []

        for game in tqdm(self.gamelog, unit="game", desc="Gathering Training Data"):
            gameDate = datetime.strptime(
                game["GAME_DATE"], "%Y-%m-%dT%H:%M:%S")
            data = {}
            self.profile_market(market, date=gameDate)

            try:
                names = list(
                    archive["NBA"][market].get(
                        gameDate.strftime("%Y-%m-%d"), {}).keys()
                )
                for name in names:
                    if (
                        game["PLAYER_NAME"]
                        == name.strip().replace("vs.", "+").split(" + ")[0]
                    ):
                        data[name] = archive["NBA"][market][
                            gameDate.strftime("%Y-%m-%d")
                        ][name]
            except:
                continue

            for name, archiveData in data.items():
                offer = {
                    "Player": name,
                    "Team": game["TEAM_ABBREVIATION"],
                    "Market": market,
                    "Opponent": game["OPP"],
                    "Home": int(game["HOME"])
                }

                if " + " in name or " vs. " in name:
                    player2 = name.replace("vs.", "+").split(" + ")[1].strip()
                    game2 = next(
                        (
                            i
                            for i in self.gamelog
                            if i["GAME_DATE"] == gameDate.strftime("%Y-%m-%dT%H:%M:%S")
                            and i["PLAYER_NAME"] == player2
                        ),
                        None,
                    )
                    if game2 is None:
                        continue
                    offer = offer | {
                        "Team": "/".join(
                            [game["TEAM_ABBREVIATION"], game2["TEAM_ABBREVIATION"]]
                        ),
                        "Opponent": "/".join([game["OPP"], game2["OPP"]]),
                        "Home": int(game["HOME"]) + 2*int(game2["HOME"])
                    }

                for line, stats in archiveData.items():
                    if not line == "Closing Lines" and not game[market] == line:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            new_get_stats = self.get_stats(
                                offer | {"Line": line}, gameDate
                            )
                            if isinstance(new_get_stats, pd.DataFrame):
                                if " + " in name:
                                    player2 = name.split(" + ")[1]
                                    game2 = next(
                                        (
                                            i
                                            for i in self.gamelog
                                            if i["PLAYER_NAME"] == player2
                                            and i["GAME_ID"] == game["GAME_ID"]
                                        ),
                                        None,
                                    )
                                    if game2 is None:
                                        continue
                                    results.append(
                                        {
                                            "Result": (game[market] + game2[market]) - line
                                        }
                                    )
                                elif " vs. " in name:
                                    player2 = name.split(" vs. ")[1]
                                    game2 = next(
                                        (
                                            i
                                            for i in self.gamelog
                                            if i["PLAYER_NAME"] == player2
                                            and i["GAME_ID"] == game["GAME_ID"]
                                        ),
                                        None,
                                    )
                                    if game2 is None:
                                        continue
                                    results.append(
                                        {
                                            "Result": (game[market] + line) - game2[market]
                                        }
                                    )
                                else:
                                    results.append(
                                        {"Result": game[market] - line}
                                    )

                                if X.empty:
                                    X = new_get_stats
                                else:
                                    X = pd.concat([X, new_get_stats])

        y = pd.DataFrame(results)
        X.reset_index(drop=True, inplace=True)
        return X, y


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
        self.season_start = datetime.strptime("2023-03-28", "%Y-%m-%d"
                                              ).date()
        self.pitchers = mlb_pitchers
        self.gameIds = []

    def parse_game(self, gameId):
        """
        Parses the game data for a given game ID.

        Args:
            gameId (int): The ID of the game to parse.
        """
        game = mlb.boxscore_data(gameId)
        new_games = []
        if game:
            self.gameIds.append(gameId)
            linescore = mlb.get("game_linescore", {"gamePk": str(gameId)})
            awayTeam = game["teamInfo"]["away"]["abbreviation"]
            homeTeam = game["teamInfo"]["home"]["abbreviation"]
            awayPitcher = game["awayPitchers"][1]["personId"]
            awayPitcher = game["away"]["players"]["ID" + str(awayPitcher)]["person"][
                "fullName"
            ]
            homePitcher = game["homePitchers"][1]["personId"]
            homePitcher = game["home"]["players"]["ID" + str(homePitcher)]["person"][
                "fullName"
            ]
            awayInning1Runs = linescore["innings"][0]["away"]["runs"]
            homeInning1Runs = linescore["innings"][0]["home"]["runs"]
            awayInning1Hits = linescore["innings"][0]["away"]["hits"]
            homeInning1Hits = linescore["innings"][0]["home"]["hits"]
            for v in game["away"]["players"].values():
                if (v["person"]["id"] == game["awayPitchers"][1]["personId"] or v["person"]["id"] in game["away"]["batters"]):
                    n = {
                        "gameId": game["gameId"],
                        "playerId": v["person"]["id"],
                        "playerName": v["person"]["fullName"],
                        "position": v.get("position", {"abbreviation": ""})["abbreviation"],
                        "team": awayTeam,
                        "opponent": homeTeam,
                        "opponent pitcher": homePitcher,
                        "home": False,
                        "starting pitcher": v["person"]["id"] == game["awayPitchers"][1]["personId"],
                        "starting batter": v["person"]["id"] in game["away"]["batters"],
                        "hits": v["stats"]["batting"].get("hits", 0),
                        "total bases": v["stats"]["batting"].get("hits", 0) + v["stats"]["batting"].get("doubles", 0) +
                        2 * v["stats"]["batting"].get("triples", 0) +
                        3 * v["stats"]["batting"].get("homeRuns", 0),
                        "singles": v["stats"]["batting"].get("hits", 0) - v["stats"]["batting"].get("doubles", 0) -
                        v["stats"]["batting"].get(
                            "triples", 0) - v["stats"]["batting"].get("homeRuns", 0),
                        "batter strikeouts": v["stats"]["batting"].get("strikeOuts", 0),
                        "runs": v["stats"]["batting"].get("runs", 0),
                        "rbi": v["stats"]["batting"].get("rbi", 0),
                        "hits+runs+rbi": v["stats"]["batting"].get("hits", 0) + v["stats"]["batting"].get("runs", 0) +
                        v["stats"]["batting"].get("rbi", 0),
                        "walks": v["stats"]["batting"].get("baseOnBalls", 0),
                        "pitcher strikeouts": v["stats"]["pitching"].get("strikeOuts", 0),
                        "walks allowed": v["stats"]["pitching"].get("baseOnBalls", 0),
                        "pitches thrown": v["stats"]["pitching"].get("numberOfPitches", 0),
                        "runs allowed": v["stats"]["pitching"].get("runs", 0),
                        "hits allowed": v["stats"]["pitching"].get("hits", 0),
                        "pitching outs": 3 * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1]),
                        "1st inning runs allowed": homeInning1Runs if v["person"]["id"] ==
                        game["awayPitchers"][1]["personId"] else 0,
                        "1st inning hits allowed": homeInning1Hits if v["person"]["id"] ==
                        game["awayPitchers"][1]["personId"] else 0,
                        "hitter fantasy score": 3*v["stats"]["batting"].get("hits", 0) +
                        2*v["stats"]["batting"].get("doubles", 0) +
                        5*v["stats"]["batting"].get("triples", 0) +
                        7*v["stats"]["batting"].get("homeRuns", 0) +
                        2*v["stats"]["batting"].get("runs", 0) +
                        2*v["stats"]["batting"].get("rbi", 0) +
                        2*v["stats"]["batting"].get("baseOnBalls", 0) +
                        5*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy score": 6*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1]) +
                        4 if int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[
                                 0]) > 5 and v["stats"]["pitching"].get("earnedRuns", 0) < 4 else 0,
                        "hitter fantasy points underdog": 3*v["stats"]["batting"].get("hits", 0) +
                        3*v["stats"]["batting"].get("doubles", 0) +
                        5*v["stats"]["batting"].get("triples", 0) +
                        7*v["stats"]["batting"].get("homeRuns", 0) +
                        2*v["stats"]["batting"].get("runs", 0) +
                        2*v["stats"]["batting"].get("rbi", 0) +
                        3*v["stats"]["batting"].get("baseOnBalls", 0) +
                        4*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points underdog": 2*v["stats"]["pitching"].get("wins", 0) +
                        v["stats"]["pitching"].get("strikeOuts", 0) -
                        v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1]) +
                        3 if int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[
                                 0]) > 5 and v["stats"]["pitching"].get("earnedRuns", 0) < 4 else 0,
                        "hitter fantasy points parlay": 3*v["stats"]["batting"].get("hits", 0) +
                        3*v["stats"]["batting"].get("doubles", 0) +
                        6*v["stats"]["batting"].get("triples", 0) +
                        9*v["stats"]["batting"].get("homeRuns", 0) +
                        3*v["stats"]["batting"].get("runs", 0) +
                        3*v["stats"]["batting"].get("rbi", 0) +
                        3*v["stats"]["batting"].get("baseOnBalls", 0) +
                        6*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points parlay": 6*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1])
                    }
                    new_games.append(n)

            for v in game["home"]["players"].values():
                if (v["person"]["id"] == game["homePitchers"][1]["personId"] or v["person"]["id"] in game["home"]["batters"]):
                    n = {
                        "gameId": game["gameId"],
                        "playerId": v["person"]["id"],
                        "playerName": v["person"]["fullName"],
                        "position": v.get("position", {"abbreviation": ""})[
                            "abbreviation"
                        ],
                        "team": homeTeam,
                        "opponent": awayTeam,
                        "opponent pitcher": awayPitcher,
                        "home": True,
                        "starting pitcher": v["person"]["id"]
                        == game["homePitchers"][1]["personId"],
                        "starting batter": v["person"]["id"] in game["home"]["batters"],
                        "hits": v["stats"]["batting"].get("hits", 0),
                        "total bases": v["stats"]["batting"].get("hits", 0)
                        + v["stats"]["batting"].get("doubles", 0)
                        + 2 * v["stats"]["batting"].get("triples", 0)
                        + 3 * v["stats"]["batting"].get("homeRuns", 0),
                        "singles": v["stats"]["batting"].get("hits", 0)
                        - v["stats"]["batting"].get("doubles", 0)
                        - v["stats"]["batting"].get("triples", 0)
                        - v["stats"]["batting"].get("homeRuns", 0),
                        "batter strikeouts": v["stats"]["batting"].get("strikeOuts", 0),
                        "runs": v["stats"]["batting"].get("runs", 0),
                        "rbi": v["stats"]["batting"].get("rbi", 0),
                        "hits+runs+rbi": v["stats"]["batting"].get("hits", 0)
                        + v["stats"]["batting"].get("runs", 0)
                        + v["stats"]["batting"].get("rbi", 0),
                        "walks": v["stats"]["batting"].get("baseOnBalls", 0),
                        "pitcher strikeouts": v["stats"]["pitching"].get(
                            "strikeOuts", 0
                        ),
                        "walks allowed": v["stats"]["pitching"].get("baseOnBalls", 0),
                        "pitches thrown": v["stats"]["pitching"].get(
                            "numberOfPitches", 0
                        ),
                        "runs allowed": v["stats"]["pitching"].get("runs", 0),
                        "hits allowed": v["stats"]["pitching"].get("hits", 0),
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
                        game["homePitchers"][1]["personId"] else 0,
                        "1st inning hits allowed": awayInning1Hits if v["person"]["id"] ==
                        game["homePitchers"][1]["personId"] else 0,
                        "hitter fantasy score": 3*v["stats"]["batting"].get("hits", 0) +
                        2*v["stats"]["batting"].get("doubles", 0) +
                        5*v["stats"]["batting"].get("triples", 0) +
                        7*v["stats"]["batting"].get("homeRuns", 0) +
                        2*v["stats"]["batting"].get("runs", 0) +
                        2*v["stats"]["batting"].get("rbi", 0) +
                        2*v["stats"]["batting"].get("baseOnBalls", 0) +
                        5*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy score": 6*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1]) +
                        4 if int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[
                                 0]) > 5 and v["stats"]["pitching"].get("earnedRuns", 0) < 4 else 0,
                        "hitter fantasy points underdog": 3*v["stats"]["batting"].get("hits", 0) +
                        3*v["stats"]["batting"].get("doubles", 0) +
                        5*v["stats"]["batting"].get("triples", 0) +
                        7*v["stats"]["batting"].get("homeRuns", 0) +
                        2*v["stats"]["batting"].get("runs", 0) +
                        2*v["stats"]["batting"].get("rbi", 0) +
                        3*v["stats"]["batting"].get("baseOnBalls", 0) +
                        4*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points underdog": 2*v["stats"]["pitching"].get("wins", 0) +
                        v["stats"]["pitching"].get("strikeOuts", 0) -
                        v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1]) +
                        3 if int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[
                                 0]) > 5 and v["stats"]["pitching"].get("earnedRuns", 0) < 4 else 0,
                        "hitter fantasy points parlay": 3*v["stats"]["batting"].get("hits", 0) +
                        3*v["stats"]["batting"].get("doubles", 0) +
                        6*v["stats"]["batting"].get("triples", 0) +
                        9*v["stats"]["batting"].get("homeRuns", 0) +
                        3*v["stats"]["batting"].get("runs", 0) +
                        3*v["stats"]["batting"].get("rbi", 0) +
                        3*v["stats"]["batting"].get("baseOnBalls", 0) +
                        6*v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points parlay": 6*v["stats"]["pitching"].get("wins", 0) +
                        3*v["stats"]["pitching"].get("strikeOuts", 0) -
                        3*v["stats"]["pitching"].get("earnedRuns", 0) +
                        3*int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]) +
                        int(v["stats"]["pitching"].get(
                            "inningsPitched", "0.0").split(".")[1])
                    }
                    new_games.append(n)

        self.gamelog = pd.concat(
            [self.gamelog, pd.DataFrame.from_records(new_games)], ignore_index=True)

    def load(self):
        """
        Loads MLB player statistics from a file.
        """
        filepath = pkg_resources.files(data) / "mlb_data.dat"
        if os.path.isfile(filepath):
            self.gamelog = pd.read_pickle(filepath)

    def update(self):
        """
        Updates the MLB player statistics.
        """
        # Get the current MLB schedule
        today = datetime.today().date()
        next_day = pd.to_datetime(
            self.gamelog.gameId.str[:10]).max() + timedelta(days=1)
        if next_day.date() > today:
            return
        mlb_games = mlb.schedule(
            start_date=next_day.strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
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
                if game["game_num"] == 1:
                    mlb_upcoming_games[awayTeam] = {
                        "Pitcher": remove_accents(game["away_probable_pitcher"]),
                        "Home": 0,
                        "Opponent": homeTeam
                    }
                    mlb_upcoming_games[homeTeam] = {
                        "Pitcher": remove_accents(game["home_probable_pitcher"]),
                        "Home": 1,
                        "Opponent": awayTeam
                    }
                elif game["game_num"] > 1:
                    mlb_upcoming_games[awayTeam + str(game["game_num"])] = {
                        "Pitcher": remove_accents(game["away_probable_pitcher"]),
                        "Home": 0,
                        "Opponent": homeTeam
                    }
                    mlb_upcoming_games[homeTeam + str(game["game_num"])] = {
                        "Pitcher": remove_accents(game["home_probable_pitcher"]),
                        "Home": 1,
                        "Opponent": awayTeam
                    }

        self.upcoming_games = mlb_upcoming_games

        mlb_game_ids = [
            game["game_id"]
            for game in mlb_games
            if game["status"] == "Final"
            and game["game_type"] != "E"
            and game["game_type"] != "S"
        ]

        # Parse the game stats
        for id in tqdm(mlb_game_ids, desc="Getting MLB Stats"):
            self.parse_game(id)

        # Remove old games to prevent file bloat
        two_years_ago = today - timedelta(days=730)
        self.gamelog = self.gamelog[self.gamelog["gameId"].str[:10].apply(
            lambda x: two_years_ago <= datetime.strptime(x, '%Y/%m/%d').date() <= today)]

        # Write to file
        self.gamelog.to_pickle((pkg_resources.files(data) / "mlb_data.dat"))

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

        # Initialize playerStats and edges
        self.playerStats = {}
        self.edges = []

        # Iterate over game log and gather stats for each player
        for game in self.gamelog:
            gameDate = datetime.strptime(
                game['gameId'][:10], '%Y/%m/%d').date()
            if gameDate > date.date():
                continue
            elif gameDate < (date - timedelta(days=365)).date():
                continue
            # Skip non-starting pitchers or non-starting batters depending on the market
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

            # Check if player exists in playerStats dictionary
            if not game["playerName"] in self.playerStats:
                self.playerStats[game["playerName"]] = {"games": []}

            # Add the stat for the current game to the player's games list
            self.playerStats[game["playerName"]]["games"].append(game[market])

        # Remove players with insufficient games or all zero stats
        self.playerStats = {
            k: v
            for k, v in self.playerStats.items()
            if len(v["games"]) > 10 and not all([g == 0 for g in v["games"]])
        }

        # Calculate average stats for each player
        averages = []
        for player, games in self.playerStats.items():
            self.playerStats[player]["avg"] = (
                np.mean(games["games"]) if games["games"] else 0
            )
            averages.append(self.playerStats[player]["avg"])

        if not len(averages):
            return

        # Determine edges for each bucket based on percentiles
        w = int(100 / buckets)
        self.edges = [np.percentile(averages, p) for p in range(0, 101, w)]

        # Calculate lines for each bucket based on average values
        lines = np.zeros(buckets)
        for i in range(1, buckets + 1):
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

        # Assign bucket and line values to each player based on their average stat
        for player, games in self.playerStats.items():
            for i in range(0, buckets):
                if games["avg"] >= self.edges[i]:
                    self.playerStats[player]["bucket"] = buckets - i
                    self.playerStats[player]["line"] = lines[i]

    def profile_market(self, market, date=datetime.today().date()):
        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.profiled_market = market
        self.profile_latest_date = date

        # Initialize playerStats and edges
        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.playerStats = {}
        self.edges = []

        # Filter gamelog for games within the date range
        one_year_ago = (date - timedelta(days=365))
        gamelog_df = self.gamelog[self.gamelog["gameId"].str[:10].apply(
            lambda x: one_year_ago <= datetime.strptime(x, '%Y/%m/%d').date() <= date)]

        # Filter non-starting pitchers or non-starting batters depending on the market
        if any([string in market for string in ["allowed", "pitch"]]):
            gamelog_df = gamelog_df[gamelog_df["starting pitcher"]]
        else:
            gamelog_df = gamelog_df[gamelog_df["starting batter"]]

        # Retrieve moneyline and totals data from archive
        gamelog_df["moneyline"] = gamelog_df.apply(lambda row: archive["MLB"]["Moneyline"].get(
            row["gameId"][:10].replace('/', '-'), {}).get(row["team"], 0.5), axis=1)
        gamelog_df["totals"] = gamelog_df.apply(lambda row: archive["MLB"]["Totals"].get(
            row["gameId"][:10].replace('/', '-'), {}).get(row["team"], 8.3), axis=1)

        # Filter players with at least 5 non-zero entries
        playerGroups = gamelog_df.groupby('playerName').filter(
            lambda x: len(x[x[market] != 0]) > 4).groupby('playerName')

        # Compute league average
        leagueavg = playerGroups[market].mean().mean()
        if np.isnan(leagueavg):
            return

        # Compute playerProfile DataFrame
        self.playerProfile = pd.DataFrame()
        self.playerProfile['avg'] = playerGroups[market].mean().div(
            leagueavg) - 1
        self.playerProfile['home'] = playerGroups.apply(
            lambda x: x.loc[x['home'], market].mean() / x[market].mean()) - 1
        self.playerProfile['away'] = playerGroups.apply(
            lambda x: x.loc[~x['home'], market].mean() / x[market].mean()) - 1

        self.playerProfile['moneyline gain'] = playerGroups.apply(
            lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float), x[market].values / x[market].mean(), 1)[1])

        self.playerProfile['totals gain'] = playerGroups.apply(
            lambda x: np.polyfit(x.totals.fillna(8.3).values.astype(float) / 8.3, x[market].values / x[market].mean(), 1)[1])

        if any([string in market for string in ["allowed", "pitch"]]):
            self.playerProfile['days off'] = playerGroups['gameId'].apply(lambda x: np.diff(
                [datetime.strptime(g[:10], "%Y/%m/%d") for g in x.iloc[-2:]])[0].days)

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

        one_year_ago = (date - timedelta(days=365))
        gamelog = self.gamelog[self.gamelog["gameId"].str[:10].apply(
            lambda x: one_year_ago <= datetime.strptime(x, '%Y/%m/%d').date() <= date)]

        # Calculate DVP (Defense Versus Position) and league average for the specified team and market
        if any([string in market for string in ["allowed", "pitch"]]):
            position = "starting pitcher"
        else:
            position = "starting batter"

        position_games = gamelog.loc[gamelog[position]]
        team_games = position_games.loc[position_games['opponent'] == team]

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
        team = offer["Team"]
        market = offer["Market"].replace("H2H ", "")
        line = offer["Line"]
        opponent = offer["Opponent"]
        home = offer.get("Home")
        if home is None:
            home = 0
            teams = team.split("/")
            for i, t in enumerate(teams):
                home += (i+1)*self.upcoming_games.get(t, {}).get("Home", 0)

        players = player.strip().replace("vs.", "+").split(" + ")
        for p in players:
            if p not in self.playerProfile.index:
                self.playerProfile.loc[p] = np.zeros_like(
                    self.playerProfile.columns)

        try:
            if datetime.strptime(date, "%Y-%m-%d").date() < datetime.today().date():
                if offer.get("Pitcher"):
                    pitcher = offer["Pitcher"]
                else:
                    if "+" in player or "vs." in player:
                        players = player.strip().replace("vs.", "+").split(" + ")
                        pitcher1 = next(
                            (
                                game["opponent pitcher"]
                                for game in self.gamelog
                                if game["playerName"] == players[0]
                                and game["gameId"][:10] == date.replace("-", "/")
                                and game["team"] == team
                            ),
                            None,
                        )["opponent pitcher"]
                        pitcher2 = next(
                            (
                                game["opponent pitcher"]
                                for game in self.gamelog
                                if game["playerName"] == players[1]
                                and game["gameId"][:10] == date.replace("-", "/")
                                and game["team"] == team
                            ),
                            None,
                        )["opponent pitcher"]
                        pitcher = "/".join([pitcher1, pitcher2])
                    else:
                        pitcher = next(
                            (
                                game["opponent pitcher"]
                                for game in self.gamelog
                                if game["playerName"] == player
                                and game["gameId"][:10] == date.replace("-", "/")
                                and game["team"] == team
                            ),
                            None,
                        )["opponent pitcher"]
            else:
                if "+" in player or "vs." in player:
                    opponents = opponent.split("/")
                    if len(opponents) == 1:
                        opponents = opponents * 2
                    pitcher1 = self.pitchers[opponents[0]]
                    pitcher2 = self.pitchers[opponents[1]]
                    pitcher = "/".join([pitcher1, pitcher2])
                else:
                    pitcher = self.pitchers[opponent]

            stats = (
                archive["MLB"][market]
                .get(date, {})
                .get(player, {})
                .get(line, [0.5] * 4)
            )
            moneyline = 0
            total = 0
            teams = team.split("/")
            for t in teams:
                t = "ARI" if t == "AZ" else t
                moneyline += archive["MLB"]["Moneyline"].get(
                    date, {}).get(t, np.nan)
                total += archive["MLB"]["Totals"].get(date, {}).get(t, np.nan)

            moneyline /= len(teams)
            total /= len(teams)
        except:
            return 0

        if np.isnan(moneyline):
            moneyline = 0.5
        if np.isnan(total):
            total = 8.3

        date = datetime.strptime(date, "%Y-%m-%d")

        if "+" in player or "vs." in player:
            players = player.strip().replace("vs.", "+").split(" + ")
            opponents = opponent.split("/")
            if len(opponents) == 1:
                opponents = opponents * 2
            pitchers = pitcher.split("/")

            if any([string in market for string in ["allowed", "pitch"]]):
                player1_games = self.gamelog.loc[(self.gamelog["playerName"] == players[0]) & (
                    pd.to_datetime(self.gamelog.gameId.str[:10]) < date) & self.gamelog["starting pitcher"]]
                player2_games = self.gamelog.loc[(self.gamelog["playerName"] == players[1]) & (
                    pd.to_datetime(self.gamelog.gameId.str[:10]) < date) & self.gamelog["starting pitcher"]]

                dvpoa1 = self.dvpoa(opponents[0], market, date=date)
                dvpoa2 = self.dvpoa(opponents[1], market, date=date)

                headtohead1 = player1_games.loc[player1_games["opponent"]
                                                == opponents[0]]
                headtohead2 = player2_games.loc[player2_games["opponent"]
                                                == opponents[1]]
            else:
                player1_games = self.gamelog.loc[(self.gamelog["playerName"] == players[0]) & (
                    pd.to_datetime(self.gamelog.gameId.str[:10]) < date) & self.gamelog["starting batter"]]
                player2_games = self.gamelog.loc[(self.gamelog["playerName"] == players[1]) & (
                    pd.to_datetime(self.gamelog.gameId.str[:10]) < date) & self.gamelog["starting batter"]]

                dvpoa1 = self.dvpoa(pitchers[0], market, date=date)
                dvpoa2 = self.dvpoa(pitchers[1], market, date=date)

                headtohead1 = player1_games.loc[player1_games["opponent pitcher"] == pitchers[0]]
                headtohead2 = player2_games.loc[player2_games["opponent pitcher"] == pitchers[1]]

            n = np.min([len(player1_games), len(player2_games)])
            m = np.min([len(headtohead1), len(headtohead2)])

            player1_games = player1_games.iloc[-n:]
            player2_games = player2_games.iloc[-n:]
            if n == 0:
                player1_games.drop(player1_games.index, inplace=True)
                player2_games.drop(player2_games.index, inplace=True)
            headtohead1 = headtohead1.iloc[-m:]
            headtohead2 = headtohead2.iloc[-m:]
            if m == 0:
                headtohead1.drop(headtohead1.index, inplace=True)
                headtohead2.drop(headtohead2.index, inplace=True)

            if "+" in player:
                game_res = (player1_games[market].to_numpy() +
                            player2_games[market].to_numpy() - line)
                h2h_res = (headtohead1[market].to_numpy() +
                           headtohead2[market].to_numpy() - line)
                try:
                    dvpoa = (2 / (1 / (dvpoa1 + 1) + 1 / (dvpoa2 + 1))) - 1
                except:
                    dvpoa = 0

            else:
                game_res = (player1_games[market].to_numpy() -
                            player2_games[market].to_numpy() + line)
                h2h_res = (headtohead1[market].to_numpy() -
                           headtohead2[market].to_numpy() + line)
                try:
                    dvpoa = (2 / (1 / (dvpoa1 + 1) - 1 / (dvpoa2 + 1))) - 1
                except:
                    dvpoa = 0

            game_res = list(game_res)
            h2h_res = list(h2h_res)

        else:
            if any([string in market for string in ["allowed", "pitch"]]):
                player_games = self.gamelog.loc[(self.gamelog["playerName"] == player) & (
                    pd.to_datetime(self.gamelog.gameId.str[:10]) < date) & self.gamelog["starting pitcher"]]

                dvpoa = self.dvpoa(opponent, market, date=date)

                headtohead = player_games.loc[player_games["opponent"] == opponent]
            else:
                player_games = self.gamelog.loc[(self.gamelog["playerName"] == player) & (
                    pd.to_datetime(self.gamelog.gameId.str[:10]) < date) & self.gamelog["starting batter"]]

                dvpoa = self.dvpoa(pitcher, market, date=date)

                headtohead = player_games.loc[player_games["opponent pitcher"] == pitcher]

            game_res = (player_games[market]-line).to_list()
            h2h_res = (headtohead[market]-line).to_list()

        stats = np.array(stats, dtype=np.float64)
        odds = np.nanmean(stats)
        if np.isnan(odds):
            odds = 0.5

        data = {
            "DVPOA": dvpoa,
            "Odds": odds - 0.5,
            "Last5": np.mean([int(i > 0) for i in game_res[-5:]]) - 0.5
            if game_res
            else 0,
            "Last10": np.mean([int(i > 0) for i in game_res[-10:]]) - 0.5
            if game_res
            else 0,
            "H2H": np.mean([int(i > 0) for i in h2h_res[-5:]] + [1, 0]) - 0.5 if h2h_res else 0,
            "Avg5": np.mean(game_res[-5:])
            if game_res
            else 0,
            "Avg10": np.mean(game_res[-10:])
            if game_res
            else 0,
            "AvgH2H": np.mean(h2h_res[-5:])
            if h2h_res
            else 0,
            "Moneyline": moneyline - 0.5,
            "Total": total / 8.3 - 1,
            "Line": line,
            "Home": home,
            "Combo": 1 if "+" in player else 0,
            "Rival": 1 if "vs." in player else 0,
        }

        if len(game_res) < 6:
            i = 6 - len(game_res)
            game_res = [0] * i + game_res
        if len(h2h_res) < 5:
            i = 5 - len(h2h_res)
            h2h_res = [0] * i + h2h_res

        # Update the data dictionary with additional values
        data.update(
            {"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
        data.update({"Game " + str(i + 1): game_res[-6 + i] for i in range(6)})

        player_data = self.playerProfile.loc[players, [
            'avg', 'home', 'away']].mean() + 1
        data.update(
            {"Player " + col: player_data[col] - 1 for col in ['avg', 'home', 'away']})

        other_player_data = np.mean(self.playerProfile.loc[players, self.playerProfile.columns.difference(
            ['avg', 'home', 'away'])], axis=0)
        data.update(
            {"Player " + col: other_player_data[col] for col in other_player_data.index})

        return data

    def get_training_matrix(self, market):
        """
        Retrieves the training data matrix and target labels for the specified market.

        Args:
            market (str): The market type to retrieve training data for.

        Returns:
            X (pd.DataFrame): The training data matrix.
            y (pd.DataFrame): The target labels.
        """
        archive.__init__(True)

        # Initialize an empty list for the target labels
        features = []
        results = []

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

            # Retrieve data from the archive based on game date and player name
            data = {}
            gameDate = datetime.strptime(game["gameId"][:10], "%Y/%m/%d")
            if gameDate < datetime(2022, 6, 5):
                continue

            self.profile_market(market, date=gameDate.date())

            try:
                names = list(
                    archive["MLB"][market].get(
                        gameDate.strftime("%Y-%m-%d"), {}).keys()
                )
                for name in names:
                    if (
                        game["playerName"]
                        == name.strip().replace("vs.", "+").split(" + ")[0]
                    ):
                        data[name] = archive["MLB"][market][
                            gameDate.strftime("%Y-%m-%d")
                        ][name]
            except:
                continue

            # Process data for each player
            for name, archiveData in data.items():
                # Construct an offer dictionary with player, team, market, opponent, and pitcher information
                offer = {
                    "Player": name,
                    "Team": game["team"],
                    "Market": market,
                    "Opponent": game["opponent"],
                    "Pitcher": game["opponent pitcher"],
                    "Home": int(game["home"])
                }

                # Modify offer for dual player markets
                if " + " in name or " vs. " in name:
                    player2 = name.replace("vs.", "+").split(" + ")[1].strip()
                    game2 = self.gamelog.loc[(self.gamelog.gameId.str[:10] == gameDate.strftime(
                        "%Y/%m/%d")) & (self.gamelog.playerName == player2)].squeeze()
                    if game2.empty:
                        continue
                    offer = offer | {
                        "Team": "/".join([game["team"], game2["team"]]),
                        "Opponent": "/".join([game["opponent"], game2["opponent"]]),
                        "Pitcher": "/".join(
                            [game["opponent pitcher"], game2["opponent pitcher"]]
                        ),
                        "Home": int(game["home"]) + 2*int(game2["home"])
                    }

                # Process stats for each line
                for line, stats in archiveData.items():
                    # Skip if line is the same as game value
                    if not line == "Closing Lines" and not game[market] == line:
                        # Retrieve stats using get_stats method
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            new_get_stats = self.get_stats(
                                offer | {"Line": line}, gameDate
                            )
                            if type(new_get_stats) is dict:
                                # Determine the result based on market type and comparison with the line
                                if " + " in name:
                                    results.append(
                                        {
                                            "Result": (game[market] + game2[market]) - line
                                        }
                                    )
                                elif " vs. " in name:
                                    results.append(
                                        {
                                            "Result": (game[market] + line) - game2[market]
                                        }
                                    )
                                else:
                                    results.append(
                                        {"Result": game[market] - line}
                                    )

                                # Concatenate retrieved stats into the training data matrix
                                features.append(new_get_stats)

        # Create the target labels DataFrame
        y = pd.DataFrame(results)
        X = pd.DataFrame(features).fillna(0.0)

        return X, y


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
        cols = ['player id', 'player display name', 'position group',
                'recent team', 'season', 'week', 'season type',
                'completions', 'attempts', 'passing yards', 'passing tds',
                'interceptions', 'sacks', 'sack fumbles', 'sack fumbles lost',
                'passing 2pt conversions', 'carries', 'rushing yards', 'rushing tds',
                'rushing fumbles', 'rushing fumbles lost', 'rushing 2pt conversions',
                'receptions', 'targets', 'receiving yards', 'receiving tds',
                'receiving fumbles', 'receiving fumbles lost', 'receiving 2pt conversions',
                'special teams tds', 'fumbles', 'fumbles lost', 'yards', 'tds',
                'fantasy points prizepicks', 'fantasy points underdog', 'fantasy points parlayplay',
                'home', 'moneyline', 'totals', 'opponent', 'gameday', 'game id']
        self.gamelog = pd.DataFrame(columns=cols)

    def load(self):
        """
        Load data from files.
        """
        filepath = pkg_resources.files(data) / "nfl_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.gamelog = pickle.load(infile)

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
                'special_teams_tds']
        try:
            nfl_data = nfl.import_weekly_data([self.season_start.year], cols)
        except:
            nfl_data = pd.DataFrame(columns=cols)

        sched = nfl.import_schedules([self.season_start.year])
        upcoming_games = sched.loc[pd.to_datetime(sched['gameday']) >= datetime.today(), [
            'gameday', 'away_team', 'home_team']]
        df1 = upcoming_games.rename(
            columns={'home_team': 'Team', 'away_team': 'Opponent'})
        df2 = upcoming_games.rename(
            columns={'away_team': 'Team', 'home_team': 'Opponent'})
        df1['Home'] = 1
        df2['Home'] = 0
        upcoming_games = pd.concat([df1, df2]).sort_values('gameday')
        self.upcoming_games = upcoming_games.groupby("Team").apply(
            lambda x: x.head(1)).droplevel(1)[['Opponent', 'Home']].to_dict(orient='index')

        nfl_data['fumbles'] = nfl_data['sack_fumbles'] + \
            nfl_data['rushing_fumbles'] + nfl_data['receiving_fumbles']
        nfl_data['fumbles_lost'] = nfl_data['sack_fumbles_lost'] + \
            nfl_data['rushing_fumbles_lost'] + \
            nfl_data['receiving_fumbles_lost']
        nfl_data['yards'] = nfl_data['receiving_yards'] + \
            nfl_data['rushing_yards']
        nfl_data['tds'] = nfl_data['rushing_tds'] + nfl_data['receiving_tds']

        nfl_data['fantasy_points_prizepicks'] = nfl_data['passing_yards']/25 + nfl_data['passing_tds']*4 - \
            nfl_data['interceptions'] + nfl_data['yards']/10 + nfl_data['tds']*6 + \
            nfl_data['receptions'] - nfl_data['fumbles_lost'] + \
            nfl_data['special_teams_tds'] + nfl_data['passing_2pt_conversions']*2 + \
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

        nfl_data.rename(
            columns=lambda x: x.replace("_", " "), inplace=True)
        self.gamelog = pd.concat(
            [self.gamelog, nfl_data], ignore_index=True).drop_duplicates().reset_index(drop=True)

        for i, row in tqdm(self.gamelog.iterrows(), desc="Updating NFL data", unit="game", total=len(self.gamelog)):
            if row['opponent'] != row['opponent']:
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

                self.gamelog.at[i, 'moneyline'] = float(archive['NFL']['Moneyline'].get(
                    self.gamelog.at[i, 'gameday'], {}).get(row['recent team'], 0.5))
                self.gamelog.at[i, 'totals'] = float(archive['NFL']['Totals'].get(
                    self.gamelog.at[i, 'gameday'], {}).get(row['recent team'], 45.0))

        self.gamelog.loc[self.gamelog['recent team']
                         == 'LA', 'recent team'] = "LAR"
        self.gamelog = self.gamelog.sort_values('gameday')

        self.players = nfl.import_ids()
        self.players = self.players.loc[self.players['position'].isin([
            'QB', 'RB', 'WR', 'TE'])]
        self.players = self.players.groupby(
            'name')['position'].apply(lambda x: x.iat[-1]).to_dict()

        # Save the updated player data
        self.gamelog.to_pickle(pkg_resources.files(data) / "nfl_data.dat")

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
        gamelog = self.gamelog.loc[(pd.to_datetime(self.gamelog["gameday"]) < date) &
                                   (pd.to_datetime(self.gamelog["gameday"]) > date - timedelta(days=365))]
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

    def profile_market(self, market, date=datetime.today()):
        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.profiled_market = market
        self.profile_latest_date = date

        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])

        gamelog = self.gamelog.loc[(pd.to_datetime(self.gamelog["gameday"]) < date) &
                                   (pd.to_datetime(self.gamelog["gameday"]) > date - timedelta(days=730))]
        # gamelog[[market, 'moneyline', 'totals']] = gamelog[[
        #     market, 'moneyline', 'totals']].apply(pd.to_numeric)
        playerGroups = gamelog.\
            groupby('player display name').\
            filter(lambda x: len(x[x[market] != 0]) > 4).\
            groupby('player display name')

        leagueavg = playerGroups[market].mean().mean()
        if np.isnan(leagueavg):
            return
        self.playerProfile['avg'] = playerGroups.apply(
            lambda x: x[market].mean()/leagueavg)-1

        self.playerProfile['home'] = playerGroups.apply(
            lambda x: x.loc[x['home'], market].mean()/x[market].mean())-1

        self.playerProfile['away'] = playerGroups.apply(
            lambda x: x.loc[~x['home'].astype(bool), market].mean()/x[market].mean())-1

        self.playerProfile['moneyline gain'] = playerGroups.\
            apply(lambda x: np.polyfit(x.moneyline.values.astype(float),
                  x[market].values/x[market].mean(), 1)[1])

        self.playerProfile['totals gain'] = playerGroups.\
            apply(lambda x: np.polyfit(x.totals.values.astype(float)/45,
                  x[market].values/x[market].mean(), 1)[1])

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
        market = offer["Market"].replace("H2H ", "")
        line = offer["Line"]
        opponent = offer["Opponent"]
        home = offer.get("Home")
        if home is None:
            home = 0
            teams = team.split("/")
            for i, t in enumerate(teams):
                home += (i+1)*self.upcoming_games.get(t, {}).get("Home", 0)

        players = player.strip().replace("vs.", "+").split(" + ")
        for p in players:
            if p not in self.playerProfile.index:
                self.playerProfile.loc[p] = np.zeros_like(
                    self.playerProfile.columns)

        try:
            stats = (
                archive["NFL"][market]
                .get(date, {})
                .get(player, {})
                .get(line, [0.5] * 4)
            )
            moneyline = 0
            total = 0
            teams = team.split("/")
            for t in teams:
                moneyline += archive["NFL"]["Moneyline"].get(
                    date, {}).get(t, np.nan)
                total += archive["NFL"]["Totals"].get(date, {}).get(t, np.nan)

            moneyline /= len(teams)
            total /= len(teams)

        except:
            return 0

        if np.isnan(moneyline):
            moneyline = 0.5
        if np.isnan(total):
            total = 45

        date = datetime.strptime(date, "%Y-%m-%d")
        if "+" in player or "vs." in player:
            players = player.strip().replace("vs.", "+").split(" + ")
            opponents = opponent.split("/")
            if len(opponents) == 1:
                opponents = opponents * 2

            player1_games = self.gamelog.loc[(self.gamelog["player display name"] == players[0]) & (
                pd.to_datetime(self.gamelog["gameday"]) < date)]
            position1 = self.players.get(players[0], "")

            headtohead1 = player1_games.loc[player1_games["opponent"]
                                            == opponents[0]]

            player2_games = self.gamelog.loc[(self.gamelog["player display name"] == players[1]) & (
                pd.to_datetime(self.gamelog["gameday"]) < date)]
            position2 = self.players.get(players[1], "")

            headtohead2 = player2_games.loc[player2_games["opponent"]
                                            == opponents[1]]

            n = min(len(player1_games), len(player2_games))
            m = min(len(headtohead1), len(headtohead2))

            player1_games = player1_games.iloc[-n:]
            player2_games = player2_games.iloc[-n:]
            if n == 0:
                player1_games.drop(player1_games.index, inplace=True)
                player2_games.drop(player2_games.index, inplace=True)
            headtohead1 = headtohead1.iloc[-m:]
            headtohead2 = headtohead2.iloc[-m:]
            if m == 0:
                headtohead1.drop(headtohead1.index, inplace=True)
                headtohead2.drop(headtohead2.index, inplace=True)

            dvpoa1 = self.dvpoa(opponents[0], position1, market, date=date)
            dvpoa2 = self.dvpoa(opponents[1], position2, market, date=date)

            if "+" in player:
                game_res = (player1_games[market].to_numpy() +
                            player2_games[market].to_numpy() - line)
                h2h_res = (headtohead1[market].to_numpy() +
                           headtohead2[market].to_numpy() - line)
                try:
                    dvpoa = (2 / (1 / (dvpoa1 + 1) + 1 / (dvpoa2 + 1))) - 1
                except:
                    dvpoa = 0

            else:
                game_res = (player1_games[market].to_numpy() -
                            player2_games[market].to_numpy() + line)
                h2h_res = (headtohead1[market].to_numpy() -
                           headtohead2[market].to_numpy() + line)
                try:
                    dvpoa = (2 / (1 / (dvpoa1 + 1) - 1 / (dvpoa2 + 1))) - 1
                except:
                    dvpoa = 0

            game_res = list(game_res)
            h2h_res = list(h2h_res)

        else:
            player_games = self.gamelog.loc[(self.gamelog["player display name"] == player) & (
                pd.to_datetime(self.gamelog["gameday"]) < date)]
            position = self.players.get(player, "")

            headtohead = player_games.loc[player_games["opponent"] == opponent]

            game_res = (player_games[market]-line).to_list()
            h2h_res = (headtohead[market]-line).to_list()

            dvpoa = self.dvpoa(opponent, position, market, date=date)

        stats[stats == None] = np.nan
        odds = np.nanmean(stats)
        if np.isnan(odds):
            odds = 0.5

        data = {
            "DVPOA": dvpoa,
            "Odds": odds - 0.5,
            "Last5": np.mean([int(i > 0) for i in game_res[-5:]]) - 0.5 if game_res else 0,
            "Last10": np.mean([int(i > 0) for i in game_res[-10:]]) - 0.5 if game_res else 0,
            "H2H": np.mean([int(i > 0) for i in h2h_res[-5:]] + [1, 0]) - 0.5 if h2h_res else 0,
            "Avg5": np.mean(game_res[-5:]) if game_res else 0,
            "Avg10": np.mean(game_res[-10:]) if game_res else 0,
            "AvgH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
            "Moneyline": moneyline - 0.5,
            "Total": total / 45 - 1,
            "Line": line,
            "Home": home,
            "Combo": 1 if "+" in player else 0,
            "Rival": 1 if "vs." in player else 0,
        }

        if len(game_res) < 6:
            i = 6 - len(game_res)
            game_res = [0] * i + game_res
        if len(h2h_res) < 5:
            i = 5 - len(h2h_res)
            h2h_res = [0] * i + h2h_res

        # Update the data dictionary with additional values
        data.update(
            {"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
        data.update({"Game " + str(i + 1): game_res[-6 + i] for i in range(6)})

        player_data = self.playerProfile.loc[players, [
            'avg', 'home', 'away']].mean() + 1
        data.update(
            {"Player " + col: player_data[col] - 1 for col in ['avg', 'home', 'away']})

        other_player_data = np.mean(self.playerProfile.loc[players, self.playerProfile.columns.difference(
            ['avg', 'home', 'away'])], axis=0)
        data.update(
            {"Player " + col: other_player_data[col] for col in other_player_data.index})

        return data

    def get_training_matrix(self, market):
        """
        Retrieves training data in the form of a feature matrix (X) and a target vector (y) for a specified market.

        Args:
            market (str): The market for which to retrieve training data.

        Returns:
            tuple: A tuple containing the feature matrix (X) and the target vector (y).
        """
        archive.__init__(True)

        # Initialize an empty list for the target labels
        features = []
        results = []

        for i, game in tqdm(self.gamelog.iterrows(), unit="game", desc="Gathering Training Data", total=len(self.gamelog)):
            gameDate = datetime.strptime(
                game["gameday"], "%Y-%m-%d")
            data = {}
            self.profile_market(market, date=gameDate)

            try:
                names = list(
                    archive["NFL"][market].get(
                        gameDate.strftime("%Y-%m-%d"), {}).keys()
                )
                for name in names:
                    if (
                        game["player display name"]
                        == name.strip().replace("vs.", "+").split(" + ")[0]
                    ):
                        data[name] = archive["NFL"][market][
                            gameDate.strftime("%Y-%m-%d")
                        ][name]
            except:
                continue

            for name, archiveData in data.items():
                offer = {
                    "Player": name,
                    "Team": game["recent team"],
                    "Market": market,
                    "Opponent": game["opponent"],
                    "Home": int(game["home"])
                }

                if " + " in name or " vs. " in name:
                    player2 = name.replace("vs.", "+").split(" + ")[1].strip()
                    game2 = self.gamelog.loc[(self.gamelog.gameday == gameDate.strftime(
                        "%Y-%m-%d")) & (self.gamelog['player display name'] == player2)].squeeze()
                    if game2.empty:
                        continue
                    offer = offer | {
                        "Team": "/".join(
                            [game["recent team"], game2["recent team"]]
                        ),
                        "Opponent": "/".join([game["opponent"], game2["opponent"]]),
                        "Home": int(game["home"]) + 2*int(game2["home"])
                    }

                for line, stats in archiveData.items():
                    if not line == "Closing Lines" and not game[market] == line:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            new_get_stats = self.get_stats(
                                offer | {"Line": line}, gameDate
                            )
                            if type(new_get_stats) is dict:
                                if " + " in name:
                                    results.append(
                                        {
                                            "Result": (game[market] + game2[market]) - line
                                        }
                                    )
                                elif " vs. " in name:
                                    results.append(
                                        {
                                            "Result": (game[market] + line) - game2[market]
                                        }
                                    )
                                else:
                                    results.append(
                                        {"Result": game[market] - line}
                                    )

                                features.append(new_get_stats)

        y = pd.DataFrame(results)
        X = pd.DataFrame(features).fillna(0.0)

        return X, y


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
        self.gameIds = []
        self.season_start = datetime.strptime("2022-10-07", "%Y-%m-%d").date()

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
            self.gameIds = nhl_data["games"]

    def parse_game(self, gameId, gameDate):
        game = scraper.get(
            f"https://statsapi.web.nhl.com/api/v1/game/{gameId}/boxscore")
        if game:
            self.gameIds.append(gameId)
            awayTeam = abbreviations['NHL'][remove_accents(game['teams']
                                            ['away']['team']['name'])]
            homeTeam = abbreviations['NHL'][remove_accents(game['teams']
                                            ['home']['team']['name'])]

            for player in game['teams']['away']['players'].values():
                n = {
                    "gameId": gameId,
                    "gameDate": gameDate,
                    "team": awayTeam,
                    "opponent": homeTeam,
                    "home": False,
                    "playerId": player['person']['id'],
                    "playerName": player['person']['fullName'],
                    "position": player['position']['code']
                }
                stats = player['stats']['goalieStats'] if n['position'] == 'G' else player['stats'].get(
                    'skaterStats', {})
                stats = stats | {
                    'sogBS': stats.get('shots', 0) + stats.get('blocked', 0) if n['position'] != "G" else 0,
                    'points': stats.get('goals', 0) + stats.get('assists', 0) if n['position'] != "G" else 0,
                    'powerPlayPoints': stats.get('powerPlayGoals', 0) + stats.get('powerPlayAssists', 0) if n['position'] != "G" else 0,
                    'goalsAgainst': stats.get('shots', 0) - stats.get('saves', 0) if n['position'] == "G" else 0
                }
                stats = stats | {
                    'fantasy score': stats.get('goals', 0)*8 + stats.get('assists', 0)*5 + stats.get('sogBS', 0)*1.5,
                    'goalie fantasy points underdog': int(stats.get('decision', 'L') == 'W')*6 + stats.get('saves', 0)*.6 - stats.get('goalsAgainst', 0)*.6,
                    'skater fantasy points underdog': stats.get('goals', 0)*6 + stats.get('assists', 0)*4 + stats.get('sogBS', 0) + stats.get('hits', 0) + stats.get('powerPlayPoints', 0)*.5,
                    'goalie fantasy points parlay': stats.get('saves', 0)*.25 - stats.get('goalsAgainst', 0),
                    'skater fantasy points parlay': stats.get('goals', 0)*3 + stats.get('assists', 0)*2 + stats.get('shots', 0)*.5 + stats.get('hits', 0) + stats.get('blocked', 0),
                }
                self.gamelog.append(n | stats)

            for player in game['teams']['home']['players'].values():
                n = {
                    "gameId": gameId,
                    "gameDate": gameDate,
                    "team": homeTeam,
                    "opponent": awayTeam,
                    "home": True,
                    "playerId": player['person']['id'],
                    "playerName": player['person']['fullName'],
                    "position": player['position']['code']
                }
                stats = player['stats']['goalieStats'] if n['position'] == 'G' else player['stats'].get(
                    'skaterStats', {})
                timeOnIce = stats.get('timeOnIce', "0:00").split(":")
                stats = stats | {
                    'timeOnIce': float(timeOnIce[0]) + float(timeOnIce[1])/60.0,
                    'sogBS': stats.get('shots', 0) + stats.get('blocked', 0) if n['position'] != "G" else 0,
                    'points': stats.get('goals', 0) + stats.get('assists', 0) if n['position'] != "G" else 0,
                    'powerPlayPoints': stats.get('powerPlayGoals', 0) + stats.get('powerPlayAssists', 0) if n['position'] != "G" else 0,
                    'goalsAgainst': stats.get('shots', 0) - stats.get('saves', 0) if n['position'] == "G" else 0
                }
                stats = stats | {
                    'fantasy score': stats.get('goals', 0)*8 + stats.get('assists', 0)*5 + stats.get('sogBS', 0)*1.5,
                    'goalie fantasy points underdog': int(stats.get('decision', 'L') == 'W')*6 + stats.get('saves', 0)*.6 - stats.get('goalsAgainst', 0)*.6,
                    'skater fantasy points underdog': stats.get('goals', 0)*6 + stats.get('assists', 0)*4 + stats.get('sogBS', 0) + stats.get('hits', 0) + stats.get('powerPlayPoints', 0)*.5,
                    'goalie fantasy points parlay': stats.get('saves', 0)*.25 - stats.get('goalsAgainst', 0),
                    'skater fantasy points parlay': stats.get('goals', 0)*3 + stats.get('assists', 0)*2 + stats.get('shots', 0)*.5 + stats.get('hits', 0) + stats.get('blocked', 0),
                }
                self.gamelog.append(n | stats)

    def update(self):
        """
        Updates the NHL skater and goalie data.

        Args:
            None

        Returns:
            None
        """

        # Get game ids
        today = datetime.today().date()
        start_date = self.season_start.strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        res = scraper.get(
            f"https://statsapi.web.nhl.com/api/v1/schedule?startDate={start_date}&endDate={end_date}")

        ids = [[(game['gamePk'], date['date']) for game in date['games'] if game['gameType']
                not in ["PR", "A"]] for date in res['dates']]

        ids = [item for sublist in ids for item in sublist]

        # Parse the game stats
        for gameId, date in tqdm(ids, desc="Getting NHL Stats"):
            if gameId not in self.gameIds:
                self.parse_game(gameId, date)

        # Remove old games to prevent file bloat
        for game in self.gamelog:
            if datetime.strptime(
                game["gameDate"], "%Y-%m-%d"
            ).date() < today - timedelta(days=730):
                self.gamelog.remove(game)

        # Write to file
        with open((pkg_resources.files(data) / "nhl_data.dat"), "wb") as outfile:
            pickle.dump(
                {"games": self.gameIds, "gamelog": self.gamelog}, outfile)

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
            elif datetime.strptime(game['gameDate'], '%Y-%m-%d').date() < (date - timedelta(days=365)).date():
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

    def profile_market(self, market, date=datetime.today()):
        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.profiled_market = market
        self.profile_latest_date = date

        # Initialize playerStats and edges
        self.playerProfile = pd.DataFrame(columns=['avg', 'home', 'away'])
        self.playerStats = {}
        self.edges = []
        gamelog = []

        # Iterate over game log and gather stats for each player
        for game in self.gamelog:
            gameDate = datetime.strptime(
                game['gameDate'][:10], '%Y-%m-%d').date()
            if gameDate > date.date():
                continue
            elif gameDate < (date - timedelta(days=365)).date():
                continue

            game["moneyline"] = float(archive["NHL"]["Moneyline"].get(
                gameDate.strftime("%Y-%m-%d"), {}).get(game["team"], 0.5))
            game["totals"] = float(archive["NHL"]["Totals"].get(
                gameDate.strftime("%Y-%m-%d"), {}).get(game["team"], 228))
            gamelog.append(game)

        gamelog = pd.DataFrame(gamelog)

        playerGroups = gamelog.\
            groupby('playerName').\
            filter(lambda x: len(x[x[market] != 0]) > 4).\
            groupby('playerName')

        leagueavg = playerGroups[market].mean().mean()
        if np.isnan(leagueavg):
            return
        self.playerProfile['avg'] = playerGroups.apply(
            lambda x: x[market].mean()/leagueavg)-1

        self.playerProfile['home'] = playerGroups.apply(
            lambda x: x.loc[x['home'], market].mean()/x[market].mean())-1

        self.playerProfile['away'] = playerGroups.apply(
            lambda x: x.loc[~x['home'].astype(bool), market].mean()/x[market].mean())-1

        self.playerProfile['moneyline gain'] = playerGroups.\
            apply(lambda x: np.polyfit(x.moneyline.fillna(0.5).values.astype(float),
                  x[market].values/x[market].mean(), 1)[1])

        self.playerProfile['totals gain'] = playerGroups.\
            apply(lambda x: np.polyfit(x.totals.fillna(8.3).values.astype(float)/8.3,
                  x[market].values/x[market].mean(), 1)[1])

        # self.playerProfile = self.playerProfile.dropna()

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
            elif datetime.strptime(game['gameDate'], '%Y-%m-%d').date() < (date - timedelta(days=365)).date():
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

        # Convert date to string format if it's a datetime object
        if type(date) is datetime:
            date = date.strftime("%Y-%m-%d")

        # Retrieve offer details
        player = offer["Player"]
        team = offer["Team"]
        market = offer["Market"].replace("H2H ", "")
        line = offer["Line"]
        opponent = offer["Opponent"]
        home = offer.get("Home")
        if home is None:
            home = 0
            teams = team.split("/")
            for i, t in enumerate(teams):
                home += (i+1)*self.upcoming_games.get(t, {}).get("Home", 0)

        # Replace team abbreviations and market names with appropriate values
        opponent = opponent.replace("NJ", "NJD").replace("TB", "TBL")
        if opponent == "LA":
            opponent = "LAK"

        if market == "PTS":
            market = "points"
        elif market == "AST":
            market = "assists"
        elif market == "BLK":
            market = "blocked"

        # Determine the bucket based on player type and line value
        players = player.strip().replace("vs.", "+").split(" + ")
        for p in players:
            if p not in self.playerProfile.index:
                self.playerProfile.loc[p] = np.zeros_like(
                    self.playerProfile.columns)

        try:
            # Retrieve stats, moneyline, and total from the archive based on date, player, and team
            stats = (
                archive["NHL"][market]
                .get(date, {})
                .get(player, {})
                .get(line, [0.5] * 4)
            )
            moneyline = 0
            total = 0
            teams = team.split("/")
            for t in teams:
                moneyline += archive["NHL"]["Moneyline"].get(
                    date, {}).get(t, np.nan)
                total += archive["NHL"]["Totals"].get(date, {}).get(t, np.nan)

            moneyline /= len(teams)
            total /= len(teams)
        except:
            return 0

        # Replace NaN values with appropriate defaults
        if np.isnan(moneyline):
            moneyline = 0.5
        if np.isnan(total):
            total = 6.5

        # Convert date to datetime object
        date = datetime.strptime(date, "%Y-%m-%d")

        if "+" in player or "vs." in player:
            # Split player and opponent information for combined or head-to-head offers
            players = player.strip().replace("vs.", "+").split(" + ")
            opponents = opponent.split("/")
            if len(opponents) == 1:
                opponents = opponents * 2

            # Filter games based on player and opponent for each player
            player1_games = [
                game
                for game in self.gamelog
                if game['playerName'] == players[0]
                and datetime.strptime(game["gameDate"], "%Y-%m-%d") < date
            ]

            headtohead1 = [
                game
                for game in player1_games
                if game["opponent"] == opponents[0]
            ]

            player2_games = [
                game
                for game in self.gamelog
                if game['playerName'] == players[1]
                and datetime.strptime(game["gameDate"], "%Y-%m-%d") < date
            ]

            headtohead2 = [
                game
                for game in player2_games
                if game["opponent"] == opponents[1]
            ]

            # Determine the minimum number of games for player and head-to-head comparisons
            n = np.min([len(player1_games), len(player2_games)])
            m = np.min([len(headtohead1), len(headtohead2)])

            # Trim the game lists to the determined minimum length
            player1_games = player1_games[-n:]
            player2_games = player2_games[-n:]
            headtohead1 = headtohead1[-m:]
            headtohead2 = headtohead2[-m:]
            if n == 0:
                player1_games = []
                player2_games = []
            if m == 0:
                headtohead1 = []
                headtohead2 = []

            # Determine the positions for player1, player2, and DVPOA calculation
            position1 = player1_games[0]["position"]
            position2 = player2_games[0]["position"]

            # Calculate DVPOA for opponent and positions
            dvpoa1 = self.dvpoa(opponents[0], position1, market, date=date)
            dvpoa2 = self.dvpoa(opponents[1], position2, market, date=date)

            if "+" in player:
                # Calculate game and head-to-head results for combined offers
                game_res = (
                    np.array([game[market] for game in player1_games])
                    + np.array([game[market] for game in player2_games])
                    - np.array([line] * n)
                )
                h2h_res = (
                    np.array([game[market] for game in headtohead1])
                    + np.array([game[market] for game in headtohead2])
                    - np.array([line] * m)
                )

                try:
                    dvpoa = (2 / (1 / (dvpoa1 + 1) + 1 / (dvpoa2 + 1))) - 1
                except:
                    dvpoa = 0

            else:
                # Calculate game and head-to-head results for head-to-head offers
                game_res = (
                    np.array([game[market] for game in player1_games])
                    - np.array([game[market] for game in player2_games])
                    + np.array([line] * n)
                )
                h2h_res = (
                    np.array([game[market] for game in headtohead1])
                    - np.array([game[market] for game in headtohead2])
                    + np.array([line] * m)
                )

                try:
                    dvpoa = (2 / (1 / (dvpoa1 + 1) + 1 / (dvpoa2 + 1))) - 1
                except:
                    dvpoa = 0

            game_res = list(game_res)
            h2h_res = list(h2h_res)

        else:
            # Filter games based on player and opponent for single player offers
            player_games = [
                game
                for game in self.gamelog
                if game['playerName'] == player
                and datetime.strptime(game["gameDate"], "%Y-%m-%d") < date
            ]

            headtohead = [
                game for game in player_games if game["opponent"] == opponent
            ]

            # Calculate game and head-to-head results for single player offers
            game_res = [game.get(market, 0) - line for game in player_games]
            h2h_res = [game[market] - line for game in headtohead]

            position = player_games[0]["position"]

            dvpoa = self.dvpoa(opponent, position, market, date=date)

        stats[stats is None] = np.nan
        odds = np.nanmean(stats)
        if np.isnan(odds):
            odds = 0.5

        # Calculate various statistics based on game and head-to-head results
        data = {
            "DVPOA": dvpoa,
            "Odds": odds - 0.5,
            "Last5": np.mean([int(i > 0) for i in game_res[-5:]]) - 0.5
            if game_res
            else 0,
            "Last10": np.mean([int(i > 0) for i in game_res[-10:]]) - 0.5
            if game_res
            else 0,
            "H2H": np.mean([int(i > 0) for i in h2h_res[-5:]] + [1, 0]) - 0.5 if h2h_res else 0,
            "Avg5": np.mean(game_res[-5:]) if game_res[-5:] else 0,
            "Avg10": np.mean(game_res[-10:]) if game_res[-10:] else 0,
            "AvgH2H": np.mean(h2h_res[-5:]) if h2h_res[-5:] else 0,
            "Moneyline": moneyline - 0.5,
            "Total": total / 6.5 - 1,
            "Line": line,
            "Home": home,
            "Combo": 1 if "+" in player else 0,
            "Rival": 1 if "vs." in player else 0,
        }

        # Pad game_res and h2h_res lists with zeros if they are shorter than required
        if len(game_res) < 6:
            i = 6 - len(game_res)
            game_res = [0] * i + game_res
        if len(h2h_res) < 5:
            i = 5 - len(h2h_res)
            h2h_res = [0] * i + h2h_res

        # Create a DataFrame with the calculated statistics
        X = pd.DataFrame(data, index=[0]).fillna(0)
        X = X.join(pd.DataFrame([h2h_res[-5:]]
                                ).fillna(0).add_prefix("Meeting "))
        X = X.join(pd.DataFrame([game_res[-6:]]).fillna(0).add_prefix("Game "))
        X = X.join(pd.DataFrame(
            [hmean(self.playerProfile.loc[players, ['avg', 'home', 'away']]+1)-1], columns=['avg', 'home', 'away']).
            add_prefix("Player "))
        X = X.join(pd.DataFrame(
            [np.mean(self.playerProfile.loc[players, self.playerProfile.columns.difference(
                ['avg', 'home', 'away'])], axis=0)],
            columns=self.playerProfile.loc[players, self.playerProfile.columns.difference(['avg', 'home', 'away'])].columns).
            add_prefix("Player "))

        return X.fillna(0.0)

    def get_training_matrix(self, market):
        """
        Retrieve the training matrix for the specified market.

        Args:
            market (str): The market for which to retrieve the training data.

        Returns:
            tuple: A tuple containing the training matrix (X) and the corresponding results (y).
        """

        # Initialize variables
        archive.__init__(True)
        X = pd.DataFrame()
        results = []

        # Iterate over each game in the gamelog
        for game in tqdm(self.gamelog, unit="game", desc="Gathering Training Data"):
            gameDate = datetime.strptime(game["gameDate"], "%Y-%m-%d")
            if gameDate < datetime(2022, 11, 7):
                continue
            if market not in game:
                continue
            data = {}
            self.profile_market(market, date=gameDate)

            try:
                names = list(
                    archive["NHL"][market].get(
                        gameDate.strftime("%Y-%m-%d"), {}).keys()
                )

                # Filter data based on the player's name in the gamelog
                for name in names:
                    if (
                        game['playerName']
                        == name.strip().replace("vs.", "+").split(" + ")[0]
                    ):
                        data[name] = archive["NHL"][market][
                            gameDate.strftime("%Y-%m-%d")
                        ][name]
            except:
                continue

            # Iterate over each name and associated archive data
            for name, archiveData in data.items():
                offer = {
                    "Player": name,
                    "Team": game["team"],
                    "Market": market,
                    "Opponent": game["opponent"],
                    "Home": int(game["home"])
                }

                # Modify offer parameters for multi-player cases
                if " + " in name or " vs. " in name:
                    player2 = name.replace("vs.", "+").split(" + ")[1]
                    game2 = next(
                        (
                            i
                            for i in self.gamelog
                            if i["gameDate"] == gameDate.strftime("%Y-%m-%d")
                            and i['playerName'] == player2
                        ),
                        None,
                    )
                    if game2 is None:
                        continue
                    offer = offer | {
                        "Team": "/".join([game["team"], game2["team"]]),
                        "Opponent": "/".join(
                            [game["opponent"],
                                game2["opponent"]]
                        ),
                        "Home": int(game["home"]) + 2*int(game2["home"])
                    }

                # Iterate over each line and stats in the archive data
                for line, stats in archiveData.items():
                    if not line == "Closing Lines" and not game[market] == line:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            new_get_stats = self.get_stats(
                                offer | {"Line": line}, gameDate
                            )

                            if type(new_get_stats) is pd.DataFrame:
                                # Compute the result and append it to the results list

                                if " + " in name:
                                    player2 = name.split(" + ")[1]
                                    game2 = next(
                                        (
                                            i
                                            for i in self.gamelog
                                            if i['playerName'] == player2
                                            and i["gameId"] == game["gameId"]
                                        ),
                                        None,
                                    )
                                    if game2 is None:
                                        continue
                                    results.append(
                                        {
                                            "Result": (game[market] + game2[market]) - line
                                        }
                                    )
                                elif " vs. " in name:
                                    player2 = name.split(" vs. ")[1]
                                    game2 = next(
                                        (
                                            i
                                            for i in self.gamelog
                                            if i['playerName'] == player2
                                            and i["gameId"] == game["gameId"]
                                        ),
                                        None,
                                    )
                                    if game2 is None:
                                        continue
                                    results.append(
                                        {
                                            "Result": (game[market] + line) - game2[market]
                                        }
                                    )
                                else:
                                    results.append(
                                        {"Result": game[market] - line}
                                    )

                                # Concatenate the new_get_stats dataframe with X
                                if X.empty:
                                    X = new_get_stats
                                else:
                                    X = pd.concat([X, new_get_stats])

        # Create the results dataframe
        y = pd.DataFrame(results)
        X.reset_index(drop=True, inplace=True)

        return X, y
