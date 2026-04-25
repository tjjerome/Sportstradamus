# ARCHIVED 2026-04-25 from src/sportstradamus/stats.py
# Reason: zero callers anywhere in the codebase (neither internal nor external).
# The obs_* family (obs_get_stats, obs_get_training_matrix, obs_profile_market,
# dvpoa, bucket_stats) was an older per-observation prediction and analysis API,
# superseded by the current vectorized offer-based API (get_stats, get_training_matrix,
# profile_market). get_fantasy was unused NFL fantasy scoring.
# Last git SHA where it was live: see git log src/sportstradamus/stats.py
# Each function includes a "# Was: ClassName.method_name" comment.

# Was: Stats._load_comps
@staticmethod
def _load_comps(filepath):
    """Load comps JSON, handling both old (list) and new (dict with distances) formats."""
    if not os.path.isfile(filepath):
        return {}
    with open(filepath) as infile:
        raw = json.load(infile)
    # Detect old format: values are lists of player names, not dicts
    for player_comps in raw.values():
        sample = next(iter(player_comps.values()), None)
        if isinstance(sample, list):
            # Convert old format to new format (distances unknown, use index-based proxy)
            for player, comp_list in player_comps.items():
                player_comps[player] = {
                    "comps": comp_list,
                    "distances": [round(i * 0.5, 4) for i in range(len(comp_list))],
                }
    return raw

# Was: StatsNBA.bucket_stats
def bucket_stats(self, market, buckets=20, date=datetime.today()):
    """Bucket player stats based on a given market.

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
        if datetime.strptime(game["GAME_DATE"][:10], "%Y-%m-%d").date() > date.date() or (
            datetime.strptime(game["GAME_DATE"][:10], "%Y-%m-%d").date()
            < (date - timedelta(days=300)).date()
        ):
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
        self.playerStats[player]["avg"] = np.mean(games["games"]) if games["games"] else 0
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
                np.mean([v for v in averages if self.edges[i - 1] <= v <= self.edges[i]]) - 0.5
            )
            + 0.5
        )

    # Assign bucket and line values to each player
    for player, games in self.playerStats.items():
        for i in range(buckets):
            if games["avg"] >= self.edges[i]:
                self.playerStats[player]["bucket"] = buckets - i
                self.playerStats[player]["line"] = lines[i]

# Was: StatsNBA.obs_profile_market
@line_profiler.profile
def obs_profile_market(self, market, date=datetime.today().date()):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, datetime):
        date = date.date()
    if market == self.profiled_market and date == self.profile_latest_date:
        return

    self.base_profile(date)
    self.profiled_market = market

    playerGroups = (
        self.short_gamelog.groupby(self.log_strings["player"])
        .filter(lambda x: (x[market].clip(0, 1).mean() > 0.1) & (x[market].count() > 1))
        .groupby(self.log_strings["player"])
    )

    leagueavg = playerGroups[market].mean().mean()
    leaguestd = playerGroups[market].mean().std()
    if np.isnan(leagueavg) or np.isnan(leaguestd):
        return

    self.playerProfile[["avg", "z", "home", "away", "moneyline gain", "totals gain"]] = 0
    self.playerProfile["avg"] = playerGroups[market].mean().div(leagueavg) - 1
    self.playerProfile["z"] = (playerGroups[market].mean() - leagueavg).div(leaguestd)
    self.playerProfile["home"] = (
        playerGroups.apply(lambda x: x.loc[x["HOME"], market].mean() / x[market].mean()) - 1
    )
    self.playerProfile["away"] = (
        playerGroups.apply(
            lambda x: x.loc[~x["HOME"].astype(bool), market].mean() / x[market].mean()
        )
        - 1
    )

    defenseGroups = self.short_gamelog.groupby([self.log_strings["opponent"], "GAME_ID"])
    defenseGames = defenseGroups[[market, "HOME", "moneyline", "totals"]].agg(
        {
            market: "sum",
            "HOME": lambda x: np.mean(x) > 0.5,
            "moneyline": "mean",
            "totals": "mean",
        }
    )
    defenseGroups = defenseGames.groupby(self.log_strings["opponent"])

    self.defenseProfile[["avg", "z", "home", "away", "moneyline gain", "totals gain"]] = 0
    leagueavg = defenseGroups[market].mean().mean()
    leaguestd = defenseGroups[market].mean().std()
    self.defenseProfile["avg"] = defenseGroups[market].mean().div(leagueavg) - 1
    self.defenseProfile["z"] = (defenseGroups[market].mean() - leagueavg).div(leaguestd)
    self.defenseProfile["home"] = (
        defenseGroups.apply(lambda x: x.loc[x["HOME"], market].mean() / x[market].mean()) - 1
    )
    self.defenseProfile["away"] = (
        defenseGroups.apply(lambda x: x.loc[~x["HOME"], market].mean() / x[market].mean()) - 1
    )

    for position in self.positions:
        positionLogs = self.short_gamelog.loc[self.short_gamelog["POS"] == position]
        positionGroups = positionLogs.groupby(self.log_strings["player"])
        positionAvg = positionGroups[market].mean().mean()
        positionStd = positionGroups[market].mean().std()
        idx = list(
            set(positionGroups.groups.keys()).intersection(set(self.playerProfile.index))
        )
        self.playerProfile.loc[idx, "position avg"] = (
            positionGroups[market].mean().div(positionAvg) - 1
        )
        self.playerProfile.loc[idx, "position z"] = (
            positionGroups[market].mean() - positionAvg
        ).div(positionStd)
        positionGroups = positionLogs.groupby([self.log_strings["opponent"], "GAME_ID"])
        positionGames = positionGroups[[market, "HOME", "moneyline", "totals"]].agg(
            {
                market: "sum",
                "HOME": lambda x: np.mean(x) > 0.5,
                "moneyline": "mean",
                "totals": "mean",
            }
        )
        positionGroups = positionGames.groupby(self.log_strings["opponent"])
        leagueavg = positionGroups[market].mean().mean()
        if leagueavg == 0:
            self.defenseProfile[position] = 0
        else:
            self.defenseProfile[position] = positionGroups[market].mean().div(leagueavg) - 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        self.playerProfile["moneyline gain"] = playerGroups.apply(
            lambda x: np.polyfit(
                x.moneyline.fillna(0.5).values.astype(float) / 0.5
                - x.moneyline.fillna(0.5).mean(),
                x[market].values.astype(float) / x[market].mean() - 1,
                1,
            )[0]
        )

        self.playerProfile["totals gain"] = playerGroups.apply(
            lambda x: np.polyfit(
                x.totals.fillna(112).values.astype(float) / 112 - x.totals.fillna(112).mean(),
                x[market].values.astype(float) / x[market].mean() - 1,
                1,
            )[0]
        )

        self.defenseProfile["moneyline gain"] = defenseGroups.apply(
            lambda x: np.polyfit(
                x.moneyline.fillna(0.5).values.astype(float) / 0.5
                - x.moneyline.fillna(0.5).mean(),
                x[market].values.astype(float) / x[market].mean() - 1,
                1,
            )[0]
        )

        self.defenseProfile["totals gain"] = defenseGroups.apply(
            lambda x: np.polyfit(
                x.totals.fillna(112).values.astype(float) / 112 - x.totals.fillna(112).mean(),
                x[market].values.astype(float) / x[market].mean() - 1,
                1,
            )[0]
        )

# Was: StatsNBA.dvpoa
def dvpoa(self, team, position, market, date=datetime.today().date()):
    """Calculate the Defense Versus Position Above Average (DVPOA) for a specific team, position, and market.

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
        if datetime.strptime(game["GAME_DATE"][:10], "%Y-%m-%d").date() > date.date() or (
            datetime.strptime(game["GAME_DATE"][:10], "%Y-%m-%d").date()
            < (date - timedelta(days=300)).date()
        ):
            continue
        if game["POS"] == position or game["POS"] == "-".join(position.split("-")[::-1]):
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

# Was: StatsNBA.obs_get_stats
@line_profiler.profile
def obs_get_stats(self, offer, date=datetime.today()):
    """Generate a pandas DataFrame with a summary of relevant stats.

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
    cv = stat_cv.get(self.league, {}).get(market, 1)
    dist = stat_dist.get(self.league, {}).get(market, "Gamma")
    # if self.defenseProfile.empty:
    #     logger.exception(f"{market} not profiled")
    #     return 0
    home = offer.get("Home")
    if home is None:
        home = self.upcoming_games.get(team, {}).get("Home", 0)

    if player not in self.playerProfile.index:
        self.playerProfile.loc[player] = np.zeros_like(self.playerProfile.columns)

    if team not in self.teamProfile.index:
        self.teamProfile.loc[team] = np.zeros_like(self.teamProfile.columns)

    if opponent not in self.defenseProfile.index:
        self.defenseProfile.loc[opponent] = np.zeros_like(self.defenseProfile.columns)

    Date = datetime.strptime(date, "%Y-%m-%d")

    player_games = self.short_gamelog.loc[(self.short_gamelog["PLAYER_NAME"] == player)]

    if len(player_games) > 0:
        position = player_games.iloc[0]["POS"]
    else:
        logger.warning(f"{player} not found")
        return 0

    one_year_ago = len(player_games)
    headtohead = player_games.loc[player_games["OPP"] == opponent]

    game_res = (player_games[market]).to_list()
    h2h_res = (headtohead[market]).to_list()

    if position in self.defenseProfile.loc[opponent]:
        dvpoa = self.defenseProfile.loc[opponent, position]
    else:
        dvpoa = 0

    if line == 0:
        line = np.median(game_res[-one_year_ago:]) if game_res else 0
        line = 0.5 if line < 1 else line

    try:
        ev = archive.get_ev(self.league, market, date, player)
        moneyline = archive.get_moneyline(self.league, date, team)
        total = archive.get_total(self.league, date, team)

    except:
        logger.exception(f"{player}, {market}")
        return 0

    if np.isnan(ev):
        if market in combo_props:
            ev = 0
            for submarket in combo_props.get(market, []):
                sub_cv = stat_cv[self.league].get(submarket, 1)
                sub_dist = stat_dist.get(self.league, {}).get(submarket, "Gamma")
                v = archive.get_ev(self.league, submarket, date, player)
                subline = archive.get_line(self.league, submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(
                        subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                    )
                if np.isnan(v) or v == 0:
                    ev = 0
                    break
                else:
                    ev += v

        elif market in ["DREB", "OREB"]:
            ev = (
                (
                    archive.get_ev(self.league, "REB", date, player)
                    * player_games.iloc[-one_year_ago:][market].sum()
                    / player_games.iloc[-one_year_ago:]["REB"].sum()
                )
                if player_games.iloc[-one_year_ago:]["REB"].sum()
                else 0
            )

        elif "fantasy" in market:
            ev = 0
            book_odds = False
            fantasy_props = [
                ("PTS", 1),
                ("REB", 1.2),
                ("AST", 1.5),
                ("BLK", 3),
                ("STL", 3),
                ("TOV", -1),
            ]
            for submarket, weight in fantasy_props:
                sub_cv = stat_cv[self.league].get(submarket, 1)
                sub_dist = stat_dist.get(self.league, {}).get(submarket, "Gamma")
                v = archive.get_ev(self.league, submarket, date, player)
                subline = archive.get_line(self.league, submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(
                        subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                    )
                if np.isnan(v) or v == 0:
                    if subline == 0 and not player_games.empty:
                        subline = np.floor(player_games.iloc[-10:][submarket].median()) + 0.5

                    if subline != 0:
                        under = (player_games.iloc[-one_year_ago:][submarket] < subline).mean()
                        ev += get_ev(subline, under, sub_cv, dist=sub_dist) * weight
                else:
                    book_odds = True
                    ev += v * weight

            if not book_odds:
                ev = 0

    odds = 0 if np.isnan(ev) or ev <= 0 else 1 - get_odds(line, ev, dist, cv=cv)

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
        "MeanYr_nonzero": (
            np.mean([x for x in game_res[-one_year_ago:] if x > 0])
            if any(x > 0 for x in game_res[-one_year_ago:])
            else max(np.mean(game_res[-one_year_ago:]) if game_res else 0.5, 0.5)
        ),
        "MeanH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
        "STD10": np.std(game_res[-10:]) if game_res else 0,
        "STDYr": np.std(game_res[-one_year_ago:]) if game_res else 0,
        "Trend3": np.polyfit(np.arange(len(game_res[-3:])), game_res[-3:], 1)[0]
        if len(game_res) > 1
        else 0,
        "Trend5": np.polyfit(np.arange(len(game_res[-5:])), game_res[-5:], 1)[0]
        if len(game_res) > 1
        else 0,
        "TrendH2H": np.polyfit(np.arange(len(h2h_res[-3:])), h2h_res[-3:], 1)[0]
        if len(h2h_res) > 1
        else 0,
        "GamesPlayed": one_year_ago,
        "DaysIntoSeason": (Date.date() - self.season_start).days,
        "DaysOff": (
            Date.date() - pd.to_datetime(player_games.iloc[-1][self.log_strings["date"]]).date()
        ).days,
        "Moneyline": moneyline,
        "Total": total,
        "Home": home,
        "Position": self.positions.index(position),
    }

    if data["Line"] <= 0:
        data["Line"] = data["AvgYr"] if data["AvgYr"] > 1 else 0.5

    if len(game_res) < 5:
        i = 5 - len(game_res)
        game_res = [0] * i + game_res
    if len(h2h_res) < 5:
        i = 5 - len(h2h_res)
        h2h_res = [0] * i + h2h_res

    # Update the data dictionary with additional values
    data.update({"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
    data.update({"Game " + str(i + 1): game_res[-5 + i] for i in range(5)})

    player_data = self.playerProfile.loc[player]

    data.update(
        {
            f"Player {col}": player_data[f"{col}"]
            for col in [
                "avg",
                "home",
                "away",
                "z",
                "moneyline gain",
                "totals gain",
                "position avg",
                "position z",
            ]
        }
    )
    data.update({f"Player {col}": player_data[f"{col}"] for col in self.stat_types})
    data.update({f"Player {col} short": player_data[f"{col} short"] for col in self.stat_types})
    data.update(
        {f"Player {col} growth": player_data[f"{col} growth"] for col in self.stat_types}
    )

    team_data = self.teamProfile.loc[team]
    data.update({"Team " + col: team_data[col] for col in team_data.index})

    defense_data = self.defenseProfile.loc[opponent]
    data.update(
        {
            "Defense " + col: defense_data[col]
            for col in defense_data.index
            if col not in self.positions
        }
    )

    return data

# Was: StatsNBA.obs_get_training_matrix
def obs_get_training_matrix(self, market, cutoff_date=None):
    """Retrieves training data in the form of a feature matrix (X) and a target vector (y) for a specified market.

    Args:
        market (str): The market for which to retrieve training data.

    Returns:
        tuple: A tuple containing the feature matrix (X) and the target vector (y).
    """
    matrix = []

    if cutoff_date is None:
        cutoff_date = (datetime.today() - timedelta(days=850)).date()

    for _i, game in tqdm(
        self.gamelog.iterrows(),
        unit="game",
        desc="Gathering Training Data",
        total=len(self.gamelog),
    ):
        gameDate = datetime.strptime(game[self.log_strings["date"]][:10], "%Y-%m-%d").date()

        if game[market] < 0:
            continue

        if gameDate <= cutoff_date:
            continue

        self.profile_market(market, date=gameDate)
        name = game[self.log_strings["player"]]

        if name not in self.playerProfile.index:
            continue

        line = archive.get_line(self.league, market, game[self.log_strings["date"]][:10], name)

        offer = {
            "Player": name,
            "Team": game["TEAM_ABBREVIATION"],
            "Market": market,
            "Opponent": game["OPP"],
            "Home": int(game["HOME"]),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_get_stats = self.get_stats(
                offer | {"Line": line}, game[self.log_strings["date"]][:10]
            )
            if type(new_get_stats) is dict:
                new_get_stats.update(
                    {"Result": game[market], "Date": gameDate, "Archived": int(line != 0)}
                )

                matrix.append(new_get_stats)

    M = pd.DataFrame(matrix).fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)

    return M

# Was: StatsMLB.bucket_stats
def bucket_stats(self, market, buckets=20, date=datetime.today()):
    """Buckets the statistics of players based on a given market (e.g., 'allowed', 'pitch').

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
    gamelog = self.gamelog.loc[
        (pd.to_datetime(self.gamelog["gameDate"]) < date)
        & (pd.to_datetime(self.gamelog["gameDate"]) > date - timedelta(days=60))
    ]

    if gamelog.empty:
        gamelog = self.gamelog.loc[
            (pd.to_datetime(self.gamelog["gameDate"]) < date)
            & (pd.to_datetime(self.gamelog["gameDate"]) > date - timedelta(days=240))
        ]

    playerGroups = (
        gamelog.groupby("playerName")
        .filter(lambda x: len(x[x[market] != 0]) > 4)
        .groupby("playerName")[market]
    )

    # Compute edges for each bucket
    w = int(100 / buckets)
    self.edges = playerGroups.mean().quantile(np.arange(0, 101, w) / 100).to_list()

    # Assign bucket and line values to each player
    self.playerStats["avg"] = playerGroups.mean()
    self.playerStats["bucket"] = np.ceil(
        playerGroups.mean().rank(pct=True, ascending=False) * 20
    ).astype(int)
    self.playerStats["line"] = playerGroups.median()
    self.playerStats.loc[self.playerStats["line"] == 0.0, "line"] = 0.5
    self.playerStats.loc[
        (np.mod(self.playerStats["line"], 1) == 0)
        & (self.playerStats["avg"] > self.playerStats["line"]),
        "line",
    ] += 0.5
    self.playerStats.loc[
        (np.mod(self.playerStats["line"], 1) == 0)
        & (self.playerStats["avg"] < self.playerStats["line"]),
        "line",
    ] -= 0.5

    self.playerStats = self.playerStats.to_dict(orient="index")

# Was: StatsMLB.dvpoa
def dvpoa(self, team, market, date=datetime.today().date()):
    """Calculates the Defense Versus Position over League-Average (DVPOA) for a given team and market.

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

    if isinstance(date, datetime):
        date = date.date()

    one_year_ago = date - timedelta(days=300)
    gamelog = self.gamelog[
        self.gamelog["gameDate"].apply(
            lambda x: one_year_ago <= datetime.strptime(x, "%Y/%m/%d").date() <= date
        )
    ]

    # Calculate DVP (Defense Versus Position) and league average for the specified team and market
    if any([string in market for string in ["allowed", "pitch"]]):
        position_games = gamelog.loc[gamelog["starting pitcher"]]
        team_games = position_games.loc[position_games["opponent"] == team]
    else:
        position_games = gamelog.loc[gamelog["starting batter"]]
        team_games = position_games.loc[position_games["opponent pitcher"] == team]

    if len(team_games) == 0:
        return 0
    else:
        dvp = team_games[market].mean()
        leagueavg = position_games[market].mean()
        dvpoa = (dvp - leagueavg) / leagueavg
        dvpoa = np.nan_to_num(dvpoa, nan=0.0, posinf=0.0, neginf=0.0)
        self.dvp_index[market][team] = dvpoa
        return dvpoa

# Was: StatsMLB.obs_get_stats
def obs_get_stats(self, offer, date=datetime.today()):
    """Calculates the relevant statistics for a given offer and date.

    Args:
        offer (dict): The offer containing player, team, market, line, and opponent information.
        date (str or datetime.datetime, optional): The date for which to calculate the statistics. Defaults to the current date.

    Returns:
        pandas.DataFrame: The calculated statistics as a DataFrame.
    """
    if isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d")

    player = offer["Player"]
    team = offer["Team"].replace("AZ", "ARI")
    market = offer["Market"]
    cv = stat_cv.get("MLB", {}).get(market, 1)
    dist = stat_dist.get("MLB", {}).get(market, "Gamma")
    # if self.defenseProfile.empty:
    #     logger.exception(f"{market} not profiled")
    #     return 0
    line = offer["Line"]
    opponent = offer["Opponent"].replace("AZ", "ARI").split(" (")[0]
    home = offer.get("Home")
    if home is None:
        home = self.upcoming_games.get(team, {}).get("Home", 0)

    if player not in self.playerProfile.index:
        self.playerProfile.loc[player] = np.zeros_like(self.playerProfile.columns)

    if team not in self.teamProfile.index:
        self.teamProfile.loc[team] = np.zeros_like(self.teamProfile.columns)

    if opponent not in self.defenseProfile.index:
        self.defenseProfile.loc[opponent] = np.zeros_like(self.defenseProfile.columns)

    Date = datetime.strptime(date, "%Y-%m-%d")

    if Date.date() < datetime.today().date():
        pitcher = offer.get("Pitcher", "")
    else:
        pitcher = self.pitchers.get(opponent, "")

    if any([string in market for string in ["allowed", "pitch"]]):
        player_games = self.short_gamelog.loc[
            (self.short_gamelog["playerName"] == player)
            & self.short_gamelog["starting pitcher"]
        ]

        headtohead = player_games.loc[player_games["opponent"] == opponent]

        pid = self.gamelog.loc[self.gamelog["playerName"] == player, "playerId"]
    else:
        player_games = self.short_gamelog.loc[
            (self.short_gamelog["playerName"] == player) & self.short_gamelog["starting batter"]
        ]

        headtohead = player_games.loc[player_games["opponent pitcher"] == pitcher]

        pid = self.gamelog.loc[
            self.gamelog["opponent pitcher"] == pitcher, "opponent pitcher id"
        ]

    if player_games.empty:
        return 0

    pid = 0 if pid.empty else pid.iat[0]

    affine_pitchers = self.comps["pitchers"].get(pid, [pid])

    one_year_ago = len(player_games)
    game_res = (player_games[market]).to_list()
    h2h_res = (headtohead[market]).to_list()

    if line == 0:
        line = np.median(game_res[-one_year_ago:]) if game_res else 0
        line = 0.5 if line < 1 else line

    try:
        if pitcher not in self.pitcherProfile.index:
            self.pitcherProfile.loc[pitcher] = np.zeros_like(self.pitcherProfile.columns)

        ev = archive.get_ev("MLB", market, date, player)
        moneyline = archive.get_moneyline("MLB", date, team)
        total = archive.get_total("MLB", date, team)

    except:
        logger.exception(f"{player}, {market}")
        return 0

    if np.isnan(ev):
        if market in combo_props:
            ev = 0
            for submarket in combo_props.get(market, []):
                sub_cv = stat_cv["MLB"].get(submarket, 1)
                sub_dist = stat_dist.get("MLB", {}).get(submarket, "Gamma")
                v = archive.get_ev("MLB", submarket, date, player)
                subline = archive.get_line("MLB", submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(
                        subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                    )
                if np.isnan(v) or v == 0:
                    ev = 0
                    break
                else:
                    ev += v

        elif "fantasy" in market:
            ev = 0
            book_odds = False
            if "pitcher" in market:
                if "underdog" in market:
                    fantasy_props = [
                        ("pitcher win", 5),
                        ("pitcher strikeouts", 3),
                        ("runs allowed", -3),
                        ("pitching outs", 1),
                        ("quality start", 5),
                    ]
                else:
                    fantasy_props = [
                        ("pitcher win", 6),
                        ("pitcher strikeouts", 3),
                        ("runs allowed", -3),
                        ("pitching outs", 1),
                        ("quality start", 4),
                    ]
            elif "underdog" in market:
                fantasy_props = [
                    ("singles", 3),
                    ("doubles", 6),
                    ("triples", 8),
                    ("home runs", 10),
                    ("walks", 3),
                    ("rbi", 2),
                    ("runs", 2),
                    ("stolen bases", 4),
                ]
            else:
                fantasy_props = [
                    ("singles", 3),
                    ("doubles", 5),
                    ("triples", 8),
                    ("home runs", 10),
                    ("walks", 2),
                    ("rbi", 2),
                    ("runs", 2),
                    ("stolen bases", 5),
                ]

            v_outs = 0
            v_runs = 0
            for submarket, weight in fantasy_props:
                sub_cv = stat_cv["MLB"].get(submarket, 1)
                sub_dist = stat_dist.get("MLB", {}).get(submarket, "Gamma")
                v = archive.get_ev("MLB", submarket, date, player)
                subline = archive.get_line("MLB", submarket, date, player)
                if submarket == "pitcher win":
                    p = 1 - get_odds(subline, v, sub_dist, cv=sub_cv)
                    ev += p * weight
                elif submarket == "quality start":
                    if v_outs > 0:
                        std = stat_cv["MLB"].get(submarket, 1) * v_outs
                        p = norm.sf(18, v_outs, std) + norm.pdf(18, v_outs, std)
                        p *= poisson.cdf(3, v_runs) if v_runs > 0 else 0.5
                        ev += p * weight
                elif submarket in ["singles", "doubles", "triples", "home runs"] and np.isnan(
                    v
                ):
                    _hits_cv = stat_cv.get("MLB", {}).get("hits", 1)
                    _hits_dist = stat_dist.get("MLB", {}).get("hits", "Gamma")
                    v = archive.get_ev("MLB", "hits", date, player)
                    subline = archive.get_line("MLB", "hits", date, player)
                    v = get_ev(
                        subline, get_odds(subline, v, _hits_dist, cv=_hits_cv), cv=cv, dist=dist
                    )
                    v *= (
                        (
                            player_games.iloc[-one_year_ago:][submarket].sum()
                            / player_games.iloc[-one_year_ago:]["hits"].sum()
                        )
                        if player_games.iloc[-one_year_ago:]["hits"].sum()
                        else 0
                    )
                    ev += v * weight
                else:
                    if sub_dist != dist and not np.isnan(v):
                        v = get_ev(
                            subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                        )

                    if np.isnan(v) or v == 0:
                        if subline == 0 and not player_games.empty:
                            subline = (
                                np.floor(player_games.iloc[-10:][submarket].median()) + 0.5
                            )

                        if subline != 0:
                            under = (
                                player_games.iloc[-one_year_ago:][submarket] < subline
                            ).mean()
                            ev += get_ev(subline, under, sub_cv, dist=sub_dist) * weight
                    else:
                        book_odds = True
                        ev += v * weight

                if submarket == "runs allowed":
                    v_runs = v if not np.isnan(v) and v > 0 else v_runs
                if submarket == "pitching outs":
                    v_outs = v if not np.isnan(v) and v > 0 else v_outs

            if not book_odds:
                ev = 0

        # elif market == "1st inning runs allowed":
        #     ev = archive.get_team_market("MLB", '1st 1 innings', date, opponent)

    odds = 0 if np.isnan(ev) or ev <= 0 else 1 - get_odds(line, ev, dist, cv=cv)

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
        "MeanYr_nonzero": (
            np.mean([x for x in game_res[-one_year_ago:] if x > 0])
            if any(x > 0 for x in game_res[-one_year_ago:])
            else max(np.mean(game_res[-one_year_ago:]) if game_res else 0.5, 0.5)
        ),
        "MeanH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
        "STD10": np.std(game_res[-10:]) if game_res else 0,
        "STDYr": np.std(game_res[-one_year_ago:]) if game_res else 0,
        "Trend3": np.polyfit(np.arange(len(game_res[-3:])), game_res[-3:], 1)[0]
        if len(game_res) > 1
        else 0,
        "Trend5": np.polyfit(np.arange(len(game_res[-5:])), game_res[-5:], 1)[0]
        if len(game_res) > 1
        else 0,
        "TrendH2H": np.polyfit(np.arange(len(h2h_res[-3:])), h2h_res[-3:], 1)[0]
        if len(h2h_res) > 1
        else 0,
        "GamesPlayed": one_year_ago,
        "DaysIntoSeason": (Date.date() - self.season_start).days,
        "DaysOff": (
            Date.date() - pd.to_datetime(player_games.iloc[-1]["gameDate"]).date()
        ).days,
        "Moneyline": moneyline,
        "Total": total,
        "Home": home,
    }

    if data["Line"] <= 0:
        data["Line"] = data["AvgYr"] if data["AvgYr"] > 1 else 0.5

    if Date.date() < datetime.today().date():
        game = self.gamelog.loc[
            (self.gamelog["playerName"] == player)
            & (pd.to_datetime(self.gamelog.gameDate) == date)
        ]
        position = game.iloc[0]["battingOrder"]
        order = self.gamelog.loc[
            (self.gamelog.gameId == game.iloc[0]["gameId"])
            & (self.gamelog.team == game.iloc[0]["team"])
            & self.gamelog["starting batter"],
            "playerName",
        ].to_list()

    else:
        order = self.upcoming_games.get(team, {}).get("Batting Order", [])
        if player in order:
            position = order.index(player) + 1
        elif player_games.empty:
            position = 10
        else:
            position = int(player_games["battingOrder"].iloc[-10:].median())

    if len(game_res) < 5:
        i = 5 - len(game_res)
        game_res = [0] * i + game_res
    if len(h2h_res) < 5:
        i = 5 - len(h2h_res)
        h2h_res = [0] * i + h2h_res

    # Update the data dictionary with additional values
    data.update({"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
    data.update({"Game " + str(i + 1): game_res[-5 + i] for i in range(5)})

    player_data = self.playerProfile.loc[player]
    data.update(
        {
            f"Player {col}": player_data[f"{col}"]
            for col in ["avg", "home", "away", "z", "moneyline gain", "totals gain"]
        }
    )

    if any([string in market for string in ["allowed", "pitch"]]):
        data.update(
            {f"Player {col}": player_data[f"{col}"] for col in self.stat_types["pitching"]}
        )
        data.update(
            {
                f"Player {col} short": player_data[f"{col} short"]
                for col in self.stat_types["pitching"]
            }
        )
        data.update(
            {
                f"Player {col} growth": player_data[f"{col} growth"]
                for col in self.stat_types["pitching"]
            }
        )

        defense_data = self.defenseProfile.loc[team]

        for batter in order:
            if batter not in self.playerProfile.index:
                self.playerProfile.loc[batter] = self.defenseProfile.loc[
                    opponent, self.stat_types["batting"]
                ]

        if len(order) > 0:
            defense_data[self.stat_types["batting"]] = self.playerProfile.loc[
                order, self.stat_types["batting"]
            ].mean()

        team_data = self.teamProfile.loc[team, self.stat_types["fielding"]]

        affine = self.gamelog.loc[
            (self.gamelog["opponent"] == opponent)
            & (pd.to_datetime(self.gamelog.gameDate) < date)
            & self.gamelog["starting pitcher"]
            & (self.gamelog["playerId"].isin(affine_pitchers))
        ]
        aff_data = affine[self.stat_types["pitching"]].mean()

        data.update({"H2H " + col: aff_data[col] for col in aff_data.index})

        data.update({"Team " + col: team_data[col] for col in self.stat_types["fielding"]})

        data.update(
            {
                "Defense " + col: defense_data[col]
                for col in ["avg", "home", "away", "z", "moneyline gain", "totals gain"]
            }
        )
        data.update({"Defense " + col: defense_data[col] for col in self.stat_types["batting"]})

    else:
        data.update(
            {f"Player {col}": player_data[f"{col}"] for col in self.stat_types["batting"]}
        )
        data.update(
            {
                f"Player {col} short": player_data[f"{col} short"]
                for col in self.stat_types["batting"]
            }
        )
        data.update(
            {
                f"Player {col} growth": player_data[f"{col} growth"]
                for col in self.stat_types["batting"]
            }
        )

        defense_data = self.pitcherProfile.loc[pitcher]
        defense_data.loc["DER"] = self.defenseProfile.loc[opponent, "DER"]

        for batter in order:
            if batter not in self.playerProfile.index:
                self.playerProfile.loc[batter] = self.teamProfile.loc[
                    team, self.stat_types["batting"]
                ]

        if len(order) > 0:
            team_data = self.playerProfile.loc[order, self.stat_types["batting"]].mean()
        else:
            team_data = self.teamProfile.loc[team, self.stat_types["batting"]]

        data.update({"Position": position})

        affine = player_games.loc[player_games["opponent pitcher id"].isin(affine_pitchers)]
        aff_data = affine[self.stat_types["batting"]].mean()

        data.update({"H2H " + col: aff_data[col] for col in aff_data.index})

        data.update(
            {
                "Team " + col: team_data[col]
                for col in team_data.index
                if col not in self.stat_types["fielding"]
            }
        )

        data.update(
            {
                "Defense " + col: defense_data[col]
                for col in defense_data.index
                if col not in self.stat_types["batting"]
            }
        )

    park = team if home else opponent
    park_factors = self.park_factors[park]
    data.update({"PF " + col: v for col, v in park_factors.items()})

    data["DVPOA"] = data.pop("Defense avg")

    return data

# Was: StatsMLB.obs_get_training_matrix
def obs_get_training_matrix(self, market, cutoff_date=None):
    """Retrieves the training data matrix and target labels for the specified market.

    Args:
        market (str): The market type to retrieve training data for.

    Returns:
        M (pd.DataFrame): The training data matrix.
    """
    # Initialize an empty list for the target labels
    matrix = []

    if cutoff_date is None:
        cutoff_date = datetime.today() - timedelta(days=850)

    # Iterate over the gamelog to collect training data
    for _i, game in tqdm(
        self.gamelog.iterrows(),
        unit="game",
        desc="Gathering Training Data",
        total=len(self.gamelog),
    ):
        # Skip games without starting pitcher or starting batter based on market type
        if (
            any([string in market for string in ["allowed", "pitch"]])
            and not game["starting pitcher"]
        ) or (
            not any([string in market for string in ["allowed", "pitch"]])
            and not game["starting batter"]
        ):
            continue

        if game[market] < 0:
            continue

        if (game["starting batter"] and game["plateAppearances"] <= 1) or (
            game["starting pitcher"] and game["pitching outs"] < 6
        ):
            continue

        # Retrieve data from the archive based on game date and player name
        gameDate = datetime.strptime(game["gameDate"], "%Y-%m-%d").date()

        if gameDate <= cutoff_date:
            continue

        self.profile_market(market, date=gameDate)
        name = game["playerName"]

        if name not in self.playerProfile.index:
            continue

        line = archive.get_line("MLB", market, game["gameDate"], name)

        # Construct an offer dictionary with player, team, market, opponent, and pitcher information
        offer = {
            "Player": name,
            "Team": game["team"],
            "Market": market,
            "Opponent": game["opponent"],
            "Pitcher": game["opponent pitcher"],
            "Home": int(game["home"]),
        }

        # Retrieve stats using get_stats method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_get_stats = self.get_stats(offer | {"Line": line}, game["gameDate"])
            if type(new_get_stats) is dict:
                # Determine the result
                new_get_stats.update(
                    {"Result": game[market], "Date": gameDate, "Archived": int(line != 0)}
                )

                # Concatenate retrieved stats into the training data matrix
                matrix.append(new_get_stats)

    # Create the target labels DataFrame
    M = pd.DataFrame(matrix).fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)

    return M

# Was: StatsNFL.bucket_stats
def bucket_stats(self, market, buckets=20, date=datetime.today()):
    """Bucket player stats based on a given market.

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
    gamelog = self.gamelog[(one_year_ago <= gameDates) & (gameDates < date)]

    playerGroups = (
        gamelog.groupby("player display name")
        .filter(lambda x: len(x[x[market] != 0]) > 4)
        .groupby("player display name")[market]
    )

    # Compute edges for each bucket
    w = int(100 / buckets)
    self.edges = playerGroups.mean().quantile(np.arange(0, 101, w) / 100).to_list()

    # Assign bucket and line values to each player
    self.playerStats["avg"] = playerGroups.mean()
    self.playerStats["bucket"] = np.ceil(
        playerGroups.mean().rank(pct=True, ascending=False) * 20
    ).astype(int)
    self.playerStats["line"] = playerGroups.median()
    self.playerStats.loc[self.playerStats["line"] == 0.0, "line"] = 0.5
    self.playerStats.loc[
        (np.mod(self.playerStats["line"], 1) == 0)
        & (self.playerStats["avg"] > self.playerStats["line"]),
        "line",
    ] += 0.5
    self.playerStats.loc[
        (np.mod(self.playerStats["line"], 1) == 0)
        & (self.playerStats["avg"] < self.playerStats["line"]),
        "line",
    ] -= 0.5

    self.playerStats = self.playerStats.to_dict(orient="index")

# Was: StatsNFL.obs_profile_market
def obs_profile_market(self, market, date=datetime.today().date()):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, datetime):
        date = date.date()
    if market == self.profiled_market and date == self.profile_latest_date:
        return

    self.base_profile(date)
    self.profiled_market = market

    one_year_ago = date - timedelta(days=300)
    gameDates = pd.to_datetime(self.gamelog["gameday"]).dt.date
    gamelog = self.gamelog[(one_year_ago <= gameDates) & (gameDates < date)].copy().dropna()
    gameDates = pd.to_datetime(self.teamlog["gameday"]).dt.date
    teamlog = self.teamlog[(one_year_ago <= gameDates) & (gameDates < date)].copy()
    teamlog.drop(columns=["gameday"], inplace=True)

    # Retrieve moneyline and totals data from archive
    gamelog.loc[:, "moneyline"] = gamelog.apply(
        lambda x: archive.get_moneyline("NFL", x["gameday"][:10], x["team"]), axis=1
    )
    gamelog.loc[:, "totals"] = gamelog.apply(
        lambda x: archive.get_total("NFL", x["gameday"][:10], x["team"]), axis=1
    )

    teamstats = teamlog.groupby("team").apply(
        lambda x: np.mean(
            x.tail(5)[list(set(self.stat_types["offense"]) | set(self.stat_types["defense"]))],
            0,
        )
    )

    playerGroups = (
        gamelog.groupby("player display name")
        .filter(lambda x: (x[market].clip(0, 1).mean() > 0.1) & (x[market].count() > 1))
        .groupby("player display name")
    )

    defenseGroups = gamelog.groupby(["opponent", "game id"])
    defenseGames = pd.DataFrame()
    defenseGames[market] = defenseGroups[market].sum()
    defenseGames["home"] = defenseGroups["home"].mean().astype(int)
    defenseGames["moneyline"] = defenseGroups["moneyline"].mean()
    defenseGames["totals"] = defenseGroups["totals"].mean()
    defenseGroups = defenseGames.groupby("opponent")

    leagueavg = playerGroups[market].mean().mean()
    leaguestd = playerGroups[market].mean().std()
    if np.isnan(leagueavg) or np.isnan(leaguestd):
        return

    self.playerProfile["avg"] = playerGroups[market].mean().div(leagueavg) - 1
    self.playerProfile["z"] = (playerGroups[market].mean() - leagueavg).div(leaguestd)
    self.playerProfile["home"] = (
        playerGroups.apply(lambda x: x.loc[x["home"], market].mean() / x[market].mean()) - 1
    )
    self.playerProfile["away"] = (
        playerGroups.apply(
            lambda x: x.loc[~x["home"].astype(bool), market].mean() / x[market].mean()
        )
        - 1
    )

    leagueavg = defenseGroups[market].mean().mean()
    leaguestd = defenseGroups[market].mean().std()
    self.defenseProfile["avg"] = defenseGroups[market].mean().div(leagueavg) - 1
    self.defenseProfile["z"] = (defenseGroups[market].mean() - leagueavg).div(leaguestd)
    self.defenseProfile["home"] = (
        defenseGroups.apply(lambda x: x.loc[x["home"] == 1, market].mean() / x[market].mean())
        - 1
    )
    self.defenseProfile["away"] = (
        defenseGroups.apply(lambda x: x.loc[x["home"] == 0, market].mean() / x[market].mean())
        - 1
    )

    if any(
        [string in market for string in ["pass", "completion", "attempts", "interceptions"]]
    ):
        positions = ["QB"]
        stat_types = self.stat_types["passing"]
    elif any([string in market for string in ["qb", "sacks"]]):
        positions = ["QB"]
        stat_types = self.stat_types["passing"] + self.stat_types["rushing"]
    elif any([string in market for string in ["rush", "carries"]]):
        positions = ["QB", "RB"]
        stat_types = self.stat_types["rushing"]
    elif any([string in market for string in ["receiving", "targets", "reception"]]):
        positions = ["WR", "RB", "TE"]
        stat_types = self.stat_types["receiving"]
    elif market == "tds":
        positions = ["QB", "WR", "RB", "TE"]
        stat_types = self.stat_types["receiving"] + self.stat_types["rushing"]
    elif market == "yards":
        positions = ["WR", "RB", "TE"]
        stat_types = self.stat_types["receiving"] + self.stat_types["rushing"]
    else:
        positions = ["QB", "WR", "RB", "TE"]
        stat_types = (
            self.stat_types["passing"]
            + self.stat_types["rushing"]
            + self.stat_types["receiving"]
        )

    playerlogs = (
        gamelog.loc[gamelog["player display name"].isin(self.playerProfile.index)]
        .fillna(0)
        .infer_objects(copy=False)
        .groupby("player display name")[stat_types]
    )
    playerstats = playerlogs.mean(numeric_only=True)
    playershortstats = (
        playerlogs.apply(lambda x: np.mean(x.tail(3), 0))
        .fillna(0)
        .infer_objects(copy=False)
        .add_suffix(" short", 1)
    )
    playertrends = (
        playerlogs.apply(
            lambda x: pd.Series(
                np.polyfit(np.arange(0, len(x.tail(3))), x.tail(3), 1)[0], index=x.columns
            )
        )
        .fillna(0)
        .infer_objects(copy=False)
        .add_suffix(" growth", 1)
    )
    playerstats = playerstats.join(playershortstats)
    playerstats = playerstats.join(playertrends)
    for position in positions:
        positionLogs = gamelog.loc[gamelog["position group"] == position]
        positionGroups = positionLogs.groupby("player display name")
        positionAvg = positionGroups[market].mean().mean()
        positionStd = positionGroups[market].mean().std()
        idx = list(
            set(positionGroups.groups.keys()).intersection(set(self.playerProfile.index))
        )
        self.playerProfile.loc[idx, "position avg"] = (
            positionGroups[market].mean().div(positionAvg) - 1
        )
        self.playerProfile.loc[idx, "position z"] = (
            positionGroups[market].mean() - positionAvg
        ).div(positionStd)
        positionGroups = positionLogs.groupby(["opponent", "game id"])
        defenseGames = pd.DataFrame()
        defenseGames[market] = positionGroups[market].sum()
        defenseGames["home"] = positionGroups["home"].mean().astype(int)
        defenseGames["moneyline"] = positionGroups["moneyline"].mean()
        defenseGames["totals"] = positionGroups["totals"].mean()
        positionGroups = defenseGames.groupby("opponent")
        leagueavg = positionGroups[market].mean().mean()
        if leagueavg == 0:
            self.defenseProfile[position] = 0
        else:
            self.defenseProfile[position] = positionGroups[market].mean().div(leagueavg) - 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        self.playerProfile["moneyline gain"] = playerGroups.apply(
            lambda x: np.polyfit(
                x.moneyline.fillna(0.5).values.astype(float) / 0.5
                - x.moneyline.fillna(0.5).mean(),
                x[market].values.astype(float) / x[market].mean() - 1,
                1,
            )[0]
        )

        self.playerProfile["totals gain"] = playerGroups.apply(
            lambda x: np.polyfit(
                x.totals.fillna(22.5).values.astype(float) / 22.5
                - x.totals.fillna(22.5).mean(),
                x[market].values.astype(float) / x[market].mean() - 1,
                1,
            )[0]
        )

        self.defenseProfile["moneyline gain"] = defenseGroups.apply(
            lambda x: np.polyfit(
                x.moneyline.fillna(0.5).values.astype(float) / 0.5
                - x.moneyline.fillna(0.5).mean(),
                x[market].values.astype(float) / x[market].mean() - 1,
                1,
            )[0]
        )

        self.defenseProfile["totals gain"] = defenseGroups.apply(
            lambda x: np.polyfit(
                x.totals.fillna(22.5).values.astype(float) / 22.5
                - x.totals.fillna(22.5).mean(),
                x[market].values.astype(float) / x[market].mean() - 1,
                1,
            )[0]
        )

    i = self.defenseProfile.index
    self.defenseProfile = self.defenseProfile.merge(
        teamstats[self.stat_types["defense"]], left_on="opponent", right_on="team"
    )
    self.defenseProfile.index = i

    self.teamProfile = teamstats[self.stat_types["offense"]]

    self.playerProfile = self.playerProfile.merge(playerstats, on="player display name")

# Was: StatsNFL.dvpoa
def dvpoa(self, team, position, market, date=datetime.today().date()):
    """Calculate the Defense Versus Position Above Average (DVPOA) for a specific team, position, and market.

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

    position_games = self.gamelog.loc[
        (self.gamelog["position group"] == position)
        & (pd.to_datetime(self.gamelog["gameday"]) < date)
    ]
    team_games = position_games.loc[position_games["opponent"] == team]

    if len(team_games) == 0:
        return 0
    else:
        dvp = team_games[market].mean()
        leagueavg = position_games[market].mean()
        dvpoa = (dvp - leagueavg) / leagueavg
        dvpoa = np.nan_to_num(dvpoa, nan=0.0, posinf=0.0, neginf=0.0)
        self.dvp_index[market][team][position] = dvpoa
        return dvpoa

# Was: StatsNFL.obs_get_stats
def obs_get_stats(self, offer, date=datetime.today()):
    """Generate a pandas DataFrame with a summary of relevant stats.

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
    cv = stat_cv.get("NFL", {}).get(market, 1)
    dist = stat_dist.get("NFL", {}).get(market, "Gamma")
    # if self.defenseProfile.empty:
    #     logger.exception(f"{market} not profiled")
    #     return 0
    home = offer.get("Home")
    if home is None:
        home = self.upcoming_games.get(team, {}).get("Home", 0)

    if player not in self.playerProfile.index:
        self.playerProfile.loc[player] = np.zeros_like(self.playerProfile.columns)

    if team not in self.teamProfile.index:
        self.teamProfile.loc[team] = np.zeros_like(self.teamProfile.columns)

    if opponent not in self.defenseProfile.index:
        self.defenseProfile.loc[opponent] = np.zeros_like(self.defenseProfile.columns)

    Date = datetime.strptime(date, "%Y-%m-%d")

    player_games = self.short_gamelog.loc[(self.short_gamelog["player display name"] == player)]
    position = self.players.get(player, "")
    one_year_ago = len(player_games)
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

    if line == 0:
        line = np.median(game_res[-one_year_ago:]) if game_res else 0
        line = 0.5 if line < 1 else line

    try:
        ev = archive.get_ev("NFL", market, date, player)
        moneyline = archive.get_moneyline("NFL", date, team)
        total = archive.get_total("NFL", date, team)

    except:
        logger.exception(f"{player}, {market}")
        return 0

    if np.isnan(ev):
        if market in combo_props:
            ev = 0
            for submarket in combo_props.get(market, []):
                sub_cv = stat_cv["NFL"].get(submarket, 1)
                sub_dist = stat_dist.get("NFL", {}).get(submarket, "Gamma")
                v = archive.get_ev("NFL", submarket, date, player)
                subline = archive.get_line("NFL", submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(
                        subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                    )
                if np.isnan(v) or v == 0:
                    ev = 0
                    break
                else:
                    ev += v

        elif market in ["rushing tds", "receiving tds"]:
            ev = (
                (
                    archive.get_ev("NFL", "tds", date, player)
                    * player_games.iloc[-one_year_ago:][market].sum()
                    / player_games.iloc[-one_year_ago:]["tds"].sum()
                )
                if player_games.iloc[-one_year_ago:]["tds"].sum()
                else 0
            )

        elif "fantasy" in market:
            ev = 0
            book_odds = False
            if "prizepicks" in market:
                fantasy_props = [
                    ("passing yards", 1 / 25),
                    ("passing tds", 4),
                    ("interceptions", -1),
                    ("rushing yards", 0.1),
                    ("receiving yards", 0.1),
                    ("tds", 6),
                    ("receptions", 1),
                ]
            else:
                fantasy_props = [
                    ("passing yards", 1 / 25),
                    ("passing tds", 4),
                    ("interceptions", -1),
                    ("rushing yards", 0.1),
                    ("receiving yards", 0.1),
                    ("tds", 6),
                    ("receptions", 0.5),
                ]
            for submarket, weight in fantasy_props:
                sub_cv = stat_cv["NFL"].get(submarket, 1)
                sub_dist = stat_dist.get("NFL", {}).get(submarket, "Gamma")
                v = archive.get_ev("NFL", submarket, date, player)
                subline = archive.get_line("NFL", submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(
                        subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                    )
                if np.isnan(v) or v == 0:
                    if subline == 0 and not player_games.empty:
                        subline = np.floor(player_games.iloc[-10:][submarket].median()) + 0.5

                    if subline != 0:
                        under = (player_games.iloc[-one_year_ago:][submarket] < subline).mean()
                        ev += get_ev(subline, under, sub_cv, dist=sub_dist)
                else:
                    book_odds = True
                    ev += v * weight

            if not book_odds:
                ev = 0

    odds = 0 if np.isnan(ev) or ev <= 0 else 1 - get_odds(line, ev, dist, cv=cv)

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
        "IQRH2H": iqr(h2h_res[-5:]) if h2h_res else 0,
        "Mean5": np.mean(game_res[-5:]) if game_res else 0,
        "Mean10": np.mean(game_res[-10:]) if game_res else 0,
        "MeanYr": np.mean(game_res[-one_year_ago:]) if game_res else 0,
        "MeanYr_nonzero": (
            np.mean([x for x in game_res[-one_year_ago:] if x > 0])
            if any(x > 0 for x in game_res[-one_year_ago:])
            else max(np.mean(game_res[-one_year_ago:]) if game_res else 0.5, 0.5)
        ),
        "MeanH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
        "STD10": np.std(game_res[-10:]) if game_res else 0,
        "STDYr": np.std(game_res[-one_year_ago:]) if game_res else 0,
        "Trend3": np.polyfit(np.arange(len(game_res[-3:])), game_res[-3:], 1)[0]
        if len(game_res) > 1
        else 0,
        "Trend5": np.polyfit(np.arange(len(game_res[-5:])), game_res[-5:], 1)[0]
        if len(game_res) > 1
        else 0,
        "TrendH2H": np.polyfit(np.arange(len(h2h_res[-3:])), h2h_res[-3:], 1)[0]
        if len(h2h_res) > 1
        else 0,
        "GamesPlayed": one_year_ago,
        "DaysIntoSeason": (Date.date() - self.season_start).days,
        "DaysOff": (Date.date() - pd.to_datetime(player_games.iloc[-1]["gameday"]).date()).days,
        "Moneyline": moneyline,
        "Total": total,
        "Home": home,
        "Position": self.positions.index(position),
    }

    if len(game_res) < 5:
        i = 5 - len(game_res)
        game_res = [0] * i + game_res
    if len(h2h_res) < 5:
        i = 5 - len(h2h_res)
        h2h_res = [0] * i + h2h_res

    # Update the data dictionary with additional values
    if any(
        [string in market for string in ["pass", "completion", "attempts", "interceptions"]]
    ):
        stat_types = self.stat_types["passing"]
    elif any([string in market for string in ["qb", "sacks"]]):
        stat_types = self.stat_types["passing"] + self.stat_types["rushing"]
    elif any([string in market for string in ["rush", "carries"]]):
        stat_types = self.stat_types["rushing"]
    elif any([string in market for string in ["receiving", "targets", "reception"]]):
        stat_types = self.stat_types["receiving"]
    elif market in ("tds", "yards"):
        stat_types = self.stat_types["receiving"] + self.stat_types["rushing"]
    else:
        stat_types = (
            self.stat_types["passing"]
            + self.stat_types["rushing"]
            + self.stat_types["receiving"]
        )

    data.update({"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
    data.update({"Game " + str(i + 1): game_res[-5 + i] for i in range(5)})

    player_data = self.playerProfile.loc[player]
    data.update(
        {
            f"Player {col}": player_data[f"{col}"]
            for col in [
                "avg",
                "home",
                "away",
                "z",
                "moneyline gain",
                "totals gain",
                "position avg",
                "position z",
            ]
        }
    )
    data.update({f"Player {col}": player_data[f"{col}"] for col in stat_types})
    data.update({f"Player {col} short": player_data[f"{col} short"] for col in stat_types})
    data.update({f"Player {col} growth": player_data[f"{col} growth"] for col in stat_types})

    team_data = self.teamProfile.loc[team]
    data.update({"Team " + col: team_data[col] for col in self.stat_types["offense"]})

    defense_data = self.defenseProfile.loc[opponent]
    data.update(
        {
            "Defense " + col: defense_data[col]
            for col in defense_data.index
            if col not in self.positions + self.stat_types["offense"]
        }
    )

    return data

# Was: StatsNFL.obs_get_training_matrix
def obs_get_training_matrix(self, market, cutoff_date=None):
    """Retrieves training data in the form of a feature matrix (X) and a target vector (y) for a specified market.

    Args:
        market (str): The market for which to retrieve training data.

    Returns:
        tuple: A tuple containing the feature matrix (X) and the target vector (y).
    """
    # Initialize an empty list for the target labels
    matrix = []

    if cutoff_date is None:
        cutoff_date = datetime.today() - timedelta(days=1200)

    for _i, game in tqdm(
        self.gamelog.iterrows(),
        unit="game",
        desc="Gathering Training Data",
        total=len(self.gamelog),
    ):
        gameDate = datetime.strptime(game["gameday"], "%Y-%m-%d").date()

        if gameDate <= cutoff_date:
            continue

        self.profile_market(market, date=gameDate)
        name = game["player display name"]

        if name not in self.playerProfile.index:
            continue

        if game[market] < 0:
            continue

        if ((game["position group"] == "QB") and (game["snap pct"] < 0.8)) or (
            game["snap pct"] < 0.3
        ):
            continue

        line = archive.get_line("NFL", market, game["gameday"], name)

        offer = {
            "Player": name,
            "Team": game["team"],
            "Market": market,
            "Opponent": game["opponent"],
            "Home": int(game["home"]),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_get_stats = self.get_stats(offer | {"Line": line}, game["gameday"])
            if type(new_get_stats) is dict:
                new_get_stats.update(
                    {"Result": game[market], "Date": gameDate, "Archived": int(line != 0)}
                )

                matrix.append(new_get_stats)

    M = pd.DataFrame(matrix).fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)

    return M

# Was: StatsNFL.get_fantasy
def get_fantasy(self):
    """Retrieves fantasy points stats.

    Args:

    Returns:
        tuple: A tuple containing the feature matrix (X) and the target vector (y).
    """
    # Initialize an empty list for the target labels
    matrix = []
    i = []
    offers = []

    self.profile_market("fantasy points underdog")
    roster = nfl.import_weekly_rosters([self.season_start.year])
    roster = roster.loc[
        (roster.status == "ACT")
        & roster.position.isin(["QB", "RB", "WR", "TE"])
        & (roster.week == roster.week.max())
    ]
    roster.loc[roster["team"] == "LA", "team"] = "LAR"
    players = (
        pd.Series(
            zip(
                roster[self.log_strings["player"]].map(remove_accents),
                roster["team"],
                strict=False,
            )
        )
        .drop_duplicates()
        .to_list()
    )

    for player, team in tqdm(players, unit="player"):
        gameDate = self.upcoming_games.get(team, {}).get(
            "gameday", datetime.today().strftime("%Y-%m-%d")
        )
        gameTime = self.upcoming_games.get(team, {}).get(
            "gametime", datetime.today().strftime("%Y-%m-%d")
        )
        opponent = self.upcoming_games.get(team, {}).get("Opponent")
        home = self.upcoming_games.get(team, {}).get("Home")
        data = (
            archive["NFL"]["fantasy points underdog"]
            .get(gameDate, {})
            .get(player, {"Lines": [0]})
        )

        lines = data["Lines"]
        line = lines[-1] if len(lines) > 0 else 0

        offer = {
            "Player": player,
            "Team": team,
            "Market": "fantasy points underdog",
            "Opponent": opponent,
            "Home": home,
            "Game": gameTime,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_get_stats = self.get_stats(offer | {"Line": line}, gameDate)
            if type(new_get_stats) is dict:
                matrix.append(new_get_stats)
                i.append(player)
                offer.pop("Market")
                offer.pop("Player")
                offer.pop("Home")
                offers.append(offer)

    M = pd.DataFrame(matrix, index=i).fillna(0.0).replace([np.inf, -np.inf], 0)

    N = pd.DataFrame(offers, index=i).fillna(0.0).replace([np.inf, -np.inf], 0)

    return M, N

# Was: StatsNHL.bucket_stats
def bucket_stats(self, market, date=datetime.today()):
    """Bucket the stats based on the specified market (e.g., 'goalsAgainst', 'saves').

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
        if datetime.strptime(game["gameDate"], "%Y-%m-%d").date() > date.date() or (
            datetime.strptime(game["gameDate"], "%Y-%m-%d").date()
            < (date - timedelta(days=300)).date()
        ):
            continue
        if (
            (market in ["goalsAgainst", "saves"] or "goalie fantasy" in market)
            and game["position"] != "G"
            or (market not in ["goalsAgainst", "saves"] and "goalie fantasy" not in market)
            and game["position"] == "G"
            or market not in game
        ):
            continue

        # Check if the player is already in the playerStats dictionary
        if game["playerName"] not in self.playerStats:
            self.playerStats[game["playerName"]] = {"games": []}

        # Append the market value to the player's games list
        self.playerStats[game["playerName"]]["games"].append(game[market])

    # Filter playerStats based on minimum games played and non-zero games
    self.playerStats = {
        k: v
        for k, v in self.playerStats.items()
        if len(v["games"]) > 10 and not all([g == 0 for g in v["games"]])
    }

    # Calculate averages and store in playerStats dictionary
    averages = []
    for player, games in self.playerStats.items():
        self.playerStats[player]["avg"] = np.mean(games["games"]) if games["games"] else 0
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
                np.mean([v for v in averages if v <= self.edges[i] and v >= self.edges[i - 1]])
                - 0.5
            )
            + 0.5
        )

    # Assign bucket and line values to each player in playerStats
    for player, games in self.playerStats.items():
        for i in range(0, 10):
            if games["avg"] >= self.edges[i]:
                self.playerStats[player]["bucket"] = 10 - i
                line = np.median(games["games"])
                if np.mod(line, 1) == 0:
                    line += 0.5 if self.playerStats[player]["avg"] >= line else -0.5
                self.playerStats[player]["line"] = line

# Was: StatsNHL.obs_profile_market
def obs_profile_market(self, market, date=datetime.today().date()):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    if isinstance(date, datetime):
        date = date.date()

    if market == self.profiled_market and date == self.profile_latest_date:
        return

    self.profiled_market = market
    self.profile_latest_date = date

    team_stat_types = [
        "Corsi",
        "Fenwick",
        "Hits",
        "Takeaways",
        "PIM",
        "Corsi_Pct",
        "Fenwick_Pct",
        "Hits_Pct",
        "Takeaways_Pct",
        "PIM_Pct",
        "Block_Pct",
        "xGoals",
        "xGoalsAgainst",
        "goalsAgainst",
        "GOE",
        "SV",
        "SOE",
        "Freeze",
        "Rebound",
        "RG",
    ]

    # Initialize playerStats and edges
    self.playerProfile = pd.DataFrame(columns=["avg", "home", "away"])
    self.defenseProfile = pd.DataFrame(columns=["avg", "home", "away"])

    # Filter gamelog for games within the date range
    one_year_ago = date - timedelta(days=300)
    gameDates = pd.to_datetime(self.gamelog["gameDate"]).dt.date
    gamelog = self.gamelog[(one_year_ago <= gameDates) & (gameDates < date)]
    gameDates = pd.to_datetime(self.teamlog["gameDate"]).dt.date
    teamlog = self.teamlog[(one_year_ago <= gameDates) & (gameDates < date)]

    # Filter non-starting goalies or non-starting skaters depending on the market
    if any([string in market for string in ["Against", "saves", "goalie"]]):
        gamelog = gamelog[gamelog["position"] == "G"].copy()
    else:
        gamelog2 = gamelog[gamelog["position"] == "G"].copy()
        gamelog = gamelog[gamelog["position"] != "G"].copy()

    # Retrieve moneyline and totals data from archive
    gamelog.loc[:, "moneyline"] = gamelog.apply(
        lambda x: archive.get_moneyline("NHL", x["gameDate"][:10], x["team"]), axis=1
    )
    gamelog.loc[:, "totals"] = gamelog.apply(
        lambda x: archive.get_total("NHL", x["gameDate"][:10], x["team"]), axis=1
    )

    teamstats = teamlog.groupby("team").apply(lambda x: np.mean(x.tail(10)[team_stat_types], 0))

    # Filter players with at least 2 entries
    playerGroups = (
        gamelog.groupby("playerName")
        .filter(lambda x: (x[market].clip(0, 1).mean() > 0.1) & (x[market].count() > 1))
        .groupby("playerName")
    )

    defenseGroups = gamelog.groupby(["opponent", "gameDate"])
    defenseGames = pd.DataFrame()
    defenseGames[market] = defenseGroups[market].sum()
    defenseGames["home"] = defenseGroups["home"].mean().astype(int)
    defenseGames["moneyline"] = defenseGroups["moneyline"].mean()
    defenseGames["totals"] = defenseGroups["totals"].mean()
    defenseGroups = defenseGames.groupby("opponent")

    # Compute league average
    leagueavg = playerGroups[market].mean().mean()
    leaguestd = playerGroups[market].mean().std()
    if np.isnan(leagueavg) or np.isnan(leaguestd):
        return

    # Compute playerProfile DataFrame
    self.playerProfile["avg"] = playerGroups[market].mean().div(leagueavg) - 1
    self.playerProfile["z"] = (playerGroups[market].mean() - leagueavg).div(leaguestd)
    self.playerProfile["home"] = (
        playerGroups.apply(lambda x: x.loc[x["home"], market].mean() / x[market].mean()) - 1
    )
    self.playerProfile["away"] = (
        playerGroups.apply(lambda x: x.loc[~x["home"], market].mean() / x[market].mean()) - 1
    )

    leagueavg = defenseGroups[market].mean().mean()
    leaguestd = defenseGroups[market].mean().std()
    self.defenseProfile["avg"] = defenseGroups[market].mean().div(leagueavg) - 1
    self.defenseProfile["z"] = (defenseGroups[market].mean() - leagueavg).div(leaguestd)
    self.defenseProfile["home"] = (
        defenseGroups.apply(lambda x: x.loc[x["home"] == 0, market].mean() / x[market].mean())
        - 1
    )
    self.defenseProfile["away"] = (
        defenseGroups.apply(lambda x: x.loc[x["home"] == 1, market].mean() / x[market].mean())
        - 1
    )

    positions = ["C", "W", "D"]
    if market == "faceOffWins":
        positions.remove("D")
    if not any([string in market for string in ["Against", "saves", "goalie"]]):
        for position in positions:
            positionLogs = gamelog.loc[gamelog["position"] == position]
            positionGroups = positionLogs.groupby("playerName")
            positionAvg = positionGroups[market].mean().mean()
            positionStd = positionGroups[market].mean().std()
            idx = list(
                set(positionGroups.groups.keys()).intersection(set(self.playerProfile.index))
            )
            self.playerProfile.loc[idx, "position avg"] = (
                positionGroups[market].mean().div(positionAvg) - 1
            )
            self.playerProfile.loc[idx, "position z"] = (
                positionGroups[market].mean() - positionAvg
            ).div(positionStd)
            positionGroups = positionLogs.groupby("opponent")
            defenseGames = pd.DataFrame()
            defenseGames[market] = positionGroups[market].sum()
            defenseGames["home"] = positionGroups["home"].mean().astype(int)
            defenseGames["moneyline"] = positionGroups["moneyline"].mean()
            defenseGames["totals"] = positionGroups["totals"].mean()
            positionGroups = defenseGames.groupby("opponent")
            leagueavg = positionGroups[market].mean().mean()
            if leagueavg == 0:
                self.defenseProfile[position] = 0
            else:
                self.defenseProfile[position] = positionGroups[market].mean().div(leagueavg) - 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        self.playerProfile["moneyline gain"] = playerGroups.apply(
            lambda x: np.polyfit(
                x.moneyline.fillna(0.5).values.astype(float) / 0.5
                - x.moneyline.fillna(0.5).mean(),
                x[market].values / x[market].mean() - 1,
                1,
            )[0]
        )

        self.playerProfile["totals gain"] = playerGroups.apply(
            lambda x: np.polyfit(
                x.totals.fillna(3).values.astype(float) / 8.3 - x.totals.fillna(3).mean(),
                x[market].values / x[market].mean() - 1,
                1,
            )[0]
        )

        self.defenseProfile["moneyline gain"] = defenseGroups.apply(
            lambda x: np.polyfit(
                x.moneyline.fillna(0.5).values.astype(float) / 0.5
                - x.moneyline.fillna(0.5).mean(),
                x[market].values / x[market].mean() - 1,
                1,
            )[0]
        )

        self.defenseProfile["totals gain"] = defenseGroups.apply(
            lambda x: np.polyfit(
                x.totals.fillna(3).values.astype(float) / 8.3 - x.totals.fillna(3).mean(),
                x[market].values / x[market].mean() - 1,
                1,
            )[0]
        )

    if any([string in market for string in ["Against", "saves", "goalie"]]):
        playerlogs = (
            gamelog.loc[gamelog["playerName"].isin(self.playerProfile.index)]
            .fillna(0)
            .infer_objects(copy=False)
            .groupby("playerName")[self.goalie_stats]
        )
        playerstats = playerlogs.mean(numeric_only=True)
        playershortstats = (
            playerlogs.apply(lambda x: np.mean(x.tail(5), 0))
            .fillna(0)
            .infer_objects(copy=False)
            .add_suffix(" short", 1)
        )
        playertrends = (
            playerlogs.apply(
                lambda x: pd.Series(
                    np.polyfit(np.arange(0, len(x.tail(5))), x.tail(5), 1)[0], index=x.columns
                )
            )
            .fillna(0)
            .infer_objects(copy=False)
            .add_suffix(" growth", 1)
        )
        playerstats = playerstats.join(playershortstats)
        playerstats = playerstats.join(playertrends)

        self.playerProfile = self.playerProfile.merge(playerstats, on="playerName")
    else:
        playerlogs = (
            gamelog.loc[gamelog["playerName"].isin(self.playerProfile.index)]
            .fillna(0)
            .infer_objects(copy=False)
            .groupby("playerName")[self.skater_stats]
        )
        playerstats = playerlogs.mean(numeric_only=True)
        playershortstats = (
            playerlogs.apply(lambda x: np.mean(x.tail(5), 0))
            .fillna(0)
            .infer_objects(copy=False)
            .add_suffix(" short", 1)
        )
        playertrends = (
            playerlogs.apply(
                lambda x: pd.Series(
                    np.polyfit(np.arange(0, len(x.tail(5))), x.tail(5), 1)[0], index=x.columns
                )
            )
            .fillna(0)
            .infer_objects(copy=False)
            .add_suffix(" growth", 1)
        )
        playerstats = playerstats.join(playershortstats)
        playerstats = playerstats.join(playertrends)

        self.playerProfile = self.playerProfile.merge(playerstats, on="playerName")

        self.goalieProfile = (
            gamelog2.fillna(0)
            .infer_objects(copy=False)
            .groupby("playerName")[self.goalie_stats]
            .mean(numeric_only=True)
        )

    i = self.defenseProfile.index
    self.defenseProfile = self.defenseProfile.merge(
        teamstats, left_on="opponent", right_on="team"
    )
    self.defenseProfile.index = i

    self.teamProfile = teamstats

# Was: StatsNHL.dvpoa
def dvpoa(self, team, position, market, date=datetime.today().date()):
    """Calculate the Defense Versus Position Above Average (DVPOA) for a specific team, position, and market.

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
        if datetime.strptime(game["gameDate"], "%Y-%m-%d").date() > date.date() or (
            datetime.strptime(game["gameDate"], "%Y-%m-%d").date()
            < (date - timedelta(days=300)).date()
        ):
            continue
        if (
            (market in ["goalsAgainst", "saves"] or "goalie fantasy" in market)
            and game["position"] != "G"
            or (market not in ["goalsAgainst", "saves"] or "goalie fantasy" not in market)
            and game["position"] == "G"
        ):
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

# Was: StatsNHL.obs_get_stats
def obs_get_stats(self, offer, date=datetime.today()):
    """Calculate various statistics for a given offer.

    Args:
        offer (dict): The offer details containing 'Player', 'Team', 'Market', 'Line', and 'Opponent'.
        date (str or datetime, optional): The date for which to calculate the statistics. Defaults to today's date.

    Returns:
        pandas.DataFrame: A DataFrame containing the calculated statistics.
    """
    if isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d")

    stat_map = {"PTS": "points", "AST": "assists", "BLK": "blocked"}

    player = offer["Player"]
    team = offer["Team"]
    market = offer["Market"]
    market = stat_map.get(market, market)
    cv = stat_cv.get("NHL", {}).get(market, 1)
    dist = stat_dist.get("NHL", {}).get(market, "Gamma")
    # if self.defenseProfile.empty:
    #     logger.exception(f"{market} not profiled")
    #     return 0
    line = offer["Line"]
    opponent = offer["Opponent"]
    home = offer.get("Home")
    if home is None:
        home = self.upcoming_games.get(team, {}).get("Home", 3)

    if player not in self.playerProfile.index:
        self.playerProfile.loc[player] = np.zeros_like(self.playerProfile.columns)

    if team not in self.teamProfile.index:
        self.teamProfile.loc[team] = np.zeros_like(self.teamProfile.columns)

    if opponent not in self.defenseProfile.index:
        self.defenseProfile.loc[opponent] = np.zeros_like(self.defenseProfile.columns)

    Date = datetime.strptime(date, "%Y-%m-%d")

    if any([string in market for string in ["Against", "saves", "goalie"]]):
        player_games = self.short_gamelog.loc[
            (self.short_gamelog["playerName"] == player)
            & (self.short_gamelog["position"] == "G")
        ]

    else:
        player_games = self.short_gamelog.loc[
            (self.short_gamelog["playerName"] == player)
            & (self.short_gamelog["position"] != "G")
        ]

    if player_games.empty:
        return 0

    headtohead = player_games.loc[player_games["opponent"] == opponent]

    one_year_ago = len(player_games)

    game_res = (player_games[market]).to_list()
    h2h_res = (headtohead[market]).to_list()

    if line == 0:
        line = np.median(game_res[-one_year_ago:]) if game_res else 0
        line = 0.5 if line < 1 else line

    try:
        if not any([string in market for string in ["Against", "saves", "goalie"]]):
            if datetime.strptime(date, "%Y-%m-%d").date() < datetime.today().date():
                goalie = offer.get("Goalie", "")
            else:
                goalie = self.upcoming_games.get(opponent, {}).get("Goalie", "")

        ev = archive.get_ev("NHL", market, date, player)
        moneyline = archive.get_moneyline("NHL", date, team)
        total = archive.get_total("NHL", date, team)

    except:
        logger.exception(f"{player}, {market}")
        return 0

    if np.isnan(ev):
        if market in combo_props:
            ev = 0
            for submarket in combo_props.get(market, []):
                sub_cv = stat_cv["NHL"].get(submarket, 1)
                sub_dist = stat_dist.get("NHL", {}).get(submarket, "Gamma")
                v = archive.get_ev("NHL", submarket, date, player)
                subline = archive.get_line("NHL", submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(
                        subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                    )
                if np.isnan(v) or v == 0:
                    ev = 0
                    break

                else:
                    ev += v

        elif market == "goalsAgainst":
            ev = archive.get_total("NHL", date, opponent)

        elif "fantasy" in market:
            ev = 0
            book_odds = False
            if "prizepicks" in market:
                fantasy_props = [("goals", 8), ("assists", 5), ("shots", 1.5), ("blocked", 1.5)]
            elif ("underdog" in market) and ("skater" in market):
                fantasy_props = [
                    ("goals", 6),
                    ("assists", 4),
                    ("shots", 1),
                    ("blocked", 1),
                    ("hits", 0.5),
                    ("powerPlayPoints", 0.5),
                ]
            else:
                fantasy_props = [("saves", 0.6), ("goalsAgainst", -3), ("Moneyline", 6)]
            for submarket, weight in fantasy_props:
                sub_cv = stat_cv["NHL"].get(submarket, 1)
                sub_dist = stat_dist.get("NHL", {}).get(submarket, "Gamma")
                v = archive.get_ev("NHL", submarket, date, player)
                subline = archive.get_line("NHL", submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(
                        subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                    )
                if np.isnan(v) or v == 0:
                    if submarket == "Moneyline":
                        p = moneyline
                        ev += p * weight
                    elif submarket == "goalsAgainst":
                        v = archive.get_total("NHL", date, opponent)
                        subline = np.floor(v) + 0.5
                        v = get_ev(
                            subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                        )
                        ev += v * weight
                    else:
                        if subline == 0 and not player_games.empty:
                            subline = (
                                np.floor(player_games.iloc[-10:][submarket].median()) + 0.5
                            )

                        if subline != 0:
                            under = (
                                player_games.iloc[-one_year_ago:][submarket] < subline
                            ).mean()
                            ev += get_ev(subline, under, sub_cv, dist=sub_dist) * weight
                else:
                    book_odds = True
                    ev += v * weight

            if not book_odds:
                ev = 0

    odds = 0 if np.isnan(ev) or ev <= 0 else 1 - get_odds(line, ev, dist, cv=cv)

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
        "MeanYr_nonzero": (
            np.mean([x for x in game_res[-one_year_ago:] if x > 0])
            if any(x > 0 for x in game_res[-one_year_ago:])
            else max(np.mean(game_res[-one_year_ago:]) if game_res else 0.5, 0.5)
        ),
        "MeanH2H": np.mean(h2h_res[-5:]) if h2h_res else 0,
        "STD10": np.std(game_res[-10:]) if game_res else 0,
        "STDYr": np.std(game_res[-one_year_ago:]) if game_res else 0,
        "Trend3": np.polyfit(np.arange(len(game_res[-3:])), game_res[-3:], 1)[0]
        if len(game_res) > 1
        else 0,
        "Trend5": np.polyfit(np.arange(len(game_res[-5:])), game_res[-5:], 1)[0]
        if len(game_res) > 1
        else 0,
        "TrendH2H": np.polyfit(np.arange(len(h2h_res[-3:])), h2h_res[-3:], 1)[0]
        if len(h2h_res) > 1
        else 0,
        "GamesPlayed": one_year_ago,
        "DaysIntoSeason": (Date.date() - self.season_start).days,
        "DaysOff": (
            Date.date() - pd.to_datetime(player_games.iloc[-1]["gameDate"]).date()
        ).days,
        "Moneyline": moneyline,
        "Total": total,
        "Home": home,
    }

    if data["Line"] <= 0:
        data["Line"] = data["AvgYr"] if data["AvgYr"] > 1 else 0.5

    if not any([string in market for string in ["Against", "saves", "goalie"]]):
        if len(player_games) > 0:
            position = player_games.iloc[0]["position"]
        else:
            logger.warning(f"{player} not found")
            return 0

        data.update({"Position": self.positions.index(position)})

    if len(game_res) < 5:
        i = 5 - len(game_res)
        game_res = [0] * i + game_res
    if len(h2h_res) < 5:
        i = 5 - len(h2h_res)
        h2h_res = [0] * i + h2h_res

    # Update the data dictionary with additional values
    data.update({"Meeting " + str(i + 1): h2h_res[-5 + i] for i in range(5)})
    data.update({"Game " + str(i + 1): game_res[-5 + i] for i in range(5)})

    player_data = self.playerProfile.loc[player]
    data.update(
        {
            f"Player {col}": player_data[f"{col}"]
            for col in ["avg", "home", "away", "z", "moneyline gain", "totals gain"]
        }
    )

    if any([string in market for string in ["Against", "saves", "goalie"]]):
        stat_types = self.stat_types["goalie"]
    else:
        stat_types = self.stat_types["skater"]
        data.update(
            {f"Player {col}": player_data[f"{col}"] for col in ["position avg", "position z"]}
        )

    data.update({f"Player {col}": player_data[f"{col}"] for col in stat_types})
    data.update({f"Player {col} short": player_data[f"{col} short"] for col in stat_types})
    data.update({f"Player {col} growth": player_data[f"{col} growth"] for col in stat_types})

    defense_data = self.defenseProfile.loc[opponent]

    data.update(
        {
            "Defense " + col: defense_data[col]
            for col in defense_data.index
            if col not in (self.positions + self.stat_types["goalie"])
        }
    )

    team_data = self.teamProfile.loc[team]

    data.update(
        {
            "Team " + col: team_data[col]
            for col in team_data.index
            if col not in self.stat_types["goalie"]
        }
    )

    if any([string in market for string in ["Against", "saves", "goalie"]]):
        data["DVPOA"] = data.pop("Defense avg")
    else:
        data["DVPOA"] = self.defenseProfile.loc[opponent, position]
        if goalie in self.playerProfile:
            goalie_data = self.playerProfile.loc[goalie]
        else:
            goalie_data = self.defenseProfile.loc[opponent]

        data.update({"Goalie " + col: goalie_data[col] for col in self.stat_types["goalie"]})

    return data

# Was: StatsNHL.obs_get_training_matrix
def obs_get_training_matrix(self, market, cutoff_date=None):
    """Retrieve the training matrix for the specified market.

    Args:
        market (str): The market for which to retrieve the training data.

    Returns:
        tuple: A tuple containing the training matrix (X) and the corresponding results (y).
    """
    matrix = []

    if cutoff_date is None:
        cutoff_date = datetime.today() - timedelta(days=850)

    # Iterate over each game in the gamelog
    for _i, game in tqdm(
        self.gamelog.iterrows(),
        unit="game",
        desc="Gathering Training Data",
        total=len(self.gamelog),
    ):
        gameDate = datetime.strptime(game["gameDate"], "%Y-%m-%d").date()

        if gameDate < cutoff_date:
            continue

        if game[market] <= 0:
            continue

        self.profile_market(market, date=gameDate)
        name = game["playerName"]

        if name not in self.playerProfile.index:
            continue

        line = archive.get_line("NHL", market, game["gameDate"], name)

        offer = {
            "Player": name,
            "Team": game["team"],
            "Market": market,
            "Opponent": game["opponent"],
            "Goalie": game["opponent goalie"],
            "Home": int(game["home"]),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_get_stats = self.get_stats(offer | {"Line": line}, game["gameDate"])
            if type(new_get_stats) is dict:
                new_get_stats.update(
                    {"Result": game[market], "Date": gameDate, "Archived": int(line != 0)}
                )

                matrix.append(new_get_stats)

    M = pd.DataFrame(matrix).fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)

    return M

