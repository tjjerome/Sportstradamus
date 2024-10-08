from sportstradamus.helpers import scraper, Archive, no_vig_odds
from sportstradamus.stats import StatsNFL
from datetime import datetime, timedelta
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np

nba_market_ids = {
    "RA": 337,
    "REB": 157,
    "AST": 151,
    "PRA": 338,
    "PA": 335,
    "PTS": 156,
    "PR": 336,
    "BLK": 152,
    "STL": 160,
    "FG3M": 162,
}

mlb_market_ids = {
    "runs allowed": 290,
    "pitcher strikeouts": 285,
    "hits": 287,
    "runs": 288,
    "rbi": 289,
    "total bases": 293,
    "singles": 295,
}

nhl_market_ids = {
    "goals": 318,
    "points": 319,
    "assists": 320,
    "shots": 321,
    "saves": 322,
}

nfl_market_ids = {
    # "passing_tds": 102,
    # "passing_yards": 103,
    # "completions": 100,
    # "attempts": 333,
    # "interceptions": 101,
    # "carries": 106,
    # "rushing_yards": 107,
    # "receiving_yards": 105,
    # "receptions": 104,
    "tds": 78
}

header = {"X-API-Key": "CHi8Hy5CEE4khd46XNYL23dCFX96oUdw6qOt1Dnh"}

nfl = StatsNFL()
nfl.load()
nfl.update()

archive = Archive()
with logging_redirect_tqdm():
    players = list(nfl.gamelog['player display name'].unique())
    for player in tqdm(players, unit="player", position=1):
        tryJr = False
        code = player.lower().replace(".", "").replace("'", "").replace(" ", "-")
        player_games = nfl.gamelog.loc[nfl.gamelog['player display name'] == player]

        for market, mid in tqdm(
            list(nfl_market_ids.items()), unit="market", position=2, leave=False
        ):
            url = f"https://api.bettingpros.com/v3/props/analysis?include_no_line_events=false&player_slug={code}&market_id={mid}&location=ALL&sort=desc&sport=NFL&limit=1000"
            try:
                res = scraper.get(url, headers=header, max_attempts=2)
                props = [
                    (i["event"], i["propOffer"])
                    for i in res["analyses"]
                    if i["propOffer"]["season"] <= 2022
                    and i["propOffer"]["line"]
                ]
            except:
                if tryJr:
                    tqdm.write(f"Error finding {player}, {market}")
                    continue
                else:
                    tryJr = True
                    if "-jr" in code:
                        code = code.replace("-jr", "")
                    elif "-iii" in code:
                        code = code.replace("-iii", "")
                    elif "-iv" in code:
                        code = code.replace("-iv", "")
                    else:
                        code = code + "-jr"

                    url = f"https://api.bettingpros.com/v3/props/analysis?include_no_line_events=false&player_slug={code}&market_id={mid}&location=ALL&sort=desc&sport=mlb&limit=1000"
                    try:
                        res = scraper.get(url, headers=header, max_attempts=2)
                        props = [
                            (i["event"], i["propOffer"])
                            for i in res["analyses"]
                            if i["propOffer"]["season"] <= 2022
                            and i["propOffer"]["line"]
                        ]
                    except:
                        tqdm.write(f"Error finding {player}, {market}")
                        continue

            for event, prop in props:
                line = prop["line"]
                date = (
                    datetime.strptime(event["scheduled"], "%Y-%m-%d %H:%M:%S")
                    - timedelta(hours=5)
                ).strftime("%Y-%m-%d")
                odds = [prop["cost"], prop["cost_inverse"]]
                if 0 in odds:
                    odds.remove(0)
                    odds = no_vig_odds(odds[0])
                else:
                    if prop["recommendation"] == "under":
                        odds.reverse()
                    odds = no_vig_odds(odds[0], odds[1])
                # game = next(
                #     (i for i in player_games if i["gameId"][:10] == date), None)
                # if game is None:
                #     tqdm.write(f"Error finding {player}, {market}, {date}")
                #     continue
                # stats = nfl.get_stats_date(game, market, line)
                stats = [odds[0]] * 4
                if market not in archive.archive["NFL"]:
                    archive["NFL"][market] = {}
                if date not in archive.archive["NFL"][market]:
                    archive.archive["NFL"][market][date] = {}
                if player not in archive.archive["NFL"][market][date]:
                    archive.archive["NFL"][market][date][player] = {
                        'Line': [line], 'EV': stats}


archive.write()
