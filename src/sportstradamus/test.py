from sportstradamus.helpers import Archive
import numpy as np
from scipy.stats import poisson, skellam, mode
from tqdm import tqdm

archive = Archive(True)

all_markets = {
    "MLB": [
        "hits+runs+rbi",
        "pitcher strikeouts",
        "pitching outs",
        "pitches thrown",
        "hits allowed",
        "runs allowed",
        # "1st inning runs allowed",
        # "1st inning hits allowed",
        "hitter fantasy score",
        "pitcher fantasy score",
        "hitter fantasy points underdog",
        "pitcher fantasy points underdog",
        "hitter fantasy points parlay",
        "pitcher fantasy points parlay",
        "total bases",
        "hits",
        "runs",
        "rbi",
        "singles",
        "batter strikeouts",
        "walks allowed",
    ],
    "NFL": [
        "passing yards",
        "rushing yards",
        "receiving yards",
        "yards",
        "fantasy points prizepicks",
        "fantasy points underdog",
        "fantasy points parlayplay",
        "passing tds",
        "rushing tds",
        "receiving tds",
        "tds",
        "completions",
        "carries",
        "receptions",
        "interceptions",
        "attempts",
        "targets",
    ],
    "NBA": [
        "PTS",
        "REB",
        "AST",
        "PRA",
        "PR",
        "RA",
        "PA",
        "FG3M",
        "fantasy score",
        "fantasy points parlay",
        "TOV",
        "BLK",
        "STL",
        "BLST",
        "FG3A",
        "FTM",
        "FGM",
        "FGA",
        "OREB",
        "DREB",
        "PF",
        "MIN",
    ],
    "NHL": [
        "points",
        "saves",
        "goalsAgainst",
        "shots",
        "sogBS",
        "fantasy score",
        "goalie fantasy points underdog",
        "skater fantasy points underdog",
        "goalie fantasy points parlay",
        "skater fantasy points parlay",
        "blocked",
        "hits",
        "goals",
        "assists",
        "faceOffWins",
        "timeOnIce",
    ],
}

for league in all_markets:
    for market in tqdm(all_markets[league]):
        for date in list(archive[league][market].keys()):
            for player in list(archive[league][market][date].keys()):
                archive[league][market][date][player] = {
                    k: archive[league][market][date][player][k] for k in archive[league][market][date][player] if k == 'Closing Lines' or not np.isnan(k)}

                # lines = list(archive[league][market][date][player].keys())
                # if 'Closing Lines' in lines:
                #     lines.remove('Closing Lines')

                #     line = mode([float(offer.get('Line', np.nan)) for offer in archive[league]
                #                 [market][date][player]['Closing Lines'] if offer], nan_policy='omit')[0]
                #     l = (np.ceil(line - 1), np.floor(line))
                #     odds = [None]*4
                #     for i, offer in enumerate(archive[league][market][date][player]['Closing Lines']):
                #         if offer:
                #             if " vs. " in player:
                #                 p = [skellam.cdf(l[0], offer["EV"][1], offer["EV"][0]),
                #                      skellam.sf(l[1], offer["EV"][1], offer["EV"][0])]
                #             else:
                #                 p = [poisson.cdf(l[0], offer["EV"]),
                #                      poisson.sf(l[1], offer["EV"])]
                #             push = 1 - p[1] - p[0]
                #             p[0] += push / 2
                #             p[1] += push / 2
                #             odds[i] = p[1]

                #     archive[league][market][date][player][line] = odds

                # for line in lines:
                #     if len(archive[league][market][date][player][line]) == 0:
                #         archive[league][market][date][player].pop(line)

                #     if line == 0 and " vs. " not in player:
                #         archive[league][market][date][player].pop(line)

archive.write()
