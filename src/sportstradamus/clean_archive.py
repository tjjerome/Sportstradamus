from sportstradamus.helpers import Archive, remove_accents, merge_dict
import numpy as np
from tqdm import tqdm

archive = Archive("All")

leagues = list(archive.archive.keys())
for league in tqdm(leagues, unit="leagues", position=0):
    markets = list(archive[league].keys())
    if "Moneyline" in markets:
        markets.remove("Moneyline")
        markets.remove("Totals")
    if "1st 1 innings" in markets:
        markets.remove("1st 1 innings")
    for market in tqdm(markets, desc=league, unit="Markets", position=1):
        # cv = stat_cv.get(league, {}).get(market, 1)
        for date in tqdm(list(archive[league][market].keys()), desc=market, unit="Gamedays", position=2):
            players = list(archive[league][market][date].keys())
            for player in players:
                if player not in archive[league][market][date]:
                    continue
                if " + " in player or " vs. " in player:
                    archive[league][market][date].pop(player)
                    continue
                if "Line" in archive[league][market][date][player]["EV"]:
                    archive[league][market][date][player]["EV"].pop("Line")

                player_name = remove_accents(player)

                if player_name != player:
                    names = [player, player_name]
                else:
                    names = [player_name]

                for name in names:
                    if name not in archive[league][market][date]:
                        continue

                    if not type(archive[league][market][date][name]) is dict:
                        archive[league][market][date].pop(name)
                    
                    if not type(archive[league][market][date][name].get("EV", {})) is dict:
                        if len(archive[league][market][date][name]["EV"]) != 4:
                            archive[league][market][date][name]["EV"] = {}
                        else:
                            ev = {}
                            for i, book in enumerate(["draftkings", "fanduel", "pinnacle", "williamhill_us"]):
                                v = archive[league][market][date][name]["EV"][i]
                                if v is None:
                                    v = np.nan
                                ev.update({book: v})

                            archive[league][market][date][name]["EV"] = ev

                    archive[league][market][date][name]["Lines"] = [line for line in archive[league][market][date][name]["Lines"] if line]

                if player_name != player:
                    archive[league][market][date][player_name] = merge_dict(archive[league][market][date].get(player_name,{}), archive[league][market][date].pop(player))

                if not len(archive[league][market][date][player_name]["EV"]) and not len(archive[league][market][date][player_name]["Lines"]):
                    archive[league][market][date].pop(player_name)

            if not len(archive[league][market][date]):
                archive[league][market].pop(date)

        if not len(archive[league][market]):
            archive[league].pop(market)


archive.write(True)