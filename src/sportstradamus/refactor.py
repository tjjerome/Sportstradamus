from sportstradamus.helpers import Archive
from datetime import datetime
from tqdm import tqdm

archive = Archive(True)

for market in ["pitcher fantasy score", "pitcher fantasy points underdog"]:
    for date in tqdm(list(archive["MLB"][market].keys())):
        if datetime.strptime(date, "%Y-%m-%d") > datetime(2023, 6, 13):
            continue
        for player in list(archive["MLB"][market][date].keys()):
            archive["MLB"][market][date].pop(player)

archive.write()
