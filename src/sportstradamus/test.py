from sportstradamus.helpers import Archive
import pickle

archive = Archive(True)

for market in list(archive["NFL"].keys()):
    if market not in ["Moneyline", "Totals"]:
        for date in list(archive["NFL"][market].keys()):
            for player in list(archive["NFL"][market][date].keys()):
                for line in list(archive["NFL"][market][date][player].keys()):
                    if len(archive["NFL"][market][date][player][line]) > 4:
                        archive["NFL"][market][date][player][line] = archive["NFL"][market][date][player][line][-4:]

with open('sportstradamus/data/archive_full.dat', 'wb') as outfile:
    pickle.dump(archive.archive, outfile)

archive.clip()
with open('sportstradamus/data/archive.dat', 'wb') as outfile:
    pickle.dump(archive.archive, outfile)
