from sportstradamus.helpers import Archive

archive = Archive(True)

for market in list(archive["NFL"].keys()):
    for date in list(archive["NFL"][market].keys()):
        if "/" in date:
            archive["NFL"][market][date.replace(
                "/", "-")] = archive["NFL"][market].pop(date)

archive.write()
