import importlib.resources as pkg_resources
import logging
import sys
from datetime import datetime

from sportstradamus import logs

# create logger
logger = logging.getLogger("log")
logger.setLevel(logging.INFO)

# create file handler; delay=True opens the file on the first record, so a
# process that never logs (e.g. each cron pull's fold step) leaves no empty file
fh = logging.FileHandler(
    pkg_resources.files(logs) / datetime.now().strftime("%Y_%m_%d_%H:%M:%S.log"),
    mode="w",
    delay=True,
)
fh.setLevel(logging.INFO)

# create stream handler
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.WARNING)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

# create formatter
formatter = logging.Formatter("%(asctime)s::%(levelname)s::%(filename)s::%(lineno)d - %(message)s")

# add formatter to ch
fh.setFormatter(formatter)
ch.setFormatter(logging.Formatter("%(message)s"))

# add ch to logger
logger.addHandler(fh)
logger.addHandler(ch)
