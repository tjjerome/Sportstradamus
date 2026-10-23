"""The shared ``log`` logger: INFO to ``sportstradamus/logs/<timestamp>.log``, WARNING+ to stdout.

Importing this module installs both handlers and a ``sys.excepthook`` that logs an uncaught
exception at CRITICAL.
"""

import importlib.resources as pkg_resources
import logging
import sys
from datetime import datetime

from sportstradamus import logs

logger = logging.getLogger("log")
logger.setLevel(logging.INFO)

# delay=True opens the file on the first record, so a process that never logs
# (e.g. each cron pull's fold step) leaves no empty file
fh = logging.FileHandler(
    pkg_resources.files(logs) / datetime.now().strftime("%Y_%m_%d_%H:%M:%S.log"),
    mode="w",
    delay=True,
)
fh.setLevel(logging.INFO)

ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.WARNING)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

formatter = logging.Formatter("%(asctime)s::%(levelname)s::%(filename)s::%(lineno)d - %(message)s")

fh.setFormatter(formatter)
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)
