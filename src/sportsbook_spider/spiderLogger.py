import logging
import datetime
import importlib.resources as pkg_resources
from sportsbook_spider import logs

# create logger
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

# create file handler
fh = logging.FileHandler(pkg_resources.files(
    logs) / datetime.datetime.now().strftime("%Y_%m_%d:%H:%M:%S.log"), mode='w')
fh.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter(
    "%(asctime)s::%(levelname)s::%(filename)s::%(lineno)d::%(message)s")

# add formatter to ch
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(fh)
