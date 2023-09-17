from sportstradamus.helpers import Archive, merge_dict
from sportstradamus.stats import StatsNBA
from datetime import datetime
from tqdm import tqdm
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json
import numpy as np
import pickle

archive = Archive()

filepath = pkg_resources.files(data) / "remote/archive.dat"
with open(filepath, 'rb') as infile:
    remote_archive = pickle.load(infile)

archive.archive = merge_dict(remote_archive, archive.archive)

archive.write()
