import pickle
import json
import importlib.resources as pkg_resources
from sportsbook_spider import data
import pandas as pd

model_list = [f.name for f in pkg_resources.files(
    data).iterdir() if '.skl' in f.name]

report = {}
for model_str in model_list:
    with open(pkg_resources.files(data) / model_str, 'rb') as infile:
        model = pickle.load(infile)

    name = model_str.split('_')
    league = name[0]
    market = name[1].replace('-', ' ').replace('.skl', '')

    if not league in report:
        report[league] = {}

    report[league][market] = pd.DataFrame(
        model['stats'], index=model['threshold'])

with open(pkg_resources.files(data) / 'training_report.json', 'w') as outfile:
    json.dump(report, outfile, indent=4)
