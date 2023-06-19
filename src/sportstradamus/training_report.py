import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import pandas as pd

model_list = [f.name for f in pkg_resources.files(
    data).iterdir() if '.skl' in f.name]

report = {}
with open(pkg_resources.files(data) / 'training_report.txt', 'w') as f:

    for model_str in model_list:
        with open(pkg_resources.files(data) / model_str, 'rb') as infile:
            model = pickle.load(infile)

        name = model_str.split('_')
        league = name[0]
        market = name[1].replace('-', ' ').replace('.skl', '')

        f.write(f" {league} {market} ".center(72, '='))
        f.write("\n")
        f.write(pd.DataFrame(model['stats'],
                index=model['threshold']).to_string())
        f.write("\n\n")
