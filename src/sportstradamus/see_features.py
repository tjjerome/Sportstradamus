import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import lightgbm as lgb
import os.path
import pandas as pd

model_list = [f.name for f in pkg_resources.files(
    data).iterdir() if ".mdl" in f.name]
model_list.sort()

data_list = []
for model_str in model_list:
    with open(pkg_resources.files(data) / model_str, "rb") as infile:
        model = pickle.load(infile)

    lgb.plot_importance(model['model'].estimator)

    # size = os.path.getsize(pkg_resources.files(data) / model_str) / 1024
    # data_list.append(model['params'] | {"size": size})

    # print(f"{model_str} - Filesize: {size} KB, Max Depth: {depth}, Num Leaves: {width}")

# df = pd.DataFrame(data_list)
# df.corr()
