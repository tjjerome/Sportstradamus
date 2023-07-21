import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import lightgbm as lgb


model_list = [f.name for f in pkg_resources.files(
    data).iterdir() if ".mdl" in f.name]
model_list.sort()
for model_str in model_list:
    with open(pkg_resources.files(data) / model_str, "rb") as infile:
        model = pickle.load(infile)

    lgb.plot_importance(model['model'].estimator,
                        title=model_str.replace(".mdl", ""),
                        importance_type="gain")
