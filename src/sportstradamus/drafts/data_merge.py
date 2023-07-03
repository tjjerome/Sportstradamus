# %%
import pandas as pd
from sportstradamus.drafts import data
import importlib.resources as pkg_resources
import numpy as np
from tqdm import tqdm
# %%
data_list = [f.name for f in pkg_resources.files(
    data).iterdir() if "teams.csv" in f.name]

# %%
df = pd.DataFrame()
for data_str in tqdm(data_list, desc='Loading Data Files', unit='file'):
    if df.empty:
        df = pd.read_csv(pkg_resources.files(data) / data_str)
    else:
        df = pd.concat([df, pd.read_csv(pkg_resources.files(data) / data_str)])

# %%
# df = df.sort_values(['draft_time', 'draft_id', 'pick_order']
#                     ).reset_index(drop=True)


# %%
# for i, item in enumerate(np.split(df, 2)):
#     item.to_csv(pkg_resources.files(data) /
#                 f"BBM_II_Regular_Season_Dump_Part_{i+1}.csv")

# %%
df.to_csv(pkg_resources.files(data) / "BBM_teams.csv")
