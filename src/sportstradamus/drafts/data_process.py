# %%
import pandas as pd
# from sportstradamus.drafts import data
import data
import importlib.resources as pkg_resources
import numpy as np
from datetime import datetime
import nfl_data_py as nfl

# %%
rosters = nfl.import_rosters([2021, 2022])
rosters = rosters.loc[rosters.position.isin(['QB', 'WR', 'TE', 'RB'])]
rosters = rosters[['player_name', 'season', 'team']]
rosters = rosters.set_index(
    rosters['player_name'] + ', ' + rosters['season'].astype(str))
rosters = rosters.to_dict()['team']

# %%
schedules = nfl.import_schedules([2021, 2022])
schedules = schedules.loc[schedules.week.isin([15, 16, 17])]
schedules = schedules[['season', 'week', 'away_team', 'home_team']]

# %%
data_list = [f.name for f in pkg_resources.files(
    data).iterdir() if ".csv" in f.name and "Regular" in f.name]
# %%
df = pd.DataFrame()
for data_str in data_list:
    if df.empty:
        df = pd.read_csv(pkg_resources.files(data) / data_str)
    else:
        df = pd.concat([df, pd.read_csv(pkg_resources.files(data) / data_str)])

# %%
df['season'] = df['draft_time'].str[:4]
df['team'] = (df['player_name'] + ', ' + df['season']
              ).map(rosters).fillna("None")
# %%
picks = df.overall_pick_number.replace(0, 216).to_numpy()
value = df.pick_points.to_numpy()
fit = np.polyfit(np.log(picks), value, 1)
df['draft_capital_spent'] = fit[0]*np.log(df['overall_pick_number'])+fit[1]
df['projection_draft_capital'] = fit[0] * \
    np.log(df['projection_adp'].replace(0, 216))+fit[1]
df['draft_capital_gained'] = df['projection_draft_capital'] - \
    df['draft_capital_spent']

# %%
draft_date = pd.to_datetime(df['draft_time'].str[:10], format='%Y-%m-%d')
df['days_until_season_start'] = (datetime.strptime(
    '2022-09-08', '%Y-%m-%d') - draft_date).dt.days
df.loc[df['days_until_season_start'] > 200, 'days_until_season_start'] = \
    df.loc[df['days_until_season_start'] >
           200, 'days_until_season_start'] - 363

# %%
teams = df.groupby(['draft_id', 'pick_order'])
teams = teams[['player_name', 'position_name', 'team_pick_number', 'roster_points', 'draft_capital_spent',
               'draft_capital_gained', 'days_until_season_start']]

for order, team in teams:
    print(order[1])
    print(team['player_name'])

# %%
